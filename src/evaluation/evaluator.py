"""Bộ đánh giá phễu retrieval, ranking và MMR."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..data.split import seen_items_before
from ..ranking.scorer import rank_candidates, retrieval_order
from ..utils import safe_mean
from .latency import summarize_latencies
from .metrics import (
    candidate_recall_at_k,
    compute_long_tail_distribution,
    compute_user_coverage,
    dcg,
    hit_rate_at_k,
    intra_list_diversity,
    mrr_at_k,
    novelty_at_k,
    ranker_ndcg_at_k,
    ranker_recall_at_k,
    target_in_catalog_rate,
)


class FullFunnelEvaluator:
    """Đánh giá chi tiết từng tầng trên cùng một tập người dùng."""

    def __init__(
        self,
        engine: Any,
        catalog_items: Sequence[int],
        popular_items: Sequence[int],
        popularity_counts: dict[int, int],
        genre_map: dict[int, set[str]],
    ) -> None:
        self.engine = engine
        self.catalog_items = set(catalog_items)
        self.popular_items = list(popular_items)
        self.popularity_counts = popularity_counts
        self.genre_map = genre_map

        catalog_size = len(self.catalog_items)
        head_cut = max(1, int(len(self.popular_items) * 0.10))
        mid_cut = max(head_cut + 1, int(len(self.popular_items) * 0.50))
        self.head_set = set(self.popular_items[:head_cut])
        self.mid_set = set(self.popular_items[head_cut:mid_cut])
        self.tail_set = set(self.popular_items[mid_cut:])

        total_interactions = sum(self.popularity_counts.values())
        self.catalog_prob = {
            item_id: float(
                (self.popularity_counts.get(item_id, 0) + 1)
                / (total_interactions + catalog_size)
            )
            for item_id in self.catalog_items
        }

    def evaluate(
        self,
        test_truth: dict[int, int],
        max_users: int | None = None,
        k: int = 10,
        seed: int = 42,
        test_events: pd.DataFrame | None = None,
        history_df: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Đánh giá retrieval, ranking, MMR và popularity baseline."""
        eligible_users = [
            user_id
            for user_id in test_truth
            if (hasattr(self.engine, "user_map") and user_id in self.engine.user_map)
            or (
                hasattr(self.engine, "retriever")
                and user_id in getattr(self.engine.retriever, "user_map", {})
            )
        ]
        if max_users is not None and len(eligible_users) > max_users:
            rng = np.random.default_rng(seed)
            eligible_users = list(
                rng.choice(eligible_users, size=max_users, replace=False)
            )

        test_targets = [test_truth[user_id] for user_id in eligible_users]
        in_catalog_rate = target_in_catalog_rate(test_targets, self.catalog_items)
        candidate_recalls = {50: [], 100: [], 200: []}
        rank_recalls: list[float] = []
        rank_ndcgs: list[float] = []
        final_recalls: list[float] = []
        final_ndcgs: list[float] = []
        final_mrrs: list[float] = []
        final_ilds: list[float] = []
        final_novelties: list[float] = []
        popularity_recalls: list[float] = []
        popularity_ndcgs: list[float] = []
        popularity_mrrs: list[float] = []
        popularity_novelties: list[float] = []
        conditional_rank_ndcgs: list[float] = []
        conditional_rank_mrrs: list[float] = []
        conditional_final_ndcgs: list[float] = []
        conditional_final_mrrs: list[float] = []
        all_final_recs: list[list[int]] = []
        recommended_unique: set[int] = set()
        source_hits = {"svd": 0, "popularity": 0, "genre": 0}
        source_rescues = {"popularity": 0, "genre": 0}
        target_retrieved = 0
        retrieval_times: list[float] = []
        ranking_times: list[float] = []
        diversity_times: list[float] = []
        total_times: list[float] = []

        candidate_k = int(getattr(self.engine, "config", {}).get("candidate_k", 200))
        rerank_pool_k = int(getattr(self.engine, "rerank_pool_k", 40))

        for user_id in eligible_users:
            target_item = test_truth[user_id]
            start = time.perf_counter()
            timestamp = _target_timestamp(test_events, user_id, target_item)
            if history_df is not None and timestamp is not None:
                seen_items = seen_items_before(history_df, user_id, timestamp)
            else:
                seen_items = self.engine.seen_by_user.get(user_id, set())
            candidates = self.engine.retriever.retrieve(
                user_id,
                k=candidate_k,
                filter_seen=True,
                seen_items_override=seen_items,
            )
            retrieval_times.append((time.perf_counter() - start) * 1000.0)

            candidate_ids = [candidate.item_id for candidate in candidates]
            matched = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.item_id == target_item
                ),
                None,
            )
            if matched is not None:
                target_retrieved += 1
                sources = set(matched.source_scores)
                for source in source_hits:
                    source_hits[source] += source in sources
                if "svd" not in sources:
                    for source in source_rescues:
                        source_rescues[source] += source in sources
            for cutoff, values in candidate_recalls.items():
                values.append(candidate_recall_at_k(candidate_ids, target_item, cutoff))

            start = time.perf_counter()
            features = self.engine.feature_builder.build_features(
                candidates,
                user_id=user_id,
                as_of_timestamp=timestamp,
            )
            if (
                getattr(self.engine, "ranker_enabled", False)
                and getattr(self.engine, "learned_ranker", None) is not None
            ):
                ranked_candidates = rank_candidates(
                    features, self.engine.learned_ranker
                )
            else:
                ranked_candidates = retrieval_order(features)
            ranking_times.append((time.perf_counter() - start) * 1000.0)

            ranked_ids = [candidate.item_id for candidate in ranked_candidates]
            rank_recalls.append(ranker_recall_at_k(ranked_ids, target_item, k))
            rank_ndcgs.append(ranker_ndcg_at_k(ranked_ids, target_item, k))
            if matched is not None:
                conditional_rank_ndcgs.append(
                    ranker_ndcg_at_k(ranked_ids, target_item, k)
                )
                conditional_rank_mrrs.append(_mrr_from_ids(ranked_ids, target_item, k))

            start = time.perf_counter()
            final_candidates = self.engine.diversity_reranker.rerank(
                ranked_candidates,
                k=k,
                rerank_pool_k=rerank_pool_k,
            )
            diversity_time = (time.perf_counter() - start) * 1000.0
            diversity_times.append(diversity_time)
            total_times.append(retrieval_times[-1] + ranking_times[-1] + diversity_time)

            final_ids = [candidate.item_id for candidate in final_candidates]
            all_final_recs.append(final_ids)
            recommended_unique.update(final_ids)
            hit = hit_rate_at_k(final_ids, target_item)
            final_recalls.append(hit)
            final_ndcgs.append(dcg(final_ids.index(target_item)) if hit else 0.0)
            final_mrrs.append(mrr_at_k(final_ids, target_item))
            final_ilds.append(intra_list_diversity(final_ids, self.genre_map))
            final_novelties.append(novelty_at_k(final_ids, self.catalog_prob))
            if matched is not None:
                conditional_final_ndcgs.append(
                    ranker_ndcg_at_k(final_ids, target_item, k)
                )
                conditional_final_mrrs.append(_mrr_from_ids(final_ids, target_item, k))

            popularity_ids = [
                item for item in self.popular_items if item not in seen_items
            ][:k]
            pop_hit = hit_rate_at_k(popularity_ids, target_item)
            popularity_recalls.append(pop_hit)
            popularity_ndcgs.append(
                dcg(popularity_ids.index(target_item)) if pop_hit else 0.0
            )
            popularity_mrrs.append(mrr_at_k(popularity_ids, target_item))
            popularity_novelties.append(novelty_at_k(popularity_ids, self.catalog_prob))

        mean_final_recall = _mean(final_recalls)
        mean_final_ndcg = _mean(final_ndcgs)
        mean_final_mrr = _mean(final_mrrs)
        mean_pop_recall = _mean(popularity_recalls)
        mean_pop_ndcg = _mean(popularity_ndcgs)
        mean_pop_mrr = _mean(popularity_mrrs)
        exposure = compute_long_tail_distribution(
            all_final_recs, self.head_set, self.mid_set, self.tail_set
        )

        return {
            "evaluation_protocol": "per_user_temporal_holdout",
            "evaluated_users": len(eligible_users),
            "target_in_catalog_rate": in_catalog_rate,
            "cold_item_test_share": 1.0 - in_catalog_rate,
            "funnel_stage_metrics": {
                "stage_1_retrieval": {
                    "candidate_recall@50": _mean(candidate_recalls[50]),
                    "candidate_recall@100": _mean(candidate_recalls[100]),
                    "candidate_recall@200": _mean(candidate_recalls[200]),
                    "target_retrieved_users": target_retrieved,
                    "source_contribution": {
                        source: count / max(1, target_retrieved)
                        for source, count in source_hits.items()
                    },
                    "source_rescue_contribution": {
                        source: count / max(1, target_retrieved)
                        for source, count in source_rescues.items()
                    },
                },
                "stage_2_ranking": {
                    "ranker_recall@10": _mean(rank_recalls),
                    "ranker_ndcg@10": _mean(rank_ndcgs),
                    "conditional_ndcg@10": _mean(conditional_rank_ndcgs),
                    "conditional_mrr@10": _mean(conditional_rank_mrrs),
                },
                "stage_3_final_post_mmr": {
                    f"recall@{k}": mean_final_recall,
                    f"ndcg@{k}": mean_final_ndcg,
                    f"mrr@{k}": mean_final_mrr,
                    "catalog_coverage": float(
                        len(recommended_unique) / max(1, len(self.catalog_items))
                    ),
                    "user_coverage": compute_user_coverage(all_final_recs, k=k),
                    "intra_list_diversity": _mean(final_ilds),
                    f"novelty@{k}": _mean(final_novelties),
                    "conditional_ndcg@10": _mean(conditional_final_ndcgs),
                    "conditional_mrr@10": _mean(conditional_final_mrrs),
                    "long_tail_exposure": exposure,
                },
            },
            "popularity_baseline": {
                f"recall@{k}": mean_pop_recall,
                f"ndcg@{k}": mean_pop_ndcg,
                f"mrr@{k}": mean_pop_mrr,
                f"novelty@{k}": _mean(popularity_novelties),
            },
            "model_lift_over_popularity": {
                "absolute_gain": {
                    f"recall@{k}": mean_final_recall - mean_pop_recall,
                    f"ndcg@{k}": mean_final_ndcg - mean_pop_ndcg,
                    f"mrr@{k}": mean_final_mrr - mean_pop_mrr,
                },
                "relative_lift_percent": {
                    f"recall@{k}": _relative_lift(mean_final_recall, mean_pop_recall),
                    f"ndcg@{k}": _relative_lift(mean_final_ndcg, mean_pop_ndcg),
                },
            },
            "latencies_ms": {
                "retrieval": summarize_latencies(retrieval_times).__dict__,
                "ranking": summarize_latencies(ranking_times).__dict__,
                "diversity": summarize_latencies(diversity_times).__dict__,
                "total": summarize_latencies(total_times).__dict__,
            },
        }


_mean = safe_mean


def _relative_lift(value: float, baseline: float) -> float:
    """Tính phần trăm cải thiện, tránh chia cho 0."""
    return (value - baseline) / baseline * 100.0 if baseline > 0 else 0.0


def _target_timestamp(
    events: pd.DataFrame | None,
    user_id: int,
    item_id: int,
) -> int | None:
    """Tìm timestamp của target để dựng seen state đúng thời điểm."""
    if events is None or events.empty:
        return None
    matches = events[(events["user_id"] == user_id) & (events["item_id"] == item_id)]
    if matches.empty or "timestamp" not in matches:
        return None
    return int(matches.iloc[0]["timestamp"])


def _mrr_from_ids(items: Sequence[int], target_item: int, k: int) -> float:
    """Tính MRR trên danh sách đã giới hạn top-k."""
    return mrr_at_k(list(items)[:k], target_item)
