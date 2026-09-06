"""Bộ điều phối đánh giá phễu hoàn chỉnh (Multi-Stage Funnel Evaluator).

Phễu đánh giá:
Test Users
   │
   ▼
Target exists in train catalog? (target_in_catalog_rate)
   │
   ▼
Retrieved in Candidates? (candidate_recall@50, @100, @200)
   │
   ▼
Ranked in top positions? (ranker_recall@10, ranker_ndcg@10)
   │
   ▼
Final Top-10 post MMR? (final_recall@10, final_ndcg@10, ILD, coverage)
"""

from __future__ import annotations

import time
from typing import Any, Sequence
import numpy as np

from .latency import summarize_latencies
from .metrics import (
    compute_long_tail_distribution,
    compute_user_coverage,
    dcg,
    hit_rate_at_k,
    intra_list_diversity,
    mrr_at_k,
    novelty_at_k,
)
from .ranking_metrics import ranker_ndcg_at_k, ranker_recall_at_k
from .retrieval_metrics import candidate_recall_at_k, target_in_catalog_rate


class FullFunnelEvaluator:
    """Đánh giá toàn diện kiến trúc 2 tầng theo phễu từng giai đoạn và so sánh đối chuẩn."""

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

        self.total_catalog_size = len(self.catalog_items)
        n_pop = len(self.popular_items)
        head_cut = max(1, int(n_pop * 0.10))
        mid_cut = max(head_cut + 1, int(n_pop * 0.50))
        self.head_set = set(self.popular_items[:head_cut])
        self.mid_set = set(self.popular_items[head_cut:mid_cut])
        self.tail_set = set(self.popular_items[mid_cut:])

        total_int = sum(self.popularity_counts.values())
        self.catalog_prob = {
            item_id: float((self.popularity_counts.get(item_id, 0) + 1) / (total_int + self.total_catalog_size))
            for item_id in self.catalog_items
        }

    def evaluate(
        self,
        test_truth: dict[int, int],
        max_users: int | None = None,
        k: int = 10,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Thực hiện đánh giá trên toàn bộ người dùng tập Test trong một lượt duy nhất (Single-Pass)."""
        eligible_users = [
            u for u in test_truth
            if (hasattr(self.engine, "user_map") and u in self.engine.user_map)
            or (hasattr(self.engine, "retriever") and u in getattr(self.engine.retriever, "user_map", {}))
        ]

        if max_users is not None and len(eligible_users) > max_users:
            rng = np.random.default_rng(seed)
            eligible_users = list(rng.choice(eligible_users, size=max_users, replace=False))

        eval_users_count = len(eligible_users)
        test_targets = [test_truth[u] for u in eligible_users]
        in_catalog_rate = target_in_catalog_rate(test_targets, self.catalog_items)

        # Stage 1 metrics
        cand_recall_50: list[float] = []
        cand_recall_100: list[float] = []
        cand_recall_200: list[float] = []

        # Stage 2 metrics
        rank_recall_10: list[float] = []
        rank_ndcg_10: list[float] = []

        # Stage 3 Final metrics
        final_recalls: list[float] = []
        final_ndcgs: list[float] = []
        final_mrrs: list[float] = []
        final_ilds: list[float] = []
        final_novelties: list[float] = []

        all_final_recs: list[list[int]] = []
        recommended_unique: set[int] = set()

        # Popularity baseline metrics
        pop_recalls: list[float] = []
        pop_ndcgs: list[float] = []
        pop_mrrs: list[float] = []
        pop_novelties: list[float] = []

        # Latencies
        t_ret_list: list[float] = []
        t_rank_list: list[float] = []
        t_div_list: list[float] = []
        t_tot_list: list[float] = []

        cand_k = int(getattr(self.engine, "config", {}).get("candidate_k", 200))
        rerank_pool_k = int(getattr(self.engine, "rerank_pool_k", 40))

        for u_id in eligible_users:
            true_item = test_truth[u_id]

            # Stage 1: Candidate Retrieval
            t0 = time.perf_counter()
            candidates = self.engine.retriever.retrieve(u_id, k=cand_k, filter_seen=True)
            t_ret = (time.perf_counter() - t0) * 1000.0
            t_ret_list.append(t_ret)

            cand_ids = [c.item_id for c in candidates]
            cand_recall_50.append(candidate_recall_at_k(cand_ids, true_item, k=50))
            cand_recall_100.append(candidate_recall_at_k(cand_ids, true_item, k=100))
            cand_recall_200.append(candidate_recall_at_k(cand_ids, true_item, k=200))

            # Stage 2: Feature Building & Ranking
            t1 = time.perf_counter()
            feat_mat = self.engine.feature_builder.build_feature_matrix(candidates, user_id=u_id)
            features = self.engine.feature_builder.build_features(candidates, user_id=u_id)
            ranked_cands = self.engine.ranker.rank(features, feature_matrix=feat_mat)
            t_rank = (time.perf_counter() - t1) * 1000.0
            t_rank_list.append(t_rank)

            ranked_ids = [r.item_id for r in ranked_cands]
            rank_recall_10.append(ranker_recall_at_k(ranked_ids, true_item, k=k))
            rank_ndcg_10.append(ranker_ndcg_at_k(ranked_ids, true_item, k=k))

            # Stage 3: MMR Diversity Reranking với rerank_pool_k đồng nhất
            t2 = time.perf_counter()
            final_recs = self.engine.diversity_reranker.rerank(
                ranked_cands,
                k=k,
                rerank_pool_k=rerank_pool_k,
            )
            t_div = (time.perf_counter() - t2) * 1000.0
            t_div_list.append(t_div)
            t_tot_list.append(t_ret + t_rank + t_div)

            items = [r.item_id for r in final_recs]
            all_final_recs.append(items)
            recommended_unique.update(items)

            # Final metrics
            hit = hit_rate_at_k(items, true_item)
            final_recalls.append(hit)
            final_ndcgs.append(dcg(items.index(true_item)) if hit else 0.0)
            final_mrrs.append(mrr_at_k(items, true_item))
            final_ilds.append(intra_list_diversity(items, self.genre_map))
            final_novelties.append(novelty_at_k(items, self.catalog_prob))

            # Baseline Popularity
            seen = self.engine.seen_by_user.get(u_id, set())
            pop_preds = [it for it in self.popular_items if it not in seen][:k]
            pop_hit = hit_rate_at_k(pop_preds, true_item)
            pop_recalls.append(pop_hit)
            pop_ndcgs.append(dcg(pop_preds.index(true_item)) if pop_hit else 0.0)
            pop_mrrs.append(mrr_at_k(pop_preds, true_item))
            pop_novelties.append(novelty_at_k(pop_preds, self.catalog_prob))

        exposure = compute_long_tail_distribution(all_final_recs, self.head_set, self.mid_set, self.tail_set)
        user_cov = compute_user_coverage(all_final_recs, k=k)

        final_rec = float(np.mean(final_recalls)) if final_recalls else 0.0
        final_ndcg = float(np.mean(final_ndcgs)) if final_ndcgs else 0.0
        final_mrr = float(np.mean(final_mrrs)) if final_mrrs else 0.0
        pop_rec = float(np.mean(pop_recalls)) if pop_recalls else 0.0
        pop_ndcg = float(np.mean(pop_ndcgs)) if pop_ndcgs else 0.0
        pop_mrr = float(np.mean(pop_mrrs)) if pop_mrrs else 0.0

        abs_gain_recall = final_rec - pop_rec
        rel_lift_recall = (abs_gain_recall / pop_rec * 100.0) if pop_rec > 0 else 0.0
        abs_gain_ndcg = final_ndcg - pop_ndcg
        rel_lift_ndcg = (abs_gain_ndcg / pop_ndcg * 100.0) if pop_ndcg > 0 else 0.0

        return {
            "evaluation_protocol": "per_user_temporal_holdout",
            "evaluated_users": eval_users_count,
            "target_in_catalog_rate": in_catalog_rate,
            "cold_item_test_share": 1.0 - in_catalog_rate,
            "funnel_stage_metrics": {
                "stage_1_retrieval": {
                    "candidate_recall@50": float(np.mean(cand_recall_50)),
                    "candidate_recall@100": float(np.mean(cand_recall_100)),
                    "candidate_recall@200": float(np.mean(cand_recall_200)),
                },
                "stage_2_ranking": {
                    "ranker_recall@10": float(np.mean(rank_recall_10)),
                    "ranker_ndcg@10": float(np.mean(rank_ndcg_10)),
                },
                "stage_3_final_post_mmr": {
                    f"recall@{k}": final_rec,
                    f"ndcg@{k}": final_ndcg,
                    f"mrr@{k}": final_mrr,
                    "catalog_coverage": float(len(recommended_unique) / max(1, self.total_catalog_size)),
                    "user_coverage": user_cov,
                    "intra_list_diversity": float(np.mean(final_ilds)),
                    f"novelty@{k}": float(np.mean(final_novelties)),
                    "long_tail_exposure": exposure,
                },
            },
            "popularity_baseline": {
                f"recall@{k}": pop_rec,
                f"ndcg@{k}": pop_ndcg,
                f"mrr@{k}": pop_mrr,
                f"novelty@{k}": float(np.mean(pop_novelties)),
            },
            "model_lift_over_popularity": {
                "absolute_gain": {
                    f"recall@{k}": abs_gain_recall,
                    f"ndcg@{k}": abs_gain_ndcg,
                    f"mrr@{k}": final_mrr - pop_mrr,
                },
                "relative_lift_percent": {
                    f"recall@{k}": rel_lift_recall,
                    f"ndcg@{k}": rel_lift_ndcg,
                },
                # Giữ nguyên trường phẳng cũ cho tương thích ngược
                f"recall@{k}": abs_gain_recall,
                f"ndcg@{k}": abs_gain_ndcg,
                f"mrr@{k}": final_mrr - pop_mrr,
            },
            "latencies_ms": {
                "retrieval": summarize_latencies(t_ret_list).__dict__,
                "ranking": summarize_latencies(t_rank_list).__dict__,
                "diversity": summarize_latencies(t_div_list).__dict__,
                "total": summarize_latencies(t_tot_list).__dict__,
            },
        }
