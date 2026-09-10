"""Chạy một experiment offline thống nhất cho toàn bộ funnel recommender."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .data import load_ratings, seen_items_before, temporal_split
from .evaluation.latency import summarize_latencies
from .evaluation.metrics import (
    compute_user_coverage,
    hit_rate_at_k,
    intra_list_diversity,
    mrr_at_k,
    novelty_at_k,
)
from .evaluation.retrieval_metrics import (
    candidate_recall_at_k,
    cold_item_test_share,
    target_in_catalog_rate,
)
from .ranking.scorer import rank_candidates, retrieval_order
from .serving.recommender import Recommender
from .utils import LOGGER, save_json, setup_logging


def _mean(values: list[float]) -> float:
    """Tính trung bình an toàn cho một metric theo user."""
    return float(np.mean(values)) if values else 0.0


def _ndcg_at_k(predictions: list[int], target: int, k: int) -> float:
    """Tính nDCG cho leave-one-out ground truth."""
    if target not in predictions[:k]:
        return 0.0
    rank = predictions.index(target)
    return float(1.0 / np.log2(rank + 2))


def _variant_metrics(
    predictions: list[list[int]],
    targets: list[int],
    genres: dict[int, set[str]],
    catalog: set[int],
    item_probabilities: dict[int, float],
    latencies: list[float],
    k: int,
) -> dict[str, float]:
    """Tổng hợp metric cho một variant trên đúng cùng danh sách user."""
    latency = summarize_latencies(latencies)
    pairs = list(zip(predictions, targets, strict=True))
    return {
        f"recall@{k}": _mean([hit_rate_at_k(preds, target) for preds, target in pairs]),
        f"ndcg@{k}": _mean([_ndcg_at_k(preds, target, k) for preds, target in pairs]),
        f"mrr@{k}": _mean([mrr_at_k(preds, target) for preds, target in pairs]),
        "catalog_coverage": len(set().union(*map(set, predictions))) / len(catalog)
        if catalog
        else 0.0,
        "user_coverage": compute_user_coverage(predictions, k),
        "intra_list_diversity": _mean(
            [intra_list_diversity(preds, genres) for preds in predictions]
        ),
        f"novelty@{k}": _mean(
            [novelty_at_k(preds, item_probabilities) for preds in predictions]
        ),
        "p50_latency_ms": latency.p50_ms,
        "p95_latency_ms": latency.p95_ms,
    }


def run_experiment(
    k: int = 10,
    max_users: int | None = None,
    output_path: str | Path = "reports/experiment.json",
    seed: int = 42,
) -> dict[str, Any]:
    """Đánh giá mọi stage trên cùng split, user set và model artifact.

    Dev selection đã được thực hiện trong ``train.py``. Hàm này chỉ chạy locked
    test một lần với các baseline/funnel variant để tránh ghép số liệu khác run.
    """
    setup_logging()
    ratings = load_ratings()
    _, _, _, test_df = temporal_split(ratings)
    recommender = Recommender()

    test_events = {
        int(row.user_id): (int(row.item_id), int(row.timestamp))
        for row in test_df.itertuples(index=False)
        if int(row.user_id) in recommender.user_map
    }
    eligible_users = sorted(test_events)
    if max_users is not None and len(eligible_users) > max_users:
        rng = np.random.default_rng(seed)
        eligible_users = sorted(
            int(user_id)
            for user_id in rng.choice(eligible_users, size=max_users, replace=False)
        )

    candidate_k = int(recommender.config.get("candidate_k", 200))
    rerank_pool_k = int(recommender.config.get("rerank_pool_k", 40))
    final_k = int(k)
    genres = recommender.genres_map
    catalog = set(int(item_id) for item_id in recommender.items.tolist())
    counts = {
        int(item_id): int(count)
        for item_id, count in recommender.popularity_counts.items()
    }
    total_count = max(1, sum(counts.values()))
    probabilities = {item_id: count / total_count for item_id, count in counts.items()}

    variant_names = (
        "Popularity",
        "SVD",
        "Multi-source RRF",
        "RRF + XGBRanker",
        "RRF + XGBRanker + MMR",
    )
    predictions: dict[str, list[list[int]]] = {name: [] for name in variant_names}
    latencies: dict[str, list[float]] = {name: [] for name in variant_names}
    candidate_recall_by_variant: dict[str, list[float]] = {
        name: [] for name in variant_names
    }
    targets: list[int] = []
    candidate_recall: dict[str, list[float]] = {
        f"recall@{cutoff}": [] for cutoff in (50, 100, candidate_k)
    }

    for user_id in eligible_users:
        target_item, timestamp = test_events[user_id]
        targets.append(target_item)
        seen = seen_items_before(ratings, user_id, timestamp)

        start = time.perf_counter()
        popularity_ids = [
            item_id for item_id in recommender.popular_items if item_id not in seen
        ][:final_k]
        popularity_candidates = [
            item_id for item_id in recommender.popular_items if item_id not in seen
        ][:candidate_k]
        predictions["Popularity"].append(popularity_ids)
        candidate_recall_by_variant["Popularity"].append(
            hit_rate_at_k(popularity_candidates, target_item)
        )
        latencies["Popularity"].append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        svd_candidates = recommender.svd_retriever.retrieve(
            user_id,
            k=candidate_k,
            filter_seen=True,
            seen_items_override=seen,
        )
        predictions["SVD"].append(
            [candidate.item_id for candidate in svd_candidates[:final_k]]
        )
        candidate_recall_by_variant["SVD"].append(
            candidate_recall_at_k(svd_candidates, target_item, candidate_k)
        )
        latencies["SVD"].append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        rrf_candidates = recommender.retriever.retrieve(
            user_id,
            k=candidate_k,
            filter_seen=True,
            seen_items_override=seen,
        )
        for cutoff in (50, 100, candidate_k):
            candidate_recall[f"recall@{cutoff}"].append(
                candidate_recall_at_k(rrf_candidates, target_item, cutoff)
            )
        features = recommender.feature_builder.build_features(
            rrf_candidates,
            user_id=user_id,
            as_of_timestamp=timestamp,
        )
        retrieval_ranked = retrieval_order(features)
        predictions["Multi-source RRF"].append(
            [candidate.item_id for candidate in rrf_candidates[:final_k]]
        )
        rrf_candidate_hit = candidate_recall_at_k(
            rrf_candidates, target_item, candidate_k
        )
        for name in variant_names[2:]:
            candidate_recall_by_variant[name].append(rrf_candidate_hit)
        latencies["Multi-source RRF"].append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        if recommender.learned_ranker is not None:
            ranked = rank_candidates(features, recommender.learned_ranker)
        else:
            ranked = retrieval_ranked
        predictions["RRF + XGBRanker"].append(
            [candidate.item_id for candidate in ranked[:final_k]]
        )
        latencies["RRF + XGBRanker"].append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        reranked = recommender.diversity_reranker.rerank(
            ranked,
            k=final_k,
            rerank_pool_k=rerank_pool_k,
        )
        predictions["RRF + XGBRanker + MMR"].append(
            [candidate.item_id for candidate in reranked]
        )
        latencies["RRF + XGBRanker + MMR"].append(
            (time.perf_counter() - start) * 1000.0
        )

    test_metrics = {
        name: _variant_metrics(
            predictions[name],
            targets,
            genres,
            catalog,
            probabilities,
            latencies[name],
            final_k,
        )
        for name in variant_names
    }
    for name in variant_names:
        test_metrics[name][f"recall@{candidate_k}"] = _mean(
            candidate_recall_by_variant[name]
        )
    dataset_config = recommender.config.get("dataset", {})
    report: dict[str, Any] = {
        "protocol": {
            "dataset": dataset_config.get("name", "MovieLens-1M"),
            "seed": seed,
            "candidate_k": candidate_k,
            "rerank_pool_k": rerank_pool_k,
            "final_k": final_k,
            "split": "per-user temporal holdout",
            "ranker": "XGBRanker(objective=rank:ndcg)",
            "feature_count": 19,
            "same_test_users": True,
        },
        "model": {
            "model_name": recommender.model_name,
            "ranker_enabled": recommender.ranker_enabled,
            "trained_ranker_available": recommender.learned_ranker is not None,
            "selected_ranker": recommender.config.get("dev", {}).get(
                "selected_ranker", "retrieval_order"
            ),
            "mmr_lambda": recommender.default_diversity_lambda,
        },
        "test": {
            "evaluated_users": len(eligible_users),
            "target_in_catalog_rate": target_in_catalog_rate(targets, catalog),
            "cold_item_test_share": cold_item_test_share(targets, catalog),
            "variants": test_metrics,
        },
        "funnel": {
            "candidate_recall": {
                key: _mean(values) for key, values in candidate_recall.items()
            },
            "raw_candidates": recommender.config.get("candidate_contract", {}).get(
                "raw_source_total_k", 250
            ),
            "canonical_candidates": candidate_k,
        },
    }
    save_json(output_path, report)
    LOGGER.info("Đã ghi unified experiment vào %s", output_path)
    return report


def main() -> None:
    """Entrypoint CLI cho experiment offline."""
    run_experiment()


if __name__ == "__main__":
    main()
