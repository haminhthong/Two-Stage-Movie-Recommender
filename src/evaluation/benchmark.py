"""Module đo kiểm độ trễ suy luận chi tiết từng giai đoạn (Stage-Level Latency Benchmark).

Phân tích hiệu năng thời gian thực của:
1. Retrieval Latency: Dot product + partition filtering.
2. Ranking Latency: Feature scaling + score fusion.
3. MMR Diversity Latency: Greedy selection + pairwise Jaccard penalty.
4. Total Latency: End-to-end recommendation request latency.
"""

from __future__ import annotations

import numpy as np

from ..serving.recommender import Recommender


def benchmark_stage_latencies(
    engine: Recommender,
    user_ids: list[int],
    k: int = 10,
    diversity_lambda: float | None = None,
) -> dict[str, dict[str, float]]:
    """Đo độ trễ p50 và p95 cho từng tầng trong quy trình gợi ý.

    Args:
        engine (Recommender): Inference engine.
        user_ids (list[int]): Danh sách user IDs để benchmark.
        k (int): Số lượng item top-K.
        diversity_lambda (float | None): Lambda MMR.

    Returns:
        dict[str, dict[str, float]]: Thống kê p50/p95 từng chặng (đơn vị: ms).
    """
    retrieval_times: list[float] = []
    ranking_times: list[float] = []
    diversity_times: list[float] = []
    total_times: list[float] = []

    for uid in user_ids:
        res = engine.recommend_detailed(
            user_id=uid,
            k=k,
            diversity_lambda=diversity_lambda,
        )
        lats = res["latencies_ms"]
        retrieval_times.append(lats["retrieval"])
        ranking_times.append(lats["ranking"])
        diversity_times.append(lats["diversity"])
        total_times.append(lats["total"])

    return {
        "retrieval": {
            "p50_ms": float(np.percentile(retrieval_times, 50)),
            "p95_ms": float(np.percentile(retrieval_times, 95)),
            "mean_ms": float(np.mean(retrieval_times)),
        },
        "ranking": {
            "p50_ms": float(np.percentile(ranking_times, 50)),
            "p95_ms": float(np.percentile(ranking_times, 95)),
            "mean_ms": float(np.mean(ranking_times)),
        },
        "diversity": {
            "p50_ms": float(np.percentile(diversity_times, 50)),
            "p95_ms": float(np.percentile(diversity_times, 95)),
            "mean_ms": float(np.mean(diversity_times)),
        },
        "total": {
            "p50_ms": float(np.percentile(total_times, 50)),
            "p95_ms": float(np.percentile(total_times, 95)),
            "mean_ms": float(np.mean(total_times)),
        },
    }
