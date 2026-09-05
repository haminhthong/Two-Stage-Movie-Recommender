"""Gói module đánh giá và benchmark (Stage 5 Offline Evaluation)."""

from .benchmark import benchmark_stage_latencies
from .metrics import (
    compute_long_tail_distribution,
    compute_user_coverage,
    dcg,
    hit_rate_at_k,
    intra_list_diversity,
    mrr_at_k,
    novelty_at_k,
)

__all__ = [
    "benchmark_stage_latencies",
    "compute_long_tail_distribution",
    "compute_user_coverage",
    "dcg",
    "hit_rate_at_k",
    "intra_list_diversity",
    "mrr_at_k",
    "novelty_at_k",
]
