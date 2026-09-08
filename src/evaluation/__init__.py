"""Package đánh giá hệ thống gợi ý và phân tích phễu từng tầng."""

from .evaluator import FullFunnelEvaluator
from .latency import StageLatencyReport, summarize_latencies
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
from .retrieval_metrics import (
    candidate_recall_at_k,
    cold_item_test_share,
    target_in_catalog_rate,
)

__all__ = [
    "FullFunnelEvaluator",
    "StageLatencyReport",
    "candidate_recall_at_k",
    "cold_item_test_share",
    "compute_long_tail_distribution",
    "compute_user_coverage",
    "dcg",
    "hit_rate_at_k",
    "intra_list_diversity",
    "mrr_at_k",
    "novelty_at_k",
    "ranker_ndcg_at_k",
    "ranker_recall_at_k",
    "summarize_latencies",
    "target_in_catalog_rate",
]
