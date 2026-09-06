"""Chỉ số đánh giá tầng xếp hạng (Stage 2 Ranking Metrics).

Đo lường năng lực của Ranker độc lập, TRƯỚC KHI qua MMR Diversity Reranking:
- Nếu target_item đã lọt vào Candidate Pool, Ranker có đưa nó lên top 10/top 50 không?
"""

from __future__ import annotations

from typing import Sequence
import numpy as np


def ranker_recall_at_k(
    ranked_items: Sequence[int],
    target_item: int,
    k: int = 10,
) -> float:
    """Đo HitRate/Recall của Ranker tại top-K (trước MMR)."""
    if not ranked_items or k <= 0:
        return 0.0
    return 1.0 if target_item in ranked_items[:k] else 0.0


def ranker_ndcg_at_k(
    ranked_items: Sequence[int],
    target_item: int,
    k: int = 10,
) -> float:
    """Đo vị trí xếp hạng có trọng số của Ranker (trước MMR)."""
    top_items = ranked_items[:k]
    if target_item in top_items:
        rank_0indexed = top_items.index(target_item)
        return float(1.0 / np.log2(rank_0indexed + 2))
    return 0.0
