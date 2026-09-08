"""Module tương thích ngược cho Diversity Reranker.

Chuyển tiếp (Re-export) từ package chuẩn src.reranking.diversity để đảm bảo
các import cũ không bị gãy.
"""

from __future__ import annotations

from ..reranking.diversity import (
    DiversityReranker,
    RankedCandidate,
    ScoredRecommendation,
)

__all__ = [
    "DiversityReranker",
    "RankedCandidate",
    "ScoredRecommendation",
]
