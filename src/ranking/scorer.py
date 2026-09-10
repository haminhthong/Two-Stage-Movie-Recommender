"""Hàm xếp hạng ứng viên từ retrieval order hoặc learned ranker."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .features import CandidateFeatures
from .model import LearnedRanker


@dataclass(frozen=True)
class RankedCandidate:
    """Ứng viên kèm score và feature dùng để giải thích kết quả."""

    item_id: int
    relevance_score: float
    features: CandidateFeatures | None = None


def rank_candidates(
    features: Sequence[CandidateFeatures], ranker: LearnedRanker
) -> list[RankedCandidate]:
    """Xếp hạng toàn bộ candidate bằng learned ranker."""
    if not features:
        return []
    scores = ranker.predict_scores(features)
    ranked = [
        RankedCandidate(
            item_id=feature.item_id,
            relevance_score=float(score),
            features=feature,
        )
        for feature, score in zip(features, scores, strict=True)
    ]
    ranked.sort(key=lambda candidate: candidate.relevance_score, reverse=True)
    return ranked


def retrieval_order(features: Sequence[CandidateFeatures]) -> list[RankedCandidate]:
    """Đóng gói retrieval order thành RankedCandidate deterministic."""
    total = max(1, len(features))
    return [
        RankedCandidate(
            item_id=feature.item_id,
            relevance_score=1.0 - index / total,
            features=feature,
        )
        for index, feature in enumerate(features)
    ]
