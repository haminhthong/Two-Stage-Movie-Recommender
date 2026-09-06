"""Áp dụng các quy tắc kinh doanh và kiểm soát thiên lệch hậu xếp hạng (Business Rules & De-biasing)."""

from __future__ import annotations

from typing import Sequence
from ..ranking.scorer import RankedCandidate


def apply_seen_filter(
    candidates: Sequence[RankedCandidate],
    seen_items: set[int],
) -> list[RankedCandidate]:
    """Loại bỏ triệt để các item người dùng đã từng xem/đánh giá trong quá khứ."""
    if not seen_items:
        return list(candidates)
    return [c for c in candidates if c.item_id not in seen_items]


def apply_popularity_calibrator(
    candidates: Sequence[RankedCandidate],
    penalty_factor: float = 0.0,
) -> list[RankedCandidate]:
    """Hiệu chỉnh điểm để giảm thiên lệch Popularity Bias và tăng phơi nhiễm phim Long-Tail.

    Công thức: Score_calibrated = Score - penalty_factor * item_popularity
    """
    if penalty_factor <= 1e-9:
        return list(candidates)

    calibrated: list[RankedCandidate] = []
    for c in candidates:
        pop_sc = c.features.popularity_score if c.features else 0.0
        new_score = c.relevance_score - (penalty_factor * pop_sc)
        calibrated.append(
            RankedCandidate(
                item_id=c.item_id,
                relevance_score=float(new_score),
                features=c.features,
            )
        )
    calibrated.sort(key=lambda r: r.relevance_score, reverse=True)
    return calibrated
