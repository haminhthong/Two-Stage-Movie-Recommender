"""Mô hình xếp hạng ứng viên (Two-Stage Relevance Ranker).

Tổng hợp điểm tương quan từ các đặc trưng ứng viên (Latent Score, Popularity Prior, Genre Affinity)
thành một điểm số mức độ phù hợp duy nhất (Relevance Score) trước khi áp dụng ràng buộc hậu xếp hạng.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .features import CandidateFeatures


@dataclass(frozen=True)
class RankedCandidate:
    """Ứng viên đã được chấm điểm xếp hạng liên quan.

    Attributes:
        item_id (int): ID định danh sản phẩm.
        relevance_score (float): Điểm tương quan tổng hợp.
        features (CandidateFeatures): Bộ đặc trưng thành phần để hỗ trợ giải thích/debug.
    """

    item_id: int
    relevance_score: float
    features: CandidateFeatures


class TwoStageRanker:
    """Xếp hạng ứng viên bằng phương pháp dung hợp tuyến tính (Weighted Linear Fusion)."""

    def __init__(
        self,
        latent_weight: float = 0.9,
        genre_affinity_weight: float = 0.0,
    ) -> None:
        """Khởi tạo TwoStageRanker.

        Args:
            latent_weight (float): Trọng số alpha cho điểm latent [0.0, 1.0].
            genre_affinity_weight (float): Trọng số bổ trợ cho độ tương hợp thể loại [0.0, 1.0].
        """
        self.latent_weight = float(latent_weight)
        self.genre_affinity_weight = float(genre_affinity_weight)

    def rank(
        self,
        features: Sequence[CandidateFeatures],
        latent_weight_override: float | None = None,
    ) -> list[RankedCandidate]:
        """Tính điểm liên quan và sắp xếp ứng viên giảm dần.

        Công thức:
            S_rel = alpha * latent_score + (1 - alpha) * popularity_score + beta * genre_affinity

        Args:
            features (Sequence[CandidateFeatures]): Danh sách đặc trưng của các ứng viên.
            latent_weight_override (float | None): Ghi đè alpha nếu có.

        Returns:
            list[RankedCandidate]: Danh sách ứng viên đã xếp hạng.
        """
        if not features:
            return []

        alpha = (
            float(latent_weight_override)
            if latent_weight_override is not None
            else self.latent_weight
        )
        alpha = min(max(alpha, 0.0), 1.0)
        pop_weight = 1.0 - alpha
        beta = self.genre_affinity_weight

        ranked: list[RankedCandidate] = []
        for feat in features:
            score = (
                alpha * feat.latent_score
                + pop_weight * feat.popularity_score
                + beta * feat.genre_affinity
            )
            ranked.append(
                RankedCandidate(
                    item_id=feat.item_id,
                    relevance_score=float(score),
                    features=feat,
                )
            )

        ranked.sort(key=lambda r: r.relevance_score, reverse=True)
        return ranked
