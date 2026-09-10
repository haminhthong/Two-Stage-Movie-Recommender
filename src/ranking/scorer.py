"""Mô hình xếp hạng ứng viên (Two-Stage Relevance Ranker).

Hỗ trợ cả:
1. Stage-2 Learned Ranker (XGBoost / Logistic Regression).
2. Weighted Linear Fusion Ranker (Baseline).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .features import CandidateFeatures
from .model import BaseRankModel, WeightedFusionRanker


@dataclass(frozen=True)
class RankedCandidate:
    """Ứng viên đã được chấm điểm xếp hạng liên quan.

    Attributes:
        item_id (int): ID định danh sản phẩm.
        relevance_score (float): Điểm tương quan tổng hợp.
        features: Bộ đặc trưng thành phần để hỗ trợ giải thích/debug.
    """

    item_id: int
    relevance_score: float
    features: CandidateFeatures | None = None


class TwoStageRanker:
    """Điều phối xếp hạng ứng viên bằng Learned Ranker hoặc Weighted Fusion Baseline."""

    def __init__(
        self,
        latent_weight: float = 0.9,
        genre_affinity_weight: float = 0.0,
        rank_model: BaseRankModel | None = None,
    ) -> None:
        self.latent_weight = float(latent_weight)
        self.genre_affinity_weight = float(genre_affinity_weight)
        self.rank_model = rank_model or WeightedFusionRanker(
            latent_weight=self.latent_weight,
            genre_affinity_weight=self.genre_affinity_weight,
        )

    def rank(
        self,
        features: Sequence[CandidateFeatures],
        latent_weight_override: float | None = None,
        feature_matrix: np.ndarray | None = None,
    ) -> list[RankedCandidate]:
        """Tính điểm liên quan và sắp xếp ứng viên giảm dần."""
        if not features:
            return []

        inp = feature_matrix if feature_matrix is not None else features

        if latent_weight_override is not None:
            baseline = WeightedFusionRanker(
                latent_weight=latent_weight_override,
                genre_affinity_weight=self.genre_affinity_weight,
            )
            scores = baseline.predict_scores(inp)
        else:
            scores = self.rank_model.predict_scores(inp)

        ranked: list[RankedCandidate] = [
            RankedCandidate(
                item_id=feat.item_id,
                relevance_score=float(score),
                features=feat,
            )
            for feat, score in zip(features, scores, strict=True)
        ]

        ranked.sort(key=lambda r: r.relevance_score, reverse=True)
        return ranked
