"""Định nghĩa các mô hình xếp hạng Tầng 2 (Stage-2 Ranking Models).

Bao gồm:
1. LearnedRanker: Mô hình học máy có giám sát (XGBoost / Logistic Regression) học từ candidate data.
2. WeightedFusionRanker: Baseline dung hợp tuyến tính Heuristic để so sánh đối chứng (Ablation).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence
import numpy as np

from .features import CandidateFeatures


class BaseRankModel(ABC):
    """Lớp cơ sở trừu tượng cho các mô hình tính điểm xếp hạng."""

    @abstractmethod
    def predict_scores(self, features: Sequence[CandidateFeatures] | np.ndarray) -> np.ndarray:
        """Dự đoán điểm số mức độ phù hợp (Relevance Scores) cho danh sách đặc trưng ứng viên."""
        raise NotImplementedError


class WeightedFusionRanker(BaseRankModel):
    """Mô hình xếp hạng baseline bằng dung hợp trọng số tuyến tính (Weighted Relevance Fusion)."""

    def __init__(
        self,
        latent_weight: float = 0.9,
        genre_affinity_weight: float = 0.0,
    ) -> None:
        self.latent_weight = float(latent_weight)
        self.genre_affinity_weight = float(genre_affinity_weight)

    def predict_scores(self, features: Sequence[CandidateFeatures] | np.ndarray) -> np.ndarray:
        if len(features) == 0:
            return np.array([], dtype=np.float32)

        alpha = min(max(self.latent_weight, 0.0), 1.0)
        pop_w = 1.0 - alpha
        beta = self.genre_affinity_weight

        if isinstance(features, np.ndarray):
            # features: [N, 15], col 1 là norm_latent, col 3 là pop, col 4 là affinity
            scores = alpha * features[:, 1] + pop_w * features[:, 3] + beta * features[:, 4]
            return scores.astype(np.float32)

        scores = [
            alpha * f.latent_score + pop_w * f.popularity_score + beta * f.genre_affinity
            for f in features
        ]
        return np.array(scores, dtype=np.float32)


class LearnedRanker(BaseRankModel):
    """Mô hình xếp hạng học máy có giám sát (Learned Ranker)."""

    def __init__(
        self,
        estimator: Any,
        model_type: str = "xgboost",
    ) -> None:
        self.estimator = estimator
        self.model_type = model_type

    def predict_scores(self, features: Sequence[CandidateFeatures] | np.ndarray) -> np.ndarray:
        if len(features) == 0:
            return np.array([], dtype=np.float32)

        if isinstance(features, np.ndarray):
            X = features
        else:
            X = np.vstack([f.to_feature_vector() for f in features])

        if hasattr(self.estimator, "predict_proba"):
            proba = self.estimator.predict_proba(X)
            if proba.shape[1] >= 2:
                return proba[:, 1].astype(np.float32)
            return proba[:, 0].astype(np.float32)

        if hasattr(self.estimator, "decision_function"):
            return self.estimator.decision_function(X).astype(np.float32)

        if hasattr(self.estimator, "predict"):
            return self.estimator.predict(X).astype(np.float32)

        raise ValueError("Estimator không hỗ trợ dự đoán xác suất hoặc điểm số.")
