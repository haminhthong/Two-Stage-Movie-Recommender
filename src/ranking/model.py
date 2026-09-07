"""Các mô hình xếp hạng Stage 2 và baseline retrieval-order."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence
import numpy as np

from .features import CandidateFeatures, N_FEATURES


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
            if features.shape[1] == N_FEATURES:
                # Feature contract mới: SVD đã normalize trong pool, popularity
                # và genre giữ score riêng theo source.
                scores = alpha * features[:, 0] + pop_w * features[:, 2] + beta * features[:, 17]
            else:
                # Tương thích vector 15 cột của artifact cũ, không dùng cho train mới.
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

        # Chỉ chuyển đổi rõ ràng cho artifact classifier 15 feature cũ. Ranker
        # mới luôn phải khớp đúng 19 cột, tránh silent feature drift.
        expected = getattr(self.estimator, "n_features_in_", X.shape[1])
        if int(expected) != X.shape[1]:
            if int(expected) == 15 and not isinstance(features, np.ndarray):
                X = np.vstack([_legacy_feature_vector(feature) for feature in features])
            else:
                raise ValueError(
                    "Feature schema của ranker không khớp: "
                    f"artifact cần {expected} cột nhưng request có {X.shape[1]}."
                )

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


def _legacy_feature_vector(feature: CandidateFeatures) -> np.ndarray:
    """Ánh xạ tường minh feature object mới về schema 15 cột legacy."""
    return np.array(
        [
            feature.raw_latent_score,
            feature.latent_score,
            feature.retrieval_rank_percentile,
            feature.popularity_score,
            feature.genre_affinity,
            feature.user_positive_count,
            feature.user_avg_rating,
            feature.item_rating_count,
            feature.item_avg_rating,
            feature.item_genre_count,
            feature.item_popularity_percentile,
            feature.retrieved_by_svd,
            feature.retrieved_by_popularity,
            feature.retrieved_by_genre,
            feature.source_count,
        ],
        dtype=np.float32,
    )
