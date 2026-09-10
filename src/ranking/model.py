"""Wrapper cho XGBRanker và ranker ablation được gọi tường minh."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .features import N_FEATURES, CandidateFeatures


class LearnedRanker:
    """Adapter thống nhất prediction của XGBRanker trên 19 feature."""

    def __init__(self, estimator: Any, model_type: str = "xgb_ranker") -> None:
        self.estimator = estimator
        self.model_type = model_type

    def predict_scores(
        self, features: Sequence[CandidateFeatures] | np.ndarray
    ) -> np.ndarray:
        """Sinh score cho ma trận feature đã đúng schema."""
        if len(features) == 0:
            return np.array([], dtype=np.float32)

        X = (
            features
            if isinstance(features, np.ndarray)
            else np.vstack([feature.to_feature_vector() for feature in features])
        )
        if X.ndim != 2 or X.shape[1] != N_FEATURES:
            raise ValueError(
                f"Ranker cần ma trận ({N_FEATURES} features), nhận shape {X.shape}."
            )

        expected = getattr(self.estimator, "n_features_in_", N_FEATURES)
        if int(expected) != N_FEATURES:
            raise ValueError(
                f"Artifact ranker cần {expected} features, contract hiện tại có {N_FEATURES}."
            )

        if hasattr(self.estimator, "predict_proba"):
            probabilities = self.estimator.predict_proba(X)
            return probabilities[:, -1].astype(np.float32)
        if hasattr(self.estimator, "decision_function"):
            return self.estimator.decision_function(X).astype(np.float32)
        if hasattr(self.estimator, "predict"):
            return self.estimator.predict(X).astype(np.float32)
        raise ValueError("Estimator không hỗ trợ sinh score.")
