"""Module huấn luyện mô hình Stage-2 Learned Ranker."""

from __future__ import annotations

import logging
from typing import Any
import numpy as np

from .model import LearnedRanker

LOGGER = logging.getLogger("recommender")


def train_learned_ranker(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None = None,
    model_type: str = "xgboost",
    seed: int = 42,
) -> LearnedRanker:
    """Huấn luyện mô hình Stage-2 Learned Ranker trên tập dữ liệu đặc trưng ứng viên.

    Args:
        X: Ma trận đặc trưng (N_samples, N_features).
        y: Nhãn nhị phân (1 = positive target, 0 = candidate negative).
        groups: Số lượng candidate cho mỗi user group.
        model_type: Loại mô hình ("xgboost", "logistic_regression").
        seed: Random seed.

    Returns:
        LearnedRanker: Đối tượng ranker đã huấn luyện.
    """
    if len(X) == 0 or len(y) == 0:
        raise ValueError("Dữ liệu huấn luyện ranking rỗng!")

    LOGGER.info(
        "Bắt đầu huấn luyện Stage-2 Learned Ranker (Type: %s) trên %d mẫu (Positive rate: %.2f%%)...",
        model_type,
        len(y),
        float(np.mean(y) * 100),
    )

    if model_type == "xgboost":
        try:
            import xgboost as xgb

            estimator = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                eval_metric="logloss",
                n_jobs=-1,
            )
            estimator.fit(X, y)
            LOGGER.info("Đã hoàn thành huấn luyện XGBoost Ranker.")
            return LearnedRanker(estimator=estimator, model_type="xgboost")
        except ImportError:
            LOGGER.warning("Không tìm thấy thư viện xgboost, tự động chuyển sang LogisticRegression.")
            model_type = "logistic_regression"

    if model_type == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced"),
        )
        estimator.fit(X, y)
        LOGGER.info("Đã hoàn thành huấn luyện Logistic Regression Ranker.")
        return LearnedRanker(estimator=estimator, model_type="logistic_regression")

    raise ValueError(f"Loại ranker không hỗ trợ: {model_type}")
