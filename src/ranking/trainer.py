"""Huấn luyện group-aware learning-to-rank cho Stage 2."""

from __future__ import annotations

import logging

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
        y: Nhãn nhị phân (1 = positive target, 0 = sampled candidate negative).
        groups: Số lượng candidate của từng query/user group.
        model_type: ``xgboost``/``xgb_ranker``/``lambdamart``. LogisticRegression
            chỉ được dùng khi gọi tường minh cho ablation.
        seed: Random seed.

    Returns:
        LearnedRanker: Đối tượng ranker đã huấn luyện.
    """
    if len(X) == 0 or len(y) == 0:
        raise ValueError("Dữ liệu huấn luyện ranking rỗng!")
    if len(X) != len(y):
        raise ValueError("Số dòng X và y không khớp.")

    if groups is None:
        groups = np.array([len(y)], dtype=np.int32)
    groups = np.asarray(groups, dtype=np.int32)
    if np.any(groups <= 0) or int(groups.sum()) != len(y):
        raise ValueError("groups phải dương và tổng groups phải bằng số dòng y.")

    LOGGER.info(
        "Bắt đầu huấn luyện Stage-2 Learned Ranker (Type: %s) trên %d mẫu (Positive rate: %.2f%%)...",
        model_type,
        len(y),
        float(np.mean(y) * 100),
    )

    if model_type in {"xgboost", "xgb_ranker", "lambdamart"}:
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise RuntimeError(
                "XGBoost là dependency bắt buộc cho ranker chính. "
                "Hãy cài requirements.txt hoặc gọi model_type='logistic_regression' "
                "chỉ cho ablation."
            ) from exc

        estimator = xgb.XGBRanker(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            objective="rank:ndcg",
            eval_metric="ndcg@10",
            n_jobs=-1,
        )
        estimator.fit(X, y, group=groups)
        LOGGER.info(
            "Đã hoàn thành huấn luyện XGBRanker rank:ndcg trên %d query groups.",
            len(groups),
        )
        return LearnedRanker(estimator=estimator, model_type="xgb_ranker")

    if model_type == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000, random_state=seed, class_weight="balanced"
            ),
        )
        estimator.fit(X, y)
        LOGGER.info("Đã hoàn thành huấn luyện Logistic Regression Ranker.")
        return LearnedRanker(estimator=estimator, model_type="logistic_regression")

    raise ValueError(f"Loại ranker không hỗ trợ: {model_type}")
