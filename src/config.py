"""Cấu hình duy nhất cho train, đánh giá và serving."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    """Cấu hình quá trình tiền xử lý dữ liệu và huấn luyện mô hình."""

    seed: int = 42
    embedding_dim: int = 64

    candidate_k: int = 200
    rerank_pool_k: int = 40
    final_k: int = 10

    # Hợp đồng dữ liệu tương tác tích cực ngầm định (Implicit-Positive Contract)
    rating_threshold: float = 4.0
    min_positive: int = 4

    # Multi-source candidate counts
    svd_candidate_k: int = 150
    popularity_candidate_k: int = 50
    genre_candidate_k: int = 50

    # LogisticRegression chỉ được gọi tường minh trong test/ablation.
    ranker_model_type: str = "xgboost"

    diversity_lambda_candidates: tuple[float, ...] = (0.8, 0.9, 0.95, 0.98, 1.0)
    # None nghĩa là dùng toàn bộ user đủ điều kiện; chỉ giảm khi chạy thử cục bộ.
    max_val_users: int | None = None
    max_rank_train_users: int | None = None
    model_dir: str = "models"

    @property
    def candidate_contract(self) -> dict[str, object]:
        """Contract duy nhất được dùng ở train, dev, test và serving."""
        return {
            "raw_source_total_k": self.svd_candidate_k
            + self.popularity_candidate_k
            + self.genre_candidate_k,
            "canonical_k": self.candidate_k,
            "sources": {
                "svd": self.svd_candidate_k,
                "popularity": self.popularity_candidate_k,
                "genre": self.genre_candidate_k,
            },
            "merge": {"method": "rrf", "rrf_k": 60},
            "seen_filter": "request_time",
        }
