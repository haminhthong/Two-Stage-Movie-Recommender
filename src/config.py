"""Cấu hình duy nhất cho train, đánh giá và serving."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    """Cấu hình quá trình tiền xử lý dữ liệu và huấn luyện mô hình."""

    seed: int = 42
    embedding_dim: int = 64

    # Cấu hình kích thước ứng viên qua từng chặng (Uniform Pipeline Protocol)
    candidate_k: int = 200  # Stage 1: Candidate Pool size
    rerank_pool_k: int = 40  # Stage 3: MMR Diversity candidate pool size (đồng nhất Train, Val, Test, Serving)
    final_k: int = 10  # Final Recommendations top-K

    # Hợp đồng dữ liệu tương tác tích cực ngầm định (Implicit-Positive Contract)
    rating_threshold: float = 4.0
    min_positive: int = 4  # Tối thiểu 4 positive ratings cho 4-Way Temporal Split (Retrieval, Rank-Train, Val, Test)

    # Multi-source candidate counts
    svd_candidate_k: int = 150
    popularity_candidate_k: int = 50
    genre_candidate_k: int = 50

    # XGBRanker là ranker chính; LogisticRegression chỉ dùng tường minh cho ablation.
    ranker_model_type: str = "xgboost"

    # Siêu tham số tìm kiếm lưới (Grid Search) cho baseline và MMR
    alpha_candidates: list[float] = field(
        default_factory=lambda: [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
    )
    diversity_lambda_candidates: list[float] = field(
        default_factory=lambda: [0.8, 0.9, 0.95, 0.98, 1.0]
    )
    max_val_users: int = 1000
    max_rank_train_users: int = 3000
    model_dir: str = "models"
    model_version: str = "v5-multisource-ranker"

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
