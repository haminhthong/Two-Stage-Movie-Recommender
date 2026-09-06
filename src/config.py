"""Module cấu hình tham số (Configuration dataclasses) cho toàn bộ hệ thống gợi ý Two-Stage.

Hỗ trợ phân chia 4 tập theo thời gian (4-Way Temporal Split), trích xuất ứng viên đa nguồn (Multi-Source Retrieval),
xếp hạng học máy Tầng 2 (Stage-2 Learned Ranker) và tái xếp hạng đa dạng hóa (MMR Diversity Reranking).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainConfig:
    """Cấu hình quá trình tiền xử lý dữ liệu và huấn luyện mô hình."""

    seed: int = 42
    embedding_dim: int = 64

    # Cấu hình kích thước ứng viên qua từng chặng (Uniform Pipeline Protocol)
    candidate_k: int = 200     # Stage 1: Candidate Pool size
    ranking_k: int = 50        # Stage 2: Ranked Candidates to consider
    rerank_pool_k: int = 40    # Stage 3: MMR Diversity candidate pool size (đồng nhất Train, Val, Test, Serving)
    final_k: int = 10          # Final Recommendations top-K

    # Hợp đồng dữ liệu tương tác tích cực ngầm định (Implicit-Positive Contract)
    rating_threshold: float = 4.0
    min_positive: int = 4      # Tối thiểu 4 positive ratings cho 4-Way Temporal Split (Retrieval, Rank-Train, Val, Test)

    # Multi-source candidate counts
    svd_candidate_k: int = 150
    popularity_candidate_k: int = 50
    genre_candidate_k: int = 50

    # Lựa chọn mô hình Stage-2 Ranker: "xgboost", "logistic_regression", "weighted_fusion"
    ranker_model_type: str = "xgboost"

    # Siêu tham số tìm kiếm lưới (Grid Search) cho baseline và MMR
    alpha_candidates: list[float] = field(
        default_factory=lambda: [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
    )
    diversity_lambda_candidates: list[float] = field(
        default_factory=lambda: [0.0, 0.02, 0.05, 0.1, 0.2]
    )
    max_val_users: int = 1000
    model_dir: str = "models"
    model_version: str = "v4-learned-ranker"


@dataclass
class RankingConfig:
    """Cấu hình cho Tầng 2: Feature scoring, learned ranker và MMR diversity reranking."""

    ranker_type: str = "xgboost"  # "xgboost", "logistic_regression", "weighted_fusion"
    latent_weight: float = 0.9
    genre_affinity_weight: float = 0.15
    diversity_lambda: float = 0.05
    popularity_scale: str = "log1p"  # 'log1p' hoặc 'linear'
    candidate_k: int = 200
    ranking_k: int = 50
    rerank_pool_k: int = 40
    final_k: int = 10


@dataclass
class ServingConfig:
    """Cấu hình phục vụ thời gian thực (Online Serving)."""

    model_dir: str | Path = "models"
    model_version: str = "v4-learned-ranker"
    candidate_k: int = 200
    ranking_k: int = 50
    rerank_pool_k: int = 40
    default_top_k: int = 10
    max_top_k: int = 50
    default_diversity_lambda: float = 0.05
