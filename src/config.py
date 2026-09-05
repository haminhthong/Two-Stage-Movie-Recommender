"""Module cấu hình tham số (Configuration dataclasses) cho toàn bộ hệ thống gợi ý.

Bao gồm cấu hình huấn luyện (TrainConfig) và cấu hình xếp hạng/phục vụ (RankingConfig).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainConfig:
    """Cấu hình quá trình tiền xử lý dữ liệu và huấn luyện mô hình."""

    seed: int = 42
    embedding_dim: int = 64
    candidate_k: int = 200
    top_k: int = 10
    rating_threshold: float = 4.0
    min_positive: int = 3
    alpha_candidates: list[float] = field(
        default_factory=lambda: [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
    )
    diversity_lambda_candidates: list[float] = field(
        default_factory=lambda: [0.0, 0.02, 0.05, 0.1, 0.2]
    )
    max_val_users: int = 1000
    model_dir: str = "models"


@dataclass
class RankingConfig:
    """Cấu hình cho Tầng 2: Feature scoring, blending weights và MMR diversity reranking."""

    latent_weight: float = 0.9
    diversity_lambda: float = 0.05
    popularity_scale: str = "log1p"  # 'log1p' hoặc 'linear'
    candidate_k: int = 200
    top_k: int = 10
    genre_affinity_weight: float = 0.0  # Trọng số tương quan thể loại bổ trợ (nếu có)


@dataclass
class ServingConfig:
    """Cấu hình phục vụ thời gian thực (Online Serving)."""

    model_dir: str | Path = "models"
    default_top_k: int = 10
    max_top_k: int = 50
    default_diversity_lambda: float = 0.05
