"""Fixture model nhỏ để test chạy được trên clone sạch."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from src.model_io import save_model
from src.ranking.features import CandidateFeatures


def make_mock_candidate_features(
    item_id: int, svd_score: float, popularity_score: float
) -> CandidateFeatures:
    """Tạo fixture CandidateFeatures đúng schema 19 cột dùng chung cho các test."""
    return CandidateFeatures(
        item_id=item_id,
        svd_score=svd_score,
        svd_rank=item_id,
        popularity_retrieval_score=popularity_score,
        popularity_rank=item_id,
        genre_retrieval_score=0.0,
        genre_rank=0,
        rrf_score=1.0,
        source_count=2.0,
        user_positive_count=0,
        user_interaction_count=0,
        user_avg_rating=4.0,
        genre_entropy=0.0,
        item_positive_count=0,
        item_rating_count=0,
        item_avg_rating=3.5,
        item_popularity_percentile=0.5,
        item_genre_count=1,
        genre_affinity=0.0,
        genre_overlap_count=0,
    )


def pytest_configure(config) -> None:
    """Đảm bảo thư mục tạm basetemp luôn có quyền ghi, tránh PermissionError trên Windows."""
    if config.option.basetemp is None:
        try:
            user = os.getenv("USERNAME") or os.getenv("USER") or "user"
            default_dir = Path(tempfile.gettempdir()) / f"pytest-of-{user}"
            if default_dir.exists():
                test_file = default_dir / ".perm_check"
                test_file.touch()
                test_file.unlink()
        except OSError:
            config.option.basetemp = Path(tempfile.mkdtemp(prefix="pytest_safe_"))


def _write_ci_fixture(project_root: Path) -> None:
    """Tạo 5 file model phẳng nếu clone chưa có model local."""
    model_root = project_root / "models"
    if (model_root / "config.json").is_file():
        return

    items = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    users = np.array([1], dtype=np.int64)
    user_embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    item_embeddings = np.array(
        [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]],
        dtype=np.float32,
    )
    counts = {10: 1, 20: 5, 30: 4, 40: 3, 50: 2}
    metadata = {
        "users": users,
        "items": items,
        "user_map": {1: 0},
        "item_map": {int(item_id): index for index, item_id in enumerate(items)},
        "popular": [20, 30, 40, 50, 10],
        "popularity_counts": counts,
        "log_popularity": {
            item_id: float(np.log1p(count)) for item_id, count in counts.items()
        },
        "seen": {1: {10}},
        "genres": {
            10: {"Action"},
            20: {"Action"},
            30: {"Drama"},
            40: {"Comedy"},
            50: {"Animation"},
        },
        "titles": {item_id: f"Fixture Movie {item_id}" for item_id in items},
        "user_genre_profiles": {1: {"Action": 1.0}},
        "user_stats": {
            1: {"positive_count": 1.0, "interaction_count": 1.0, "avg_rating": 5.0}
        },
        "item_stats": {
            item_id: {
                "positive_count": float(count),
                "rating_count": float(count),
                "avg_rating": 4.0,
            }
            for item_id, count in counts.items()
        },
    }
    config = {
        "model_name": "ci-fixture",
        "embedding_dimension": 2,
        "candidate_k": 200,
        "rerank_pool_k": 40,
        "final_k": 10,
        "svd_candidate_k": 150,
        "popularity_candidate_k": 50,
        "genre_candidate_k": 50,
        "feature_schema_version": "rank-features-v1",
        "ranker_enabled": False,
        "diversity_lambda": 0.95,
        "candidate_contract": {
            "raw_source_total_k": 250,
            "canonical_k": 200,
            "sources": {"svd": 150, "popularity": 50, "genre": 50},
            "merge": {"method": "rrf", "rrf_k": 60},
            "seen_filter": "request_time",
        },
    }
    save_model(model_root, user_embeddings, item_embeddings, metadata, config)


def pytest_sessionstart(session) -> None:
    """Đảm bảo smoke test có model tối thiểu trên CI clone sạch."""
    _write_ci_fixture(Path(__file__).resolve().parents[1])
