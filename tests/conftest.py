"""Fixture production tối thiểu để bộ test chạy được trên clone sạch.

Repository không commit dữ liệu MovieLens hoặc model binary. Khi chạy local đã
có ``models/production.json`` thì fixture này không can thiệp vào artifact thật.
Trên CI/clone mới, fixture tạo một schema-5 release nhỏ, đủ để kiểm thử API,
retrieval, seen-filter và cold-start mà không làm giả kết quả benchmark.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.artifacts.writer import save_versioned_bundle


def _write_ci_fixture(project_root: Path) -> None:
    """Tạo release test cục bộ nếu workspace chưa có production artifact."""
    model_root = project_root / "models"
    pointer_path = model_root / "production.json"
    if pointer_path.is_file():
        return

    release_dir = model_root / "ci-fixture"
    if release_dir.exists():
        raise RuntimeError(
            "CI fixture bị thiếu production.json nhưng thư mục release đã tồn tại; "
            "hãy xóa artifact test dở dang rồi chạy lại."
        )

    items = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    users = np.array([1], dtype=np.int64)
    user_embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    item_embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
            [0.6, 0.4],
        ],
        dtype=np.float32,
    )

    metadata = {
        "users": users,
        "items": items,
        "user_map": {1: 0},
        "item_map": {int(item_id): index for index, item_id in enumerate(items)},
        "popular": [20, 30, 40, 50, 10],
        "popularity_counts": {10: 1, 20: 5, 30: 4, 40: 3, 50: 2},
        "log_popularity": {
            item_id: float(np.log1p(count))
            for item_id, count in {10: 1, 20: 5, 30: 4, 40: 3, 50: 2}.items()
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
        "user_stats": {1: {"positive_count": 1.0, "avg_rating": 5.0}},
        "item_stats": {
            item_id: {
                "positive_count": float(count),
                "rating_count": float(count),
                "avg_rating": 4.0,
            }
            for item_id, count in {10: 1, 20: 5, 30: 4, 40: 3, 50: 2}.items()
        },
    }
    config = {
        "schema_version": 5,
        "version": "ci-fixture",
        "feature_schema_version": "rank-features-v1",
        "ranker_model_type": "xgb_ranker",
        "candidate_contract": {
            "raw_source_total_k": 250,
            "canonical_k": 200,
            "sources": {"svd": 150, "popularity": 50, "genre": 50},
            "merge": {"method": "rrf", "rrf_k": 60},
            "seen_filter": "request_time",
        },
        "candidate_k": 200,
        "rerank_pool_k": 40,
        "final_k": 10,
        "ranker_enabled": False,
        "multi_source_retrieval": False,
        "latent_weight": 0.9,
        "diversity_lambda": 0.95,
        "svd_candidate_k": 150,
        "popularity_candidate_k": 50,
        "genre_candidate_k": 50,
    }

    save_versioned_bundle(
        base_dir=model_root,
        version="ci-fixture",
        user_embeddings=user_embeddings,
        item_embeddings=item_embeddings,
        metadata=metadata,
        config_payload=config,
        publish=True,
    )


def pytest_sessionstart(session) -> None:
    """Đảm bảo smoke test có artifact tối thiểu trên CI clone sạch."""
    _write_ci_fixture(Path(__file__).resolve().parents[1])
