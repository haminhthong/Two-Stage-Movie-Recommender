"""Tạo báo cáo nguồn gốc dữ liệu và phân chia tập (Data & Split Provenance Manifests)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
import pandas as pd
from .loader import compute_file_sha256


def create_data_manifest(
    ratings_path: str | Path = "data/raw/ml-1m/ratings.dat",
    movies_path: str | Path = "data/raw/ml-1m/movies.dat",
    df_ratings: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Tạo bảng kê chứng nhận nguồn gốc dữ liệu (Data Manifest) với mã băm SHA256."""
    ratings_p = Path(ratings_path)
    movies_p = Path(movies_path)

    ratings_sha = compute_file_sha256(ratings_p) if ratings_p.exists() else "N/A"
    movies_sha = compute_file_sha256(movies_p) if movies_p.exists() else "N/A"

    n_rows = len(df_ratings) if df_ratings is not None else 0
    n_users = int(df_ratings["user_id"].nunique()) if df_ratings is not None else 0
    n_items = int(df_ratings["item_id"].nunique()) if df_ratings is not None else 0

    return {
        "dataset_name": "MovieLens 1M",
        "ratings_file": str(ratings_p),
        "ratings_sha256": ratings_sha,
        "movies_file": str(movies_p),
        "movies_sha256": movies_sha,
        "total_interactions": n_rows,
        "unique_users": n_users,
        "unique_items": n_items,
        "python_version": sys.version,
    }


def create_split_manifest(
    retrieval_train_df: pd.DataFrame,
    rank_train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    rating_threshold: float = 4.0,
    min_positive: int = 4,
) -> dict[str, Any]:
    """Tạo bảng kê phân chia tập dữ liệu (Split Manifest) bảo đảm tính lặp lại."""
    pos_retrieval = int((retrieval_train_df["rating"] >= rating_threshold).sum())
    return {
        "protocol": "per_user_temporal_holdout",
        "description": (
            "Per-user leave-last-3 positive split. Target timestamps: "
            "t_retrieval < t_rank < t_val < t_test per user. Eliminates future lookahead leakage."
        ),
        "rating_threshold": float(rating_threshold),
        "min_positive": int(min_positive),
        "counts": {
            "retrieval_train_rows": len(retrieval_train_df),
            "retrieval_train_positive_rows": pos_retrieval,
            "retrieval_train_users": int(retrieval_train_df["user_id"].nunique()),
            "retrieval_train_items": int(retrieval_train_df["item_id"].nunique()),
            "rank_train_users": len(rank_train_df),
            "validation_users": len(val_df),
            "test_users": len(test_df),
        },
        "timestamps": {
            "retrieval_train_min_ts": int(retrieval_train_df["timestamp"].min()),
            "retrieval_train_max_ts": int(retrieval_train_df["timestamp"].max()),
            "rank_train_min_ts": int(rank_train_df["timestamp"].min()),
            "rank_train_max_ts": int(rank_train_df["timestamp"].max()),
            "val_min_ts": int(val_df["timestamp"].min()),
            "val_max_ts": int(val_df["timestamp"].max()),
            "test_min_ts": int(test_df["timestamp"].min()),
            "test_max_ts": int(test_df["timestamp"].max()),
        },
    }
