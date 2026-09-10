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
