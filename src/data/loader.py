"""Module tải dữ liệu MovieLens 1M kèm tính toán SHA256 Checksum phục vụ Provenance Manifest."""

from __future__ import annotations

import hashlib
from pathlib import Path
import pandas as pd


def compute_file_sha256(path: str | Path) -> str:
    """Tính toán mã băm SHA256 của một tệp tin."""
    file_path = Path(path)
    if not file_path.exists():
        return ""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def load_ratings(path: str | Path = "data/raw/ml-1m/ratings.dat") -> pd.DataFrame:
    """Đọc dữ liệu lượt tương tác/đánh giá phim MovieLens 1M từ tệp raw.

    Định dạng tệp raw: User_ID::Movie_ID::Rating::Timestamp

    Args:
        path: Đường dẫn tới tệp ratings.dat.

    Returns:
        pd.DataFrame: ['user_id', 'item_id', 'rating', 'timestamp'] sắp xếp theo user_id, timestamp.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy tệp ratings tại: {file_path}. "
            "Vui lòng chạy 'python scripts/download_data.py' trước."
        )

    df = pd.read_csv(
        file_path,
        sep="::",
        engine="python",
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={"user_id": int, "item_id": int, "rating": float, "timestamp": int},
    )
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    return df


def load_movies(path: str | Path = "data/raw/ml-1m/movies.dat") -> pd.DataFrame:
    """Đọc metadata thông tin phim (Tiêu đề, Thể loại) từ tệp movies.dat.

    Args:
        path: Đường dẫn tới tệp movies.dat.

    Returns:
        pd.DataFrame: ['item_id', 'title', 'genres']
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy tệp metadata phim tại: {file_path}. "
            "Vui lòng chạy 'python scripts/download_data.py' trước."
        )

    return pd.read_csv(
        file_path,
        sep="::",
        engine="python",
        names=["item_id", "title", "genres"],
        encoding="latin-1",
        dtype={"item_id": int, "title": str, "genres": str},
    )
