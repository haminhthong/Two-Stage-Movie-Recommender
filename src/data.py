"""Module xử lý dữ liệu cho hệ thống gợi ý MovieLens 1M.

Bao gồm tải tập tương tác (ratings), danh mục phim (movies) và thực hiện
phân chia dữ liệu theo chuỗi thời gian (time-based leave-last-two split).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_ratings(path: str | Path = "data/raw/ml-1m/ratings.dat") -> pd.DataFrame:
    """Đọc dữ liệu lượt tương tác/đánh giá phim MovieLens 1M từ tệp raw.

    Định dạng tệp raw MovieLens 1M: `User_ID::Movie_ID::Rating::Timestamp`

    Args:
        path (str | Path): Đường dẫn tới tệp ratings.dat. (Mặc định: 'data/raw/ml-1m/ratings.dat')

    Returns:
        pd.DataFrame: DataFrame gồm các cột ['user_id', 'item_id', 'rating', 'timestamp'],
                      được sắp xếp tăng dần theo user_id và timestamp.
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
    # Sắp xếp theo thứ tự thời gian của từng user để hỗ trợ time-based split
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    return df


def load_movies(path: str | Path = "data/raw/ml-1m/movies.dat") -> pd.DataFrame:
    """Đọc metadata thông tin phim (Tiêu đề, Thể loại) từ tệp movies.dat.

    Lưu ý: MovieLens 1M sử dụng bảng mã `latin-1` đặc thù.

    Args:
        path (str | Path): Đường dẫn tới tệp movies.dat.

    Returns:
        pd.DataFrame: DataFrame gồm các cột ['item_id', 'title', 'genres'].
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


def time_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Phân chia dữ liệu theo chiến lược Leave-Last-Two Per User theo thời gian.

    Chiến lược này phản ánh bài toán thế giới thực tốt hơn Random Split vì:
    - 2 tương tác cuối cùng theo thời gian của mỗi user được tách làm Validation và Test set.
    - Tất cả các tương tác trước đó được đưa vào Tập Huấn luyện (Train set).
    - Tránh lộ thông tin tương lai (Data Leakage / Lookahead Bias).

    Args:
        df (pd.DataFrame): DataFrame lượt đánh giá đã sắp xếp theo timestamp.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df)
    """
    # Tính thứ tự ngược theo thời gian cho từng user (0: tương tác mới nhất, 1: kề cuối, >=2: tương tác cũ)
    rank = df.groupby("user_id").cumcount(ascending=False)

    test_df = df[rank == 0].copy()
    val_df = df[rank == 1].copy()
    train_df = df[rank >= 2].copy()

    return train_df, val_df, test_df
