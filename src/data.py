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
    rating_threshold: float = 4.0,
    min_positive: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Phân chia dữ liệu theo chiến lược Leave-Last-Two Positive Per User theo thời gian.

    Chiến lược này phản ánh bài toán thế giới thực và đồng nhất với mục tiêu Implicit Recommender:
    - Lọc các tương tác tích cực (rating >= rating_threshold).
    - Với các user có ít nhất min_positive tương tác tích cực:
      * Tương tác tích cực mới nhất làm Test set (ground truth tích cực cho Test).
      * Tương tác tích cực kề cuối làm Validation set (ground truth tích cực cho Validation).
      * Tất cả tương tác diễn ra trước mốc thời gian của validation set được đưa vào
        Tập Huấn luyện (Train set), loại bỏ hoàn toàn Lookahead Data Leakage.
    - Với user có ít hơn min_positive tương tác tích cực, toàn bộ tương tác được giữ trong Train set.

    Args:
        df (pd.DataFrame): DataFrame lượt đánh giá đã sắp xếp theo timestamp.
        rating_threshold (float): Ngưỡng rating để coi là tương tác tích cực. (Mặc định: 4.0)
        min_positive (int): Số tương tác tích cực tối thiểu để user có mặt trong val/test. (Mặc định: 3)

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df)
    """
    positive_df = df[df["rating"] >= rating_threshold].copy()
    user_pos_counts = positive_df.groupby("user_id").size()
    eligible_users = user_pos_counts[user_pos_counts >= min_positive].index

    pos_eligible = positive_df[positive_df["user_id"].isin(eligible_users)].copy()
    pos_rank = pos_eligible.groupby("user_id").cumcount(ascending=False)

    test_df = pos_eligible[pos_rank == 0].copy()
    val_df = pos_eligible[pos_rank == 1].copy()

    val_timestamps = val_df.set_index("user_id")["timestamp"]
    cutoff_series = df["user_id"].map(val_timestamps)
    train_mask = cutoff_series.isna() | (df["timestamp"] < cutoff_series)
    train_df = df[train_mask].copy()

    return train_df, val_df, test_df

