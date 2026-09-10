"""Phân chia dữ liệu theo giao thức per-user temporal holdout."""

from __future__ import annotations

import pandas as pd


def seen_items_before(
    df: pd.DataFrame,
    user_id: int,
    as_of_timestamp: int,
) -> set[int]:
    """Lấy toàn bộ item user đã tương tác trước mốc request/target."""
    history = df[(df["user_id"] == user_id) & (df["timestamp"] < int(as_of_timestamp))]
    return set(history["item_id"].astype(int).tolist())


def temporal_split(
    df: pd.DataFrame,
    rating_threshold: float = 4.0,
    min_positive: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Phân chia 4 tập độc lập theo thời gian (4-Way Per-User Temporal Holdout).

    Mục tiêu: Đảm bảo tập Rank-Train hoàn toàn độc lập với Validation (dùng tune ranker/MMR)
    và Test (chỉ report một lần duy nhất).

    Cấu trúc phân chia cho user có >= min_positive tương tác tích cực:
    - Test set (pos_rank = 0): Tương tác tích cực mới nhất (mốc t_test).
    - Validation set (pos_rank = 1): Tương tác tích cực thứ hai từ dưới lên (mốc t_val).
    - Rank-Train set (pos_rank = 2): Tương tác tích cực thứ ba từ dưới lên (mốc t_rank).
    - Retrieval-Train set: Toàn bộ tương tác diễn ra TRƯỚC mốc t_rank của user (timestamp < t_rank).

    Với user có < min_positive tương tác tích cực, toàn bộ tương tác của họ được lưu trong Retrieval-Train set.

    Returns:
        tuple: (retrieval_train_df, rank_train_df, val_df, test_df)
    """
    positive_df = df.sort_values(["user_id", "timestamp", "item_id"]).copy()
    positive_df = positive_df[positive_df["rating"] >= rating_threshold].copy()
    user_pos_counts = positive_df.groupby("user_id").size()
    eligible_users = user_pos_counts[user_pos_counts >= min_positive].index

    pos_eligible = positive_df[positive_df["user_id"].isin(eligible_users)].copy()
    pos_rank = pos_eligible.groupby("user_id").cumcount(ascending=False)

    test_df = pos_eligible[pos_rank == 0].copy().reset_index(drop=True)
    val_df = pos_eligible[pos_rank == 1].copy().reset_index(drop=True)
    rank_train_df = pos_eligible[pos_rank == 2].copy().reset_index(drop=True)

    # Retrieval history tại thời điểm huấn luyện Ranker: chỉ lấy trước t_rank
    rank_timestamps = rank_train_df.set_index("user_id")["timestamp"]
    cutoff_series = df["user_id"].map(rank_timestamps)
    retrieval_train_mask = cutoff_series.isna() | (df["timestamp"] < cutoff_series)
    retrieval_train_df = df[retrieval_train_mask].copy().reset_index(drop=True)

    return retrieval_train_df, rank_train_df, val_df, test_df
