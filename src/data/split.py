"""Module phân chia dữ liệu theo thời gian (Temporal Split Strategies).

Quy ước:
1. Giao thức là 'Per-User Temporal Holdout': đảm bảo thứ tự thời gian cho từng người dùng,
   loại bỏ rò rỉ tương lai (no future leakage). Không overclaim là strict global point-in-time snapshot.
2. Hỗ trợ 2 chiến lược:
   - 4-Way Temporal Split: Retrieval History -> Rank-Train Target -> Validation Target -> Test Target.
   - 3-Way Temporal Split: Train -> Validation -> Test (Tương thích ngược).
"""

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


def global_temporal_windows(
    df: pd.DataFrame,
    train_fraction: float = 0.60,
    rank_fraction: float = 0.15,
    validation_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tạo expanding global temporal windows cho backtest cross-user.

    Các dòng được sắp theo timestamp toàn cục; vì vậy snapshot của mỗi window
    không thể lấy dữ liệu từ tương lai của user khác.
    """
    if not 0 < train_fraction < 1 or rank_fraction < 0 or validation_fraction < 0:
        raise ValueError("Các tỷ lệ temporal window không hợp lệ.")
    if train_fraction + rank_fraction + validation_fraction >= 1:
        raise ValueError("Tổng train/rank/validation phải nhỏ hơn 1.")

    ordered = df.sort_values(["timestamp", "user_id", "item_id"]).reset_index(drop=True)
    row_count = len(ordered)
    train_end = max(1, int(row_count * train_fraction))
    rank_end = max(train_end, int(row_count * (train_fraction + rank_fraction)))
    val_end = max(
        rank_end,
        int(row_count * (train_fraction + rank_fraction + validation_fraction)),
    )
    return (
        ordered.iloc[:train_end].copy(),
        ordered.iloc[train_end:rank_end].copy(),
        ordered.iloc[rank_end:val_end].copy(),
        ordered.iloc[val_end:].copy(),
    )


def time_split(
    df: pd.DataFrame,
    rating_threshold: float = 4.0,
    min_positive: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Phân chia dữ liệu theo chiến lược Leave-Last-Two Positive Per User (3-Way Split).

    - Test: Tương tác tích cực mới nhất (pos_rank = 0).
    - Validation: Tương tác tích cực kề cuối (pos_rank = 1).
    - Train: Toàn bộ tương tác diễn ra trước mốc thời gian của Validation (timestamp < val_timestamp).
    """
    positive_df = df.sort_values(["user_id", "timestamp", "item_id"]).copy()
    positive_df = positive_df[positive_df["rating"] >= rating_threshold].copy()
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


def temporal_split_four_way(
    df: pd.DataFrame,
    rating_threshold: float = 4.0,
    min_positive: int = 4,
    protocol: str = "per_user",
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
    if protocol == "global":
        return global_temporal_windows(df)
    if protocol != "per_user":
        raise ValueError("protocol phải là 'per_user' hoặc 'global'.")

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
