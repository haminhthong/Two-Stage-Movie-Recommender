"""Xử lý ma trận tương tác và hồ sơ sở thích người dùng tuân thủ Implicit-Positive Contract."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def build_positive_interaction_matrix(
    train_df: pd.DataFrame,
    user_map: dict[int, int],
    item_map: dict[int, int],
    rating_threshold: float = 4.0,
) -> csr_matrix:
    """Tạo ma trận tương tác ngầm định (Implicit CSR Matrix) chỉ từ các đánh giá tích cực.

    Rating dưới ngưỡng rating_threshold không được lưu trữ dưới dạng explicit zero.
    """
    positive = train_df[train_df["rating"] >= rating_threshold]
    rows = positive["user_id"].map(user_map).to_numpy()
    columns = positive["item_id"].map(item_map).to_numpy()
    values = np.ones(len(positive), dtype=np.float32)
    matrix = csr_matrix((values, (rows, columns)), shape=(len(user_map), len(item_map)))
    matrix.eliminate_zeros()
    return matrix


def build_user_genre_profiles(
    train_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    rating_threshold: float = 4.0,
) -> dict[int, dict[str, float]]:
    """Xây dựng hồ sơ phân phối thể loại của người dùng CHỈ từ tương tác tích cực.

    Khắc phục triệt để lỗi logic P0.1:
    - Rating >= rating_threshold mới được tính là sở thích tích cực.
    - Rating < rating_threshold (phim dở, không thích) tuyệt đối không được gộp vào hồ sơ sở thích.
    """
    # LỌC CHẶT CHẼ: Chỉ lấy positive interactions
    positive_train = train_df[train_df["rating"] >= rating_threshold].copy()
    train_with_genres = positive_train.merge(
        movies_df[["item_id", "genres"]], on="item_id", how="left"
    )

    user_genre_profiles: dict[int, dict[str, float]] = {}
    for u_id, group in train_with_genres.groupby("user_id"):
        g_counts: dict[str, int] = {}
        total_g = 0
        for g_str in group["genres"].dropna():
            for g in str(g_str).split("|"):
                g = g.strip()
                if g:
                    g_counts[g] = g_counts.get(g, 0) + 1
                    total_g += 1
        if total_g > 0:
            user_genre_profiles[int(u_id)] = {
                g: count / total_g for g, count in g_counts.items()
            }

    return user_genre_profiles


def extract_seen_items(train_df: pd.DataFrame) -> dict[int, set[int]]:
    """Trích xuất tập sản phẩm đã từng xem/tương tác trong quá khứ của từng user.

    Theo Seen Filter Guardrail: Mọi item đã xem (kể cả rating < rating_threshold)
    đều thuộc tập seen để hệ thống loại bỏ, không lãng phí slot gợi ý sản phẩm cũ.
    """
    return (
        train_df.groupby("user_id")["item_id"]
        .apply(lambda x: set(map(int, x)))
        .to_dict()
    )
