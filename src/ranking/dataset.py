"""Xây dựng tập dữ liệu huấn luyện xếp hạng (Stage-2 Rank Training Dataset Builder).

Tuân thủ hợp đồng dữ liệu:
- Dữ liệu lịch sử tại thời điểm tạo ứng viên CHỈ lấy trước mốc t_rank (không rò rỉ tương lai).
- Nhãn y = 1 cho next positive target item.
- Nhãn y = 0 cho các candidate khác trong retrieval pool (Sampled Non-Interaction Proxy).
  LƯU Ý MINH BẠCH: MovieLens không có impression logs, do đó nhãn 0 là proxy không tương tác,
  không khẳng định chắc chắn người dùng đã nhìn thấy và từ chối item.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd

from ..retrieval.base import Candidate, CandidateRetriever
from .features import CandidateFeatureBuilder, CandidateFeatures


class RankDatasetBuilder:
    """Tạo tập dữ liệu ma trận đặc trưng (X, y, groups) phục vụ huấn luyện mô hình Ranker."""

    def __init__(
        self,
        feature_builder: CandidateFeatureBuilder,
        candidate_k: int = 100,
    ) -> None:
        """Khởi tạo RankDatasetBuilder.

        Args:
            feature_builder: Bộ trích xuất CandidateFeatureBuilder.
            candidate_k: Số lượng ứng viên trích xuất cho mỗi user để tạo mẫu negative.
        """
        self.feature_builder = feature_builder
        self.candidate_k = candidate_k

    def build_dataset(
        self,
        rank_train_df: pd.DataFrame,
        retriever: CandidateRetriever,
        max_users: int | None = None,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Tạo ma trận (X, y, groups) cho Ranker.

        Args:
            rank_train_df: DataFrame chứa ground truth target tích cực (user_id, item_id).
            retriever: Bộ trích xuất Stage 1 đã được fit trên retrieval_train_df.
            max_users: Giới hạn số lượng user huấn luyện (None nếu dùng toàn bộ).
            seed: Seed ngẫu nhiên.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                X (N_samples, N_features),
                y (N_samples,),
                groups (N_users,) - số lượng mẫu cho mỗi user query group.
        """
        user_target_map = dict(zip(rank_train_df["user_id"], rank_train_df["item_id"], strict=True))
        users = list(user_target_map.keys())

        if max_users is not None and len(users) > max_users:
            rng = np.random.default_rng(seed)
            users = list(rng.choice(users, size=max_users, replace=False))

        X_rows: list[np.ndarray] = []
        y_rows: list[int] = []
        group_counts: list[int] = []

        for u_id in users:
            target_item = user_target_map[u_id]
            candidates = retriever.retrieve(user_id=u_id, k=self.candidate_k, filter_seen=True)
            if not candidates:
                continue

            # Kiểm tra xem target_item có lọt vào candidates không
            cand_item_ids = {c.item_id for c in candidates}
            if target_item not in cand_item_ids:
                # Bổ sung target item vào pool để luôn có positive sample cho ranker học phân biệt
                candidates = list(candidates)
                candidates.append(
                    Candidate(
                        item_id=target_item,
                        retrieval_score=0.0,
                        retrieval_source="ground_truth",
                        retrieval_rank=len(candidates),
                        source_scores={},
                    )
                )

            features = self.feature_builder.build_features(candidates, user_id=u_id)
            if not features:
                continue

            num_cands = len(features)
            group_counts.append(num_cands)

            for feat in features:
                X_rows.append(feat.to_feature_vector())
                y_rows.append(1 if feat.item_id == target_item else 0)

        X = np.vstack(X_rows).astype(np.float32) if X_rows else np.empty((0, 15), dtype=np.float32)
        y = np.array(y_rows, dtype=np.int32)
        groups = np.array(group_counts, dtype=np.int32)

        return X, y, groups
