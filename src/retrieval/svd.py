"""Triển khai trích xuất ứng viên bằng phân rã nhân tử ẩn (TruncatedSVD Latent Dot Product).

Stage 1 Candidate Retriever sử dụng tích vô hướng trên không gian vector nhúng.
Độ phức tạp hiện tại: Exact brute-force dot product O(num_items * dim).
Lộ trình mở rộng: Chỉ mục tìm kiếm tiệm cận ANN (FAISS / ScaNN / HNSW).
"""

from __future__ import annotations

import numpy as np

from .base import Candidate, CandidateRetriever


class SVDRetriever(CandidateRetriever):
    """Trích xuất ứng viên dựa trên Tích vô hướng (Dot Product) từ TruncatedSVD embeddings."""

    def __init__(
        self,
        user_embeddings: np.ndarray,
        item_embeddings: np.ndarray,
        user_map: dict[int, int],
        item_map: dict[int, int],
        items: np.ndarray,
        seen_by_user: dict[int, set[int]] | None = None,
    ) -> None:
        """Khởi tạo SVDRetriever.

        Args:
            user_embeddings (np.ndarray): Ma trận nhúng người dùng [N, d].
            item_embeddings (np.ndarray): Ma trận nhúng sản phẩm [M, d].
            user_map (dict[int, int]): Ánh xạ user_id thực tế sang chỉ mục hàng ma trận.
            item_map (dict[int, int]): Ánh xạ item_id thực tế sang chỉ mục cột ma trận.
            items (np.ndarray): Mảng các item_id theo đúng thứ tự hàng của item_embeddings.
            seen_by_user (dict[int, set[int]] | None): Lịch sử sản phẩm đã tương tác trong train.
        """
        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings
        self.user_map = user_map
        self.item_map = item_map
        self.items = np.asarray(items)
        self.seen_by_user = seen_by_user or {}

        # Mảng tra cứu nhanh boolean mask hoặc hash map
        self._num_items = len(self.items)

    def retrieve(
        self,
        user_id: int,
        k: int = 200,
        filter_seen: bool = True,
    ) -> list[Candidate]:
        """Trích xuất top-k ứng viên có điểm tích vô hướng cao nhất.

        Args:
            user_id (int): ID người dùng.
            k (int): Số lượng ứng viên cần trích xuất (Mặc định: 200).
            filter_seen (bool): Loại trừ các item trong tập train của user (Mặc định: True).

        Returns:
            list[Candidate]: Danh sách Candidate sắp xếp giảm dần theo retrieval_score.
        """
        if user_id not in self.user_map or k <= 0:
            return []

        user_idx = self.user_map[user_id]
        user_vec = self.user_embeddings[user_idx]

        # Brute-force dot product: O(num_items * dim)
        latent_scores = self.item_embeddings @ user_vec

        # Lọc bỏ sản phẩm đã xem (Seen Filter guardrail)
        if filter_seen:
            seen_items = self.seen_by_user.get(user_id)
            if seen_items:
                seen_indices = np.isin(self.items, list(seen_items))
                latent_scores[seen_indices] = -np.inf

        available_count = int(np.isfinite(latent_scores).sum())
        if available_count == 0:
            return []

        retrieval_count = min(k, available_count)
        if retrieval_count == self._num_items:
            partition_indices = np.argsort(-latent_scores)
        else:
            top_partition = np.argpartition(-latent_scores, retrieval_count - 1)[
                :retrieval_count
            ]
            partition_indices = top_partition[np.argsort(-latent_scores[top_partition])]

        return [
            Candidate(
                item_id=int(self.items[idx]),
                retrieval_score=float(latent_scores[idx]),
            )
            for idx in partition_indices
        ]
