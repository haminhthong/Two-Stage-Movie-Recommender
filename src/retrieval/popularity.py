"""Triển khai trích xuất ứng viên dựa trên độ phổ biến toàn cục (Global Popularity Retriever).

Được sử dụng làm Baseline so sánh và làm chính sách Fallback cho trường hợp Cold-Start.
"""

from __future__ import annotations

from .base import Candidate, CandidateRetriever


class PopularityRetriever(CandidateRetriever):
    """Trích xuất ứng viên phổ biến nhất hệ thống."""

    def __init__(
        self,
        popular_items: list[int],
        popularity_scores: dict[int, float] | None = None,
        seen_by_user: dict[int, set[int]] | None = None,
    ) -> None:
        """Khởi tạo PopularityRetriever.

        Args:
            popular_items (list[int]): Danh sách ID phim sắp xếp giảm dần theo lượt tương tác.
            popularity_scores (dict[int, float] | None): Điểm số phổ biến chuẩn hóa [0, 1].
            seen_by_user (dict[int, set[int]] | None): Tập sản phẩm đã tương tác trong train.
        """
        self.popular_items = [int(x) for x in popular_items]
        self.popularity_scores = popularity_scores or {}
        self.seen_by_user = seen_by_user or {}

    def retrieve(
        self,
        user_id: int,
        k: int = 200,
        filter_seen: bool = True,
    ) -> list[Candidate]:
        """Trích xuất top-k phim phổ biến nhất (loại bỏ phim đã xem nếu filter_seen=True).

        Args:
            user_id (int): ID người dùng.
            k (int): Số lượng item cần lấy.
            filter_seen (bool): Có loại bỏ item trong train hay không.

        Returns:
            list[Candidate]: Danh sách Candidate theo độ phổ biến.
        """
        if k <= 0:
            return []

        seen = self.seen_by_user.get(user_id, set()) if filter_seen else set()
        candidates: list[Candidate] = []

        for item_id in self.popular_items:
            if item_id in seen:
                continue
            score = self.popularity_scores.get(item_id, 0.0)
            candidates.append(Candidate(item_id=item_id, retrieval_score=float(score)))
            if len(candidates) >= k:
                break

        return candidates
