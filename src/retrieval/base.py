"""Trừu tượng hóa Candidate Generator (Candidate Retrieval Abstraction).

Định nghĩa lớp dữ liệu Candidate và lớp cơ sở trừu tượng CandidateRetriever
cho tất cả các thuật toán trích xuất ứng viên trong Tầng 1 (Stage 1).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    """Đối tượng ứng viên được trích xuất từ Tầng 1.

    Attributes:
        item_id (int): ID định danh sản phẩm/phim.
        retrieval_score (float): Điểm tương quan trích xuất (ví dụ: Dot product hoặc Popularity).
    """

    item_id: int
    retrieval_score: float


class CandidateRetriever(ABC):
    """Lớp cơ sở trừu tượng cho các mô hình trích xuất ứng viên (Retrieval Models)."""

    @abstractmethod
    def retrieve(
        self,
        user_id: int,
        k: int = 200,
        filter_seen: bool = True,
    ) -> list[Candidate]:
        """Trích xuất top-K ứng viên tiềm năng nhất cho người dùng.

        Args:
            user_id (int): ID người dùng.
            k (int): Số lượng ứng viên cần trích xuất (Mặc định: 200).
            filter_seen (bool): Có loại trừ các item người dùng đã xem trong quá khứ hay không.

        Returns:
            list[Candidate]: Danh sách ứng viên kèm điểm số.
        """
        raise NotImplementedError
