"""Trừu tượng hóa Candidate Generator (Candidate Retrieval Abstraction).

Định nghĩa lớp dữ liệu Candidate phong phú (Rich Metadata) và lớp cơ sở trừu tượng
CandidateRetriever cho các thuật toán trích xuất ứng viên Tầng 1 (Stage 1).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Candidate:
    """Đối tượng ứng viên được trích xuất từ Tầng 1 kèm siêu dữ liệu nguồn.

    Attributes:
        item_id (int): ID định danh sản phẩm/phim.
        retrieval_score (float): Điểm tương quan trích xuất chính.
        retrieval_source (str): Nguồn trích xuất (ví dụ: 'svd', 'popularity', 'genre', 'multi_source').
        retrieval_rank (int): Thứ hạng xuất phát của ứng viên tại nguồn (0-indexed).
        source_scores (dict[str, float]): Điểm số từ từng nguồn riêng lẻ nếu được hợp nhất.
    """

    item_id: int
    retrieval_score: float
    retrieval_source: str = "svd"
    retrieval_rank: int = 0
    source_scores: dict[str, float] = field(default_factory=dict)


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
            list[Candidate]: Danh sách ứng viên kèm điểm số và thông tin nguồn.
        """
        raise NotImplementedError
