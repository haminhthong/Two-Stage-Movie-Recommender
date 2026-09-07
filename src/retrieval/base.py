"""Hợp đồng dùng chung cho Stage 1 candidate retrieval."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class Candidate:
    """Ứng viên kèm tín hiệu riêng của từng nguồn retrieval.

    Bốn trường đầu tiên được giữ lại để tương thích với code cũ. Các trường
    ``*_score``/``*_rank`` mới là contract chính; ``retrieval_score`` chỉ là
    alias thuận tiện (với ứng viên hợp nhất, nó là ``rrf_score``).
    Rank của từng source là 1-based vì được dùng trực tiếp trong RRF.
    """

    item_id: int
    retrieval_score: float = 0.0
    retrieval_source: str = "svd"
    retrieval_rank: int = 0
    source_scores: dict[str, float] = field(default_factory=dict)
    svd_score: float | None = None
    svd_rank: int | None = None
    popularity_score: float | None = None
    popularity_rank: int | None = None
    genre_score: float | None = None
    genre_rank: int | None = None
    rrf_score: float | None = None
    source_count: int | None = None

    def __post_init__(self) -> None:
        """Chuẩn hóa object cũ sang contract source-specific mới."""
        source_scores = dict(self.source_scores)
        explicit_scores = {
            "svd": self.svd_score,
            "popularity": self.popularity_score,
            "genre": self.genre_score,
        }

        # Khi khởi tạo theo contract mới, không để giá trị mặc định ``svd``
        # tạo ra một source giả cho candidate popularity/genre.
        if any(value is not None for value in explicit_scores.values()):
            source_scores = {
                source: float(value)
                for source, value in explicit_scores.items()
                if value is not None
            }
            if self.retrieval_score == 0.0 and source_scores:
                first_score = next(iter(source_scores.values()))
                object.__setattr__(self, "retrieval_score", first_score)
            if len(source_scores) == 1 and self.retrieval_source == "svd":
                object.__setattr__(self, "retrieval_source", next(iter(source_scores)))
        elif self.retrieval_source in {"svd", "popularity", "genre"}:
            source_scores.setdefault(self.retrieval_source, float(self.retrieval_score))

        svd_score = self.svd_score
        popularity_score = self.popularity_score
        genre_score = self.genre_score
        if svd_score is None and "svd" in source_scores:
            svd_score = float(source_scores["svd"])
        if popularity_score is None and "popularity" in source_scores:
            popularity_score = float(source_scores["popularity"])
        if genre_score is None and "genre" in source_scores:
            genre_score = float(source_scores["genre"])

        source_ranks = {
            "svd": self.svd_rank,
            "popularity": self.popularity_rank,
            "genre": self.genre_rank,
        }
        rank_values = {
            source: (int(rank) if rank is not None else self.retrieval_rank + 1)
            for source, rank in source_ranks.items()
            if source in source_scores
        }

        object.__setattr__(self, "source_scores", source_scores)
        object.__setattr__(self, "svd_score", svd_score)
        object.__setattr__(self, "popularity_score", popularity_score)
        object.__setattr__(self, "genre_score", genre_score)
        object.__setattr__(self, "svd_rank", rank_values.get("svd"))
        object.__setattr__(self, "popularity_rank", rank_values.get("popularity"))
        object.__setattr__(self, "genre_rank", rank_values.get("genre"))
        object.__setattr__(
            self,
            "rrf_score",
            self.rrf_score
            if self.rrf_score is not None
            else self.retrieval_score
            if self.retrieval_source == "multi_source"
            else None,
        )
        object.__setattr__(self, "source_count", len(source_scores))


class CandidateRetriever(ABC):
    """Lớp cơ sở trừu tượng cho các mô hình trích xuất ứng viên (Retrieval Models)."""

    @abstractmethod
    def retrieve(
        self,
        user_id: int,
        k: int = 200,
        filter_seen: bool = True,
        seen_items_override: Iterable[int] | None = None,
    ) -> list[Candidate]:
        """Trích xuất top-K ứng viên tiềm năng nhất cho người dùng.

        Args:
            user_id (int): ID người dùng.
            k (int): Số lượng ứng viên cần trích xuất (Mặc định: 200).
            filter_seen (bool): Có loại trừ item đã xem hay không.
            seen_items_override: Tập seen chỉ có hiệu lực trong request hiện tại.
                Retriever không được mutate state dùng chung khi nhận override.

        Returns:
            list[Candidate]: Danh sách ứng viên kèm điểm số và thông tin nguồn.
        """
        raise NotImplementedError
