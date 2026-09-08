"""Schema định nghĩa dữ liệu và hợp đồng tương tác (Data Contracts & Schemas)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class InteractionEvent:
    """Sự kiện tương tác người dùng - sản phẩm."""

    user_id: int
    item_id: int
    rating: float
    timestamp: int


@dataclass(frozen=True)
class MovieMetadata:
    """Thông tin chi tiết một bộ phim."""

    item_id: int
    title: str
    genres: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DataContract:
    """Quy ước định nghĩa dữ liệu và đảm bảo tính toàn vẹn (Data Contract).

    Quy tắc 1: Implicit Positive Feedback -> Chỉ Rating >= rating_threshold (mặc định 4.0) mới là positive.
    Quy tắc 2: Seen Filter -> Mọi item user đã từng xem (kể cả rating < 4.0) đều bị loại bỏ khi suy luận.
    Quy tắc 3: Per-User Temporal Holdout -> Lịch sử user tại thời điểm suy luận không chứa tương tác tương lai.
    """

    feedback_type: str = "implicit_positive"
    rating_threshold: float = 4.0
    split_protocol: str = "per_user_temporal_holdout"
    seen_filter_scope: str = "all_interactions_before_cutoff"
    negative_sampling_proxy: str = "unobserved_candidates_without_impressions"


@dataclass(frozen=True)
class RecommendationContext:
    """Trạng thái bất biến của một request tại mốc ``as_of_timestamp``."""

    user_id: int
    as_of_timestamp: int | None = None
    seen_item_ids: frozenset[int] = frozenset()
    positive_item_ids: frozenset[int] = frozenset()
    recent_item_ids: frozenset[int] = frozenset()

    @classmethod
    def from_items(
        cls,
        user_id: int,
        seen_item_ids: Iterable[int] = (),
        positive_item_ids: Iterable[int] = (),
        recent_item_ids: Iterable[int] = (),
        as_of_timestamp: int | None = None,
    ) -> RecommendationContext:
        """Tạo context và chuẩn hóa ID về int/frozen set."""
        return cls(
            user_id=int(user_id),
            as_of_timestamp=(
                int(as_of_timestamp) if as_of_timestamp is not None else None
            ),
            seen_item_ids=frozenset(int(item_id) for item_id in seen_item_ids),
            positive_item_ids=frozenset(int(item_id) for item_id in positive_item_ids),
            recent_item_ids=frozenset(int(item_id) for item_id in recent_item_ids),
        )

    @property
    def effective_seen_items(self) -> frozenset[int]:
        """Seen history cộng recent session items, không mutate context."""
        return self.seen_item_ids | self.recent_item_ids
