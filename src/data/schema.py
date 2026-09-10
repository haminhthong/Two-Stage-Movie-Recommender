"""Context bất biến dùng trong một lần suy luận."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


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
