"""Chính sách gợi ý cho người dùng mới (Unified Cold-Start Policies).

Bao gồm:
1. Chiến lược Phổ biến Toàn cục (Global Popularity Fallback) cho người dùng không cung cấp sở thích.
2. Chiến lược Phổ biến theo Thể loại (Genre-Aware Popularity Policy) cho người dùng khai báo sở thích ban đầu.
"""

from __future__ import annotations

from typing import Any


class ColdStartPolicy:
    """Điều phối và sinh danh sách gợi ý cho các kịch bản người dùng Cold-Start."""

    def __init__(
        self,
        popular_items: list[int],
        genre_map: dict[int, set[str]] | None = None,
        title_map: dict[int, str] | None = None,
        popularity_counts: dict[int, int] | None = None,
    ) -> None:
        """Khởi tạo ColdStartPolicy.

        Args:
            popular_items (list[int]): Danh sách ID phim xếp theo độ phổ biến giảm dần.
            genre_map (dict[int, set[str]] | None): Bản đồ thể loại của từng phim.
            title_map (dict[int, str] | None): Bản đồ tiêu đề phim.
            popularity_counts (dict[int, int] | None): Số lượt tương tác thực tế của từng phim.
        """
        self.popular_items = [int(x) for x in popular_items]
        self.genre_map = genre_map or {}
        self.title_map = title_map or {}
        self.popularity_counts = popularity_counts or {}

    def get_recommendations(
        self,
        preferred_genres: list[str] | None = None,
        k: int = 10,
        seen_items: set[int] | None = None,
    ) -> tuple[list[int], str]:
        """Tạo danh sách item gợi ý cho user mới.

        Args:
            preferred_genres (list[str] | None): Danh sách thể loại mong muốn.
            k (int): Số lượng item cần lấy.
            seen_items (set[int] | None): Tập sản phẩm cần bỏ qua nếu có.

        Returns:
            tuple[list[int], str]: (Danh sách ID phim, Tên chiến lược áp dụng)
        """
        if k <= 0:
            return [], "cold_start_empty"

        seen = seen_items or set()
        selected: list[int] = []

        if preferred_genres:
            pref_set = {g.strip().lower() for g in preferred_genres if g.strip()}
            if pref_set:
                for item in self.popular_items:
                    if item in seen:
                        continue
                    item_genres = {g.lower() for g in self.genre_map.get(item, set())}
                    if item_genres & pref_set:
                        selected.append(item)
                        if len(selected) >= k:
                            break

                # Nếu chưa đủ k items, bù bằng các phim phổ biến chung chưa có trong selected
                if len(selected) < k:
                    for item in self.popular_items:
                        if item not in seen and item not in selected:
                            selected.append(item)
                            if len(selected) >= k:
                                break

                return selected[:k], "cold_start_genre_aware"

        # Kịch bản phổ biến chung (Global Popularity)
        for item in self.popular_items:
            if item not in seen:
                selected.append(item)
                if len(selected) >= k:
                    break

        return selected[:k], "cold_start_popularity"

    def enrich_items(self, item_ids: list[int]) -> list[dict[str, Any]]:
        """Gắn kèm thông tin metadata chi tiết cho danh sách phim."""
        return [
            {
                "item_id": item_id,
                "title": self.title_map.get(item_id, f"Movie {item_id}"),
                "genres": sorted(list(self.genre_map.get(item_id, set()))),
                "interaction_count": int(self.popularity_counts.get(item_id, 0)),
                "explanation": "Cold-start recommendation based on system popularity",
            }
            for item_id in item_ids
        ]
