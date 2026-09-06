"""Triển khai trích xuất ứng viên theo thể loại yêu thích (Genre-Based Content Retriever)."""

from __future__ import annotations

from .base import Candidate, CandidateRetriever


class GenreRetriever(CandidateRetriever):
    """Trích xuất ứng viên phổ biến nhất thuộc các thể loại yêu thích của người dùng."""

    def __init__(
        self,
        popular_items: list[int],
        genre_map: dict[int, set[str]],
        user_genre_profiles: dict[int, dict[str, float]],
        popularity_scores: dict[int, float] | None = None,
        seen_by_user: dict[int, set[int]] | None = None,
    ) -> None:
        """Khởi tạo GenreRetriever.

        Args:
            popular_items: Danh sách item sắp xếp theo độ phổ biến giảm dần.
            genre_map: Ánh xạ item_id sang tập hợp các thể loại.
            user_genre_profiles: Phân phối thể loại tích cực của từng user.
            popularity_scores: Điểm phổ biến chuẩn hóa [0, 1].
            seen_by_user: Lịch sử sản phẩm đã tương tác trong train.
        """
        self.popular_items = [int(x) for x in popular_items]
        self.genre_map = genre_map
        self.user_genre_profiles = user_genre_profiles
        self.popularity_scores = popularity_scores or {}
        self.seen_by_user = seen_by_user or {}

    def retrieve(
        self,
        user_id: int,
        k: int = 50,
        filter_seen: bool = True,
    ) -> list[Candidate]:
        """Trích xuất top-k phim thuộc thể loại ưa thích nhất của user."""
        if k <= 0:
            return []

        user_profile = self.user_genre_profiles.get(user_id, {})
        seen = self.seen_by_user.get(user_id, set()) if filter_seen else set()

        if not user_profile:
            # Fallback nếu user chưa có genre profile
            cands: list[Candidate] = []
            rank = 0
            for it in self.popular_items:
                if it in seen:
                    continue
                sc = float(self.popularity_scores.get(it, 0.0))
                cands.append(
                    Candidate(
                        item_id=it,
                        retrieval_score=sc,
                        retrieval_source="genre",
                        retrieval_rank=rank,
                        source_scores={"genre": sc},
                    )
                )
                rank += 1
                if len(cands) >= k:
                    break
            return cands

        # Lấy top thể loại có tỷ trọng cao nhất
        top_genres = set(
            sorted(user_profile.keys(), key=lambda g: user_profile[g], reverse=True)[:3]
        )

        candidates: list[Candidate] = []
        rank = 0
        for item_id in self.popular_items:
            if item_id in seen:
                continue
            item_genres = self.genre_map.get(item_id, set())
            overlap = len(item_genres & top_genres)
            if overlap > 0:
                affinity = sum(user_profile.get(g, 0.0) for g in item_genres) / max(1, len(item_genres))
                pop_sc = self.popularity_scores.get(item_id, 0.0)
                genre_score = float(0.6 * affinity + 0.4 * pop_sc)
                candidates.append(
                    Candidate(
                        item_id=item_id,
                        retrieval_score=genre_score,
                        retrieval_source="genre",
                        retrieval_rank=rank,
                        source_scores={"genre": genre_score},
                    )
                )
                rank += 1
                if len(candidates) >= k:
                    break

        return candidates
