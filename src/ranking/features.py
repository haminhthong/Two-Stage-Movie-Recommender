"""Trích xuất và chuẩn hóa đặc trưng ứng viên (Candidate Feature Engineering).

Chuyển đổi điểm thô từ Tầng 1 (Latent Dot Product) và kết hợp các tín hiệu
độ phổ biến (Log-transformed Popularity Prior) cùng sở thích thể loại (User Genre Affinity).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..retrieval.base import Candidate


@dataclass(frozen=True)
class CandidateFeatures:
    """Tập hợp đặc trưng của một ứng viên tại Tầng 2.

    Attributes:
        item_id (int): ID định danh sản phẩm/phim.
        latent_score (float): Điểm tương quan ẩn chuẩn hóa trong khoảng [0, 1].
        popularity_score (float): Điểm phổ biến chuẩn hóa [0, 1] (thường dùng log1p).
        genre_affinity (float): Độ tương hợp thể loại với lịch sử người dùng [0, 1].
    """

    item_id: int
    latent_score: float
    popularity_score: float
    genre_affinity: float = 0.0


class CandidateFeatureBuilder:
    """Xây dựng và chuẩn hóa vector đặc trưng cho danh sách ứng viên."""

    def __init__(
        self,
        popularity_scores: dict[int, float],
        genre_map: dict[int, set[str]] | None = None,
        user_genre_profiles: dict[int, dict[str, float]] | None = None,
    ) -> None:
        """Khởi tạo CandidateFeatureBuilder.

        Args:
            popularity_scores (dict[int, float]): Điểm phổ biến chuẩn hóa của từng item.
            genre_map (dict[int, set[str]] | None): Bản đồ thể loại của từng item.
            user_genre_profiles (dict[int, dict[str, float]] | None): Phân phối thể loại lịch sử của user.
        """
        self.popularity_scores = popularity_scores
        self.genre_map = genre_map or {}
        self.user_genre_profiles = user_genre_profiles or {}

    def build_features(
        self,
        candidates: list[Candidate],
        user_id: int | None = None,
    ) -> list[CandidateFeatures]:
        """Tính toán và chuẩn hóa đặc trưng cho toàn bộ ứng viên trong pool.

        Args:
            candidates (list[Candidate]): Danh sách ứng viên từ Stage 1.
            user_id (int | None): ID người dùng để tra cứu genre affinity (tùy chọn).

        Returns:
            list[CandidateFeatures]: Danh sách đặc trưng đã chuẩn hóa.
        """
        if not candidates:
            return []

        raw_scores = np.array([c.retrieval_score for c in candidates], dtype=np.float32)
        min_s = float(raw_scores.min())
        max_s = float(raw_scores.max())
        denom = max_s - min_s
        if denom <= 1e-9:
            norm_latent = np.ones(len(candidates), dtype=np.float32)
        else:
            norm_latent = (raw_scores - min_s) / denom

        user_profile = self.user_genre_profiles.get(user_id, {}) if user_id is not None else {}

        features_list: list[CandidateFeatures] = []
        for i, cand in enumerate(candidates):
            item_id = cand.item_id
            pop_sc = float(self.popularity_scores.get(item_id, 0.0))

            # Tính User Genre Affinity: tổng xác suất các thể loại phim mà user từng xem
            affinity = 0.0
            if user_profile:
                item_genres = self.genre_map.get(item_id, set())
                if item_genres:
                    overlap_sum = sum(user_profile.get(g, 0.0) for g in item_genres)
                    affinity = float(overlap_sum / len(item_genres))

            features_list.append(
                CandidateFeatures(
                    item_id=item_id,
                    latent_score=float(norm_latent[i]),
                    popularity_score=pop_sc,
                    genre_affinity=min(max(affinity, 0.0), 1.0),
                )
            )

        return features_list
