"""Trích xuất và chuẩn hóa đặc trưng ứng viên (Candidate Feature Engineering).

Kết hợp đa chiều:
1. User features (hoạt động, số lượng positive, điểm trung bình)
2. Item features (phổ biến, đánh giá trung bình, số thể loại, percentile)
3. User x Item interaction (raw latent score, normalized score, rank, genre affinity)
4. Multi-source signals (svd, popularity, genre, source count)

Tối ưu hiệu năng: Tiền tính toán percentile từ điển O(1) và sinh ma trận 2D trực tiếp.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np

from ..retrieval.base import Candidate

FEATURE_NAMES = [
    "raw_latent_score",
    "normalized_latent_score",
    "retrieval_rank_percentile",
    "popularity_score",
    "genre_affinity",
    "user_positive_count",
    "user_avg_rating",
    "item_rating_count",
    "item_avg_rating",
    "item_genre_count",
    "item_popularity_percentile",
    "retrieved_by_svd",
    "retrieved_by_popularity",
    "retrieved_by_genre",
    "source_count",
]


@dataclass(frozen=True)
class CandidateFeatures:
    """Tập hợp đặc trưng của một ứng viên tại Tầng 2.

    Duy trì 100% tương thích ngược các trường cốt lõi (item_id, latent_score, popularity_score, genre_affinity).
    """

    item_id: int
    latent_score: float           # Normalized latent score [0, 1]
    popularity_score: float       # Log1p popularity score [0, 1]
    genre_affinity: float = 0.0

    # Extended Stage-2 Features
    raw_latent_score: float = 0.0
    retrieval_rank: int = 0
    retrieval_rank_percentile: float = 0.0
    user_positive_count: int = 0
    user_avg_rating: float = 4.0
    item_rating_count: int = 0
    item_avg_rating: float = 3.5
    item_genre_count: int = 1
    item_popularity_percentile: float = 0.5
    retrieved_by_svd: float = 1.0
    retrieved_by_popularity: float = 0.0
    retrieved_by_genre: float = 0.0
    source_count: float = 1.0

    def to_feature_vector(self) -> np.ndarray:
        """Chuyển đổi thành vector số học dùng cho Scikit-Learn hoặc XGBoost."""
        return np.array(
            [
                self.raw_latent_score,
                self.latent_score,
                self.retrieval_rank_percentile,
                self.popularity_score,
                self.genre_affinity,
                float(self.user_positive_count),
                self.user_avg_rating,
                float(self.item_rating_count),
                self.item_avg_rating,
                float(self.item_genre_count),
                self.item_popularity_percentile,
                self.retrieved_by_svd,
                self.retrieved_by_popularity,
                self.retrieved_by_genre,
                self.source_count,
            ],
            dtype=np.float32,
        )


class CandidateFeatureBuilder:
    """Xây dựng và chuẩn hóa vector đặc trưng phong phú cho danh sách ứng viên."""

    def __init__(
        self,
        popularity_scores: dict[int, float],
        genre_map: dict[int, set[str]] | None = None,
        user_genre_profiles: dict[int, dict[str, float]] | None = None,
        user_stats: dict[int, dict[str, float]] | None = None,
        item_stats: dict[int, dict[str, float]] | None = None,
    ) -> None:
        """Khởi tạo CandidateFeatureBuilder."""
        self.popularity_scores = popularity_scores
        self.genre_map = genre_map or {}
        self.user_genre_profiles = user_genre_profiles or {}
        self.user_stats = user_stats or {}
        self.item_stats = item_stats or {}

        # Tiền tính toán O(1) lookups cho percentile và genre count
        pop_values = sorted(self.popularity_scores.values())
        num_pop = max(1, len(pop_values))
        arr_pop = np.array(pop_values)

        self.pop_percentiles: dict[int, float] = {}
        for it_id, sc in self.popularity_scores.items():
            idx = int(np.searchsorted(arr_pop, sc))
            self.pop_percentiles[it_id] = float(idx / num_pop)

        self.item_genre_counts: dict[int, int] = {
            it_id: max(1, len(genres)) for it_id, genres in self.genre_map.items()
        }

    def build_feature_matrix(
        self,
        candidates: Sequence[Candidate],
        user_id: int | None = None,
    ) -> np.ndarray:
        """Sinh trực tiếp ma trận 2D (N, 15) float32 liên tục trong RAM để tối ưu tốc độ inference."""
        n = len(candidates)
        if n == 0:
            return np.empty((0, 15), dtype=np.float32)

        raw_scores = np.fromiter((c.retrieval_score for c in candidates), dtype=np.float32, count=n)
        min_s = float(raw_scores.min())
        max_s = float(raw_scores.max())
        denom = max_s - min_s
        norm_latent = (raw_scores - min_s) / denom if denom > 1e-9 else np.ones(n, dtype=np.float32)

        user_profile = self.user_genre_profiles.get(user_id, {}) if user_id is not None else {}
        u_stat = self.user_stats.get(user_id, {}) if user_id is not None else {}
        user_pos_count = float(u_stat.get("positive_count", 10.0))
        user_avg_rat = float(u_stat.get("avg_rating", 3.8))

        mat = np.empty((n, 15), dtype=np.float32)
        inv_pool = 1.0 / n

        for i, cand in enumerate(candidates):
            item_id = cand.item_id
            pop_sc = self.popularity_scores.get(item_id, 0.0)

            # Genre affinity
            affinity = 0.0
            if user_profile:
                item_genres = self.genre_map.get(item_id)
                if item_genres:
                    overlap_sum = sum(user_profile.get(g, 0.0) for g in item_genres)
                    affinity = overlap_sum / len(item_genres)

            i_stat = self.item_stats.get(item_id, {})
            item_cnt = float(i_stat.get("rating_count", 50.0))
            item_avg_rat = float(i_stat.get("avg_rating", 3.5))
            genre_cnt = float(self.item_genre_counts.get(item_id, 1))
            pop_perc = self.pop_percentiles.get(item_id, 0.5)

            src_scores = cand.source_scores
            by_svd = 1.0 if (src_scores and "svd" in src_scores) or cand.retrieval_source == "svd" else 0.0
            by_pop = 1.0 if (src_scores and "popularity" in src_scores) or cand.retrieval_source == "popularity" else 0.0
            by_genre = 1.0 if (src_scores and "genre" in src_scores) or cand.retrieval_source == "genre" else 0.0
            src_cnt = float(len(src_scores)) if src_scores else 1.0

            mat[i, 0] = raw_scores[i]
            mat[i, 1] = norm_latent[i]
            mat[i, 2] = 1.0 - (i * inv_pool)
            mat[i, 3] = pop_sc
            mat[i, 4] = affinity
            mat[i, 5] = user_pos_count
            mat[i, 6] = user_avg_rat
            mat[i, 7] = item_cnt
            mat[i, 8] = item_avg_rat
            mat[i, 9] = genre_cnt
            mat[i, 10] = pop_perc
            mat[i, 11] = by_svd
            mat[i, 12] = by_pop
            mat[i, 13] = by_genre
            mat[i, 14] = src_cnt

        return mat

    def build_features(
        self,
        candidates: Sequence[Candidate],
        user_id: int | None = None,
    ) -> list[CandidateFeatures]:
        """Tính toán và chuẩn hóa vector đặc trưng cho toàn bộ ứng viên trong pool."""
        n = len(candidates)
        if n == 0:
            return []

        mat = self.build_feature_matrix(candidates, user_id=user_id)

        features_list: list[CandidateFeatures] = []
        for i, cand in enumerate(candidates):
            features_list.append(
                CandidateFeatures(
                    item_id=cand.item_id,
                    latent_score=float(mat[i, 1]),
                    popularity_score=float(mat[i, 3]),
                    genre_affinity=min(max(float(mat[i, 4]), 0.0), 1.0),
                    raw_latent_score=float(mat[i, 0]),
                    retrieval_rank=i,
                    retrieval_rank_percentile=float(mat[i, 2]),
                    user_positive_count=int(mat[i, 5]),
                    user_avg_rating=float(mat[i, 6]),
                    item_rating_count=int(mat[i, 7]),
                    item_avg_rating=float(mat[i, 8]),
                    item_genre_count=int(mat[i, 9]),
                    item_popularity_percentile=float(mat[i, 10]),
                    retrieved_by_svd=float(mat[i, 11]),
                    retrieved_by_popularity=float(mat[i, 12]),
                    retrieved_by_genre=float(mat[i, 13]),
                    source_count=float(mat[i, 14]),
                )
            )

        return features_list
