"""Feature contract cho Stage 2.

Mọi feature retrieval giữ semantics riêng của source. Không dùng một
``retrieval_score`` chung để so sánh trực tiếp SVD, popularity và genre.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..retrieval.base import Candidate

FEATURE_SCHEMA_VERSION = "rank-features-v1"
FEATURE_NAMES = [
    "svd_score",
    "svd_rank",
    "popularity_retrieval_score",
    "popularity_rank",
    "genre_retrieval_score",
    "genre_rank",
    "rrf_score",
    "source_count",
    "user_positive_count",
    "user_interaction_count",
    "user_avg_rating",
    "genre_entropy",
    "item_positive_count",
    "item_rating_count",
    "item_avg_rating",
    "item_popularity_percentile",
    "item_genre_count",
    "genre_affinity",
    "genre_overlap_count",
]
N_FEATURES = len(FEATURE_NAMES)


@dataclass(frozen=True)
class CandidateFeatures:
    """Đúng 19 feature model tại một snapshot thời gian."""

    item_id: int
    svd_score: float
    svd_rank: int
    popularity_retrieval_score: float
    popularity_rank: int
    genre_retrieval_score: float
    genre_rank: int
    rrf_score: float
    source_count: float
    user_positive_count: int
    user_interaction_count: int
    user_avg_rating: float
    genre_entropy: float
    item_positive_count: int
    item_rating_count: int
    item_avg_rating: float
    item_popularity_percentile: float
    item_genre_count: int
    genre_affinity: float
    genre_overlap_count: int

    def to_feature_vector(self) -> np.ndarray:
        """Chuyển object thành vector float32 đúng thứ tự feature contract."""
        return np.array(
            [
                self.svd_score,
                self.svd_rank,
                self.popularity_retrieval_score,
                self.popularity_rank,
                self.genre_retrieval_score,
                self.genre_rank,
                self.rrf_score,
                self.source_count,
                self.user_positive_count,
                self.user_interaction_count,
                self.user_avg_rating,
                self.genre_entropy,
                self.item_positive_count,
                self.item_rating_count,
                self.item_avg_rating,
                self.item_popularity_percentile,
                self.item_genre_count,
                self.genre_affinity,
                self.genre_overlap_count,
            ],
            dtype=np.float32,
        )


class CandidateFeatureBuilder:
    """Sinh feature theo pool và có thể tính lại tại một mốc ``as_of``."""

    def __init__(
        self,
        popularity_scores: dict[int, float],
        genre_map: dict[int, set[str]] | None = None,
        user_genre_profiles: dict[int, dict[str, float]] | None = None,
        user_stats: dict[int, dict[str, float]] | None = None,
        item_stats: dict[int, dict[str, float]] | None = None,
        interactions_df: pd.DataFrame | None = None,
        rating_threshold: float = 4.0,
    ) -> None:
        self.popularity_scores = {
            int(key): float(value) for key, value in popularity_scores.items()
        }
        self.genre_map = genre_map or {}
        self.user_genre_profiles = user_genre_profiles or {}
        self.user_stats = user_stats or {}
        self.item_stats = item_stats or {}
        self.interactions_df = interactions_df
        self.rating_threshold = float(rating_threshold)
        self._user_timelines: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._item_timelines: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._item_positive_timelines: dict[int, np.ndarray] = {}
        if interactions_df is not None:
            self._build_timeline_indexes(interactions_df)

        self.pop_percentiles = _popularity_percentiles(self.popularity_scores)
        self.item_genre_counts = {
            int(item_id): max(1, len(genres))
            for item_id, genres in self.genre_map.items()
        }

    def _build_timeline_indexes(self, interactions_df: pd.DataFrame) -> None:
        """Lập index theo user/item để truy vấn lịch sử trước một cutoff."""
        required = {"user_id", "item_id", "rating", "timestamp"}
        missing = required.difference(interactions_df.columns)
        if missing:
            raise ValueError(f"interactions_df thiếu cột: {sorted(missing)}")

        for user_id, group in interactions_df.groupby("user_id", sort=False):
            ordered = group.sort_values("timestamp")
            self._user_timelines[int(user_id)] = (
                ordered["timestamp"].to_numpy(dtype=np.int64),
                ordered["item_id"].to_numpy(dtype=np.int64),
                ordered["rating"].to_numpy(dtype=np.float32),
            )

        for item_id, group in interactions_df.groupby("item_id", sort=False):
            ordered = group.sort_values("timestamp")
            timestamps = ordered["timestamp"].to_numpy(dtype=np.int64)
            ratings = ordered["rating"].to_numpy(dtype=np.float32)
            item_key = int(item_id)
            self._item_timelines[item_key] = (timestamps, ratings)
            self._item_positive_timelines[item_key] = timestamps[
                ratings >= self.rating_threshold
            ]

    def _snapshot_statistics(
        self,
        as_of_timestamp: int | None,
        user_id: int | None = None,
        item_ids: Sequence[int] = (),
    ) -> tuple[
        dict[int, dict[str, float]],
        dict[int, dict[str, float]],
        dict[int, float],
        dict[int, dict[str, float]],
    ]:
        """Tạo user/item stats chỉ từ interaction có timestamp nhỏ hơn ``as_of``."""
        if as_of_timestamp is None or self.interactions_df is None:
            return (
                self.user_stats,
                self.item_stats,
                self.popularity_scores,
                self.user_genre_profiles,
            )

        cutoff = int(as_of_timestamp)
        user_snapshot: dict[int, dict[str, float]] = {}
        profile_snapshot: dict[int, dict[str, float]] = {}
        if user_id is not None and user_id in self._user_timelines:
            timestamps, item_ids_seen, ratings = self._user_timelines[user_id]
            end = int(np.searchsorted(timestamps, cutoff, side="left"))
            history_ratings = ratings[:end]
            history_items = item_ids_seen[:end]
            if end:
                user_snapshot[user_id] = {
                    "positive_count": float(
                        np.count_nonzero(history_ratings >= self.rating_threshold)
                    ),
                    "interaction_count": float(end),
                    "avg_rating": float(history_ratings.mean()),
                }
                genre_counts: dict[str, int] = {}
                total_genres = 0
                for item_id in history_items[history_ratings >= self.rating_threshold]:
                    for genre in self.genre_map.get(int(item_id), set()):
                        genre_counts[genre] = genre_counts.get(genre, 0) + 1
                        total_genres += 1
                if total_genres:
                    profile_snapshot[user_id] = {
                        genre: count / total_genres
                        for genre, count in genre_counts.items()
                    }

        item_snapshot: dict[int, dict[str, float]] = {}
        for item_id in set(int(value) for value in item_ids):
            timeline = self._item_timelines.get(item_id)
            if timeline is None:
                continue
            timestamps, ratings = timeline
            end = int(np.searchsorted(timestamps, cutoff, side="left"))
            history_ratings = ratings[:end]
            if end:
                item_snapshot[item_id] = {
                    "positive_count": float(
                        np.count_nonzero(history_ratings >= self.rating_threshold)
                    ),
                    "rating_count": float(end),
                    "avg_rating": float(history_ratings.mean()),
                }

        positive_counts: dict[int, int] = {}
        for item_id, timestamps in self._item_positive_timelines.items():
            count = int(np.searchsorted(timestamps, cutoff, side="left"))
            if count:
                positive_counts[item_id] = count
        max_count = max(positive_counts.values(), default=1)
        max_log = max(float(np.log1p(max_count)), 1.0)
        popularity = {
            item_id: float(np.log1p(count) / max_log)
            for item_id, count in positive_counts.items()
        }
        return user_snapshot, item_snapshot, popularity, profile_snapshot

    def build_feature_matrix(
        self,
        candidates: Sequence[Candidate],
        user_id: int | None = None,
        as_of_timestamp: int | None = None,
    ) -> np.ndarray:
        """Sinh ma trận ``(N, 19)`` không dùng dữ liệu sau mốc ``as_of``."""
        n = len(candidates)
        if n == 0:
            return np.empty((0, N_FEATURES), dtype=np.float32)

        user_stats, item_stats, popularity_scores, profiles = self._snapshot_statistics(
            as_of_timestamp,
            user_id=user_id,
            item_ids=[candidate.item_id for candidate in candidates],
        )
        pop_percentiles = (
            self.pop_percentiles
            if as_of_timestamp is None or self.interactions_df is None
            else _popularity_percentiles(popularity_scores)
        )
        svd_scores = np.asarray(
            [
                float(candidate.source_scores.get("svd", 0.0))
                for candidate in candidates
            ],
            dtype=np.float32,
        )
        user_profile = profiles.get(user_id, {}) if user_id is not None else {}
        user_stat = user_stats.get(user_id, {}) if user_id is not None else {}
        user_positive_count = float(user_stat.get("positive_count", 0.0))
        user_interaction_count = float(
            user_stat.get("interaction_count", user_positive_count)
        )
        user_avg_rating = float(user_stat.get("avg_rating", 3.8))
        genre_entropy = _genre_entropy(user_profile)

        matrix = np.zeros((n, N_FEATURES), dtype=np.float32)
        for index, candidate in enumerate(candidates):
            item_id = int(candidate.item_id)
            source_scores = candidate.source_scores
            # Đây là điểm retrieval của source popularity, không phải prior item.
            # Candidate không được popularity retriever trả về phải nhận 0 ở cột này;
            # prior theo thời gian đã được phản ánh riêng qua percentile/count.
            pop_score = float(source_scores.get("popularity", 0.0))
            genre_score = float(source_scores.get("genre", 0.0))
            item_genres = self.genre_map.get(item_id, set())
            affinity = (
                sum(user_profile.get(genre, 0.0) for genre in item_genres)
                / max(1, len(item_genres))
                if user_profile and item_genres
                else 0.0
            )
            overlap = sum(
                1 for genre in item_genres if user_profile.get(genre, 0.0) > 0
            )
            item_stat = item_stats.get(item_id, {})
            item_rating_count = float(item_stat.get("rating_count", 0.0))
            item_positive_count = float(item_stat.get("positive_count", 0.0))
            item_avg_rating = float(item_stat.get("avg_rating", 3.5))
            source_count = len(source_scores) or 1

            matrix[index] = [
                svd_scores[index],
                float(candidate.svd_rank or 0),
                pop_score,
                float(candidate.popularity_rank or 0),
                genre_score,
                float(candidate.genre_rank or 0),
                float(candidate.rrf_score or 0.0),
                float(source_count),
                user_positive_count,
                user_interaction_count,
                user_avg_rating,
                genre_entropy,
                item_positive_count,
                item_rating_count,
                item_avg_rating,
                float(pop_percentiles.get(item_id, 0.0)),
                float(self.item_genre_counts.get(item_id, max(1, len(item_genres)))),
                affinity,
                float(overlap),
            ]
        return matrix

    def build_features(
        self,
        candidates: Sequence[Candidate],
        user_id: int | None = None,
        as_of_timestamp: int | None = None,
    ) -> list[CandidateFeatures]:
        """Sinh object feature để ranker, explanation và debug cùng dùng."""
        matrix = self.build_feature_matrix(
            candidates,
            user_id=user_id,
            as_of_timestamp=as_of_timestamp,
        )
        features: list[CandidateFeatures] = []
        for index, candidate in enumerate(candidates):
            row = matrix[index]
            features.append(
                CandidateFeatures(
                    item_id=int(candidate.item_id),
                    svd_score=float(row[0]),
                    svd_rank=int(row[1]),
                    popularity_retrieval_score=float(row[2]),
                    popularity_rank=int(row[3]),
                    genre_retrieval_score=float(row[4]),
                    genre_rank=int(row[5]),
                    rrf_score=float(row[6]),
                    source_count=float(row[7]),
                    user_positive_count=int(row[8]),
                    user_interaction_count=int(row[9]),
                    user_avg_rating=float(row[10]),
                    genre_entropy=float(row[11]),
                    item_positive_count=int(row[12]),
                    item_rating_count=int(row[13]),
                    item_avg_rating=float(row[14]),
                    item_popularity_percentile=float(row[15]),
                    item_genre_count=int(row[16]),
                    genre_affinity=float(row[17]),
                    genre_overlap_count=int(row[18]),
                )
            )
        return features


def _genre_entropy(profile: dict[str, float]) -> float:
    """Tính entropy của hồ sơ genre; profile rỗng có entropy bằng 0."""
    if not profile:
        return 0.0
    values = np.asarray(list(profile.values()), dtype=np.float32)
    values = values[values > 0]
    return float(-(values * np.log(values)).sum()) if values.size else 0.0


def _popularity_percentiles(scores: dict[int, float]) -> dict[int, float]:
    """Tính percentile popularity trên đúng snapshot đang dùng cho sample."""
    if not scores:
        return {}

    values = np.asarray(sorted(scores.values()), dtype=np.float32)
    return {
        int(item_id): float(np.searchsorted(values, score, side="right") / values.size)
        for item_id, score in scores.items()
    }
