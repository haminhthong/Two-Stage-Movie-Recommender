"""Feature contract cho Stage 2.

Mọi feature retrieval giữ semantics riêng của source. Không dùng một
``retrieval_score`` chung để so sánh trực tiếp SVD, popularity và genre.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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
    """Đặc trưng của một candidate tại một snapshot thời gian.

    Ba trường đầu được giữ nguyên để các baseline cũ vẫn khởi tạo được object.
    Vector model dùng feature contract 19 cột ở ``FEATURE_NAMES``.
    """

    item_id: int
    latent_score: float
    popularity_score: float
    genre_affinity: float = 0.0

    # Trường tương thích ngược và metadata hỗ trợ debug.
    raw_latent_score: float = 0.0
    retrieval_rank: int = 0
    retrieval_rank_percentile: float = 0.0
    user_positive_count: int = 0
    user_avg_rating: float = 4.0
    item_rating_count: int = 0
    item_avg_rating: float = 3.5
    item_genre_count: int = 1
    item_popularity_percentile: float = 0.5
    retrieved_by_svd: float = 0.0
    retrieved_by_popularity: float = 0.0
    retrieved_by_genre: float = 0.0
    source_count: float = 1.0

    # Feature contract mới, source-specific và group-aware.
    svd_score: float = 0.0
    svd_rank: int = 0
    popularity_retrieval_score: float = 0.0
    popularity_rank: int = 0
    genre_retrieval_score: float = 0.0
    genre_rank: int = 0
    rrf_score: float = 0.0
    user_interaction_count: int = 0
    genre_entropy: float = 0.0
    item_positive_count: int = 0
    genre_overlap_count: int = 0

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
        self.popularity_scores = {int(key): float(value) for key, value in popularity_scores.items()}
        self.genre_map = genre_map or {}
        self.user_genre_profiles = user_genre_profiles or {}
        self.user_stats = user_stats or {}
        self.item_stats = item_stats or {}
        self.interactions_df = interactions_df
        self.rating_threshold = float(rating_threshold)
        self._snapshot_cache: dict[int, tuple[dict, dict, dict, dict]] = {}

        values = np.asarray(sorted(self.popularity_scores.values()), dtype=np.float32)
        self.pop_percentiles: dict[int, float] = {}
        if values.size:
            for item_id, score in self.popularity_scores.items():
                self.pop_percentiles[item_id] = float(
                    np.searchsorted(values, score, side="right") / values.size
                )
        self.item_genre_counts = {
            int(item_id): max(1, len(genres))
            for item_id, genres in self.genre_map.items()
        }

    def _snapshot_statistics(
        self,
        as_of_timestamp: int | None,
    ) -> tuple[dict[int, dict[str, float]], dict[int, dict[str, float]], dict[int, float], dict[int, dict[str, float]]]:
        """Tạo user/item stats chỉ từ interaction có timestamp nhỏ hơn ``as_of``."""
        if as_of_timestamp is None or self.interactions_df is None:
            return (
                self.user_stats,
                self.item_stats,
                self.popularity_scores,
                self.user_genre_profiles,
            )

        cutoff = int(as_of_timestamp)
        if cutoff in self._snapshot_cache:
            return self._snapshot_cache[cutoff]

        history = self.interactions_df[self.interactions_df["timestamp"] < cutoff]
        positive = history[history["rating"] >= self.rating_threshold]

        user_stats: dict[int, dict[str, float]] = {}
        for user_id, group in history.groupby("user_id"):
            positive_count = int((group["rating"] >= self.rating_threshold).sum())
            user_stats[int(user_id)] = {
                "positive_count": float(positive_count),
                "interaction_count": float(len(group)),
                "avg_rating": float(group["rating"].mean()),
            }

        item_stats: dict[int, dict[str, float]] = {}
        for item_id, group in history.groupby("item_id"):
            item_stats[int(item_id)] = {
                "positive_count": float((group["rating"] >= self.rating_threshold).sum()),
                "rating_count": float(len(group)),
                "avg_rating": float(group["rating"].mean()),
            }

        counts = positive.groupby("item_id").size().to_dict()
        max_count = max(counts.values(), default=1)
        max_log = max(float(np.log1p(max_count)), 1.0)
        popularity = {
            int(item_id): float(np.log1p(count) / max_log)
            for item_id, count in counts.items()
        }

        profiles: dict[int, dict[str, float]] = {}
        for user_id, group in positive.groupby("user_id"):
            counts_by_genre: dict[str, int] = {}
            total = 0
            for item_id in group["item_id"]:
                for genre in self.genre_map.get(int(item_id), set()):
                    counts_by_genre[genre] = counts_by_genre.get(genre, 0) + 1
                    total += 1
            if total:
                profiles[int(user_id)] = {
                    genre: count / total for genre, count in counts_by_genre.items()
                }

        snapshot = (user_stats, item_stats, popularity, profiles)
        self._snapshot_cache[cutoff] = snapshot
        return snapshot

    @staticmethod
    def _normalise_source_scores(
        candidates: Sequence[Candidate],
        source: str,
    ) -> np.ndarray:
        """Min-max normalize score trong đúng source; candidate khác source nhận 0."""
        values = np.asarray(
            [
                float(candidate.source_scores[source])
                if source in candidate.source_scores
                else np.nan
                for candidate in candidates
            ],
            dtype=np.float32,
        )
        present = np.isfinite(values)
        if not present.any():
            return np.zeros(len(candidates), dtype=np.float32)
        low = float(np.nanmin(values))
        high = float(np.nanmax(values))
        if high - low <= 1e-9:
            values[present] = 1.0
        else:
            values[present] = (values[present] - low) / (high - low)
        values[~present] = 0.0
        return values

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
            as_of_timestamp
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
        user_interaction_count = float(user_stat.get("interaction_count", user_positive_count))
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
            overlap = sum(1 for genre in item_genres if user_profile.get(genre, 0.0) > 0)
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
                float(self.pop_percentiles.get(item_id, 0.0)),
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
        n = len(candidates)
        normalized_svd_scores = self._normalise_source_scores(candidates, "svd")
        for index, candidate in enumerate(candidates):
            row = matrix[index]
            source_scores = candidate.source_scores
            features.append(
                CandidateFeatures(
                    item_id=int(candidate.item_id),
                    latent_score=float(normalized_svd_scores[index]),
                    popularity_score=float(row[2]),
                    genre_affinity=float(row[17]),
                    raw_latent_score=float(
                        candidate.svd_score
                        if candidate.svd_score is not None
                        else candidate.retrieval_score
                    ),
                    retrieval_rank=int(candidate.retrieval_rank),
                    retrieval_rank_percentile=float(1.0 - index / max(1, n)),
                    user_positive_count=int(row[8]),
                    user_avg_rating=float(row[10]),
                    item_rating_count=int(row[13]),
                    item_avg_rating=float(row[14]),
                    item_genre_count=int(row[16]),
                    item_popularity_percentile=float(row[15]),
                    retrieved_by_svd=float("svd" in source_scores),
                    retrieved_by_popularity=float("popularity" in source_scores),
                    retrieved_by_genre=float("genre" in source_scores),
                    source_count=float(row[7]),
                    svd_score=float(row[0]),
                    svd_rank=int(row[1]),
                    popularity_retrieval_score=float(row[2]),
                    popularity_rank=int(row[3]),
                    genre_retrieval_score=float(row[4]),
                    genre_rank=int(row[5]),
                    rrf_score=float(row[6]),
                    user_interaction_count=int(row[9]),
                    genre_entropy=float(row[11]),
                    item_positive_count=int(row[12]),
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
