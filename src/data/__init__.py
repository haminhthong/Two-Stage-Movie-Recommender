"""Package xử lý dữ liệu hệ thống gợi ý."""

from .interactions import (
    build_positive_interaction_matrix,
    build_user_genre_profiles,
    extract_seen_items,
)
from .loader import compute_file_sha256, load_movies, load_ratings
from .manifest import create_data_manifest, create_split_manifest
from .schema import DataContract, InteractionEvent, MovieMetadata, RecommendationContext
from .split import (
    global_temporal_windows,
    seen_items_before,
    temporal_split_four_way,
    time_split,
)

__all__ = [
    "DataContract",
    "InteractionEvent",
    "MovieMetadata",
    "RecommendationContext",
    "build_positive_interaction_matrix",
    "build_user_genre_profiles",
    "compute_file_sha256",
    "create_data_manifest",
    "create_split_manifest",
    "extract_seen_items",
    "global_temporal_windows",
    "load_movies",
    "load_ratings",
    "seen_items_before",
    "temporal_split_four_way",
    "time_split",
]
