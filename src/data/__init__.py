"""Package xử lý dữ liệu hệ thống gợi ý."""

from .interactions import (
    build_positive_interaction_matrix,
    build_user_genre_profiles,
    extract_seen_items,
)
from .loader import compute_file_sha256, load_movies, load_ratings
from .manifest import create_data_manifest, create_split_manifest
from .schema import DataContract, InteractionEvent, MovieMetadata
from .split import temporal_split_four_way, time_split

__all__ = [
    "load_ratings",
    "load_movies",
    "compute_file_sha256",
    "time_split",
    "temporal_split_four_way",
    "build_positive_interaction_matrix",
    "build_user_genre_profiles",
    "extract_seen_items",
    "create_data_manifest",
    "create_split_manifest",
    "InteractionEvent",
    "MovieMetadata",
    "DataContract",
]
