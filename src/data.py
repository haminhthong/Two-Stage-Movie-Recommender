"""Module tương thích ngược cho dữ liệu (Data Compatibility Facade).

Re-exports toàn bộ hàm và lớp từ package src.data để duy trì 100% tương thích ngược.
"""

from __future__ import annotations

from .data.interactions import (
    build_positive_interaction_matrix,
    build_user_genre_profiles,
    extract_seen_items,
)
from .data.loader import compute_file_sha256, load_movies, load_ratings
from .data.manifest import create_data_manifest, create_split_manifest
from .data.schema import DataContract, InteractionEvent, MovieMetadata
from .data.split import temporal_split_four_way, time_split

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
