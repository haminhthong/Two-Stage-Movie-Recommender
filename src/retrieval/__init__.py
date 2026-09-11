"""Package trích xuất ứng viên Tầng 1 (Stage 1 Candidate Retrieval)."""

from .base import Candidate, CandidateRetriever, resolve_seen_items
from .genre import GenreRetriever
from .merger import MultiSourceRetriever
from .popularity import PopularityRetriever
from .svd import SVDRetriever

__all__ = [
    "Candidate",
    "CandidateRetriever",
    "GenreRetriever",
    "MultiSourceRetriever",
    "PopularityRetriever",
    "SVDRetriever",
    "resolve_seen_items",
]
