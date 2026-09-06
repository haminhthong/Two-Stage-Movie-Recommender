"""Schema định nghĩa dữ liệu và hợp đồng tương tác (Data Contracts & Schemas)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class InteractionEvent:
    """Sự kiện tương tác người dùng - sản phẩm."""

    user_id: int
    item_id: int
    rating: float
    timestamp: int


@dataclass(frozen=True)
class MovieMetadata:
    """Thông tin chi tiết một bộ phim."""

    item_id: int
    title: str
    genres: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DataContract:
    """Quy ước định nghĩa dữ liệu và đảm bảo tính toàn vẹn (Data Contract).

    Quy tắc 1: Implicit Positive Feedback -> Chỉ Rating >= rating_threshold (mặc định 4.0) mới là positive.
    Quy tắc 2: Seen Filter -> Mọi item user đã từng xem (kể cả rating < 4.0) đều bị loại bỏ khi suy luận.
    Quy tắc 3: Per-User Temporal Holdout -> Lịch sử user tại thời điểm suy luận không chứa tương tác tương lai.
    """

    feedback_type: str = "implicit_positive"
    rating_threshold: float = 4.0
    split_protocol: str = "per_user_temporal_holdout"
    seen_filter_scope: str = "all_interactions_before_cutoff"
    negative_sampling_proxy: str = "unobserved_candidates_without_impressions"
