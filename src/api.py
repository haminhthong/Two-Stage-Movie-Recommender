"""Dịch vụ HTTP REST API phục vụ hệ thống gợi ý xem phim (Two-Stage Recommender API).

Sử dụng khung ứng dụng FastAPI với Pydantic schemas để tự động kiểm tra kiểu dữ liệu,
sinh tài liệu OpenAPI (Swagger UI) và tối ưu hiệu năng lazy-loading model artifact.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Path, Query
from pydantic import BaseModel, Field

from .recommender import Recommender
from .utils import load_json

app = FastAPI(
    title="Two-Stage Recommendation System API",
    description="REST API gợi ý phim cá nhân hóa 2 tầng (TruncatedSVD + MMR Genre Diversity Reranking)",
    version="2.0.0",
)

_recommender: Recommender | None = None


def get_recommender() -> Recommender:
    """Hàm Lazy-load Model Artifact và tái sử dụng Recommender instance giữa các request."""
    global _recommender
    if _recommender is None:
        _recommender = Recommender()
    return _recommender


# --- Pydantic Data Schemas ---


class RecommendationItemSchema(BaseModel):
    """Schema dữ liệu chi tiết của từng sản phẩm gợi ý."""

    item_id: int = Field(..., description="ID định danh phim")
    title: str | None = Field(None, description="Tên bộ phim")
    genres: list[str] = Field(
        default_factory=list, description="Danh sách thể loại phim"
    )
    interaction_count: int = Field(0, description="Tổng số lượt đánh giá/tương tác")


class RecommendationResponseSchema(BaseModel):
    """Schema phản hồi kết quả gợi ý cá nhân hóa."""

    user_id: int = Field(..., description="ID người dùng được gợi ý")
    items: list[int] | list[RecommendationItemSchema] = Field(
        ...,
        description="Danh sách ID phim hoặc danh sách đối tượng phim đầy đủ metadata",
    )
    includes_metadata: bool = Field(
        ..., description="Trạng thái có kèm metadata hay không"
    )
    model_version: str = Field(..., description="Phiên bản mô hình đang chạy")


class ColdStartRequestSchema(BaseModel):
    """Schema yêu cầu gợi ý cho người dùng mới (Cold Start)."""

    preferred_genres: list[str] | None = Field(
        None, description="Danh sách thể loại yêu thích (ví dụ: ['Action', 'Sci-Fi'])"
    )
    k: int = Field(10, ge=1, le=50, description="Số lượng gợi ý (1-50)")


class HealthResponseSchema(BaseModel):
    """Schema kiểm tra sức khỏe dịch vụ."""

    status: str = Field(..., description="Trạng thái hệ thống ('ok' hoặc 'degraded')")
    model_ready: bool = Field(..., description="Mô hình sẵn sàng phục vụ")
    model_version: str = Field(..., description="Phiên bản mô hình")


# --- API Endpoints ---


@app.get("/health", response_model=HealthResponseSchema, tags=["System Health"])
def health_check() -> dict[str, Any]:
    """Kiểm tra trạng thái sẵn sàng hoạt động của ứng dụng và mô hình AI."""
    try:
        recommender = get_recommender()
    except (OSError, ValueError, KeyError) as exc:
        return {
            "status": "degraded",
            "model_ready": False,
            "model_version": "not_trained",
            "detail": type(exc).__name__,
        }
    return {
        "status": "ok",
        "model_ready": True,
        "model_version": recommender.config.get("version", "v2.0.0"),
    }


@app.get(
    "/recommend/{user_id}",
    response_model=RecommendationResponseSchema,
    tags=["Recommendations"],
)
def get_recommendation(
    user_id: Annotated[int, Path(ge=1, description="ID người dùng cần gợi ý")],
    k: int = Query(10, ge=1, le=50, description="Số lượng gợi ý tối đa (1-50)"),
    diversity: float = Query(
        0.05, ge=0.0, le=1.0, description="Hệ số phạt đa dạng thể loại MMR"
    ),
    include_metadata: bool = Query(
        False, description="Đặt True để trả về chi tiết tên và thể loại phim"
    ),
) -> dict[str, Any]:
    """Tạo danh sách gợi ý phim cá nhân hóa cho một người dùng cụ thể.

    - Nếu user_id tồn tại: Trả về kết quả 2-Stage Retrieval + Reranking.
    - Nếu user_id mới (Cold-Start): Tự động fallback về danh sách phim phổ biến nhất.
    """
    try:
        recommender = get_recommender()
        items = (
            recommender.recommend_with_metadata(user_id, k, diversity_lambda=diversity)
            if include_metadata
            else recommender.recommend(user_id, k, diversity_lambda=diversity)
        )
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=503, detail="Mô hình gợi ý chưa sẵn sàng hoặc gặp lỗi artifact."
        ) from exc

    return {
        "user_id": user_id,
        "items": items,
        "includes_metadata": include_metadata,
        "model_version": recommender.config.get("version", "v2.0.0"),
    }


@app.post(
    "/recommend/cold-start",
    tags=["Recommendations"],
)
def cold_start_recommendation(
    body: ColdStartRequestSchema,
) -> dict[str, Any]:
    """Gợi ý phim thông minh cho Người dùng mới (Cold Start) dựa trên thể loại yêu thích."""
    try:
        recommender = get_recommender()
        items = recommender.recommend_cold_start(
            preferred_genres=body.preferred_genres, k=body.k
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail="Không thể khởi tạo gợi ý cold-start."
        ) from exc

    return {
        "preferred_genres": body.preferred_genres or [],
        "items": items,
        "model_version": recommender.config.get("version", "v2.0.0"),
    }


@app.get("/metrics", tags=["System Performance"])
def get_metrics() -> dict[str, Any]:
    """Trả về kết quả báo cáo Offline Evaluation Metrics mới nhất."""
    try:
        return load_json("reports/test_metrics.json")
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Chưa có báo cáo metrics. Vui lòng chạy 'python -m src.evaluate' trước.",
        ) from None
