"""Dịch vụ HTTP REST API phục vụ hệ thống gợi ý xem phim (Two-Stage Recommender API).

Sử dụng khung ứng dụng FastAPI với Pydantic schemas để tự động kiểm tra kiểu dữ liệu,
sinh tài liệu OpenAPI (Swagger UI), trả về chiến lược phục vụ (strategy observability),
phân tích điểm số (debug scores) và giải thích đề xuất (light explanations),
đồng thời hỗ trợ nhận diện hành vi phiên gần đây (recent session items).
"""

from __future__ import annotations

from typing import Annotated, Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Path, Query
from pydantic import BaseModel, Field

from .artifacts.loader import ArtifactValidationError
from .recommender import Recommender
from .utils import load_json

app = FastAPI(
    title="Two-Stage Recommendation System API",
    description=(
        "REST API gợi ý phim cá nhân hóa 2 tầng (TruncatedSVD Retrieval + "
        "Stage-2 Learned Ranker + MMR Genre Diversity Reranking) kèm giải thích "
        "và giám sát chiến lược phục vụ thời gian thực."
    ),
    version="5.0.0",
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
    scores: dict[str, float] | None = Field(
        None, description="Bảng phân rã điểm số chi tiết (retrieval, popularity, ranking, diversity_penalty, final)"
    )
    explanation: str | None = Field(
        None, description="Lý do giải thích tại sao sản phẩm này được gợi ý"
    )
    rank: int | None = Field(None, description="Thứ hạng trong danh sách cuối")
    reason_codes: list[str] = Field(default_factory=list, description="Mã lý do có bằng chứng")


class RecommendationResponseSchema(BaseModel):
    """Schema phản hồi kết quả gợi ý cá nhân hóa."""

    user_id: int = Field(..., description="ID người dùng được gợi ý")
    strategy: str = Field(
        ...,
        description="Chiến lược phục vụ ('two_stage_personalized', 'cold_start_popularity', 'fallback_popularity')",
    )
    items: list[int] | list[RecommendationItemSchema] = Field(
        ...,
        description="Danh sách ID phim hoặc danh sách đối tượng phim đầy đủ metadata",
    )
    includes_metadata: bool = Field(
        ..., description="Trạng thái có kèm metadata hay không"
    )
    model_version: str = Field(..., description="Phiên bản mô hình đang chạy")
    latencies_ms: dict[str, float] | None = Field(
        None, description="Độ trễ xử lý từng giai đoạn đo được trong request"
    )
    recommendations: list[RecommendationItemSchema] = Field(
        default_factory=list, description="Payload recommendation chuẩn hóa"
    )
    pipeline: dict[str, int] | None = Field(None, description="Số lượng item qua từng tầng")
    request_id: str | None = None


class ColdStartRequestSchema(BaseModel):
    """Schema yêu cầu gợi ý cho người dùng mới (Cold Start)."""

    preferred_genres: list[str] | None = Field(
        None, description="Danh sách thể loại yêu thích (ví dụ: ['Action', 'Sci-Fi'])"
    )
    k: int = Field(10, ge=1, le=50, description="Số lượng gợi ý (1-50)")


class ColdStartResponseSchema(BaseModel):
    """Schema phản hồi kết quả gợi ý người dùng mới."""

    strategy: str = Field(
        ..., description="Chiến lược cold-start ('cold_start_genre_aware' hoặc 'cold_start_popularity')"
    )
    preferred_genres: list[str] = Field(default_factory=list, description="Thể loại đã lọc")
    items: list[RecommendationItemSchema] = Field(..., description="Danh sách phim gợi ý")
    model_version: str = Field(..., description="Phiên bản mô hình")


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
    except (ArtifactValidationError, OSError, ValueError, KeyError) as exc:
        return {
            "status": "degraded",
            "model_ready": False,
            "model_version": "not_trained",
            "detail": type(exc).__name__,
        }
    return {
        "status": "ok",
        "model_ready": True,
        "model_version": recommender.config.get("version", "unknown"),
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
        0.95, ge=0.0, le=1.0, description="Trọng số giữ độ liên quan trong MMR; càng cao càng ít hy sinh relevance"
    ),
    include_metadata: bool = Query(
        False, description="Đặt True để trả về chi tiết tên và thể loại phim"
    ),
    include_scores: bool = Query(
        False, description="Đặt True để trả về bảng phân rã điểm chi tiết và độ trễ"
    ),
    debug: bool = Query(
        False, description="Bật thông tin debug scores/latency; mặc định tắt"
    ),
    recent_items: str | None = Query(
        None, description="Danh sách ID phim xem gần đây, phân tách bởi dấu phẩy (ví dụ: '1,2,3')"
    ),
) -> dict[str, Any]:
    """Tạo danh sách gợi ý phim cá nhân hóa cho một người dùng cụ thể.

    - Người dùng đã biết: Trả về kết quả Two-Stage Personalized (`strategy="two_stage_personalized"`).
    - Người dùng mới: Tự động fallback sang Popularity (`strategy="cold_start_popularity"`).
    """
    recent_item_ids: list[int] | None = None
    if recent_items:
        try:
            recent_item_ids = [
                int(value.strip())
                for value in recent_items.split(",")
                if value.strip()
            ]
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail="recent_items phải là danh sách ID số, phân tách bởi dấu phẩy.",
            ) from exc

    try:
        recommender = get_recommender()
        detailed_res = recommender.recommend_detailed(
            user_id=user_id,
            k=k,
            diversity_lambda=diversity,
            recent_item_ids=recent_item_ids,
            debug=include_scores or debug,
            request_id=uuid4().hex,
        )
    except (ArtifactValidationError, OSError, ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=503, detail="Mô hình gợi ý chưa sẵn sàng hoặc gặp lỗi artifact."
        ) from exc

    strategy = detailed_res["strategy"]
    items_raw = detailed_res["items"]

    if not include_metadata:
        output_items = [item["item_id"] for item in items_raw]
    else:
        output_items = []
        for item in items_raw:
            item_dict = {
                "item_id": item["item_id"],
                "title": item["title"],
                "genres": item["genres"],
                "interaction_count": item["interaction_count"],
                "scores": item["scores"] if include_scores else None,
                "explanation": item.get("explanation"),
                "rank": item.get("rank"),
                "reason_codes": item.get("reason_codes", []),
            }
            output_items.append(item_dict)

    return {
        "user_id": user_id,
        "strategy": strategy,
        "items": output_items,
        "includes_metadata": include_metadata,
        "model_version": recommender.config.get("version", "unknown"),
        # `debug` cũng bật đo lường trong engine; giữ response nhất quán với
        # mô tả query thay vì chỉ trả latency khi include_scores=True.
        "latencies_ms": (
            detailed_res["latencies_ms"] if debug or include_scores else None
        ),
        "recommendations": [
            {
                "item_id": item["item_id"],
                "title": item.get("title"),
                "genres": item.get("genres", []),
                "interaction_count": item.get("interaction_count", 0),
                "scores": item.get("scores") if debug or include_scores else None,
                "explanation": item.get("explanation"),
                "rank": item.get("rank"),
                "reason_codes": item.get("reason_codes", []),
            }
            for item in items_raw
        ],
        "pipeline": detailed_res.get("pipeline"),
        "request_id": detailed_res.get("request_id"),
    }


@app.post(
    "/recommend/cold-start",
    response_model=ColdStartResponseSchema,
    tags=["Recommendations"],
)
def cold_start_recommendation(
    body: ColdStartRequestSchema,
) -> dict[str, Any]:
    """Gợi ý phim thông minh cho Người dùng mới (Cold Start) dựa trên thể loại yêu thích."""
    try:
        recommender = get_recommender()
        item_ids, strategy, enriched = recommender._engine.cold_start_recommend(
            preferred_genres=body.preferred_genres, k=body.k
        )
    except (ArtifactValidationError, OSError, ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=503, detail="Mô hình gợi ý chưa sẵn sàng hoặc gặp lỗi artifact."
        ) from exc

    return {
        "strategy": strategy,
        "preferred_genres": body.preferred_genres or [],
        "items": enriched,
        "model_version": recommender.config.get("version", "unknown"),
    }
