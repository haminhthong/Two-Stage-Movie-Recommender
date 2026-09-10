"""FastAPI service cho recommender MovieLens."""

from __future__ import annotations

from typing import Annotated, Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Path, Query
from pydantic import BaseModel, Field

from .model_io import ModelLoadError
from .serving.recommender import Recommender

app = FastAPI(
    title="Two-Stage Movie Recommender API",
    description="SVD + multi-source RRF + XGBRanker + genre MMR.",
    version="1.0.0",
)

_recommender: Recommender | None = None


def get_recommender() -> Recommender:
    """Lazy-load một Recommender và tái sử dụng giữa các request."""
    global _recommender
    if _recommender is None:
        _recommender = Recommender()
    return _recommender


class RecommendationItemSchema(BaseModel):
    """Metadata tối thiểu của một movie; score chỉ xuất hiện khi debug=true."""

    item_id: int
    title: str | None = None
    genres: list[str] = Field(default_factory=list)
    interaction_count: int = 0
    scores: dict[str, float] | None = None
    explanation: str | None = None
    rank: int | None = None
    reason_codes: list[str] = Field(default_factory=list)


class RecommendationDebugSchema(BaseModel):
    """Thông tin chẩn đoán không nằm trong payload recommendation mặc định."""

    latencies_ms: dict[str, float]
    pipeline: dict[str, int] | None = None


class RecommendationResponseSchema(BaseModel):
    """Response chính, không lặp lại cùng danh sách dưới hai field."""

    user_id: int
    strategy: str
    recommendations: list[RecommendationItemSchema]
    model_version: str
    debug: RecommendationDebugSchema | None = None


class ColdStartRequestSchema(BaseModel):
    """Input cho user mới."""

    preferred_genres: list[str] | None = None
    k: int = Field(10, ge=1, le=50)


class ColdStartResponseSchema(BaseModel):
    """Response cold-start dùng cùng tên field recommendations."""

    strategy: str
    preferred_genres: list[str] = Field(default_factory=list)
    recommendations: list[RecommendationItemSchema]
    model_version: str


class HealthResponseSchema(BaseModel):
    """Trạng thái model local."""

    status: str
    model_ready: bool
    model_version: str


@app.get("/health", response_model=HealthResponseSchema, tags=["system"])
def health_check() -> dict[str, Any]:
    """Kiểm tra artifact đã sẵn sàng hay chưa."""
    try:
        recommender = get_recommender()
    except (ModelLoadError, OSError, ValueError, KeyError):
        return {
            "status": "degraded",
            "model_ready": False,
            "model_version": "not_trained",
        }
    return {
        "status": "ok",
        "model_ready": True,
        "model_version": recommender.model_name,
    }


@app.get(
    "/recommend/{user_id}",
    response_model=RecommendationResponseSchema,
    tags=["recommendations"],
)
def get_recommendation(
    user_id: Annotated[int, Path(ge=1, description="ID user")],
    k: int = Query(10, ge=1, le=50),
    diversity: float = Query(0.95, ge=0.0, le=1.0),
    debug: bool = Query(False, description="Trả score và latency chẩn đoán"),
    recent_items: str | None = Query(
        None, description="Movie IDs gần đây, ví dụ 1,2,3"
    ),
) -> dict[str, Any]:
    """Gợi ý movie cho user đã biết hoặc tự động cold-start."""
    recent_item_ids: list[int] | None = None
    if recent_items:
        try:
            recent_item_ids = [
                int(value.strip()) for value in recent_items.split(",") if value.strip()
            ]
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail="recent_items phải là danh sách ID số phân tách bởi dấu phẩy.",
            ) from exc

    try:
        recommender = get_recommender()
        detail = recommender.recommend_detailed(
            user_id=user_id,
            k=k,
            diversity_lambda=diversity,
            recent_item_ids=recent_item_ids,
            debug=debug,
            request_id=uuid4().hex,
        )
    except (ModelLoadError, OSError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng.") from exc

    recommendations = [
        {
            "item_id": item["item_id"],
            "title": item.get("title"),
            "genres": item.get("genres", []),
            "interaction_count": item.get("interaction_count", 0),
            "scores": item.get("scores") if debug else None,
            "explanation": item.get("explanation") if debug else None,
            "rank": item.get("rank") if debug else None,
            "reason_codes": item.get("reason_codes", []) if debug else [],
        }
        for item in detail["items"]
    ]
    return {
        "user_id": user_id,
        "strategy": detail["strategy"],
        "recommendations": recommendations,
        "model_version": recommender.model_name,
        "debug": (
            {
                "latencies_ms": detail["latencies_ms"],
                "pipeline": detail.get("pipeline"),
            }
            if debug
            else None
        ),
    }


@app.post(
    "/recommend/cold-start",
    response_model=ColdStartResponseSchema,
    tags=["recommendations"],
)
def cold_start_recommendation(body: ColdStartRequestSchema) -> dict[str, Any]:
    """Gợi ý cho user mới theo genre ưu tiên hoặc popularity."""
    try:
        recommender = get_recommender()
        _ids, strategy, items = recommender.cold_start_recommend(
            preferred_genres=body.preferred_genres,
            k=body.k,
        )
    except (ModelLoadError, OSError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng.") from exc

    return {
        "strategy": strategy,
        "preferred_genres": body.preferred_genres or [],
        "recommendations": items,
        "model_version": recommender.model_name,
    }
