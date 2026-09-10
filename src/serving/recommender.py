"""Động cơ gợi ý hai tầng phục vụ thời gian thực (Two-Stage Recommender Engine).

Điều phối hoàn chỉnh luồng trực tuyến (Online Serving Flow):
1. Cold-Start Check -> Nếu user chưa có trong hệ thống, chuyển sang ColdStartPolicy.
2. Stage 1: Candidate Retrieval (Multi-Source / SVD Dot Product) -> Rút trích ~200 ứng viên, lọc phim đã xem.
3. Stage 2: Feature Engineering & Learned Ranking -> Chấm điểm bằng Stage-2 Learned Ranker (XGBoost).
   Nếu Dev model selection tắt ranker, giữ nguyên retrieval order.
4. Stage 3: MMR Diversity Reranking với `rerank_pool_k = 40` đồng nhất với Offline Validation.
5. Enrichment: Bổ sung metadata (Title, Genres, Popularity, Scores, Latencies).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..data.schema import RecommendationContext
from ..model_io import load_model
from ..ranking.features import CandidateFeatureBuilder
from ..ranking.scorer import rank_candidates, retrieval_order
from ..reranking.diversity import DiversityReranker, ScoredRecommendation
from ..retrieval.genre import GenreRetriever
from ..retrieval.merger import MultiSourceRetriever
from ..retrieval.popularity import PopularityRetriever
from ..retrieval.svd import SVDRetriever
from .cold_start import ColdStartPolicy

LOGGER = logging.getLogger("recommender")


class Recommender:
    """Động cơ điều phối toàn diện cho quá trình suy luận gợi ý thời gian thực."""

    def __init__(self, model_dir: str | Path | None = None) -> None:
        """Nạp model phẳng và khởi tạo các stage của recommender."""
        target_dir = Path(model_dir) if model_dir else Path("models")
        bundle = load_model(target_dir)

        self.user_embeddings: np.ndarray = bundle["user_embeddings"]
        self.item_embeddings: np.ndarray = bundle["item_embeddings"]
        self.metadata: dict[str, Any] = bundle["metadata"]
        self.config: dict[str, Any] = bundle["config"]
        self.learned_ranker: Any | None = bundle.get("ranker")
        self.model_name: str = self.config.get("model_name", "local-model")

        # Phân rã metadata
        self.user_map: dict[int, int] = self.metadata.get("user_map", {})
        self.item_map: dict[int, int] = self.metadata.get("item_map", {})
        self.items: np.ndarray = np.asarray(self.metadata.get("items", []))
        self.popular_items: list[int] = [
            int(x) for x in self.metadata.get("popular", [])
        ]
        self.popularity_counts: dict[int, int] = self.metadata.get(
            "popularity_counts", {}
        )
        self.log_popularity_scores: dict[int, float] = self.metadata.get(
            "log_popularity", {}
        )
        self.seen_by_user: dict[int, set[int]] = self.metadata.get("seen", {})
        self.genres_map: dict[int, set[str]] = self.metadata.get("genres", {})
        self.titles_map: dict[int, str] = self.metadata.get("titles", {})
        self.user_genre_profiles: dict[int, dict[str, float]] = self.metadata.get(
            "user_genre_profiles", {}
        )
        self.user_stats: dict[int, dict[str, float]] = self.metadata.get(
            "user_stats", {}
        )
        self.item_stats: dict[int, dict[str, float]] = self.metadata.get(
            "item_stats", {}
        )

        # Khởi tạo Stage 1 Retrievers
        self.svd_retriever = SVDRetriever(
            user_embeddings=self.user_embeddings,
            item_embeddings=self.item_embeddings,
            user_map=self.user_map,
            item_map=self.item_map,
            items=self.items,
            seen_by_user=self.seen_by_user,
        )

        self.popularity_retriever = PopularityRetriever(
            popular_items=self.popular_items,
            popularity_scores=self.log_popularity_scores,
            seen_by_user=self.seen_by_user,
        )

        self.genre_retriever = GenreRetriever(
            popular_items=self.popular_items,
            genre_map=self.genres_map,
            user_genre_profiles=self.user_genre_profiles,
            popularity_scores=self.log_popularity_scores,
            seen_by_user=self.seen_by_user,
        )

        self.multi_retriever = MultiSourceRetriever(
            retrievers={
                "svd": (
                    self.svd_retriever,
                    int(self.config.get("svd_candidate_k", 150)),
                ),
                "popularity": (
                    self.popularity_retriever,
                    int(self.config.get("popularity_candidate_k", 50)),
                ),
                "genre": (
                    self.genre_retriever,
                    int(self.config.get("genre_candidate_k", 50)),
                ),
            }
        )

        self.retriever = self.multi_retriever

        # Khởi tạo Feature Builder
        self.feature_builder = CandidateFeatureBuilder(
            popularity_scores=self.log_popularity_scores,
            genre_map=self.genres_map,
            user_genre_profiles=self.user_genre_profiles,
            user_stats=self.user_stats,
            item_stats=self.item_stats,
        )

        default_lambda = float(self.config.get("diversity_lambda", 0.95))
        self.rerank_pool_k = int(self.config.get("rerank_pool_k", 40))

        # Khởi tạo Stage 2 Ranker
        ranker_enabled = bool(self.config.get("ranker_enabled", False))
        self.ranker_enabled = ranker_enabled and self.learned_ranker is not None

        # Khởi tạo Stage 3 MMR Reranker với rerank_pool_k đồng nhất
        self.diversity_reranker = DiversityReranker(
            genre_map=self.genres_map,
            default_lambda=default_lambda,
            default_rerank_pool_k=self.rerank_pool_k,
        )

        # Cold-Start Policy
        self.cold_start_policy = ColdStartPolicy(
            popular_items=self.popular_items,
            genre_map=self.genres_map,
            title_map=self.titles_map,
            popularity_counts=self.popularity_counts,
            log_popularity=self.log_popularity_scores,
        )

        self.default_diversity_lambda = default_lambda

    def cold_start_recommend(
        self,
        preferred_genres: list[str] | None = None,
        k: int = 10,
        seen_items: set[int] | None = None,
    ) -> tuple[list[int], str, list[dict[str, Any]]]:
        """Entrypoint cho người dùng mới."""
        item_ids, strategy = self.cold_start_policy.get_recommendations(
            preferred_genres=preferred_genres, k=k, seen_items=seen_items
        )
        return item_ids, strategy, self.cold_start_policy.enrich_items(item_ids)

    def recommend_detailed(
        self,
        user_id: int,
        k: int = 10,
        diversity_lambda: float | None = None,
        recent_item_ids: list[int] | None = None,
        debug: bool = False,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Tạo gợi ý chi tiết kèm đo độ trễ và phân tích điểm số từng chặng."""
        t_start = time.perf_counter()

        # Kiểm tra Cold Start
        request_context = RecommendationContext.from_items(
            user_id=user_id,
            seen_item_ids=self.seen_by_user.get(user_id, set()),
            recent_item_ids=recent_item_ids or (),
        )

        if user_id not in self.user_map:
            item_ids, strategy = self.cold_start_policy.get_recommendations(
                k=k, seen_items=set(request_context.effective_seen_items)
            )
            total_ms = (time.perf_counter() - t_start) * 1000.0
            enriched = self.cold_start_policy.enrich_items(item_ids)
            return {
                "user_id": user_id,
                "strategy": strategy,
                "items": enriched,
                "latencies_ms": {
                    "retrieval": 0.0,
                    "ranking": 0.0,
                    "diversity": 0.0,
                    "total": total_ms,
                },
                "model_version": self.model_name,
                "pipeline": {
                    "candidate_count": 0,
                    "ranked_count": 0,
                    "rerank_pool_count": 0,
                },
            }

        # Stage 1: Candidate Retrieval
        t_ret_start = time.perf_counter()
        candidate_k = int(self.config.get("candidate_k", 200))

        candidates = self.retriever.retrieve(
            user_id=user_id,
            k=candidate_k,
            filter_seen=True,
            seen_items_override=request_context.effective_seen_items,
        )

        retrieval_ms = (time.perf_counter() - t_ret_start) * 1000.0
        retrieval_source_counts = {
            source: sum(
                1 for candidate in candidates if source in candidate.source_scores
            )
            for source in ("svd", "popularity", "genre")
        }

        if not candidates:
            item_ids, strategy = self.cold_start_policy.get_recommendations(
                k=k,
                seen_items=set(request_context.effective_seen_items),
            )
            total_ms = (time.perf_counter() - t_start) * 1000.0
            return {
                "user_id": user_id,
                "strategy": "fallback_popularity",
                "items": self.cold_start_policy.enrich_items(item_ids),
                "latencies_ms": {
                    "retrieval": retrieval_ms,
                    "ranking": 0.0,
                    "diversity": 0.0,
                    "total": total_ms,
                },
                "model_version": self.model_name,
                "pipeline": {
                    "candidate_count": 0,
                    "ranked_count": 0,
                    "rerank_pool_count": 0,
                },
            }

        # Stage 2: Feature Engineering & Ranking
        t_rank_start = time.perf_counter()
        features = self.feature_builder.build_features(
            candidates,
            user_id=user_id,
            as_of_timestamp=request_context.as_of_timestamp,
        )
        if self.ranker_enabled:
            ranked_candidates = rank_candidates(features, self.learned_ranker)
        else:
            # Khi ranker bị tắt bởi Dev model selection hoặc artifact không sẵn sàng,
            # fallback phải là thứ tự retrieval đã freeze, không phải một
            # heuristic khác làm thay đổi contract offline/online.
            ranked_candidates = retrieval_order(features)
        ranking_ms = (time.perf_counter() - t_rank_start) * 1000.0

        # Stage 3: MMR Diversity Reranking với rerank_pool_k đồng nhất (P0.2 fix)
        t_div_start = time.perf_counter()
        final_recommendations: list[ScoredRecommendation] = (
            self.diversity_reranker.rerank(
                ranked_candidates,
                k=k,
                diversity_lambda_override=diversity_lambda,
                rerank_pool_k=self.rerank_pool_k,
            )
        )
        diversity_ms = (time.perf_counter() - t_div_start) * 1000.0
        total_ms = (time.perf_counter() - t_start) * 1000.0
        LOGGER.info(
            "recommendation_request request_id=%s model_version=%s strategy=personalized "
            "candidate_count=%d ranking_count=%d top_k=%d retrieval_ms=%.3f "
            "ranking_ms=%.3f rerank_ms=%.3f total_ms=%.3f source_counts=%s",
            request_id or "-",
            self.model_name,
            len(candidates),
            len(ranked_candidates),
            len(final_recommendations),
            retrieval_ms,
            ranking_ms,
            diversity_ms,
            total_ms,
            retrieval_source_counts,
        )

        enriched_items: list[dict[str, Any]] = []
        for rank_idx, rec in enumerate(final_recommendations):
            item_id = rec.item_id
            genres = sorted(self.genres_map.get(item_id, set()))
            pop_cnt = int(self.popularity_counts.get(item_id, 0))

            pop_sc = rec.features.popularity_retrieval_score if rec.features else 0.0
            lat_sc = rec.features.svd_score if rec.features else 0.0
            aff_sc = rec.features.genre_affinity if rec.features else 0.0

            # Explanation
            reason_codes: list[str] = []
            if rec.features and rec.features.svd_rank > 0 and lat_sc >= 0.5:
                reason_codes.append("COLLABORATIVE_MATCH")
            if aff_sc > 0.0:
                reason_codes.append("GENRE_MATCH")
            if rec.features and rec.features.popularity_rank > 0:
                reason_codes.append("POPULARITY_SIGNAL")
            if not reason_codes:
                reason_codes.append("RETRIEVAL_MATCH")
            explanation = _build_evidence_explanation(reason_codes, genres)

            score_payload = {
                "retrieval": round(lat_sc, 4),
                "popularity": round(pop_sc, 4),
                "ranking": round(rec.relevance_score, 4),
                "diversity_penalty": round(rec.diversity_penalty, 4),
                "final": round(rec.final_score, 4),
            }

            enriched_items.append(
                {
                    "item_id": item_id,
                    "title": self.titles_map.get(item_id, f"Movie {item_id}"),
                    "genres": genres,
                    "interaction_count": pop_cnt,
                    "rank": rank_idx + 1,
                    "reason_codes": reason_codes,
                    "scores": score_payload if debug else None,
                    "explanation": explanation,
                }
            )

        return {
            "user_id": user_id,
            "strategy": "two_stage_personalized",
            "items": enriched_items,
            "model_version": self.model_name,
            "latencies_ms": {
                "retrieval": retrieval_ms,
                "ranking": ranking_ms,
                "diversity": diversity_ms,
                "reranking": diversity_ms,
                "total": total_ms,
            },
            "pipeline": {
                "candidate_count": len(candidates),
                "ranked_count": len(ranked_candidates),
                "rerank_pool_count": min(self.rerank_pool_k, len(ranked_candidates)),
            },
            "retrieval_source_counts": retrieval_source_counts,
            "fallback_reason": None,
            "request_id": request_id,
        }

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        diversity_lambda: float | None = None,
        recent_item_ids: list[int] | None = None,
    ) -> list[int]:
        """Tạo danh sách top-K ID phim gợi ý cho user."""
        detail = self.recommend_detailed(
            user_id=user_id,
            k=k,
            diversity_lambda=diversity_lambda,
            recent_item_ids=recent_item_ids,
        )
        return [item["item_id"] for item in detail["items"]]

    def recommend_with_metadata(
        self,
        user_id: int,
        k: int = 10,
        diversity_lambda: float | None = None,
        recent_item_ids: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Tạo danh sách gợi ý kèm metadata chuẩn UI."""
        detail = self.recommend_detailed(
            user_id=user_id,
            k=k,
            diversity_lambda=diversity_lambda,
            recent_item_ids=recent_item_ids,
        )
        return [
            {
                "item_id": item["item_id"],
                "title": item["title"],
                "genres": item["genres"],
                "interaction_count": item["interaction_count"],
            }
            for item in detail["items"]
        ]


def _build_evidence_explanation(reason_codes: list[str], genres: list[str]) -> str:
    """Tạo explanation dựa trên tín hiệu thực tế, không kể chuyện causal."""
    if "COLLABORATIVE_MATCH" in reason_codes:
        return "High collaborative retrieval score"
    if "GENRE_MATCH" in reason_codes:
        labels = ", ".join(genres[:2]) if genres else "your preferred genres"
        return f"Matches your historical genre preferences: {labels}"
    if "POPULARITY_SIGNAL" in reason_codes:
        return "Popular among active users"
    return "Retrieved as a relevant unseen title"
