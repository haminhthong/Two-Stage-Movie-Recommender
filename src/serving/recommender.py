"""Động cơ điều phối hệ thống gợi ý 2 tầng (Two-Stage Recommendation Engine).

Orchestrator kết nối tuần tự:
Candidate Retrieval (Stage 1) -> Feature Building -> Two-Stage Ranking -> MMR Diversity Reranking (Stage 2)
đồng thời hỗ trợ đo lường độ trễ từng chặng, bóc tách điểm số phục vụ debug và sinh giải thích nhẹ (light explanation).
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import joblib
import numpy as np

from ..config import RankingConfig
from ..ranking.diversity import DiversityReranker, ScoredRecommendation
from ..ranking.features import CandidateFeatureBuilder
from ..ranking.scorer import TwoStageRanker
from ..retrieval.svd import SVDRetriever
from ..utils import LOGGER, load_json
from .cold_start import ColdStartPolicy


class TwoStageRecommenderEngine:
    """Hệ thống gợi ý 2 tầng hoàn chỉnh chuẩn Production."""

    def __init__(self, model_dir: str | Path | None = None) -> None:
        """Khởi tạo engine và nạp các artifacts mô hình vào RAM."""
        project_root = Path(__file__).resolve().parents[2]
        artifact_dir = Path(model_dir) if model_dir else project_root / "models"

        if not artifact_dir.exists():
            raise FileNotFoundError(
                f"Thư mục chứa model artifacts không tồn tại: {artifact_dir}. "
                "Vui lòng chạy 'python -m src.train' để huấn luyện mô hình."
            )

        LOGGER.info("Đang tải model artifacts từ: %s", artifact_dir)
        self.user_embeddings: np.ndarray = np.load(artifact_dir / "user_emb.npy")
        self.item_embeddings: np.ndarray = np.load(artifact_dir / "item_emb.npy")
        self.metadata: dict[str, Any] = joblib.load(artifact_dir / "meta.joblib")
        self.config: dict[str, Any] = load_json(artifact_dir / "config.json")

        self.users: np.ndarray = self.metadata["users"]
        self.items: np.ndarray = self.metadata["items"]
        self.user_map: dict[int, int] = self.metadata["user_map"]
        self.item_map: dict[int, int] = self.metadata["item_map"]
        self.popular_items: list[int] = [int(x) for x in self.metadata["popular"]]
        self.popularity_counts: dict[int, int] = self.metadata.get("popularity_counts", {})
        self.seen_by_user: dict[int, set[int]] = self.metadata.get("seen", {})
        self.genres_map: dict[int, set[str]] = self.metadata.get("genres", {})
        self.titles_map: dict[int, str] = self.metadata.get("titles", {})
        self.user_genre_profiles: dict[int, dict[str, float]] = self.metadata.get(
            "user_genre_profiles", {}
        )

        # Tính toán điểm phổ biến chuẩn hóa dạng log1p (ưu việt hơn rank-linear)
        max_pop = max((self.popularity_counts.get(i, 0) for i in self.items), default=1)
        max_log_pop = float(np.log1p(max_pop)) if max_pop > 0 else 1.0
        self.log_popularity_scores: dict[int, float] = {
            int(i): float(np.log1p(self.popularity_counts.get(i, 0)) / max_log_pop)
            for i in self.items
        }

        # Lưu thêm linear rank để hỗ trợ tính tương thích ngược
        max_denom = max(1, len(self.popular_items) - 1)
        self.popularity_rank: dict[int, float] = {
            int(item): 1.0 - (rank / max_denom)
            for rank, item in enumerate(self.popular_items)
        }

        # Khởi tạo các module con
        self.retriever = SVDRetriever(
            user_embeddings=self.user_embeddings,
            item_embeddings=self.item_embeddings,
            user_map=self.user_map,
            item_map=self.item_map,
            items=self.items,
            seen_by_user=self.seen_by_user,
        )

        self.feature_builder = CandidateFeatureBuilder(
            popularity_scores=self.log_popularity_scores,
            genre_map=self.genres_map,
            user_genre_profiles=self.user_genre_profiles,
        )

        default_alpha = float(self.config.get("latent_weight", 0.9))
        default_lambda = float(self.config.get("diversity_lambda", 0.05))

        self.ranker = TwoStageRanker(
            latent_weight=default_alpha,
            genre_affinity_weight=0.0,
        )

        self.diversity_reranker = DiversityReranker(
            genre_map=self.genres_map,
            default_lambda=default_lambda,
        )

        self.cold_start_policy = ColdStartPolicy(
            popular_items=self.popular_items,
            genre_map=self.genres_map,
            title_map=self.titles_map,
            popularity_counts=self.popularity_counts,
        )

        if len(self.item_embeddings) != len(self.items):
            raise ValueError("Kích thước Item Embeddings và danh sách Items không khớp!")

    def recommend_detailed(
        self,
        user_id: int,
        k: int = 10,
        diversity_lambda: float | None = None,
        latent_weight: float | None = None,
    ) -> dict[str, Any]:
        """Tạo gợi ý chi tiết kèm thống kê thời gian từng chặng và phân tích điểm số."""
        t_start = time.perf_counter()

        # Kiểm tra kịch bản Cold Start
        if user_id not in self.user_map:
            t_cs = time.perf_counter()
            item_ids, strategy = self.cold_start_policy.get_recommendations(
                k=k, seen_items=self.seen_by_user.get(user_id, set())
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
            }

        # Stage 1: Candidate Retrieval
        t_ret_start = time.perf_counter()
        candidate_k = int(self.config.get("candidate_k", 200))
        candidates = self.retriever.retrieve(user_id=user_id, k=candidate_k, filter_seen=True)
        retrieval_ms = (time.perf_counter() - t_ret_start) * 1000.0

        if not candidates:
            # Fallback nếu không còn candidate khả dụng
            item_ids, strategy = self.cold_start_policy.get_recommendations(k=k)
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
            }

        # Stage 2: Feature Building & Ranking
        t_rank_start = time.perf_counter()
        features = self.feature_builder.build_features(candidates, user_id=user_id)
        ranked_candidates = self.ranker.rank(features, latent_weight_override=latent_weight)
        ranking_ms = (time.perf_counter() - t_rank_start) * 1000.0

        # Stage 2.5: MMR-Style Diversity Reranking
        t_div_start = time.perf_counter()
        final_recommendations: list[ScoredRecommendation] = self.diversity_reranker.rerank(
            ranked_candidates,
            k=k,
            diversity_lambda_override=diversity_lambda,
        )
        diversity_ms = (time.perf_counter() - t_div_start) * 1000.0
        total_ms = (time.perf_counter() - t_start) * 1000.0

        enriched_items: list[dict[str, Any]] = []
        for rank_idx, rec in enumerate(final_recommendations):
            item_id = rec.item_id
            genres = sorted(list(self.genres_map.get(item_id, set())))
            pop_cnt = int(self.popularity_counts.get(item_id, 0))

            explanation = (
                f"Rank #{rank_idx + 1}: Matched your latent preference profile; "
                f"diversified across {', '.join(genres[:2]) if genres else 'genres'}."
            )

            scores_debug = {
                "retrieval": round(float(rec.features.latent_score if rec.features else 0.0), 4),
                "popularity": round(float(rec.features.popularity_score if rec.features else 0.0), 4),
                "ranking": round(float(rec.relevance_score), 4),
                "diversity_penalty": round(float(rec.diversity_penalty), 4),
                "final": round(float(rec.final_score), 4),
            }

            enriched_items.append(
                {
                    "item_id": item_id,
                    "title": self.titles_map.get(item_id, f"Movie {item_id}"),
                    "genres": genres,
                    "interaction_count": pop_cnt,
                    "scores": scores_debug,
                    "explanation": explanation,
                }
            )

        return {
            "user_id": user_id,
            "strategy": "two_stage_personalized",
            "items": enriched_items,
            "latencies_ms": {
                "retrieval": retrieval_ms,
                "ranking": ranking_ms,
                "diversity": diversity_ms,
                "total": total_ms,
            },
        }

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        diversity_lambda: float | None = None,
        latent_weight: float | None = None,
    ) -> list[int]:
        """Tạo danh sách top-K ID phim cho một người dùng (nhẹ, nhanh cho benchmark/eval)."""
        res = self.recommend_detailed(
            user_id=user_id,
            k=k,
            diversity_lambda=diversity_lambda,
            latent_weight=latent_weight,
        )
        return [item["item_id"] for item in res["items"]]

    def recommend_with_metadata(
        self,
        user_id: int,
        k: int = 10,
        diversity_lambda: float | None = None,
    ) -> list[dict[str, Any]]:
        """Tạo danh sách gợi ý kèm metadata định dạng chuẩn cho API client."""
        res = self.recommend_detailed(user_id=user_id, k=k, diversity_lambda=diversity_lambda)
        return [
            {
                "item_id": item["item_id"],
                "title": item["title"],
                "genres": item["genres"],
                "interaction_count": item["interaction_count"],
            }
            for item in res["items"]
        ]

    def recommend_cold_start(
        self, preferred_genres: list[str] | None = None, k: int = 10
    ) -> list[dict[str, Any]]:
        """Gợi ý cho người dùng mới qua ColdStartPolicy."""
        item_ids, _ = self.cold_start_policy.get_recommendations(
            preferred_genres=preferred_genres, k=k
        )
        return [
            {
                "item_id": item_id,
                "title": self.titles_map.get(item_id, f"Movie {item_id}"),
                "genres": sorted(list(self.genres_map.get(item_id, set()))),
                "interaction_count": int(self.popularity_counts.get(item_id, 0)),
            }
            for item_id in item_ids
        ]
