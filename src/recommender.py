"""Module hệ thống gợi ý 2 tầng (Two-Stage Recommender Inference Facade).

Cung cấp giao diện tương thích ngược cho Recommender class, đồng thời tích hợp
động cơ điều phối TwoStageRecommenderEngine mới nhất.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .serving.cold_start import ColdStartPolicy
from .serving.recommender import TwoStageRecommenderEngine


class Recommender:
    """Hệ thống gợi ý 2 tầng hỗ trợ cá nhân hóa, đa dạng hóa và xử lý người dùng mới (Cold-Start)."""

    def __init__(self, model_dir: str | Path | None = None) -> None:
        """Khởi tạo Recommender và nạp toàn bộ Model Artifacts vào bộ nhớ RAM."""
        self._engine = TwoStageRecommenderEngine(model_dir=model_dir)

        # Expose attributes để tương thích hoàn toàn với mã nguồn cũ
        self.user_embeddings: np.ndarray = self._engine.user_embeddings
        self.item_embeddings: np.ndarray = self._engine.item_embeddings
        self.metadata: dict[str, Any] = self._engine.metadata
        self.config: dict[str, Any] = self._engine.config
        self.popularity_rank: dict[int, float] = self._engine.popularity_rank

    def _genre_similarity(self, item_a: int, item_b: int) -> float:
        """Tính chỉ số tương đồng thể loại Jaccard Index giữa 2 bộ phim."""
        if hasattr(self, "_engine"):
            return self._engine.diversity_reranker.genre_similarity(item_a, item_b)

        genres_a = self.metadata.get("genres", {}).get(item_a, set())
        genres_b = self.metadata.get("genres", {}).get(item_b, set())
        if not genres_a or not genres_b:
            return 0.0
        intersection = len(genres_a & genres_b)
        union = len(genres_a | genres_b)
        return float(intersection / union) if union > 0 else 0.0

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        diversity_lambda: float | None = None,
        latent_weight: float | None = None,
        recent_item_ids: list[int] | None = None,
    ) -> list[int]:
        """Tạo danh sách top-K gợi ý cá nhân hóa cho một người dùng."""
        if hasattr(self, "_engine"):
            return self._engine.recommend(
                user_id=user_id,
                k=k,
                diversity_lambda=diversity_lambda,
                latent_weight=latent_weight,
                recent_item_ids=recent_item_ids,
            )
        # Fallback cho mock/unit test __new__
        return []

    def recommend_detailed(
        self,
        user_id: int,
        k: int = 10,
        diversity_lambda: float | None = None,
        latent_weight: float | None = None,
        recent_item_ids: list[int] | None = None,
        debug: bool = False,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Tạo gợi ý chi tiết kèm phân tích điểm và đo lường độ trễ."""
        if hasattr(self, "_engine"):
            return self._engine.recommend_detailed(
                user_id=user_id,
                k=k,
                diversity_lambda=diversity_lambda,
                latent_weight=latent_weight,
                recent_item_ids=recent_item_ids,
                debug=debug,
                request_id=request_id,
            )
        return {"user_id": user_id, "strategy": "mock", "items": []}

    def recommend_with_metadata(
        self,
        user_id: int,
        k: int = 10,
        diversity_lambda: float | None = None,
        recent_item_ids: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Tạo gợi ý đầy đủ thông tin metadata (Title, Genres, Popularity Count)."""
        if hasattr(self, "_engine"):
            return self._engine.recommend_with_metadata(
                user_id=user_id,
                k=k,
                diversity_lambda=diversity_lambda,
                recent_item_ids=recent_item_ids,
            )

        # Hỗ trợ mock instance trong unit test
        recommended_item_ids = self.recommend(user_id=user_id, k=k, diversity_lambda=diversity_lambda)
        titles: dict[int, str] = self.metadata.get("titles", {})
        genres: dict[int, set[str]] = self.metadata.get("genres", {})
        popularity_counts: dict[int, int] = self.metadata.get("popularity_counts", {})

        return [
            {
                "item_id": item_id,
                "title": titles.get(item_id, f"Movie {item_id}"),
                "genres": sorted(list(genres.get(item_id, set()))),
                "interaction_count": int(popularity_counts.get(item_id, 0)),
            }
            for item_id in recommended_item_ids
        ]

    def recommend_cold_start(
        self, preferred_genres: list[str] | None = None, k: int = 10
    ) -> list[dict[str, Any]]:
        """Gợi ý thông minh cho Người dùng mới (Cold-Start User) dựa trên sở thích thể loại."""
        if hasattr(self, "_engine"):
            return self._engine.cold_start_recommend(preferred_genres=preferred_genres, k=k)[2]

        # Hỗ trợ mock instance trong unit test
        policy = ColdStartPolicy(
            popular_items=self.metadata.get("popular", []),
            genre_map=self.metadata.get("genres", {}),
            title_map=self.metadata.get("titles", {}),
            popularity_counts=self.metadata.get("popularity_counts", {}),
        )
        item_ids, _ = policy.get_recommendations(preferred_genres=preferred_genres, k=k)
        return policy.enrich_items(item_ids)
