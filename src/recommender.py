"""Module hệ thống gợi ý 2 tầng (Two-Stage Recommender Inference Engine).

Triển khai quy trình gợi ý cá nhân hóa gồm 2 giai đoạn:
1. Giai đoạn 1 (Candidate Retrieval): Tìm kiếm nhanh top ứng viên bằng Tích vô hướng (Dot Product)
   trên không gian nhúng ẩn (Latent Vector Space).
2. Giai đoạn 2 (Reranking & MMR-Style Diversity): Cân bằng giữa điểm sở thích cá nhân (Latent score),
   độ phổ biến sản phẩm (Popularity Prior) và phạt mức độ trùng lặp thể loại (MMR-style Genre Diversity Reranking)
   để chống hiện tượng Filter Bubble.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .utils import LOGGER, load_json


class Recommender:
    """Hệ thống gợi ý 2 tầng hỗ trợ cá nhân hóa, đa dạng hóa và xử lý người dùng mới (Cold-Start)."""

    def __init__(self, model_dir: str | Path | None = None) -> None:
        """Khởi tạo Recommender và nạp toàn bộ Model Artifacts vào bộ nhớ RAM.

        Args:
            model_dir (str | Path | None): Thư mục chứa artifacts. Nếu None, mặc định sử dụng 'models/'.

        Raises:
            FileNotFoundError: Khi tệp artifact không tồn tại.
            ValueError: Khi kích thước embedding không tương thích với metadata.
        """
        project_root = Path(__file__).resolve().parents[1]
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

        popular_items: list[int] = self.metadata["popular"]
        max_denom = max(1, len(popular_items) - 1)
        self.popularity_rank: dict[int, float] = {
            int(item): 1.0 - (rank / max_denom)
            for rank, item in enumerate(popular_items)
        }

        if len(self.item_embeddings) != len(self.metadata["items"]):
            raise ValueError(
                "Kích thước Item Embeddings và danh sách Items trong Metadata không khớp nhau!"
            )


    def _genre_similarity(self, item_a: int, item_b: int) -> float:
        """Tính chỉ số tương đồng thể loại Jaccard Index giữa 2 bộ phim.

        Công thức: Jaccard(A, B) = |A ∩ B| / |A ∪ B|

        Args:
            item_a (int): ID phim thứ nhất.
            item_b (int): ID phim thứ hai.

        Returns:
            float: Điểm tương đồng trong khoảng [0.0, 1.0].
        """
        genres_a: set[str] = self.metadata.get("genres", {}).get(item_a, set())
        genres_b: set[str] = self.metadata.get("genres", {}).get(item_b, set())

        if not genres_a or not genres_b:
            return 0.0

        intersection = len(genres_a & genres_b)
        union = len(genres_a | genres_b)
        return intersection / union if union > 0 else 0.0

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        diversity_lambda: float | None = None,
        latent_weight: float | None = None,
    ) -> list[int]:
        """Tạo danh sách top-K gợi ý cá nhân hóa cho một người dùng.

        Quy trình xử lý:
        1. Xử lý Cold-Start: Nếu user_id chưa từng xuất hiện trong tập Train,
           trả về top-K phim phổ biến nhất hệ thống.
        2. Stage 1 (Retrieval): Tính toán tích vô hướng `item_embeddings @ user_vector`
           và dùng `np.argpartition` trích xuất nhanh Top-200 candidate tốt nhất.
        3. Stage 2 (Reranking): Chuẩn hóa điểm latent score về [0, 1] và kết hợp với
           Popularity Rank theo trọng số `latent_weight` (alpha).
        4. Diversity Penalty (MMR-style): Duyệt chọn từng item tối ưu hóa hàm mục tiêu
           phạt mức độ trùng lặp thể loại (MMR-style genre diversity reranking).

        Args:
            user_id (int): ID người dùng cần gợi ý.
            k (int): Số lượng sản phẩm gợi ý tối đa. (Mặc định: 10)
            diversity_lambda (float | None): Hệ số phạt trùng lặp thể loại [0.0, 1.0].
                Nếu None, lấy mặc định từ config (0.05).
            latent_weight (float | None): Trọng số điểm latent score [0.0, 1.0].
                Nếu None, lấy mặc định từ config.

        Returns:
            list[int]: Danh sách ID các bộ phim được gợi ý theo thứ tự ưu tiên.
        """
        if k <= 0:
            return []

        user_map: dict[int, int] = self.metadata["user_map"]
        items: np.ndarray = self.metadata["items"]

        # Xử lý Cold Start: Người dùng mới
        if user_id not in user_map:
            return [int(item) for item in self.metadata["popular"][:k]]

        # Giai đoạn 1: Latent Retrieval qua Tích vô hướng (Dot Product)
        user_idx = user_map[user_id]
        user_vec = self.user_embeddings[user_idx]
        latent_scores = self.item_embeddings @ user_vec

        # Lọc bỏ các sản phẩm người dùng đã tương tác trước đó
        seen_items: set[int] = self.metadata.get("seen", {}).get(user_id, set())
        if seen_items:
            seen_mask = np.isin(items, list(seen_items))
            latent_scores[seen_mask] = -np.inf

        available_count = int(np.isfinite(latent_scores).sum())
        if available_count == 0:
            return []

        candidate_k = max(
            1, min(int(self.config.get("candidate_k", 200)), available_count)
        )
        candidate_indices = np.argpartition(-latent_scores, candidate_k - 1)[
            :candidate_k
        ]

        # Giai đoạn 2: Reranking (Kết hợp Latent Score + Popularity Prior)
        raw_scores = latent_scores[candidate_indices]
        min_score, max_score = float(raw_scores.min()), float(raw_scores.max())
        normalized_scores = (raw_scores - min_score) / (max_score - min_score + 1e-9)

        pop_scores = np.array(
            [
                self.popularity_rank.get(int(items[idx]), 0.0)
                for idx in candidate_indices
            ]
        )

        effective_latent_weight = float(
            self.config.get("latent_weight", 0.9)
            if latent_weight is None
            else latent_weight
        )
        final_scores = (
            effective_latent_weight * normalized_scores
            + (1.0 - effective_latent_weight) * pop_scores
        )

        pool = list(candidate_indices[np.argsort(-final_scores)])
        score_by_index = {
            int(idx): float(sc)
            for idx, sc in zip(candidate_indices, final_scores, strict=True)
        }

        # Áp dụng MMR-style Genre Diversity Reranking (Diversity Penalty)
        diversity = float(
            self.config.get("diversity_lambda", 0.0)
            if diversity_lambda is None
            else diversity_lambda
        )
        diversity = min(max(diversity, 0.0), 1.0)

        selected_indices: list[int] = []

        while pool and len(selected_indices) < k:
            best_cand = max(
                pool,
                key=lambda j: (
                    score_by_index[int(j)]
                    - diversity
                    * max(
                        (
                            self._genre_similarity(int(items[j]), int(items[s]))
                            for s in selected_indices
                        ),
                        default=0.0,
                    )
                ),
            )
            selected_indices.append(best_cand)
            pool.remove(best_cand)

        return [int(items[j]) for j in selected_indices]

    def recommend_with_metadata(
        self,
        user_id: int,
        k: int = 10,
        diversity_lambda: float | None = None,
    ) -> list[dict[str, Any]]:
        """Tạo gợi ý đầy đủ thông tin metadata (Title, Genres, Popularity Count).

        Phù hợp trực tiếp cho giao diện Front-end / Client tiêu dùng API.

        Args:
            user_id (int): ID người dùng.
            k (int): Số gợi ý.
            diversity_lambda (float | None): Trọng số phạt đa dạng thể loại.

        Returns:
            list[dict[str, Any]]: Danh sách dictionary chứa thông tin chi tiết từng phim.
        """
        recommended_item_ids = self.recommend(
            user_id=user_id, k=k, diversity_lambda=diversity_lambda
        )
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
        """Gợi ý thông minh cho Người dùng mới (Cold-Start User) dựa trên sở thích thể loại.

        Args:
            preferred_genres (list[str] | None): Danh sách các thể loại phim ưa thích.
            k (int): Số lượng gợi ý.

        Returns:
            list[dict[str, Any]]: Danh sách phim gợi ý phổ biến lọc theo thể loại.
        """
        popular_items = self.metadata["popular"]
        genres_map = self.metadata.get("genres", {})
        titles_map = self.metadata.get("titles", {})
        popularity_counts = self.metadata.get("popularity_counts", {})

        selected: list[int] = []

        if preferred_genres:
            preferred_set = {g.strip().lower() for g in preferred_genres}
            for item in popular_items:
                item_g = {g.lower() for g in genres_map.get(item, set())}
                if item_g & preferred_set:
                    selected.append(item)
                    if len(selected) >= k:
                        break

        # Nếu không chọn thể loại hoặc không đủ phim, bù bằng top popular chung
        if len(selected) < k:
            for item in popular_items:
                if item not in selected:
                    selected.append(item)
                    if len(selected) >= k:
                        break

        return [
            {
                "item_id": item_id,
                "title": titles_map.get(item_id, f"Movie {item_id}"),
                "genres": sorted(list(genres_map.get(item_id, set()))),
                "interaction_count": int(popularity_counts.get(item_id, 0)),
            }
            for item_id in selected[:k]
        ]
