"""Tái xếp hạng đảm bảo tính đa dạng thể loại (MMR-Style Genre Diversity Reranker).

Áp dụng cơ chế phạt độ trùng lặp thể loại (Diversity Penalty) theo phong cách MMR
(Maximal Marginal Relevance) nhằm giải quyết hiện tượng Filter Bubble / Echo Chamber.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .features import CandidateFeatures
from .scorer import RankedCandidate


@dataclass(frozen=True)
class ScoredRecommendation:
    """Sản phẩm gợi ý cuối cùng kèm bảng phân rã điểm số chi tiết.

    Attributes:
        item_id (int): ID định danh sản phẩm.
        final_score (float): Điểm số xếp hạng cuối cùng sau phạt đa dạng.
        relevance_score (float): Điểm tương quan tổng hợp từ Ranker.
        diversity_penalty (float): Giá trị phạt độ trùng lặp thể loại.
        features: Đặc trưng thành phần (latent, popularity, genre affinity).
    """

    item_id: int
    final_score: float
    relevance_score: float
    diversity_penalty: float
    features: CandidateFeatures | None = None


class DiversityReranker:
    """Tái xếp hạng bằng thuật toán tham lam phạt trùng lặp thể loại kiểu MMR."""

    def __init__(
        self,
        genre_map: dict[int, set[str]] | None = None,
        default_lambda: float = 0.05,
    ) -> None:
        """Khởi tạo DiversityReranker.

        Args:
            genre_map (dict[int, set[str]] | None): Ánh xạ item_id sang tập hợp các thể loại.
            default_lambda (float): Hệ số phạt đa dạng mặc định [0.0, 1.0].
        """
        self.genre_map = genre_map or {}
        self.default_lambda = float(default_lambda)

    def genre_similarity(self, item_a: int, item_b: int) -> float:
        """Tính hệ số tương đồng Jaccard giữa hai bộ phim dựa trên thể loại.

        Jaccard = |A ∩ B| / |A ∪ B|
        """
        genres_a = self.genre_map.get(item_a, set())
        genres_b = self.genre_map.get(item_b, set())

        if not genres_a or not genres_b:
            return 0.0

        intersection = len(genres_a & genres_b)
        union = len(genres_a | genres_b)
        return float(intersection / union) if union > 0 else 0.0

    def rerank(
        self,
        candidates: Sequence[RankedCandidate],
        k: int = 10,
        diversity_lambda_override: float | None = None,
    ) -> list[ScoredRecommendation]:
        """Thực hiện quy trình chọn tham lam MMR-style để lấy Top-K.

        Công thức mục tiêu tại mỗi bước:
            Score(i) = Relevance(i) - lambda * max_{s in Selected} Jaccard(i, s)

        Args:
            candidates (Sequence[RankedCandidate]): Danh sách ứng viên đã xếp hạng sơ bộ.
            k (int): Số lượng item gợi ý mong muốn.
            diversity_lambda_override (float | None): Ghi đè lambda nếu có.

        Returns:
            list[ScoredRecommendation]: Danh sách Top-K đã tái sắp xếp kèm phân tích điểm.
        """
        if not candidates or k <= 0:
            return []

        diversity_lambda = (
            float(diversity_lambda_override)
            if diversity_lambda_override is not None
            else self.default_lambda
        )
        diversity_lambda = min(max(diversity_lambda, 0.0), 1.0)

        # Nếu lambda == 0.0, trả về nguyên trạng thứ tự xếp hạng của ranker
        if diversity_lambda <= 1e-9:
            return [
                ScoredRecommendation(
                    item_id=c.item_id,
                    final_score=c.relevance_score,
                    relevance_score=c.relevance_score,
                    diversity_penalty=0.0,
                    features=c.features,
                )
                for c in candidates[:k]
            ]

        pool = list(candidates)
        selected: list[ScoredRecommendation] = []
        selected_item_ids: list[int] = []

        while pool and len(selected) < k:
            best_cand_idx = -1
            best_marginal_score = -float("inf")
            best_penalty = 0.0

            for idx, cand in enumerate(pool):
                max_sim = 0.0
                if selected_item_ids:
                    max_sim = max(
                        self.genre_similarity(cand.item_id, s_id)
                        for s_id in selected_item_ids
                    )

                penalty = diversity_lambda * max_sim
                marginal_score = cand.relevance_score - penalty

                if marginal_score > best_marginal_score:
                    best_marginal_score = marginal_score
                    best_cand_idx = idx
                    best_penalty = penalty

            chosen = pool.pop(best_cand_idx)
            selected_item_ids.append(chosen.item_id)
            selected.append(
                ScoredRecommendation(
                    item_id=chosen.item_id,
                    final_score=float(best_marginal_score),
                    relevance_score=float(chosen.relevance_score),
                    diversity_penalty=float(best_penalty),
                    features=chosen.features,
                )
            )

        return selected
