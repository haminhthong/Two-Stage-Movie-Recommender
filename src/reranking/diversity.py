"""Tái xếp hạng đảm bảo tính đa dạng thể loại (MMR Genre Diversity Reranking).

Áp dụng cơ chế phạt độ trùng lặp thể loại (Diversity Penalty) theo thuật toán MMR
(Maximal Marginal Relevance) nhằm giải quyết hiện tượng Filter Bubble / Echo Chamber.

Tối ưu hóa hiệu năng cấp phần cứng:
1. Precomputed Genre Bitmask: Mã hóa thể loại phim thành số nguyên 32-bit (Bitmask).
   Tương đồng Jaccard tính bằng phép toán bitwise AND / OR và CPU instruction `.bit_count()`.
2. MMR Candidate-Pool Approximation: Giới hạn pool_k (mặc định 40) đồng nhất xuyên suốt Train/Val/Serving.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..ranking.features import CandidateFeatures
from ..ranking.scorer import RankedCandidate


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
    """Tái xếp hạng bằng thuật toán tham lam phạt trùng lặp thể loại kiểu MMR tối ưu bằng Bitmask."""

    def __init__(
        self,
        genre_map: dict[int, set[str]] | None = None,
        default_lambda: float = 0.95,
        default_rerank_pool_k: int = 40,
    ) -> None:
        """Khởi tạo DiversityReranker."""
        self.genre_map = genre_map or {}
        self.default_lambda = float(default_lambda)
        self.default_rerank_pool_k = int(default_rerank_pool_k)

        # 1. Tiền tính toán bảng ánh xạ thể loại sang Bitmask số nguyên
        unique_genres = sorted(list({g for genres in self.genre_map.values() for g in genres}))
        genre_to_bit = {g: (1 << i) for i, g in enumerate(unique_genres)}

        self._genre_bitmask: dict[int, int] = {
            item_id: sum(genre_to_bit[g] for g in genres if g in genre_to_bit)
            for item_id, genres in self.genre_map.items()
        }

    def genre_similarity(self, item_a: int, item_b: int) -> float:
        """Tính hệ số tương đồng Jaccard giữa hai bộ phim bằng phép toán bitwise nanosecond."""
        if item_a == item_b:
            return 1.0

        bm_a = self._genre_bitmask.get(item_a, 0)
        bm_b = self._genre_bitmask.get(item_b, 0)

        if bm_a == 0 or bm_b == 0:
            return 0.0

        inter = (bm_a & bm_b).bit_count()
        union = (bm_a | bm_b).bit_count()
        return float(inter / union) if union > 0 else 0.0

    def rerank(
        self,
        candidates: Sequence[RankedCandidate],
        k: int = 10,
        diversity_lambda_override: float | None = None,
        rerank_pool_k: int | None = None,
    ) -> list[ScoredRecommendation]:
        """Thực hiện quy trình chọn tham lam MMR-style để lấy Top-K.

        Công thức: MMR(i) = lambda * R_norm(i) - (1 - lambda) * maxSim(i, S).
        R_norm được normalize trong chính pool để không phụ thuộc scale model.
        """
        if not candidates or k <= 0:
            return []

        diversity_lambda = (
            float(diversity_lambda_override)
            if diversity_lambda_override is not None
            else self.default_lambda
        )
        diversity_lambda = min(max(diversity_lambda, 0.0), 1.0)

        # Cắt pool ứng viên theo rerank_pool_k để đồng nhất pipeline Train/Val/Serving
        pool_k = rerank_pool_k if rerank_pool_k is not None else self.default_rerank_pool_k
        pool = list(candidates[:pool_k])

        # Lambda 0 là compatibility mode: tắt MMR và giữ retrieval/ranking order.
        if diversity_lambda <= 1e-9:
            return [
                ScoredRecommendation(
                    item_id=c.item_id,
                    final_score=c.relevance_score,
                    relevance_score=c.relevance_score,
                    diversity_penalty=0.0,
                    features=c.features,
                )
                for c in pool[:k]
            ]

        raw_relevance = np.asarray(
            [candidate.relevance_score for candidate in pool], dtype=np.float32
        )
        min_relevance = float(raw_relevance.min())
        max_relevance = float(raw_relevance.max())
        relevance_range = max_relevance - min_relevance
        if relevance_range <= 1e-9:
            normalized_relevance = np.ones(len(pool), dtype=np.float32)
        else:
            normalized_relevance = (raw_relevance - min_relevance) / relevance_range

        selected: list[ScoredRecommendation] = []
        selected_item_ids: list[int] = []

        while pool and len(selected) < k:
            best_cand_idx = -1
            best_marginal_score = -float("inf")
            best_penalty = 0.0

            for idx, cand in enumerate(pool):
                max_sim = 0.0
                if selected_item_ids:
                    cand_id = cand.item_id
                    bm_cand = self._genre_bitmask.get(cand_id, 0)
                    if bm_cand > 0:
                        for s_id in selected_item_ids:
                            bm_s = self._genre_bitmask.get(s_id, 0)
                            if bm_s > 0:
                                inter = (bm_cand & bm_s).bit_count()
                                union = (bm_cand | bm_s).bit_count()
                                sim = inter / union if union > 0 else 0.0
                                if sim > max_sim:
                                    max_sim = sim
                                    if max_sim == 1.0:
                                        break

                relevance_weight = diversity_lambda
                diversity_weight = 1.0 - diversity_lambda
                penalty = diversity_weight * max_sim
                marginal_score = (
                    relevance_weight * float(normalized_relevance[idx]) - penalty
                )

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
