"""Thư viện tính toán các chỉ số đánh giá hệ thống gợi ý (Recommender Evaluation Metrics).

Bao gồm:
1. Relevance & Ranking Accuracy: Recall@K / HitRate@K, NDCG@K, MRR@K.
2. Diversity & Serendipity: Intra-List Diversity (ILD), Novelty@K.
3. Distribution & Long-tail Exposure: Catalog Coverage, User Coverage, Head/Mid/Tail Exposure.
"""

from __future__ import annotations

import numpy as np


def dcg(rank: int) -> float:
    """Tính điểm Discounted Cumulative Gain (DCG) tại vị trí rank (0-indexed).

    Công thức: DCG = 1 / log2(rank + 2)
    """
    return float(1.0 / np.log2(rank + 2))


def hit_rate_at_k(preds: list[int], true_item: int) -> float:
    """Tính HitRate@K cho một người dùng.

    Lưu ý quan trọng: Trong giao thức đánh giá Leave-One-Out (mỗi người dùng chỉ có duy nhất
    1 ground truth item trong tập Test), Recall@K đồng nhất tuyệt đối với HitRate@K:
        Recall@K = Hit / 1 = HitRate@K
    """
    return 1.0 if true_item in preds else 0.0


def mrr_at_k(preds: list[int], true_item: int) -> float:
    """Tính Mean Reciprocal Rank (MRR@K) cho một người dùng.

    Công thức: 1 / (rank + 1) nếu hit, ngược lại 0.0.
    """
    if true_item in preds:
        rank_1indexed = preds.index(true_item) + 1
        return 1.0 / rank_1indexed
    return 0.0


def intra_list_diversity(items: list[int], genres_map: dict[int, set[str]]) -> float:
    """Tính chỉ số đa dạng thể loại trong danh sách (Intra-List Diversity - ILD).

    Đo bằng khoảng cách Jaccard trung bình giữa tất cả các cặp item trong top-K:
        Dist(A, B) = 1 - |A ∩ B| / |A ∪ B|
    Bỏ qua các cặp nếu thiếu metadata thể loại để tránh thổi phồng điểm ILD giả tạo.
    """
    if len(items) <= 1:
        return 0.0

    pairwise_distances: list[float] = []
    for i, first in enumerate(items):
        for second in items[i + 1 :]:
            set_a = genres_map.get(first, set())
            set_b = genres_map.get(second, set())
            if set_a and set_b:
                union = len(set_a | set_b)
                if union > 0:
                    jaccard_sim = len(set_a & set_b) / union
                    pairwise_distances.append(1.0 - jaccard_sim)

    return float(np.mean(pairwise_distances)) if pairwise_distances else 0.0


def novelty_at_k(
    items: list[int],
    item_probabilities: dict[int, float],
) -> float:
    """Tính điểm Novelty@K (Độ mới lạ / bất ngờ của danh sách gợi ý).

    Được đo lường bằng thông tin tự thân (Self-Information) trung bình:
        Novelty(items) = -1/K * sum_{i in items} log2(P(i))
    Item càng ít phổ biến (P(i) nhỏ), -log2(P(i)) càng lớn, điểm Novelty càng cao.
    """
    if not items:
        return 0.0

    self_info: list[float] = []
    for item in items:
        prob = item_probabilities.get(item, 1e-6)
        prob = max(prob, 1e-9)
        self_info.append(-float(np.log2(prob)))

    return float(np.mean(self_info))


def compute_long_tail_distribution(
    all_recommendations: list[list[int]],
    head_set: set[int],
    mid_set: set[int],
    tail_set: set[int],
) -> dict[str, float]:
    """Đo lường phân bổ phơi nhiễm (Exposure Distribution) trên các phân khúc:

    - Head (Top 10% phim phổ biến nhất).
    - Mid (40% phim phổ biến trung bình).
    - Tail (50% phim ít phổ biến nhất / Long-Tail).
    """
    total_slots = sum(len(preds) for preds in all_recommendations)
    if total_slots == 0:
        return {"head_share": 0.0, "mid_share": 0.0, "tail_share": 0.0}

    head_count = 0
    mid_count = 0
    tail_count = 0

    for preds in all_recommendations:
        for item in preds:
            if item in head_set:
                head_count += 1
            elif item in mid_set:
                mid_count += 1
            elif item in tail_set:
                tail_count += 1
            else:
                # Nếu chưa phân loại, mặc định tail
                tail_count += 1

    return {
        "head_exposure_share": float(head_count / total_slots),
        "mid_exposure_share": float(mid_count / total_slots),
        "tail_exposure_share": float(tail_count / total_slots),
    }


def compute_user_coverage(all_recommendations: list[list[int]], k: int) -> float:
    """Tính tỷ lệ người dùng nhận được đủ K gợi ý hợp lệ."""
    if not all_recommendations:
        return 0.0
    full_k_users = sum(1 for preds in all_recommendations if len(preds) >= k)
    return float(full_k_users / len(all_recommendations))
