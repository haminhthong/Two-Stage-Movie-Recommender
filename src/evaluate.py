"""Module đánh giá chỉ số Offline (Offline Metrics Evaluation) cho hệ thống gợi ý.

Các chỉ số đo lường bao gồm:
1. Recall@K: Tỷ lệ phim thực sự tương tác nằm trong top-K gợi ý.
2. NDCG@K (Normalized Discounted Cumulative Gain): Đánh giá độ chính xác xếp hạng có trọng số vị trí.
3. Catalog Coverage: Tỷ lệ phim trong danh mục được hệ thống gợi ý cho người dùng.
4. Intra-List Diversity (ILD): Độ đa dạng thể loại trung bình trong cùng danh sách top-K (chống Filter Bubble).
5. Popular-Item Share: Tỷ lệ lượt hiển thị rơi vào Top-100 phim phổ biến nhất (Đo lường Popularity Bias guardrail).
6. Ablation Study: Nghiên cứu tác động từng thành phần kèm thống kê độ trễ p50/p95 latency.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .data import load_ratings, time_split
from .recommender import Recommender
from .utils import LOGGER, save_json, setup_logging


def dcg(rank: int) -> float:
    """Tính điểm Discounted Cumulative Gain (DCG) tại vị trí rank (0-indexed).

    Công thức: DCG = 1 / log2(rank + 2)

    Args:
        rank (int): Vị trí xuất hiện của item đúng (bắt đầu từ 0).

    Returns:
        float: Điểm DCG giảm dần theo vị trí.
    """
    return 1.0 / np.log2(rank + 2)


def intra_list_diversity(items: list[int], genres_map: dict[int, set[str]]) -> float:
    """Tính chỉ số đa dạng thể loại trong danh sách (Intra-List Diversity - ILD).

    Được đo bằng khoảng cách Jaccard trung bình giữa tất cả các cặp item hợp lệ trong top-K list.
    Công thức khoảng cách Jaccard: Dist(A, B) = 1 - |A ∩ B| / |A ∪ B|
    Lưu ý: Nếu một trong hai item thiếu metadata thể loại, cặp đó sẽ được bỏ qua (skip)
    thay vì mặc định gán 1.0 (nhằm tránh thổi phồng điểm ILD giả tạo).

    Args:
        items (list[int]): Danh sách ID các sản phẩm được gợi ý.
        genres_map (dict[int, set[str]]): Từ điển ánh xạ item_id sang tập thể loại.

    Returns:
        float: Điểm ILD trong khoảng [0.0, 1.0] (Càng gần 1.0 danh sách càng đa dạng).
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


def evaluate_recommender(
    k: int = 10,
    max_users: int = 2000,
    output_path: str = "reports/test_metrics.json",
    seed: int = 42,
) -> dict[str, Any]:
    """Đánh giá toàn diện hệ thống gợi ý trên tập Test (Positive-Sequence Time Split).

    Lưu ý: Tập Test hoàn toàn độc lập và không tham gia vào quá trình tuning siêu tham số.

    Args:
        k (int): Số lượng item top-K được gợi ý. (Mặc định: 10)
        max_users (int): Số lượng user tối đa đánh giá. (Mặc định: 2000)
        output_path (str): Đường dẫn lưu báo cáo JSON. (Mặc định: 'reports/test_metrics.json')
        seed (int): Seed ngẫu nhiên cố định để tái lập kết quả. (Mặc định: 42)

    Returns:
        dict[str, Any]: Kết quả chi tiết các chỉ số offline evaluation.
    """
    setup_logging()
    LOGGER.info("Bắt đầu đánh giá mô hình offline trên tập Test với Top-K = %d...", k)

    df_ratings = load_ratings()
    _, _, test_df = time_split(df_ratings)

    # Ground truth từ tập Test: {user_id: true_item_id} (chỉ gồm rating >= 4.0)
    test_truth = dict(zip(test_df.user_id, test_df.item_id, strict=True))

    recommender = Recommender()
    genres_map = recommender.metadata.get("genres", {})
    popular_set = set(recommender.metadata["popular"][:100])
    total_catalog_size = len(recommender.metadata["items"])

    recall_scores: list[float] = []
    ndcg_scores: list[float] = []
    diversity_scores: list[float] = []

    recommended_items_unique: set[int] = set()
    total_recommendations_count: int = 0
    popular_recommendations_count: int = 0

    eligible_users = np.array(
        [
            user_id
            for user_id in test_truth
            if user_id in recommender.metadata["user_map"]
        ]
    )
    rng = np.random.default_rng(seed)
    if len(eligible_users) > max_users:
        eligible_users = rng.choice(eligible_users, size=max_users, replace=False)

    popularity_recall: list[float] = []
    popularity_ndcg: list[float] = []
    for user_id in eligible_users:
        true_item = test_truth[int(user_id)]

        preds = recommender.recommend(int(user_id), k=k)
        recommended_items_unique.update(preds)
        total_recommendations_count += len(preds)
        popular_recommendations_count += sum(1 for item in preds if item in popular_set)

        # Tính toán Intra-List Diversity
        diversity = intra_list_diversity(preds, genres_map)
        diversity_scores.append(diversity)

        # Tính toán Recall và NDCG
        hit = true_item in preds
        recall_scores.append(float(hit))

        if hit:
            hit_position = preds.index(true_item)
            ndcg_scores.append(dcg(hit_position))
        else:
            ndcg_scores.append(0.0)

        # Baseline phổ biến toàn cục, loại item user đã xem trong train.
        seen = recommender.metadata.get("seen", {}).get(int(user_id), set())
        popular_predictions = [
            int(item)
            for item in recommender.metadata["popular"]
            if int(item) not in seen
        ][:k]
        popular_hit = true_item in popular_predictions
        popularity_recall.append(float(popular_hit))
        popularity_ndcg.append(
            dcg(popular_predictions.index(true_item)) if popular_hit else 0.0
        )

    # Tính toán các chỉ số tổng hợp
    metrics_summary = {
        f"recall@{k}": float(np.mean(recall_scores)),
        f"ndcg@{k}": float(np.mean(ndcg_scores)),
        "catalog_coverage": float(len(recommended_items_unique) / total_catalog_size),
        "intra_list_diversity": float(np.mean(diversity_scores)),
        "popular_item_share": float(
            popular_recommendations_count / max(1, total_recommendations_count)
        ),
        "evaluated_users": len(recall_scores),
        "sampling": "seeded_random_without_replacement",
        "sampling_seed": seed,
        "popularity_baseline": {
            f"recall@{k}": float(np.mean(popularity_recall)),
            f"ndcg@{k}": float(np.mean(popularity_ndcg)),
        },
        "model_lift_over_popularity": {
            f"recall@{k}": float(np.mean(recall_scores) - np.mean(popularity_recall)),
            f"ndcg@{k}": float(np.mean(ndcg_scores) - np.mean(popularity_ndcg)),
        },
    }

    save_json(output_path, metrics_summary)
    LOGGER.info("Kết quả đánh giá Offline Metrics: %s", metrics_summary)
    return metrics_summary


def run_ablation_study(
    k: int = 10,
    max_users: int = 2000,
    output_path: str = "reports/ablation.json",
    seed: int = 42,
) -> dict[str, Any]:
    """Nghiên cứu đóng góp từng thành phần (Ablation Study) trên tập Test.

    So sánh 4 cấu hình:
    1. Popularity: Gợi ý phổ biến toàn cục (loại item đã xem trong train).
    2. SVD only: Chỉ dùng Latent Dot Product (latent_weight=1.0, diversity_lambda=0.0).
    3. SVD + popularity: Kết hợp Latent Score + Popularity Prior (diversity_lambda=0.0).
    4. SVD + popularity + MMR: Mô hình hoàn chỉnh với MMR-style genre diversity reranking.

    Args:
        k (int): Số gợi ý top-K. (Mặc định: 10)
        max_users (int): Số user đánh giá. (Mặc định: 2000)
        output_path (str): Đường dẫn lưu tệp JSON kết quả. (Mặc định: 'reports/ablation.json')
        seed (int): Seed ngẫu nhiên tái lập. (Mặc định: 42)

    Returns:
        dict[str, Any]: Kết quả chi tiết của 4 cấu hình ablation kèm latency p50/p95.
    """
    setup_logging()
    LOGGER.info("Bắt đầu chạy Ablation Study trên %d người dùng tập Test...", max_users)

    df_ratings = load_ratings()
    _, _, test_df = time_split(df_ratings)
    test_truth = dict(zip(test_df.user_id, test_df.item_id, strict=True))

    recommender = Recommender()
    genres_map = recommender.metadata.get("genres", {})
    total_catalog_size = len(recommender.metadata["items"])

    eligible_users = np.array(
        [
            user_id
            for user_id in test_truth
            if user_id in recommender.metadata["user_map"]
        ]
    )
    rng = np.random.default_rng(seed)
    if len(eligible_users) > max_users:
        eligible_users = rng.choice(eligible_users, size=max_users, replace=False)

    variants: dict[str, dict[str, Any]] = {
        "Popularity": {},
        "SVD only": {},
        "SVD + popularity": {},
        "SVD + popularity + MMR": {},
    }

    for variant_name in variants:
        recalls: list[float] = []
        ndcgs: list[float] = []
        diversities: list[float] = []
        unique_items: set[int] = set()
        latencies_ms: list[float] = []

        for user_id in eligible_users:
            u_id = int(user_id)
            true_item = test_truth[u_id]

            start_t = time.perf_counter()
            if variant_name == "Popularity":
                seen = recommender.metadata.get("seen", {}).get(u_id, set())
                preds = [
                    int(item)
                    for item in recommender.metadata["popular"]
                    if int(item) not in seen
                ][:k]
            elif variant_name == "SVD only":
                preds = recommender.recommend(
                    u_id, k=k, diversity_lambda=0.0, latent_weight=1.0
                )
            elif variant_name == "SVD + popularity":
                preds = recommender.recommend(
                    u_id, k=k, diversity_lambda=0.0
                )
            elif variant_name == "SVD + popularity + MMR":
                preds = recommender.recommend(
                    u_id, k=k
                )
            else:
                preds = []

            lat_ms = (time.perf_counter() - start_t) * 1000.0
            latencies_ms.append(lat_ms)

            unique_items.update(preds)
            hit = true_item in preds
            recalls.append(float(hit))
            ndcgs.append(dcg(preds.index(true_item)) if hit else 0.0)
            diversities.append(intra_list_diversity(preds, genres_map))

        variants[variant_name] = {
            f"recall@{k}": float(np.mean(recalls)),
            f"ndcg@{k}": float(np.mean(ndcgs)),
            "intra_list_diversity": float(np.mean(diversities)),
            "catalog_coverage": float(len(unique_items) / total_catalog_size),
            "p50_latency_ms": float(np.percentile(latencies_ms, 50)),
            "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
        }
        LOGGER.info(
            "Hoàn thành variant '%s': Recall@%d=%.4f, NDCG@%d=%.4f, ILD=%.4f, Coverage=%.2f%%, p50=%.3fms, p95=%.3fms",
            variant_name,
            k,
            variants[variant_name][f"recall@{k}"],
            k,
            variants[variant_name][f"ndcg@{k}"],
            variants[variant_name]["intra_list_diversity"],
            variants[variant_name]["catalog_coverage"] * 100,
            variants[variant_name]["p50_latency_ms"],
            variants[variant_name]["p95_latency_ms"],
        )

    results = {
        "k": k,
        "evaluated_users": len(eligible_users),
        "seed": seed,
        "variants": variants,
    }
    save_json(output_path, results)
    LOGGER.info("Đã lưu kết quả Ablation Study vào: %s", output_path)
    return results


def main() -> None:
    """Hàm main thực thi script đánh giá và ablation khi gọi từ CLI."""
    evaluate_recommender()
    run_ablation_study()


if __name__ == "__main__":
    main()

