"""Module đánh giá chỉ số Offline (Offline Metrics Evaluation) cho hệ thống gợi ý.

Các chỉ số đo lường toàn diện bao gồm:
1. HitRate@K / Recall@K: Tỷ lệ tìm đúng item thực sự tương tác trong top-K (trong leave-one-out, Recall@K == HitRate@K).
2. NDCG@K: Đánh giá độ chính xác xếp hạng có trọng số vị trí.
3. MRR@K (Mean Reciprocal Rank): Nghịch đảo vị trí xuất hiện đầu tiên của item đúng.
4. Catalog Coverage: Tỷ lệ phim trong toàn bộ kho được hệ thống khai phá gợi ý.
5. Intra-List Diversity (ILD): Độ đa dạng thể loại trung bình trong cùng danh sách top-K (chống Filter Bubble).
6. Novelty@K: Thông tin tự thân trung bình -log2(P(item)) đo lường mức độ bất ngờ, ít thiên lệch phổ biến.
7. Long-Tail Exposure: Tỷ lệ phơi nhiễm phân bổ theo phân khúc Head (Top 10%), Mid (Next 40%), Tail (Bottom 50%).
8. Stage-Level Latency Benchmark: Phân tích độ trễ p50/p95 từng tầng (Retrieval, Ranking, MMR, Total).
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import numpy as np

from .data import load_ratings, time_split
from .evaluation.metrics import (
    compute_long_tail_distribution,
    compute_user_coverage,
    dcg,
    hit_rate_at_k,
    intra_list_diversity,
    mrr_at_k,
    novelty_at_k,
)
from .recommender import Recommender
from .utils import LOGGER, save_json, setup_logging


def evaluate_recommender(
    k: int = 10,
    max_users: int | None = None,  # Mặc định None để đánh giá toàn bộ users tập Test
    output_path: str = "reports/test_metrics.json",
    seed: int = 42,
) -> dict[str, Any]:
    """Đánh giá toàn diện hệ thống gợi ý trên tập Test (Official Evaluation).

    Giao thức đánh giá:
    - Đánh giá trên toàn bộ người dùng hợp lệ trong tập Test (~6,035 users).
    - Tập Test hoàn toàn độc lập, không tham gia vào tuning siêu tham số.

    Args:
        k (int): Số lượng item top-K gợi ý.
        max_users (int | None): Số lượng user đánh giá (None = toàn bộ user hợp lệ).
        output_path (str): Đường dẫn lưu báo cáo.
        seed (int): Seed ngẫu nhiên tái lập nếu lấy mẫu.

    Returns:
        dict[str, Any]: Báo cáo chỉ số offline.
    """
    setup_logging()
    LOGGER.info("Bắt đầu đánh giá mô hình offline trên tập Test với Top-K = %d...", k)

    df_ratings = load_ratings()
    _, _, test_df = time_split(df_ratings)

    # Ground truth từ tập Test: {user_id: true_item_id} (chỉ gồm rating >= 4.0)
    test_truth = dict(zip(test_df.user_id, test_df.item_id, strict=True))

    recommender = Recommender()
    genres_map = recommender.metadata.get("genres", {})
    popular_items: list[int] = recommender.metadata["popular"]
    popular_set_100 = set(popular_items[:100])
    total_catalog_size = len(recommender.metadata["items"])

    # Phân chia Head (10%), Mid (40%), Tail (50%)
    n_items = len(popular_items)
    head_cutoff = max(1, int(n_items * 0.10))
    mid_cutoff = max(head_cutoff + 1, int(n_items * 0.50))
    head_set = set(popular_items[:head_cutoff])
    mid_set = set(popular_items[head_cutoff:mid_cutoff])
    tail_set = set(popular_items[mid_cutoff:])

    # Tính xác suất phổ biến cho chỉ số Novelty
    pop_counts = recommender.metadata.get("popularity_counts", {})
    total_interactions = sum(pop_counts.values())
    catalog_prob = {
        item_id: float((pop_counts.get(item_id, 0) + 1) / (total_interactions + total_catalog_size))
        for item_id in recommender.metadata["items"]
    }

    eligible_users = np.array(
        [
            int(user_id)
            for user_id in test_truth
            if int(user_id) in recommender.metadata["user_map"]
        ]
    )
    if max_users is not None and len(eligible_users) > max_users:
        rng = np.random.default_rng(seed)
        eligible_users = rng.choice(eligible_users, size=max_users, replace=False)

    total_eval_users = len(eligible_users)
    LOGGER.info("Số lượng người dùng đánh giá chính thức: %d", total_eval_users)

    recall_scores: list[float] = []
    ndcg_scores: list[float] = []
    mrr_scores: list[float] = []
    diversity_scores: list[float] = []
    novelty_scores: list[float] = []
    all_recommendations: list[list[int]] = []

    recommended_items_unique: set[int] = set()
    total_recommendations_count: int = 0
    popular_recommendations_count: int = 0

    # Baseline Popularity
    pop_recalls: list[float] = []
    pop_ndcgs: list[float] = []
    pop_mrrs: list[float] = []
    pop_novelties: list[float] = []

    for user_id in eligible_users:
        u_id = int(user_id)
        true_item = test_truth[u_id]

        preds = recommender.recommend(u_id, k=k)
        all_recommendations.append(preds)
        recommended_items_unique.update(preds)
        total_recommendations_count += len(preds)
        popular_recommendations_count += sum(1 for item in preds if item in popular_set_100)

        hit = hit_rate_at_k(preds, true_item)
        recall_scores.append(hit)
        ndcg_scores.append(dcg(preds.index(true_item)) if hit else 0.0)
        mrr_scores.append(mrr_at_k(preds, true_item))
        diversity_scores.append(intra_list_diversity(preds, genres_map))
        novelty_scores.append(novelty_at_k(preds, catalog_prob))

        # Popularity baseline (loại seen)
        seen = recommender.metadata.get("seen", {}).get(u_id, set())
        pop_preds = [item for item in popular_items if item not in seen][:k]
        pop_hit = hit_rate_at_k(pop_preds, true_item)
        pop_recalls.append(pop_hit)
        pop_ndcgs.append(dcg(pop_preds.index(true_item)) if pop_hit else 0.0)
        pop_mrrs.append(mrr_at_k(pop_preds, true_item))
        pop_novelties.append(novelty_at_k(pop_preds, catalog_prob))

    exposure = compute_long_tail_distribution(
        all_recommendations, head_set, mid_set, tail_set
    )
    user_cov = compute_user_coverage(all_recommendations, k=k)

    mean_recall = float(np.mean(recall_scores))
    mean_ndcg = float(np.mean(ndcg_scores))
    mean_mrr = float(np.mean(mrr_scores))
    mean_div = float(np.mean(diversity_scores))
    mean_novelty = float(np.mean(novelty_scores))

    pop_mean_recall = float(np.mean(pop_recalls))
    pop_mean_ndcg = float(np.mean(pop_ndcgs))
    pop_mean_mrr = float(np.mean(pop_mrrs))
    pop_mean_novelty = float(np.mean(pop_novelties))

    metrics_summary = {
        f"recall@{k}": mean_recall,
        f"hit_rate@{k}": mean_recall,  # Leave-one-out equivalence
        f"ndcg@{k}": mean_ndcg,
        f"mrr@{k}": mean_mrr,
        "catalog_coverage": float(len(recommended_items_unique) / total_catalog_size),
        "user_coverage": user_cov,
        "intra_list_diversity": mean_div,
        f"novelty@{k}": mean_novelty,
        "popular_item_share": float(
            popular_recommendations_count / max(1, total_recommendations_count)
        ),
        "long_tail_exposure": exposure,
        "evaluated_users": total_eval_users,
        "evaluation_protocol": (
            "full_eligible_test_users" if max_users is None else f"sampled_{max_users}_users"
        ),
        "popularity_baseline": {
            f"recall@{k}": pop_mean_recall,
            f"hit_rate@{k}": pop_mean_recall,
            f"ndcg@{k}": pop_mean_ndcg,
            f"mrr@{k}": pop_mean_mrr,
            f"novelty@{k}": pop_mean_novelty,
        },
        "model_lift_over_popularity": {
            f"recall@{k}": mean_recall - pop_mean_recall,
            f"ndcg@{k}": mean_ndcg - pop_mean_ndcg,
            f"mrr@{k}": mean_mrr - pop_mean_mrr,
            f"novelty@{k}": mean_novelty - pop_mean_novelty,
        },
    }

    save_json(output_path, metrics_summary)
    LOGGER.info("Kết quả đánh giá Official Test Metrics: %s", metrics_summary)
    return metrics_summary


def run_ablation_study(
    k: int = 10,
    max_users: int = 2000,
    output_path: str = "reports/ablation.json",
    seed: int = 42,
) -> dict[str, Any]:
    """Nghiên cứu đóng góp từng thành phần và đo lường độ trễ chi tiết từng giai đoạn.

    So sánh 4 cấu hình:
    1. Popularity Baseline
    2. SVD only (latent_weight=1.0, diversity_lambda=0.0)
    3. SVD + Popularity (latent_weight=alpha, diversity_lambda=0.0)
    4. SVD + Popularity + MMR (Full Two-Stage Pipeline)
    """
    setup_logging()
    LOGGER.info(
        "Bắt đầu chạy Ablation Study & Stage-level Latency Benchmark trên %d người dùng...",
        max_users,
    )

    df_ratings = load_ratings()
    _, _, test_df = time_split(df_ratings)
    test_truth = dict(zip(test_df.user_id, test_df.item_id, strict=True))

    recommender = Recommender()
    genres_map = recommender.metadata.get("genres", {})
    total_catalog_size = len(recommender.metadata["items"])
    popular_items = recommender.metadata["popular"]

    eligible_users = np.array(
        [
            int(user_id)
            for user_id in test_truth
            if int(user_id) in recommender.metadata["user_map"]
        ]
    )
    rng = np.random.default_rng(seed)
    if len(eligible_users) > max_users:
        eligible_users = rng.choice(eligible_users, size=max_users, replace=False)

    variants: dict[str, dict[str, Any]] = {
        "Popularity Baseline": {},
        "SVD only": {},
        "SVD + popularity": {},
        "SVD + popularity + MMR": {},
    }

    engine = recommender._engine
    best_alpha = float(recommender.config.get("latent_weight", 0.9))

    for variant_name in variants:
        recalls: list[float] = []
        ndcgs: list[float] = []
        mrrs: list[float] = []
        diversities: list[float] = []
        unique_items: set[int] = set()

        retrieval_latencies: list[float] = []
        ranking_latencies: list[float] = []
        diversity_latencies: list[float] = []
        total_latencies: list[float] = []

        for user_id in eligible_users:
            u_id = int(user_id)
            true_item = test_truth[u_id]

            if variant_name == "Popularity Baseline":
                t0 = time.perf_counter()
                seen = recommender.metadata.get("seen", {}).get(u_id, set())
                preds = [item for item in popular_items if item not in seen][:k]
                tot_ms = (time.perf_counter() - t0) * 1000.0
                ret_ms, rank_ms, div_ms = tot_ms, 0.0, 0.0
            elif variant_name == "SVD only":
                res = engine.recommend_detailed(
                    user_id=u_id, k=k, diversity_lambda=0.0, latent_weight=1.0
                )
                preds = [item["item_id"] for item in res["items"]]
                ret_ms = res["latencies_ms"]["retrieval"]
                rank_ms = res["latencies_ms"]["ranking"]
                div_ms = res["latencies_ms"]["diversity"]
                tot_ms = res["latencies_ms"]["total"]
            elif variant_name == "SVD + popularity":
                res = engine.recommend_detailed(
                    user_id=u_id, k=k, diversity_lambda=0.0, latent_weight=best_alpha
                )
                preds = [item["item_id"] for item in res["items"]]
                ret_ms = res["latencies_ms"]["retrieval"]
                rank_ms = res["latencies_ms"]["ranking"]
                div_ms = res["latencies_ms"]["diversity"]
                tot_ms = res["latencies_ms"]["total"]
            elif variant_name == "SVD + popularity + MMR":
                res = engine.recommend_detailed(user_id=u_id, k=k)
                preds = [item["item_id"] for item in res["items"]]
                ret_ms = res["latencies_ms"]["retrieval"]
                rank_ms = res["latencies_ms"]["ranking"]
                div_ms = res["latencies_ms"]["diversity"]
                tot_ms = res["latencies_ms"]["total"]
            else:
                preds = []
                ret_ms, rank_ms, div_ms, tot_ms = 0.0, 0.0, 0.0, 0.0

            retrieval_latencies.append(ret_ms)
            ranking_latencies.append(rank_ms)
            diversity_latencies.append(div_ms)
            total_latencies.append(tot_ms)

            unique_items.update(preds)
            hit = hit_rate_at_k(preds, true_item)
            recalls.append(hit)
            ndcgs.append(dcg(preds.index(true_item)) if hit else 0.0)
            mrrs.append(mrr_at_k(preds, true_item))
            diversities.append(intra_list_diversity(preds, genres_map))

        variants[variant_name] = {
            f"recall@{k}": float(np.mean(recalls)),
            f"hit_rate@{k}": float(np.mean(recalls)),
            f"ndcg@{k}": float(np.mean(ndcgs)),
            f"mrr@{k}": float(np.mean(mrrs)),
            "intra_list_diversity": float(np.mean(diversities)),
            "catalog_coverage": float(len(unique_items) / total_catalog_size),
            "retrieval_p50_ms": float(np.percentile(retrieval_latencies, 50)),
            "retrieval_p95_ms": float(np.percentile(retrieval_latencies, 95)),
            "ranking_p50_ms": float(np.percentile(ranking_latencies, 50)),
            "ranking_p95_ms": float(np.percentile(ranking_latencies, 95)),
            "diversity_p50_ms": float(np.percentile(diversity_latencies, 50)),
            "diversity_p95_ms": float(np.percentile(diversity_latencies, 95)),
            "p50_latency_ms": float(np.percentile(total_latencies, 50)),
            "p95_latency_ms": float(np.percentile(total_latencies, 95)),
        }

        LOGGER.info(
            "Hoàn thành variant '%s': Recall=%.4f, NDCG=%.4f, ILD=%.4f, Cov=%.2f%% | p50=%.2fms (ret=%.2f, rank=%.2f, div=%.2f)",
            variant_name,
            variants[variant_name][f"recall@{k}"],
            variants[variant_name][f"ndcg@{k}"],
            variants[variant_name]["intra_list_diversity"],
            variants[variant_name]["catalog_coverage"] * 100,
            variants[variant_name]["p50_latency_ms"],
            variants[variant_name]["retrieval_p50_ms"],
            variants[variant_name]["ranking_p50_ms"],
            variants[variant_name]["diversity_p50_ms"],
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
