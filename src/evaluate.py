"""Module đánh giá chỉ số Offline (Offline Metrics Evaluation) và nghiên cứu đóng góp từng thành phần (Ablation Study).

Cung cấp:
1. Đánh giá phễu hoàn chỉnh (Multi-Stage Funnel Evaluation):
   - Stage 1: Target-in-Catalog Rate, Candidate Recall@50/100/200
   - Stage 2: Ranker Recall@10, Ranker NDCG@10
   - Stage 3 (Final Post-MMR): Recall@10, NDCG@10, MRR@10, ILD, Coverage, Novelty
   - Lift: Báo cáo cả `absolute_gain` và `relative_lift`
2. Nghiên cứu thực nghiệm (Ablation Benchmark):
   - So sánh Popularity vs SVD vs Heuristic Weighted Fusion vs Stage-2 Learned Ranker vs Full Pipeline
   - Nghiên cứu xấp xỉ MMR Candidate Pool: Pool 20 vs 40 vs 100 vs 200 (Recall vs ILD vs Latency).
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import numpy as np

from .data import load_ratings, temporal_split_four_way, time_split
from .evaluation.evaluator import FullFunnelEvaluator
from .evaluation.latency import summarize_latencies
from .evaluation.metrics import (
    compute_long_tail_distribution,
    compute_user_coverage,
    dcg,
    hit_rate_at_k,
    intra_list_diversity,
    mrr_at_k,
    novelty_at_k,
)
from .ranking.scorer import TwoStageRanker
from .recommender import Recommender
from .utils import LOGGER, save_json, setup_logging


def evaluate_recommender(
    k: int = 10,
    max_users: int | None = None,
    output_path: str = "reports/test_metrics.json",
    seed: int = 42,
) -> dict[str, Any]:
    """Đánh giá toàn diện hệ thống gợi ý trên tập Test (Official Funnel Evaluation)."""
    setup_logging()
    LOGGER.info("Bắt đầu đánh giá mô hình offline theo phễu từng tầng trên tập Test với Top-K = %d...", k)

    df_ratings = load_ratings()
    _, _, _, test_df = temporal_split_four_way(df_ratings)
    test_truth = dict(zip(test_df.user_id, test_df.item_id, strict=True))

    recommender = Recommender()
    engine = recommender._engine

    evaluator = FullFunnelEvaluator(
        engine=engine,
        catalog_items=recommender.metadata["items"],
        popular_items=recommender.metadata["popular"],
        popularity_counts=recommender.metadata.get("popularity_counts", {}),
        genre_map=recommender.metadata.get("genres", {}),
    )

    summary = evaluator.evaluate(test_truth=test_truth, max_users=max_users, k=k, seed=seed)

    final_stage = summary["funnel_stage_metrics"]["stage_3_final_post_mmr"]
    pop_baseline = summary["popularity_baseline"]
    lift = summary["model_lift_over_popularity"]

    flat_summary = {
        f"recall@{k}": final_stage[f"recall@{k}"],
        f"hit_rate@{k}": final_stage[f"recall@{k}"],
        f"ndcg@{k}": final_stage[f"ndcg@{k}"],
        f"mrr@{k}": final_stage[f"mrr@{k}"],
        "catalog_coverage": final_stage["catalog_coverage"],
        "user_coverage": final_stage["user_coverage"],
        "intra_list_diversity": final_stage["intra_list_diversity"],
        f"novelty@{k}": final_stage[f"novelty@{k}"],
        "target_in_catalog_rate": summary["target_in_catalog_rate"],
        "cold_item_test_share": summary["cold_item_test_share"],
        "evaluated_users": summary["evaluated_users"],
        "evaluation_protocol": summary["evaluation_protocol"],
        "funnel_stage_metrics": summary["funnel_stage_metrics"],
        "popularity_baseline": pop_baseline,
        "model_lift_over_popularity": lift,
        "latencies_ms": summary["latencies_ms"],
    }

    save_json(output_path, flat_summary)
    LOGGER.info("Đã xuất kết quả đánh giá phễu chính thức tới '%s'!", output_path)
    return flat_summary


def run_ablation_study(
    k: int = 10,
    max_users: int = 1000,
    output_path: str = "reports/ablation.json",
    seed: int = 42,
) -> dict[str, Any]:
    """Nghiên cứu đóng góp từng thành phần và benchmark kích thước pool MMR."""
    setup_logging()
    LOGGER.info("Bắt đầu chạy Ablation Study & MMR Candidate Pool Benchmark trên %d người dùng...", max_users)

    df_ratings = load_ratings()
    _, _, _, test_df = temporal_split_four_way(df_ratings)
    test_truth = dict(zip(test_df.user_id, test_df.item_id, strict=True))

    recommender = Recommender()
    engine = recommender._engine
    genres_map = recommender.metadata.get("genres", {})
    popular_items = recommender.metadata["popular"]

    eligible_users = [u for u in test_truth if u in engine.user_map]
    rng = np.random.default_rng(seed)
    if len(eligible_users) > max_users:
        eligible_users = list(rng.choice(eligible_users, size=max_users, replace=False))

    # Pre-extract candidates và features cho 1000 users 1 lần duy nhất
    LOGGER.info("Tiền trích xuất candidates & features cho %d ablation users...", len(eligible_users))
    user_data: dict[int, dict[str, Any]] = {}
    cand_k = engine.config.get("candidate_k", 200)

    for u_id in eligible_users:
        cands = engine.retriever.retrieve(u_id, k=cand_k, filter_seen=True)
        feats = engine.feature_builder.build_features(cands, user_id=u_id)
        user_data[u_id] = {
            "cands": cands,
            "feats": feats,
            "seen": engine.seen_by_user.get(u_id, set()),
            "true_item": test_truth[u_id],
        }

    # Pre-rank learned candidates
    if engine.learned_ranker is not None:
        for u_id, d in user_data.items():
            d["learned_ranked"] = engine.ranker.rank(d["feats"])

    variants = {
        "Popularity Baseline": {"type": "pop"},
        "SVD Only": {"type": "weighted", "alpha": 1.0},
        "SVD + Popularity (Weighted Baseline)": {"type": "weighted", "alpha": 0.85},
        "Stage-2 Learned Ranker": {"type": "learned_no_div"},
        "Full Pipeline (Learned + MMR)": {"type": "learned_mmr", "div_lambda": 0.05, "pool_k": 40},
        "MMR Pool 20": {"type": "learned_mmr", "div_lambda": 0.05, "pool_k": 20},
        "MMR Pool 40": {"type": "learned_mmr", "div_lambda": 0.05, "pool_k": 40},
        "MMR Pool 100": {"type": "learned_mmr", "div_lambda": 0.05, "pool_k": 100},
        "MMR Pool 200": {"type": "learned_mmr", "div_lambda": 0.05, "pool_k": 200},
    }

    results: dict[str, Any] = {}

    for var_name, var_cfg in variants.items():
        recalls: list[float] = []
        ilds: list[float] = []
        latencies: list[float] = []
        v_type = var_cfg["type"]

        for u_id in eligible_users:
            t0 = time.perf_counter()
            d = user_data[u_id]
            true_item = d["true_item"]

            if v_type == "pop":
                seen = d["seen"]
                preds = [it for it in popular_items if it not in seen][:k]
            elif v_type == "weighted":
                alpha = var_cfg["alpha"]
                scorer = TwoStageRanker(latent_weight=alpha)
                ranked = scorer.rank(d["feats"])
                preds = [r.item_id for r in ranked[:k]]
            elif v_type == "learned_no_div":
                ranked = d.get("learned_ranked", engine.ranker.rank(d["feats"]))
                preds = [r.item_id for r in ranked[:k]]
            elif v_type == "learned_mmr":
                ranked = d.get("learned_ranked", engine.ranker.rank(d["feats"]))
                lam = var_cfg["div_lambda"]
                pool_k = var_cfg["pool_k"]
                final_recs = engine.diversity_reranker.rerank(
                    ranked, k=k, diversity_lambda_override=lam, rerank_pool_k=pool_k
                )
                preds = [r.item_id for r in final_recs]

            t_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(t_ms)
            recalls.append(hit_rate_at_k(preds, true_item))
            ilds.append(intra_list_diversity(preds, genres_map))

        lat_summary = summarize_latencies(latencies)
        results[var_name] = {
            "recall@10": float(np.mean(recalls)),
            "intra_list_diversity": float(np.mean(ilds)),
            "p50_latency_ms": lat_summary.p50_ms,
            "p95_latency_ms": lat_summary.p95_ms,
        }

    save_json(output_path, results)
    LOGGER.info("Kết quả Ablation Study & Pool Benchmark: %s", results)
    return results


def main() -> None:
    import sys
    evaluate_recommender()
    run_ablation_study()
    sys.exit(0)


if __name__ == "__main__":
    main()
