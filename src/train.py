"""Module huấn luyện hệ thống gợi ý 2 tầng hoàn chỉnh (Canonical Two-Stage Training Pipeline).

Quy trình chuẩn công nghiệp (Production Recommender Pipeline):
1. Nạp MovieLens 1M và sinh bảng kê nguồn gốc dữ liệu (Data Manifest kèm SHA256).
2. Phân chia 4 tập độc lập theo thời gian (4-Way Temporal Split: Retrieval -> Rank-Train -> Val -> Test)
   và sinh Split Manifest (Per-User Temporal Holdout).
3. Huấn luyện Stage-1 Retrieval: Xây dựng ma trận tương tác ngầm định (Implicit-Positive: Rating >= 4.0),
   phân rã nhân tử ẩn TruncatedSVD, tính Log-transformed Popularity Prior và User Genre Profiles (chỉ từ positive ratings).
4. Tạo Candidate Dataset tại mốc t_rank: Sinh tập huấn luyện (X, y, groups) cho Ranker với positive target
   và sampled negative proxies từ candidate pool.
5. Huấn luyện Stage-2 Learned Ranker (XGBoost / Logistic Regression) với 15 features phân cấp.
6. Validation Tuning: Tối ưu hóa siêu tham số alpha (cho baseline) và diversity_lambda (cho MMR) với
   kích thước rerank_pool_k = 40 đồng nhất, áp dụng pre-ranking caching để tối ưu tốc độ.
7. Đóng gói và lưu trữ Versioned Artifacts (models/<version>/), cập nhật production.json và đồng bộ backward-compatible.
8. Báo cáo đánh giá phễu (Stage Funnel Metrics) và kết thúc quy trình.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from .artifacts.writer import save_versioned_bundle
from .config import TrainConfig
from .data import (
    build_positive_interaction_matrix,
    build_user_genre_profiles,
    create_data_manifest,
    create_split_manifest,
    extract_seen_items,
    load_movies,
    load_ratings,
    temporal_split_four_way,
)
from .ranking.dataset import RankDatasetBuilder
from .ranking.features import CandidateFeatureBuilder
from .ranking.model import LearnedRanker, WeightedFusionRanker
from .ranking.scorer import RankedCandidate, TwoStageRanker
from .ranking.trainer import train_learned_ranker
from .reranking.diversity import DiversityReranker
from .retrieval.svd import SVDRetriever
from .utils import LOGGER, save_json, set_seed, setup_logging


def train_model(config: TrainConfig | None = None) -> dict[str, Any]:
    """Quy trình huấn luyện hệ thống gợi ý 2 tầng hoàn chỉnh."""
    cfg = config or TrainConfig()
    setup_logging()
    set_seed(cfg.seed)
    LOGGER.info("=== Bắt đầu quy trình huấn luyện Two-Stage Recommender Pipeline (Version: %s) ===", cfg.model_version)

    # Step 1: Nạp dữ liệu & Sinh Data Manifest
    df_ratings = load_ratings()
    df_movies = load_movies()
    data_manifest = create_data_manifest(df_ratings=df_ratings)
    LOGGER.info("Đã nạp MovieLens 1M: %d tương tác từ %d users và %d items.", len(df_ratings), data_manifest["unique_users"], data_manifest["unique_items"])

    # Step 2: 4-Way Temporal Split & Sinh Split Manifest
    LOGGER.info("Thực hiện 4-Way Temporal Split (min_positive=%d, threshold=%.1f)...", cfg.min_positive, cfg.rating_threshold)
    retrieval_train_df, rank_train_df, val_df, test_df = temporal_split_four_way(
        df_ratings,
        rating_threshold=cfg.rating_threshold,
        min_positive=cfg.min_positive,
    )
    split_manifest = create_split_manifest(
        retrieval_train_df=retrieval_train_df,
        rank_train_df=rank_train_df,
        val_df=val_df,
        test_df=test_df,
        rating_threshold=cfg.rating_threshold,
        min_positive=cfg.min_positive,
    )
    LOGGER.info(
        "Phân chia hoàn tất: Retrieval-Train (%d dòng, %d positive), Rank-Train (%d users), Val (%d users), Test (%d users).",
        len(retrieval_train_df),
        split_manifest["counts"]["retrieval_train_positive_rows"],
        len(rank_train_df),
        len(val_df),
        len(test_df),
    )

    # Step 3: Huấn luyện Stage-1 Retrieval
    users = np.sort(retrieval_train_df.user_id.unique())
    items = np.sort(retrieval_train_df.item_id.unique())
    user_map = {int(u): i for i, u in enumerate(users)}
    item_map = {int(m): i for i, m in enumerate(items)}

    interaction_matrix = build_positive_interaction_matrix(
        retrieval_train_df, user_map, item_map, cfg.rating_threshold
    )
    LOGGER.info("Kích thước ma trận tương tác: %s với %d tương tác tích cực.", interaction_matrix.shape, interaction_matrix.nnz)

    svd = TruncatedSVD(n_components=cfg.embedding_dim, random_state=cfg.seed)
    user_embeddings = svd.fit_transform(interaction_matrix)
    item_embeddings = svd.components_.T
    LOGGER.info("Trích xuất Embeddings thành công: User %s, Item %s.", user_embeddings.shape, item_embeddings.shape)

    # Tính toán Popularity Priors & Seen Filter
    positive_retrieval = retrieval_train_df[retrieval_train_df.rating >= cfg.rating_threshold]
    popularity_counts = positive_retrieval.groupby("item_id").size().sort_values(ascending=False)
    popular_items = [int(x) for x in popularity_counts.index.to_list()]

    max_count = max(popularity_counts.values, default=1)
    max_log_count = float(np.log1p(max_count)) if max_count > 0 else 1.0
    log_popularity: dict[int, float] = {
        int(item_id): float(np.log1p(cnt) / max_log_count)
        for item_id, cnt in popularity_counts.items()
    }

    seen_by_user = extract_seen_items(retrieval_train_df)

    genre_map: dict[int, set[str]] = {
        int(row.item_id): set(str(row.genres).split("|"))
        for row in df_movies.itertuples()
    }
    title_map: dict[int, str] = {
        int(row.item_id): str(row.title) for row in df_movies.itertuples()
    }

    # SỬA TRIỆT ĐỂ P0.1: User Genre Profiles CHỈ tính từ positive interactions (rating >= 4.0)
    user_genre_profiles = build_user_genre_profiles(
        retrieval_train_df, df_movies, rating_threshold=cfg.rating_threshold
    )
    LOGGER.info("Đã xây dựng %d User Genre Profiles tuân thủ Implicit-Positive contract.", len(user_genre_profiles))

    # Thống kê User và Item stats phục vụ feature engineering
    user_pos_counts = positive_retrieval.groupby("user_id").size().to_dict()
    user_avg_ratings = retrieval_train_df.groupby("user_id")["rating"].mean().to_dict()
    user_stats: dict[int, dict[str, float]] = {
        int(u): {
            "positive_count": float(user_pos_counts.get(u, 0)),
            "avg_rating": float(user_avg_ratings.get(u, 3.8)),
        }
        for u in users
    }

    item_rating_counts = retrieval_train_df.groupby("item_id").size().to_dict()
    item_avg_ratings = retrieval_train_df.groupby("item_id")["rating"].mean().to_dict()
    item_stats: dict[int, dict[str, float]] = {
        int(m): {
            "rating_count": float(item_rating_counts.get(m, 0)),
            "avg_rating": float(item_avg_ratings.get(m, 3.5)),
        }
        for m in items
    }

    # Step 4: Tạo tập huấn luyện Stage-2 Ranker
    feature_builder = CandidateFeatureBuilder(
        popularity_scores=log_popularity,
        genre_map=genre_map,
        user_genre_profiles=user_genre_profiles,
        user_stats=user_stats,
        item_stats=item_stats,
    )

    svd_retriever = SVDRetriever(
        user_embeddings=user_embeddings,
        item_embeddings=item_embeddings,
        user_map=user_map,
        item_map=item_map,
        items=items,
        seen_by_user=seen_by_user,
    )

    LOGGER.info("Sinh tập dữ liệu giám sát Stage-2 Ranker từ tập Rank-Train (candidate_k=%d)...", cfg.candidate_k)
    dataset_builder = RankDatasetBuilder(feature_builder=feature_builder, candidate_k=100)
    X_rank, y_rank, groups_rank = dataset_builder.build_dataset(
        rank_train_df=rank_train_df,
        retriever=svd_retriever,
        max_users=3000,
        seed=cfg.seed,
    )

    # Step 5: Huấn luyện Stage-2 Learned Ranker
    learned_ranker = train_learned_ranker(
        X=X_rank,
        y=y_rank,
        groups=groups_rank,
        model_type=cfg.ranker_model_type,
        seed=cfg.seed,
    )

    # Step 6: Validation Tuning (với rerank_pool_k = 40 đồng nhất)
    LOGGER.info("Bắt đầu điều chỉnh (tuning) siêu tham số trên Validation với rerank_pool_k = %d...", cfg.rerank_pool_k)
    val_truth = dict(zip(val_df.user_id, val_df.item_id, strict=True))
    eligible_val_users = np.array([u for u in val_truth if u in user_map])

    rng = np.random.default_rng(cfg.seed)
    if len(eligible_val_users) > cfg.max_val_users:
        sampled_val_users = rng.choice(eligible_val_users, size=cfg.max_val_users, replace=False)
    else:
        sampled_val_users = eligible_val_users

    # Chuẩn bị candidates cho tập Validation
    user_val_cands: dict[int, tuple[int, list]] = {}
    val_alpha_precomputed: dict[int, tuple[int, np.ndarray, np.ndarray, np.ndarray]] = {}
    val_learned_ranked: dict[int, tuple[int, list[RankedCandidate]]] = {}

    ranker_scorer = TwoStageRanker(rank_model=learned_ranker)

    for uid in sampled_val_users:
        u_id = int(uid)
        true_item = val_truth[u_id]
        cands = svd_retriever.retrieve(u_id, k=cfg.candidate_k, filter_seen=True)
        if not cands:
            continue

        user_val_cands[u_id] = (true_item, cands)

        # Precompute cho alpha tuning
        cand_indices = np.array([c.item_id for c in cands])
        raw_sc = np.array([c.retrieval_score for c in cands], dtype=np.float32)
        min_s, max_s = float(raw_sc.min()), float(raw_sc.max())
        denom = max_s - min_s
        norm_sc = (raw_sc - min_s) / (denom + 1e-9) if denom > 1e-9 else np.ones_like(raw_sc)
        pop_sc = np.array([log_popularity.get(int(m), 0.0) for m in cand_indices], dtype=np.float32)
        val_alpha_precomputed[u_id] = (true_item, cand_indices, norm_sc, pop_sc)

        # Precompute ranked candidates bằng learned ranker (chạy 1 lần duy nhất!)
        feats = feature_builder.build_features(cands, user_id=u_id)
        ranked = ranker_scorer.rank(feats)
        val_learned_ranked[u_id] = (true_item, ranked)

    # Tuning alpha cho Baseline Heuristic Fusion (vectorized, chạy cực nhanh)
    alpha_recalls: dict[str, float] = {}
    for alpha in cfg.alpha_candidates:
        hits: list[float] = []
        for u_id, (true_item, cand_indices, norm_sc, pop_sc) in val_alpha_precomputed.items():
            combined = alpha * norm_sc + (1.0 - alpha) * pop_sc
            top_cand = cand_indices[np.argsort(-combined)[:cfg.final_k]]
            hits.append(float(true_item in top_cand))
        alpha_recalls[f"{alpha:.2f}"] = float(np.mean(hits)) if hits else 0.0

    best_alpha = max(cfg.alpha_candidates, key=lambda a: alpha_recalls[f"{a:.2f}"])
    LOGGER.info("Kết quả tuning alpha (Baseline Ranker) trên Validation: %s -> Chọn best_alpha = %.2f", alpha_recalls, best_alpha)

    # Tuning diversity_lambda cho Stage 3 MMR Reranking trên các candidates đã pre-ranked
    diversity_tuning_results: dict[str, dict[str, float]] = {}
    best_tradeoff_score = -1.0
    best_lambda = 0.05

    for div_lambda in cfg.diversity_lambda_candidates:
        hits = []
        ilds = []
        reranker = DiversityReranker(
            genre_map=genre_map,
            default_lambda=div_lambda,
            default_rerank_pool_k=cfg.rerank_pool_k,
        )

        for u_id, (true_item, ranked) in val_learned_ranked.items():
            final_recs = reranker.rerank(
                ranked,
                k=cfg.final_k,
                diversity_lambda_override=div_lambda,
                rerank_pool_k=cfg.rerank_pool_k,
            )
            top_ids = [r.item_id for r in final_recs]
            hit = float(true_item in top_ids)
            hits.append(hit)

            if len(top_ids) > 1:
                dists: list[float] = []
                for i_idx, it1 in enumerate(top_ids):
                    for it2 in top_ids[i_idx + 1 :]:
                        g1, g2 = genre_map.get(it1, set()), genre_map.get(it2, set())
                        if g1 and g2:
                            u_cnt = len(g1 | g2)
                            if u_cnt > 0:
                                dists.append(1.0 - len(g1 & g2) / u_cnt)
                ilds.append(float(np.mean(dists)) if dists else 0.0)

        mean_recall = float(np.mean(hits)) if hits else 0.0
        mean_ild = float(np.mean(ilds)) if ilds else 0.0
        tradeoff = mean_recall + 0.01 * mean_ild

        key_str = f"lambda_{div_lambda:.2f}"
        diversity_tuning_results[key_str] = {
            "recall@10": round(mean_recall, 4),
            "intra_list_diversity": round(mean_ild, 4),
            "tradeoff_objective": round(tradeoff, 4),
        }

        if tradeoff > best_tradeoff_score:
            best_tradeoff_score = tradeoff
            best_lambda = div_lambda

    LOGGER.info("Kết quả tuning diversity_lambda: %s -> Chọn best_diversity_lambda = %.2f", diversity_tuning_results, best_lambda)

    # Step 7: Đóng gói và lưu trữ Versioned Artifacts
    metadata_payload = {
        "users": users,
        "items": items,
        "user_map": user_map,
        "item_map": item_map,
        "popular": popular_items,
        "popularity_counts": popularity_counts.to_dict(),
        "log_popularity": log_popularity,
        "seen": seen_by_user,
        "genres": genre_map,
        "titles": title_map,
        "user_genre_profiles": user_genre_profiles,
        "user_stats": user_stats,
        "item_stats": item_stats,
    }

    config_payload = {
        "schema_version": 4,
        "version": cfg.model_version,
        "seed": cfg.seed,
        "embedding_dimension": cfg.embedding_dim,
        "candidate_k": cfg.candidate_k,
        "ranking_k": cfg.ranking_k,
        "rerank_pool_k": cfg.rerank_pool_k,
        "final_k": cfg.final_k,
        "top_k": cfg.final_k,
        "latent_weight": float(best_alpha),
        "diversity_lambda": float(best_lambda),
        "ranker_model_type": cfg.ranker_model_type,
        "interaction_contract": {
            "feedback": "implicit_positive",
            "rating_threshold": cfg.rating_threshold,
            "explicit_zeros_stored": False,
            "split": "per_user_temporal_holdout",
            "seen_filter": "all_train_interactions",
            "negative_sampling_proxy": "unobserved_candidates_without_impressions",
        },
        "validation_tuning_grid": {
            "alpha_tuning": alpha_recalls,
            "diversity_tuning": diversity_tuning_results,
        },
        "positive_interactions": int(interaction_matrix.nnz),
    }

    saved_version_dir = save_versioned_bundle(
        base_dir=cfg.model_dir,
        version=cfg.model_version,
        user_embeddings=user_embeddings,
        item_embeddings=item_embeddings,
        metadata=metadata_payload,
        config_payload=config_payload,
        ranker=learned_ranker,
        data_manifest=data_manifest,
        split_manifest=split_manifest,
    )
    LOGGER.info("Đã lưu trữ thành công toàn bộ Versioned Model Bundle tại '%s'!", saved_version_dir)

    return config_payload


def main() -> None:
    """Hàm main thực thi script huấn luyện khi gọi từ CLI."""
    train_model()


if __name__ == "__main__":
    main()
