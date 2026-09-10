"""Module huấn luyện hệ thống gợi ý 2 tầng hoàn chỉnh (Canonical Two-Stage Training Pipeline).

Quy trình chuẩn công nghiệp (Production Recommender Pipeline):
1. Nạp MovieLens 1M và sinh bảng kê nguồn gốc dữ liệu (Data Manifest kèm SHA256).
2. Phân chia 4 tập độc lập theo thời gian (4-Way Temporal Split: Retrieval -> Rank-Train -> Val -> Test)
   và sinh Split Manifest (Per-User Temporal Holdout).
3. Huấn luyện Stage-1 Retrieval: SVD + Popularity + Genre, sau đó hợp nhất bằng RRF.
4. Tạo Candidate Dataset tại mốc t_rank: Sinh tập huấn luyện (X, y, groups) cho Ranker với positive target
   và sampled negative proxies từ candidate pool.
5. Huấn luyện Stage-2 XGBRanker (rank:ndcg) với query groups và 19 feature.
6. Validation Tuning: Tối ưu hóa siêu tham số alpha (cho baseline) và diversity_lambda (cho MMR) với
   kích thước rerank_pool_k = 40 đồng nhất, áp dụng pre-ranking caching để tối ưu tốc độ.
7. Đóng gói Release Candidate; promotion là bước explicit sau Locked Test.
8. Báo cáo đánh giá phễu (Stage Funnel Metrics) và kết thúc quy trình.
"""

from __future__ import annotations

from typing import Any

import numpy as np
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
    temporal_split,
)
from .evaluation.ranking_metrics import ranker_ndcg_at_k, ranker_recall_at_k
from .ranking.dataset import RankDatasetBuilder
from .ranking.features import CandidateFeatureBuilder
from .ranking.scorer import RankedCandidate, TwoStageRanker
from .ranking.trainer import train_learned_ranker
from .reranking.diversity import DiversityReranker
from .retrieval.genre import GenreRetriever
from .retrieval.merger import MultiSourceRetriever
from .retrieval.popularity import PopularityRetriever
from .retrieval.svd import SVDRetriever
from .utils import LOGGER, set_seed, setup_logging


def train_model(config: TrainConfig | None = None) -> dict[str, Any]:
    """Quy trình huấn luyện hệ thống gợi ý 2 tầng hoàn chỉnh."""
    cfg = config or TrainConfig()
    setup_logging()
    set_seed(cfg.seed)
    LOGGER.info(
        "=== Bắt đầu quy trình huấn luyện Two-Stage Recommender Pipeline (Version: %s) ===",
        cfg.model_version,
    )

    # Step 1: Nạp dữ liệu & Sinh Data Manifest
    df_ratings = load_ratings()
    df_movies = load_movies()
    data_manifest = create_data_manifest(df_ratings=df_ratings)
    LOGGER.info(
        "Đã nạp MovieLens 1M: %d tương tác từ %d users và %d items.",
        len(df_ratings),
        data_manifest["unique_users"],
        data_manifest["unique_items"],
    )

    # Step 2: 4-Way Temporal Split & Sinh Split Manifest
    LOGGER.info(
        "Thực hiện 4-Way Temporal Split (min_positive=%d, threshold=%.1f)...",
        cfg.min_positive,
        cfg.rating_threshold,
    )
    retrieval_train_df, rank_train_df, val_df, test_df = temporal_split(
        df_ratings,
        rating_threshold=cfg.rating_threshold,
        min_positive=cfg.min_positive,
        protocol="per_user",
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
    LOGGER.info(
        "Kích thước ma trận tương tác: %s với %d tương tác tích cực.",
        interaction_matrix.shape,
        interaction_matrix.nnz,
    )

    svd = TruncatedSVD(n_components=cfg.embedding_dim, random_state=cfg.seed)
    user_embeddings = svd.fit_transform(interaction_matrix)
    item_embeddings = svd.components_.T
    LOGGER.info(
        "Trích xuất Embeddings thành công: User %s, Item %s.",
        user_embeddings.shape,
        item_embeddings.shape,
    )

    # Tính toán Popularity Priors & Seen Filter
    positive_retrieval = retrieval_train_df[
        retrieval_train_df.rating >= cfg.rating_threshold
    ]
    popularity_counts = (
        positive_retrieval.groupby("item_id").size().sort_values(ascending=False)
    )
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
    LOGGER.info(
        "Đã xây dựng %d User Genre Profiles tuân thủ Implicit-Positive contract.",
        len(user_genre_profiles),
    )

    # Thống kê User và Item stats phục vụ feature engineering
    user_pos_counts = positive_retrieval.groupby("user_id").size().to_dict()
    user_avg_ratings = retrieval_train_df.groupby("user_id")["rating"].mean().to_dict()
    user_stats: dict[int, dict[str, float]] = {
        int(u): {
            "positive_count": float(user_pos_counts.get(u, 0)),
            "interaction_count": float((retrieval_train_df.user_id == u).sum()),
            "avg_rating": float(user_avg_ratings.get(u, 3.8)),
        }
        for u in users
    }

    item_rating_counts = retrieval_train_df.groupby("item_id").size().to_dict()
    item_positive_counts = positive_retrieval.groupby("item_id").size().to_dict()
    item_avg_ratings = retrieval_train_df.groupby("item_id")["rating"].mean().to_dict()
    item_stats: dict[int, dict[str, float]] = {
        int(m): {
            "positive_count": float(item_positive_counts.get(m, 0)),
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
        interactions_df=retrieval_train_df,
        rating_threshold=cfg.rating_threshold,
    )

    svd_retriever = SVDRetriever(
        user_embeddings=user_embeddings,
        item_embeddings=item_embeddings,
        user_map=user_map,
        item_map=item_map,
        items=items,
        seen_by_user=seen_by_user,
    )

    popularity_retriever = PopularityRetriever(
        popular_items=popular_items,
        popularity_scores=log_popularity,
        seen_by_user=seen_by_user,
    )
    genre_retriever = GenreRetriever(
        popular_items=popular_items,
        genre_map=genre_map,
        user_genre_profiles=user_genre_profiles,
        popularity_scores=log_popularity,
        seen_by_user=seen_by_user,
    )
    multi_retriever = MultiSourceRetriever(
        retrievers={
            "svd": (svd_retriever, cfg.svd_candidate_k),
            "popularity": (popularity_retriever, cfg.popularity_candidate_k),
            "genre": (genre_retriever, cfg.genre_candidate_k),
        },
        rrf_k=60,
    )

    LOGGER.info(
        "Sinh dữ liệu Stage-2 từ candidate pool đa nguồn (candidate_k=%d)...",
        cfg.candidate_k,
    )
    dataset_builder = RankDatasetBuilder(
        feature_builder=feature_builder,
        candidate_k=cfg.candidate_k,
    )
    X_rank, y_rank, groups_rank = dataset_builder.build_dataset(
        rank_train_df=rank_train_df,
        retriever=multi_retriever,
        max_users=cfg.max_rank_train_users,
        seed=cfg.seed,
    )
    LOGGER.info(
        "Coverage target retrieval của rank-train: %.2f%%",
        100.0
        * float(dataset_builder.last_build_stats.get("target_retrieval_rate", 0.0)),
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
    LOGGER.info(
        "Bắt đầu điều chỉnh (tuning) siêu tham số trên Validation với rerank_pool_k = %d...",
        cfg.rerank_pool_k,
    )
    val_truth = dict(zip(val_df.user_id, val_df.item_id, strict=True))
    eligible_val_users = np.array([u for u in val_truth if u in user_map])

    rng = np.random.default_rng(cfg.seed)
    if len(eligible_val_users) > cfg.max_val_users:
        sampled_val_users = rng.choice(
            eligible_val_users, size=cfg.max_val_users, replace=False
        )
    else:
        sampled_val_users = eligible_val_users

    # Chuẩn bị candidates cho tập Validation
    val_alpha_precomputed: dict[
        int, tuple[int, np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    val_learned_ranked: dict[int, tuple[int, list[RankedCandidate]]] = {}
    val_retrieval_ranked: dict[int, list[RankedCandidate]] = {}
    val_stage1_ndcgs: list[float] = []
    val_stage1_recalls: list[float] = []
    val_learned_ndcgs: list[float] = []
    val_learned_recalls: list[float] = []

    ranker_scorer = TwoStageRanker(rank_model=learned_ranker)

    for uid in sampled_val_users:
        u_id = int(uid)
        true_item = val_truth[u_id]
        val_timestamp = int(val_df.loc[val_df["user_id"] == u_id, "timestamp"].iloc[0])
        val_seen = extract_seen_items(
            df_ratings[
                (df_ratings["user_id"] == u_id)
                & (df_ratings["timestamp"] < val_timestamp)
            ]
        ).get(u_id, set())
        cands = multi_retriever.retrieve(
            u_id,
            k=cfg.candidate_k,
            filter_seen=True,
            seen_items_override=val_seen,
        )
        if not cands:
            continue

        # Precompute cho alpha tuning
        cand_indices = np.array([c.item_id for c in cands])
        val_features = feature_builder.build_features(
            cands,
            user_id=u_id,
            as_of_timestamp=val_timestamp,
        )
        val_matrix = np.vstack(
            [feature.to_feature_vector() for feature in val_features]
        )
        norm_sc = val_matrix[:, 0]
        pop_sc = val_matrix[:, 2]
        val_alpha_precomputed[u_id] = (true_item, cand_indices, norm_sc, pop_sc)

        # Precompute ranked candidates bằng learned ranker (chạy 1 lần duy nhất!)
        feats = val_features
        ranked = ranker_scorer.rank(feats)
        val_learned_ranked[u_id] = (true_item, ranked)
        val_retrieval_ranked[u_id] = [
            RankedCandidate(
                item_id=feature.item_id,
                relevance_score=1.0 - index / max(1, len(feats)),
                features=feature,
            )
            for index, feature in enumerate(feats)
        ]
        cand_ids = [candidate.item_id for candidate in cands]
        ranked_ids = [candidate.item_id for candidate in ranked]
        val_stage1_ndcgs.append(ranker_ndcg_at_k(cand_ids, true_item, k=cfg.final_k))
        val_stage1_recalls.append(
            ranker_recall_at_k(cand_ids, true_item, k=cfg.final_k)
        )
        val_learned_ndcgs.append(ranker_ndcg_at_k(ranked_ids, true_item, k=cfg.final_k))
        val_learned_recalls.append(
            ranker_recall_at_k(ranked_ids, true_item, k=cfg.final_k)
        )

    # Tuning alpha cho Baseline Heuristic Fusion (vectorized, chạy cực nhanh)
    alpha_recalls: dict[str, float] = {}
    for alpha in cfg.alpha_candidates:
        hits: list[float] = []
        for _u_id, (
            true_item,
            cand_indices,
            norm_sc,
            pop_sc,
        ) in val_alpha_precomputed.items():
            combined = alpha * norm_sc + (1.0 - alpha) * pop_sc
            top_cand = cand_indices[np.argsort(-combined)[: cfg.final_k]]
            hits.append(float(true_item in top_cand))
        alpha_recalls[f"{alpha:.2f}"] = float(np.mean(hits)) if hits else 0.0

    best_alpha = max(cfg.alpha_candidates, key=lambda a: alpha_recalls[f"{a:.2f}"])
    LOGGER.info(
        "Kết quả tuning alpha (Baseline Ranker) trên Validation: %s -> Chọn best_alpha = %.2f",
        alpha_recalls,
        best_alpha,
    )

    # Gate Stage 2: chỉ bật ranker nếu thắng retrieval order trên Dev.
    stage1_ndcg = float(np.mean(val_stage1_ndcgs)) if val_stage1_ndcgs else 0.0
    stage1_recall = float(np.mean(val_stage1_recalls)) if val_stage1_recalls else 0.0
    learned_ndcg = float(np.mean(val_learned_ndcgs)) if val_learned_ndcgs else 0.0
    learned_recall = float(np.mean(val_learned_recalls)) if val_learned_recalls else 0.0
    ranker_enabled = learned_ndcg > stage1_ndcg and learned_recall >= stage1_recall
    LOGGER.info(
        "Dev model selection: order NDCG=%.4f/Recall=%.4f; ranker NDCG=%.4f/Recall=%.4f; enabled=%s",
        stage1_ndcg,
        stage1_recall,
        learned_ndcg,
        learned_recall,
        ranker_enabled,
    )

    # Tuning diversity theo constraint: NDCG không giảm quá 2%, sau đó chọn ILD cao nhất.
    diversity_tuning_results: dict[str, dict[str, float]] = {}
    # Lambda=1 là fallback an toàn: thuần relevance nếu không có lambda nào
    # vượt guardrail đa dạng hóa. Lambda=0 vẫn là MMR hợp lệ, không phải cờ tắt.
    best_lambda = 1.0
    best_ild = -1.0
    relevance_baseline_ndcg = learned_ndcg if ranker_enabled else stage1_ndcg
    relevance_floor = relevance_baseline_ndcg * 0.98

    for div_lambda in cfg.diversity_lambda_candidates:
        hits: list[float] = []
        ndcgs: list[float] = []
        ilds: list[float] = []
        reranker = DiversityReranker(
            genre_map=genre_map,
            default_lambda=div_lambda,
            default_rerank_pool_k=cfg.rerank_pool_k,
        )

        for u_id, (true_item, ranked) in val_learned_ranked.items():
            source_ranked = ranked if ranker_enabled else val_retrieval_ranked[u_id]
            final_recs = reranker.rerank(
                source_ranked,
                k=cfg.final_k,
                diversity_lambda_override=div_lambda,
                rerank_pool_k=cfg.rerank_pool_k,
            )
            top_ids = [r.item_id for r in final_recs]
            hit = float(true_item in top_ids)
            hits.append(hit)
            ndcgs.append(ranker_ndcg_at_k(top_ids, true_item, k=cfg.final_k))

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
        mean_ndcg = float(np.mean(ndcgs)) if ndcgs else 0.0
        mean_ild = float(np.mean(ilds)) if ilds else 0.0
        valid = mean_ndcg >= relevance_floor

        key_str = f"lambda_{div_lambda:.2f}"
        diversity_tuning_results[key_str] = {
            "recall@10": round(mean_recall, 4),
            "ndcg@10": round(mean_ndcg, 4),
            "intra_list_diversity": round(mean_ild, 4),
            "relevance_floor": round(relevance_floor, 4),
            "within_relevance_guardrail": valid,
        }

        if valid and mean_ild > best_ild:
            best_ild = mean_ild
            best_lambda = div_lambda

    LOGGER.info(
        "Kết quả tuning diversity_lambda: %s -> Chọn best_diversity_lambda = %.2f",
        diversity_tuning_results,
        best_lambda,
    )

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
        "schema_version": 5,
        "version": cfg.model_version,
        "seed": cfg.seed,
        "embedding_dimension": cfg.embedding_dim,
        "candidate_k": cfg.candidate_k,
        "max_rank_train_users": cfg.max_rank_train_users,
        "rerank_pool_k": cfg.rerank_pool_k,
        "final_k": cfg.final_k,
        "top_k": cfg.final_k,
        "candidate_contract": cfg.candidate_contract,
        "multi_source_retrieval": True,
        "retrieval_fusion": {"method": "rrf", "rrf_k": 60},
        "latent_weight": float(best_alpha),
        "diversity_lambda": float(best_lambda),
        "ranker_model_type": learned_ranker.model_type,
        "ranker_enabled": ranker_enabled,
        "feature_schema_version": "rank-features-v1",
        "interaction_contract": {
            "feedback": "implicit_positive",
            "rating_threshold": cfg.rating_threshold,
            "explicit_zeros_stored": False,
            "split": "per_user_temporal_holdout",
            "seen_filter": "all_interactions_before_request_timestamp",
            "negative_sampling_proxy": "unobserved_candidates_without_impressions",
        },
        "validation_tuning_grid": {
            "alpha_tuning": alpha_recalls,
            "diversity_tuning": diversity_tuning_results,
        },
        "positive_interactions": int(interaction_matrix.nnz),
        "rank_train_target_retrieval": dataset_builder.last_build_stats,
        "dev_metrics": {
            "stage1_order_ndcg@10": stage1_ndcg,
            "stage1_order_recall@10": stage1_recall,
            "ranker_ndcg@10": learned_ndcg,
            "ranker_recall@10": learned_recall,
            "ranker_dev_selection_passed": ranker_enabled,
        },
    }

    saved_version_dir = save_versioned_bundle(
        base_dir=cfg.model_dir,
        version=f"candidates/{cfg.model_version}",
        user_embeddings=user_embeddings,
        item_embeddings=item_embeddings,
        metadata=metadata_payload,
        config_payload=config_payload,
        # Chỉ đóng gói ranker khi Dev model selection đã bật; artifact bị loại không nên
        # trở thành một dependency runtime hoặc bị hiểu nhầm là model active.
        ranker=learned_ranker if ranker_enabled else None,
        data_manifest=data_manifest,
        split_manifest=split_manifest,
        evaluation_report={"dev_metrics": config_payload["dev_metrics"]},
        publish=False,
    )
    LOGGER.info(
        "Đã lưu Release Candidate (chưa active) tại '%s'. Locked Test và promotion là bước riêng.",
        saved_version_dir,
    )

    return config_payload


def main() -> None:
    """Hàm main thực thi script huấn luyện khi gọi từ CLI."""
    train_model()


if __name__ == "__main__":
    main()
