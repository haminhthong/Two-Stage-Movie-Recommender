"""Huấn luyện pipeline retrieval đa nguồn, ranking và MMR."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import TruncatedSVD

from .config import TrainConfig
from .data import (
    build_positive_interaction_matrix,
    build_user_genre_profiles,
    create_data_manifest,
    extract_seen_items,
    load_movies,
    load_ratings,
    seen_items_before,
    temporal_split,
)
from .evaluation import intra_list_diversity, ranker_ndcg_at_k, ranker_recall_at_k
from .model_io import save_model
from .ranking.dataset import RankDatasetBuilder
from .ranking.features import CandidateFeatureBuilder
from .ranking.scorer import rank_candidates, retrieval_order
from .ranking.trainer import train_learned_ranker
from .reranking.diversity import DiversityReranker
from .retrieval.genre import GenreRetriever
from .retrieval.merger import MultiSourceRetriever
from .retrieval.popularity import PopularityRetriever
from .retrieval.svd import SVDRetriever
from .utils import LOGGER, safe_mean, set_seed, setup_logging


def _build_retrievers(
    cfg: TrainConfig,
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    users: np.ndarray,
    items: np.ndarray,
    seen_by_user: dict[int, set[int]],
    popular_items: list[int],
    log_popularity: dict[int, float],
    genre_map: dict[int, set[str]],
    user_genre_profiles: dict[int, dict[str, float]],
) -> MultiSourceRetriever:
    """Tạo cùng một retrieval contract cho train và evaluation."""
    svd_retriever = SVDRetriever(
        user_embeddings=user_embeddings,
        item_embeddings=item_embeddings,
        user_map={int(user_id): index for index, user_id in enumerate(users)},
        item_map={int(item_id): index for index, item_id in enumerate(items)},
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
    return MultiSourceRetriever(
        retrievers={
            "svd": (svd_retriever, cfg.svd_candidate_k),
            "popularity": (popularity_retriever, cfg.popularity_candidate_k),
            "genre": (genre_retriever, cfg.genre_candidate_k),
        },
        rrf_k=60,
    )


_mean = safe_mean


def train_model(config: TrainConfig | None = None) -> dict[str, Any]:
    """Train và ghi model trực tiếp vào ``models/``."""
    cfg = config or TrainConfig()
    setup_logging()
    set_seed(cfg.seed)
    LOGGER.info("Bắt đầu training Two-Stage Recommender.")

    ratings = load_ratings()
    movies = load_movies()
    data_manifest = create_data_manifest(df_ratings=ratings)
    retrieval_df, rank_train_df, val_df, _test_df = temporal_split(
        ratings,
        rating_threshold=cfg.rating_threshold,
        min_positive=cfg.min_positive,
    )

    users = np.sort(retrieval_df.user_id.unique())
    items = np.sort(retrieval_df.item_id.unique())
    user_map = {int(user_id): index for index, user_id in enumerate(users)}
    item_map = {int(item_id): index for index, item_id in enumerate(items)}
    interaction_matrix = build_positive_interaction_matrix(
        retrieval_df, user_map, item_map, cfg.rating_threshold
    )
    svd = TruncatedSVD(n_components=cfg.embedding_dim, random_state=cfg.seed)
    user_embeddings = svd.fit_transform(interaction_matrix)
    item_embeddings = svd.components_.T

    positive_retrieval = retrieval_df[retrieval_df.rating >= cfg.rating_threshold]
    popularity_counts = (
        positive_retrieval.groupby("item_id").size().sort_values(ascending=False)
    )
    popular_items = [int(item_id) for item_id in popularity_counts.index]
    max_count = max(popularity_counts.values, default=1)
    max_log_count = max(float(np.log1p(max_count)), 1.0)
    log_popularity = {
        int(item_id): float(np.log1p(count) / max_log_count)
        for item_id, count in popularity_counts.items()
    }
    seen_by_user = extract_seen_items(retrieval_df)
    genre_map = {
        int(row.item_id): set(str(row.genres).split("|")) for row in movies.itertuples()
    }
    title_map = {int(row.item_id): str(row.title) for row in movies.itertuples()}
    user_genre_profiles = build_user_genre_profiles(
        retrieval_df, movies, rating_threshold=cfg.rating_threshold
    )

    user_pos_counts = positive_retrieval.groupby("user_id").size().to_dict()
    user_avg_ratings = retrieval_df.groupby("user_id")["rating"].mean().to_dict()
    user_stats = {
        int(user_id): {
            "positive_count": float(user_pos_counts.get(user_id, 0)),
            "interaction_count": float((retrieval_df.user_id == user_id).sum()),
            "avg_rating": float(user_avg_ratings.get(user_id, 3.8)),
        }
        for user_id in users
    }
    item_rating_counts = retrieval_df.groupby("item_id").size().to_dict()
    item_positive_counts = positive_retrieval.groupby("item_id").size().to_dict()
    item_avg_ratings = retrieval_df.groupby("item_id")["rating"].mean().to_dict()
    item_stats = {
        int(item_id): {
            "positive_count": float(item_positive_counts.get(item_id, 0)),
            "rating_count": float(item_rating_counts.get(item_id, 0)),
            "avg_rating": float(item_avg_ratings.get(item_id, 3.5)),
        }
        for item_id in items
    }

    feature_builder = CandidateFeatureBuilder(
        popularity_scores=log_popularity,
        genre_map=genre_map,
        user_genre_profiles=user_genre_profiles,
        user_stats=user_stats,
        item_stats=item_stats,
        interactions_df=retrieval_df,
        rating_threshold=cfg.rating_threshold,
    )
    multi_retriever = _build_retrievers(
        cfg,
        user_embeddings,
        item_embeddings,
        users,
        items,
        seen_by_user,
        popular_items,
        log_popularity,
        genre_map,
        user_genre_profiles,
    )

    dataset_builder = RankDatasetBuilder(feature_builder, candidate_k=cfg.candidate_k)
    X_rank, y_rank, groups_rank = dataset_builder.build_dataset(
        rank_train_df=rank_train_df,
        retriever=multi_retriever,
        max_users=cfg.max_rank_train_users,
        seed=cfg.seed,
    )
    learned_ranker = train_learned_ranker(
        X_rank,
        y_rank,
        groups=groups_rank,
        model_type=cfg.ranker_model_type,
        seed=cfg.seed,
    )

    val_truth = dict(zip(val_df.user_id, val_df.item_id, strict=True))
    val_users = [user_id for user_id in val_truth if user_id in user_map]
    if cfg.max_val_users is not None and len(val_users) > cfg.max_val_users:
        rng = np.random.default_rng(cfg.seed)
        val_users = list(rng.choice(val_users, cfg.max_val_users, replace=False))

    retrieval_ndcgs: list[float] = []
    retrieval_recalls: list[float] = []
    ranker_ndcgs: list[float] = []
    ranker_recalls: list[float] = []
    val_ranked: dict[int, tuple[int, list]] = {}

    for user_id in val_users:
        timestamp = int(val_df.loc[val_df.user_id == user_id, "timestamp"].iloc[0])
        seen = seen_items_before(ratings, user_id, timestamp)
        candidates = multi_retriever.retrieve(
            user_id,
            k=cfg.candidate_k,
            filter_seen=True,
            seen_items_override=seen,
        )
        features = feature_builder.build_features(
            candidates, user_id=user_id, as_of_timestamp=timestamp
        )
        ordered = retrieval_order(features)
        ranked = rank_candidates(features, learned_ranker)
        target = int(val_truth[user_id])
        retrieval_ids = [candidate.item_id for candidate in ordered]
        ranked_ids = [candidate.item_id for candidate in ranked]
        retrieval_ndcgs.append(ranker_ndcg_at_k(retrieval_ids, target, cfg.final_k))
        retrieval_recalls.append(ranker_recall_at_k(retrieval_ids, target, cfg.final_k))
        ranker_ndcgs.append(ranker_ndcg_at_k(ranked_ids, target, cfg.final_k))
        ranker_recalls.append(ranker_recall_at_k(ranked_ids, target, cfg.final_k))
        # Dev MMR chỉ nhìn rerank_pool_k; không giữ toàn bộ 200 object cho
        # hàng nghìn user vì validation không cần chúng sau khi tính metric.
        val_ranked[user_id] = (
            target,
            ranked[: cfg.rerank_pool_k],
            ordered[: cfg.rerank_pool_k],
        )

    retrieval_ndcg = _mean(retrieval_ndcgs)
    retrieval_recall = _mean(retrieval_recalls)
    learned_ndcg = _mean(ranker_ndcgs)
    learned_recall = _mean(ranker_recalls)
    ranker_enabled = (
        learned_ndcg > retrieval_ndcg and learned_recall >= retrieval_recall
    )

    selected_ranker = "xgb_ranker" if ranker_enabled else "retrieval_order"
    selected_ranked = {
        user_id: ranked if ranker_enabled else ordered
        for user_id, (target, ranked, ordered) in val_ranked.items()
    }

    diversity_results: dict[str, dict[str, float | bool]] = {}
    best_lambda = 1.0
    best_ild = -1.0
    relevance_floor = (learned_ndcg if ranker_enabled else retrieval_ndcg) * 0.98
    for diversity_lambda in cfg.diversity_lambda_candidates:
        reranker = DiversityReranker(
            genre_map=genre_map,
            default_lambda=diversity_lambda,
            default_rerank_pool_k=cfg.rerank_pool_k,
        )
        ndcgs: list[float] = []
        ilds: list[float] = []
        for user_id, (target, _, _) in val_ranked.items():
            ranked = selected_ranked[user_id]
            recommendations = reranker.rerank(
                ranked,
                k=cfg.final_k,
                diversity_lambda_override=diversity_lambda,
                rerank_pool_k=cfg.rerank_pool_k,
            )
            ids = [recommendation.item_id for recommendation in recommendations]
            ndcgs.append(ranker_ndcg_at_k(ids, target, cfg.final_k))
            ilds.append(intra_list_diversity(ids, genre_map))
        mean_ndcg = _mean(ndcgs)
        mean_ild = _mean(ilds)
        valid = mean_ndcg >= relevance_floor
        diversity_results[f"lambda_{diversity_lambda:.2f}"] = {
            "ndcg@10": round(mean_ndcg, 4),
            "intra_list_diversity": round(mean_ild, 4),
            "relevance_floor": round(relevance_floor, 4),
            "within_relevance_constraint": valid,
        }
        if valid and mean_ild > best_ild:
            best_lambda = diversity_lambda
            best_ild = mean_ild

    metadata = {
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
        "model_name": "movielens-two-stage-recommender",
        "seed": cfg.seed,
        "embedding_dimension": cfg.embedding_dim,
        "candidate_k": cfg.candidate_k,
        "rerank_pool_k": cfg.rerank_pool_k,
        "final_k": cfg.final_k,
        "svd_candidate_k": cfg.svd_candidate_k,
        "popularity_candidate_k": cfg.popularity_candidate_k,
        "genre_candidate_k": cfg.genre_candidate_k,
        "candidate_contract": cfg.candidate_contract,
        "ranker_model_type": learned_ranker.model_type,
        "ranker_enabled": ranker_enabled,
        "diversity_lambda": best_lambda,
        "feature_schema_version": "rank-features-v1",
        "interaction_contract": {
            "rating_threshold": cfg.rating_threshold,
            "split": "per_user_temporal_holdout",
            "seen_filter": "all_interactions_before_request_timestamp",
            "negative_sampling_proxy": "unobserved_candidates_without_impressions",
        },
        "dataset": {
            "name": data_manifest["dataset_name"],
            "ratings_sha256": data_manifest["ratings_sha256"],
            "movies_sha256": data_manifest["movies_sha256"],
        },
        "dev": {
            "evaluated_users": len(val_ranked),
            "retrieval_order_ndcg@10": retrieval_ndcg,
            "retrieval_order_recall@10": retrieval_recall,
            "xgb_ranker_ndcg@10": learned_ndcg,
            "xgb_ranker_recall@10": learned_recall,
            "selected_ranker": selected_ranker,
            "mmr_lambda": best_lambda,
            "mmr_tuning": diversity_results,
        },
        "rank_training": dataset_builder.last_build_stats,
    }
    save_model(
        cfg.model_dir,
        user_embeddings,
        item_embeddings,
        metadata,
        config_payload,
        # Lưu ranker để unified experiment có thể báo cáo ablation.
        # Serving vẫn chỉ dùng nó khi Dev selection bật ranker_enabled.
        ranker=learned_ranker,
    )
    LOGGER.info(
        "Đã ghi model vào %s; selected_ranker=%s, mmr_lambda=%.2f",
        cfg.model_dir,
        selected_ranker,
        best_lambda,
    )
    return config_payload


def main() -> None:
    train_model()


if __name__ == "__main__":
    main()
