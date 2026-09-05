"""Module huấn luyện mô hình gợi ý 2 tầng (Two-Stage Recommender Training Pipeline).

Quy trình chuẩn:
1. Nạp dữ liệu MovieLens 1M và chia tập theo thời gian (Time-based Leave-Last-Two Positive Split).
2. Xây dựng ma trận tương tác ngầm định (Implicit Positive Feedback: Rating >= 4.0).
3. Phân rã ma trận nhân tử ẩn (Latent Factor Decomposition using TruncatedSVD).
4. Tính toán phân phối phổ biến (Log-transformed Popularity Prior) và hồ sơ thể loại người dùng (User Genre Profiles).
5. Tối ưu hóa đa mục tiêu (Multi-Objective Grid Search) cho alpha (latent_weight) và diversity_lambda trên tập Validation.
6. Lưu trữ Model Artifacts và Config đầy đủ phiên bản.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from .config import TrainConfig
from .data import load_movies, load_ratings, time_split
from .utils import LOGGER, save_json, set_seed, setup_logging


def build_positive_interaction_matrix(
    train_df: Any,
    user_map: dict[int, int],
    item_map: dict[int, int],
    rating_threshold: float = 4.0,
) -> csr_matrix:
    """Tạo ma trận implicit chỉ từ rating dương, không lưu explicit zero."""
    positive = train_df[train_df["rating"] >= rating_threshold]
    rows = positive["user_id"].map(user_map).to_numpy()
    columns = positive["item_id"].map(item_map).to_numpy()
    values = np.ones(len(positive), dtype=np.float32)
    matrix = csr_matrix((values, (rows, columns)), shape=(len(user_map), len(item_map)))
    matrix.eliminate_zeros()
    return matrix


def train_model(
    config: TrainConfig | None = None,
) -> dict[str, Any]:
    """Quy trình chính huấn luyện hệ thống gợi ý 2 tầng.

    Args:
        config (TrainConfig | None): Cấu hình huấn luyện. Nếu None, dùng mặc định.

    Returns:
        dict[str, Any]: Payload cấu hình và siêu tham số tốt nhất.
    """
    cfg = config or TrainConfig()
    setup_logging()
    set_seed(cfg.seed)
    LOGGER.info("Bắt đầu quy trình huấn luyện Canonical Two-Stage Recommender System...")

    # Step 1: Nạp dữ liệu và chia tập Temporal Split
    df_ratings = load_ratings()
    train_df, val_df, _ = time_split(
        df_ratings,
        rating_threshold=cfg.rating_threshold,
        min_positive=cfg.min_positive,
    )

    users = np.sort(train_df.user_id.unique())
    items = np.sort(train_df.item_id.unique())

    user_map = {int(u): i for i, u in enumerate(users)}
    item_map = {int(m): i for i, m in enumerate(items)}

    # Step 2: Xây dựng Sparse User-Item Matrix
    interaction_matrix = build_positive_interaction_matrix(
        train_df, user_map, item_map, cfg.rating_threshold
    )
    LOGGER.info(
        "Kích thước ma trận tương tác: %s với %d tương tác tích cực (>= %.1f)",
        interaction_matrix.shape,
        interaction_matrix.nnz,
        cfg.rating_threshold,
    )

    # Step 3: Latent Factor Decomposition bằng TruncatedSVD
    svd = TruncatedSVD(n_components=cfg.embedding_dim, random_state=cfg.seed)
    user_embeddings = svd.fit_transform(interaction_matrix)
    item_embeddings = svd.components_.T
    LOGGER.info("Đã trích xuất embeddings với số chiều: %d", cfg.embedding_dim)

    # Step 4: Tính toán Popularity Prior (Log1p Transformation) & Seen Filter
    positive_train = train_df[train_df.rating >= cfg.rating_threshold]
    popularity_counts = (
        positive_train.groupby("item_id").size().sort_values(ascending=False)
    )
    popular_items = [int(x) for x in popularity_counts.index.to_list()]

    # Điểm log-popularity chuẩn hóa
    max_count = max(popularity_counts.values, default=1)
    max_log_count = float(np.log1p(max_count)) if max_count > 0 else 1.0
    log_popularity: dict[int, float] = {
        int(item_id): float(np.log1p(cnt) / max_log_count)
        for item_id, cnt in popularity_counts.items()
    }

    # Bảng linear rank cho tương thích ngược
    max_rank_denom = max(1, len(popular_items) - 1)
    popularity_rank: dict[int, float] = {
        int(item_id): 1.0 - (rank / max_rank_denom)
        for rank, item_id in enumerate(popular_items)
    }

    # Seen Filter: Lưu TOÀN BỘ item user đã đánh giá trong train (kể cả rating < threshold)
    seen_by_user: dict[int, set[int]] = (
        train_df.groupby("user_id")["item_id"]
        .apply(lambda x: set(map(int, x)))
        .to_dict()
    )

    # Nạp metadata phim và sinh User Genre Profiles
    df_movies = load_movies()
    genre_map: dict[int, set[str]] = {
        int(row.item_id): set(str(row.genres).split("|"))
        for row in df_movies.itertuples()
    }
    title_map: dict[int, str] = {
        int(row.item_id): str(row.title) for row in df_movies.itertuples()
    }

    # Xây dựng User Genre Profiles từ train set
    user_genre_profiles: dict[int, dict[str, float]] = {}
    train_with_genres = train_df.merge(df_movies[["item_id", "genres"]], on="item_id", how="left")
    for u_id, group in train_with_genres.groupby("user_id"):
        g_counts: dict[str, int] = {}
        total_g = 0
        for g_str in group["genres"].dropna():
            for g in str(g_str).split("|"):
                g_counts[g] = g_counts.get(g, 0) + 1
                total_g += 1
        if total_g > 0:
            user_genre_profiles[int(u_id)] = {
                g: count / total_g for g, count in g_counts.items()
            }

    # Step 5: Chuẩn bị ứng viên trên tập Validation để tuning
    validation_truth = dict(zip(val_df.user_id, val_df.item_id, strict=True))
    eligible_val_users = np.array([u for u in validation_truth if u in user_map])

    rng = np.random.default_rng(cfg.seed)
    if len(eligible_val_users) > cfg.max_val_users:
        sampled_val_users = rng.choice(
            eligible_val_users, size=cfg.max_val_users, replace=False
        )
    else:
        sampled_val_users = eligible_val_users

    user_val_candidates: dict[int, tuple[int, np.ndarray, np.ndarray, np.ndarray]] = {}
    for uid_val in sampled_val_users:
        u_id = int(uid_val)
        true_item = validation_truth[u_id]
        u_idx = user_map[u_id]
        latent_scores = item_embeddings @ user_embeddings[u_idx]

        user_seen = seen_by_user.get(u_id, set())
        if user_seen:
            seen_indices = np.isin(items, list(user_seen))
            latent_scores[seen_indices] = -np.inf

        valid_count = min(cfg.candidate_k, int(np.isfinite(latent_scores).sum()))
        if valid_count <= 0:
            continue

        cand_indices = np.argpartition(-latent_scores, valid_count - 1)[:valid_count]
        raw_sc = latent_scores[cand_indices]
        min_s, max_s = float(raw_sc.min()), float(raw_sc.max())
        norm_sc = (raw_sc - min_s) / (max_s - min_s + 1e-9)
        pop_sc = np.array([log_popularity.get(int(items[j]), 0.0) for j in cand_indices])

        user_val_candidates[u_id] = (true_item, cand_indices, norm_sc, pop_sc)

    # Step 5a: Tối ưu siêu tham số alpha (latent_weight) trên Validation (Relevance Maximization)
    LOGGER.info(
        "Bắt đầu điều chỉnh (tuning) siêu tham số alpha trên Validation..."
    )
    alpha_recalls: dict[str, float] = {}
    for alpha in cfg.alpha_candidates:
        hits: list[float] = []
        for u_id, (true_item, cand_indices, norm_sc, pop_sc) in user_val_candidates.items():
            combined = alpha * norm_sc + (1.0 - alpha) * pop_sc
            top_cand = cand_indices[np.argsort(-combined)[:cfg.top_k]]
            hits.append(float(true_item in items[top_cand]))
        alpha_recalls[f"{alpha:.2f}"] = float(np.mean(hits)) if hits else 0.0

    best_alpha = max(cfg.alpha_candidates, key=lambda a: alpha_recalls[f"{a:.2f}"])
    LOGGER.info(
        "Kết quả tuning alpha trên Validation: %s -> Chọn best_alpha = %.2f",
        alpha_recalls,
        best_alpha,
    )

    # Step 5b: Tối ưu hóa diversity_lambda trên best_alpha (Trade-off Relevance vs Intra-List Diversity)
    LOGGER.info(
        "Bắt đầu tuning diversity_lambda trên Validation với best_alpha = %.2f...",
        best_alpha,
    )
    diversity_tuning_results: dict[str, dict[str, float]] = {}
    best_tradeoff_score = -1.0
    best_lambda = 0.05

    # Lấy top 40 ứng viên để MMR re-rank (tối ưu tốc độ, toán học chính xác vì lambda <= 0.2)
    mmr_pool_size = min(40, cfg.candidate_k)

    for div_lambda in cfg.diversity_lambda_candidates:
        hits = []
        ilds = []

        for u_id, (true_item, cand_indices, norm_sc, pop_sc) in user_val_candidates.items():
            combined = best_alpha * norm_sc + (1.0 - best_alpha) * pop_sc
            sort_order = np.argsort(-combined)
            pool = list(cand_indices[sort_order][:mmr_pool_size])
            score_map = {int(idx): float(sc) for idx, sc in zip(cand_indices, combined)}

            selected_idx: list[int] = []
            while pool and len(selected_idx) < cfg.top_k:
                if div_lambda <= 1e-9:
                    selected_idx.append(pool.pop(0))
                else:
                    best_cand = max(
                        pool,
                        key=lambda j: (
                            score_map[int(j)]
                            - div_lambda
                            * max(
                                (
                                    len(genre_map.get(int(items[j]), set()) & genre_map.get(int(items[s]), set()))
                                    / max(1, len(genre_map.get(int(items[j]), set()) | genre_map.get(int(items[s]), set())))
                                    for s in selected_idx
                                ),
                                default=0.0,
                            )
                        ),
                    )
                    selected_idx.append(best_cand)
                    pool.remove(best_cand)

            hit = float(true_item in items[selected_idx])
            hits.append(hit)

            rec_items = [int(items[j]) for j in selected_idx]
            if len(rec_items) > 1:
                dists: list[float] = []
                for i_idx, it1 in enumerate(rec_items):
                    for it2 in rec_items[i_idx + 1 :]:
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

    LOGGER.info(
        "Kết quả tuning diversity_lambda: %s -> Chọn best_diversity_lambda = %.2f (Score: %.4f)",
        diversity_tuning_results,
        best_lambda,
        best_tradeoff_score,
    )

    # Step 6: Lưu trữ Artifacts
    artifact_path = Path(cfg.model_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)

    np.save(artifact_path / "user_emb.npy", user_embeddings)
    np.save(artifact_path / "item_emb.npy", item_embeddings)

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
    }
    joblib.dump(metadata_payload, artifact_path / "meta.joblib")

    config_payload = {
        "schema_version": 3,
        "version": "movielens-svd-two-stage-v3",
        "seed": cfg.seed,
        "embedding_dimension": cfg.embedding_dim,
        "candidate_k": cfg.candidate_k,
        "top_k": cfg.top_k,
        "latent_weight": float(best_alpha),
        "diversity_lambda": float(best_lambda),
        "validation_tuning_grid": {
            "alpha_tuning": alpha_recalls,
            "diversity_tuning": diversity_tuning_results,
        },
        "interaction_contract": {
            "feedback": "implicit_positive",
            "rating_threshold": cfg.rating_threshold,
            "explicit_zeros_stored": False,
            "split": "leave_last_two_positive_per_user",
            "seen_filter": "all_train_interactions",
        },
        "positive_interactions": int(interaction_matrix.nnz),
    }
    save_json(artifact_path / "config.json", config_payload)

    LOGGER.info(
        "Đã huấn luyện và lưu trữ artifacts thành công tại '%s'!", artifact_path
    )
    return config_payload


def main() -> None:
    """Hàm main thực thi script huấn luyện khi gọi từ CLI."""
    train_model()


if __name__ == "__main__":
    main()
