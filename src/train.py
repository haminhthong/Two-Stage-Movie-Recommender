"""Module huấn luyện mô hình gợi ý 2 tầng (Two-Stage Recommender).

Sử dụng phương pháp TruncatedSVD (Matrix Factorization) để trích xuất User/Item Latent Embeddings,
tính toán Popularity Rank và tìm kiếm trọng số cân bằng (alpha) tối ưu trên tập Validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from .data import load_movies, load_ratings, time_split
from .utils import LOGGER, save_json, set_seed, setup_logging

SEED: int = 42
EMBEDDING_DIMENSION: int = 64
CANDIDATE_K: int = 200
TOP_K: int = 10
RATING_THRESHOLD: float = 4.0


def build_positive_interaction_matrix(
    train_df: Any,
    user_map: dict[int, int],
    item_map: dict[int, int],
    rating_threshold: float = RATING_THRESHOLD,
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
    model_dir: str | Path = "models",
    embedding_dim: int = EMBEDDING_DIMENSION,
    seed: int = SEED,
) -> dict[str, Any]:
    """Quy trình chính huấn luyện hệ thống gợi ý 2 tầng.

    Các bước thực hiện:
    1. Thiết lập random seed và nạp dữ liệu MovieLens 1M.
    2. Thực hiện time-based split (Train/Val/Test).
    3. Xây dựng Ma trận thưa User-Item (Implicit feedback: Rating >= 4.0).
    4. Phân rã ma trận bằng TruncatedSVD để thu về User Embeddings và Item Embeddings.
    5. Tính toán chỉ số phổ biến (Popularity Score / Rank) của từng sản phẩm.
    6. Tối ưu hóa siêu tham số latent_weight (alpha) trên tập Validation (không dùng Test set).
    7. Trích xuất thông tin phim (Title, Genres) và lưu trữ Artifacts vào ổ đĩa.

    Args:
        model_dir (str | Path): Thư mục lưu trữ artifacts mô hình. (Mặc định: 'models')
        embedding_dim (int): Số chiều không gian nhúng ẩn. (Mặc định: 64)
        seed (int): Random seed cố định. (Mặc định: 42)

    Returns:
        dict[str, Any]: Cấu hình và kết quả huấn luyện mô hình.
    """
    setup_logging()
    set_seed(seed)
    LOGGER.info("Bắt đầu quy trình huấn luyện Two-Stage Recommender System...")

    # Step 1: Nạp dữ liệu và chia tập
    df_ratings = load_ratings()
    train_df, val_df, _ = time_split(df_ratings)

    users = np.sort(train_df.user_id.unique())
    items = np.sort(train_df.item_id.unique())

    user_map = {u: i for i, u in enumerate(users)}
    item_map = {m: i for i, m in enumerate(items)}

    # Step 2: Xây dựng Sparse User-Item Matrix (Implicit Feedback)
    interaction_matrix = build_positive_interaction_matrix(
        train_df, user_map, item_map, RATING_THRESHOLD
    )
    LOGGER.info(
        "Kích thước ma trận tương tác User-Item: %s với %d lượt thích (>= %.1f)",
        interaction_matrix.shape,
        interaction_matrix.nnz,
        RATING_THRESHOLD,
    )

    # Step 3: Phân rã TruncatedSVD (Latent Vector Extraction)
    svd = TruncatedSVD(n_components=embedding_dim, random_state=seed)
    user_embeddings = svd.fit_transform(interaction_matrix)
    item_embeddings = svd.components_.T
    LOGGER.info("Đã trích xuất embeddings với số chiều: %d", embedding_dim)

    # Step 4: Tính toán Popularity Rank
    positive_train = train_df[train_df.rating >= RATING_THRESHOLD]
    popularity_counts = (
        positive_train.groupby("item_id").size().sort_values(ascending=False)
    )
    popular_items = popularity_counts.index.to_list()

    max_rank_denom = max(1, len(popular_items) - 1)
    popularity_rank = {
        int(item_id): 1.0 - (rank / max_rank_denom)
        for rank, item_id in enumerate(popular_items)
    }

    # Tập hợp các sản phẩm user đã tương tác trong train để lọc bớt lúc gợi ý
    seen_by_user = (
        train_df.groupby("user_id")["item_id"]
        .apply(lambda x: set(map(int, x)))
        .to_dict()
    )

    # Step 5: Tối ưu siêu tham số alpha (latent_weight) trên tập Validation
    LOGGER.info(
        "Bắt đầu điều chỉnh (tuning) siêu tham số latent_weight trên Validation..."
    )
    alpha_candidates = [0.8, 0.9, 1.0]
    hits: dict[float, list[float]] = {alpha: [] for alpha in alpha_candidates}
    validation_truth = dict(zip(val_df.user_id, val_df.item_id, strict=True))

    for idx, (user_id, true_item) in enumerate(validation_truth.items()):
        # Giới hạn 1000 user trong tập val để tăng tốc độ tuning
        if idx >= 1000 or user_id not in user_map:
            continue

        u_idx = user_map[user_id]
        latent_scores = item_embeddings @ user_embeddings[u_idx]

        # Đánh dấu -inf cho các item người dùng đã xem trong Train set
        user_seen = seen_by_user.get(user_id, set())
        if user_seen:
            seen_indices = np.isin(items, list(user_seen))
            latent_scores[seen_indices] = -np.inf

        valid_count = min(CANDIDATE_K, int(np.isfinite(latent_scores).sum()))
        if valid_count <= 0:
            continue

        top_candidate_indices = np.argpartition(-latent_scores, valid_count - 1)[
            :valid_count
        ]
        raw_scores = latent_scores[top_candidate_indices]
        min_s, max_s = float(raw_scores.min()), float(raw_scores.max())
        norm_scores = (raw_scores - min_s) / (max_s - min_s + 1e-9)

        pop_scores = np.array(
            [popularity_rank.get(int(items[j]), 0.0) for j in top_candidate_indices]
        )

        for alpha in alpha_candidates:
            combined = alpha * norm_scores + (1.0 - alpha) * pop_scores
            top_reranked = top_candidate_indices[np.argsort(-combined)[:TOP_K]]
            hit = float(true_item in items[top_reranked])
            hits[alpha].append(hit)

    validation_recall = {
        str(alpha): float(np.mean(hits[alpha])) if hits[alpha] else 0.0
        for alpha in alpha_candidates
    }

    best_alpha = max(alpha_candidates, key=lambda a: validation_recall[str(a)])
    LOGGER.info(
        "Kết quả tuning alpha trên Validation: %s -> Chọn best_alpha = %.2f",
        validation_recall,
        best_alpha,
    )

    # Step 6: Đọc thông tin phim và chuẩn bị Metadata
    df_movies = load_movies()
    genre_map = {
        int(row.item_id): set(str(row.genres).split("|"))
        for row in df_movies.itertuples()
    }
    title_map = {int(row.item_id): str(row.title) for row in df_movies.itertuples()}

    # Step 7: Lưu trữ Artifacts
    artifact_path = Path(model_dir)
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
        "seen": seen_by_user,
        "genres": genre_map,
        "titles": title_map,
    }
    joblib.dump(metadata_payload, artifact_path / "meta.joblib")

    config_payload = {
        "schema_version": 2,
        "version": "movielens-svd-two-stage-v3",
        "seed": seed,
        "embedding_dimension": embedding_dim,
        "candidate_k": CANDIDATE_K,
        "top_k": TOP_K,
        "latent_weight": float(best_alpha),
        "diversity_lambda": 0.05,
        "validation_recall@10": validation_recall,
        "interaction_contract": {
            "feedback": "implicit_positive",
            "rating_threshold": RATING_THRESHOLD,
            "explicit_zeros_stored": False,
            "split": "leave_last_two_per_user",
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
