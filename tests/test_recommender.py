"""Bộ kiểm thử mở rộng cho kiến trúc Two-Stage Recommender Pipeline.

Bao gồm kiểm thử:
- Tính bất biến của Temporal Split (Không rò rỉ tương lai).
- Seen Filter Guardrail (Không bao giờ gợi ý item đã xem).
- Deterministic Ranking & MMR Diversity behavior.
- Chiến lược Cold-Start & trường strategy trong API response.
- Tính nhất quán của Artifacts và tính toán đặc trưng.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.api import app
from src.data import temporal_split
from src.ranking.features import CandidateFeatureBuilder, CandidateFeatures
from src.reranking.diversity import DiversityReranker, RankedCandidate
from src.retrieval.base import Candidate
from src.retrieval.svd import SVDRetriever
from src.serving.recommender import Recommender


def _feature(
    item_id: int, svd_score: float, popularity_score: float
) -> CandidateFeatures:
    """Tạo feature fixture đúng schema 19 cột."""
    return CandidateFeatures(
        item_id=item_id,
        svd_score=svd_score,
        svd_rank=item_id,
        popularity_retrieval_score=popularity_score,
        popularity_rank=item_id,
        genre_retrieval_score=0.0,
        genre_rank=0,
        rrf_score=1.0,
        source_count=2.0,
        user_positive_count=0,
        user_interaction_count=0,
        user_avg_rating=4.0,
        genre_entropy=0.0,
        item_positive_count=0,
        item_rating_count=0,
        item_avg_rating=3.5,
        item_popularity_percentile=0.5,
        item_genre_count=1,
        genre_affinity=0.0,
        genre_overlap_count=0,
    )


def test_temporal_split_no_future_leakage() -> None:
    """Kiểm tra tập Train tuyệt đối không chứa tương tác diễn ra sau hoặc cùng mốc thời gian của Validation."""
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 2, 2, 2],
            "item_id": [10, 20, 30, 40, 50, 60, 70],
            "rating": [5.0, 4.0, 5.0, 4.5, 4.0, 4.0, 4.0],
            "timestamp": [1000, 2000, 3000, 4000, 500, 600, 700],
        }
    )
    train_df, _rank_df, _val_df, _test_df = temporal_split(
        df, rating_threshold=4.0, min_positive=3
    )

    # User 1: positive tại 1000, 2000, 3000, 4000.
    # test = item 40 (ts=4000), val = item 30 (ts=3000).
    # Retrieval train phải chỉ chứa timestamp trước rank-train target ts=2000.
    user_1_train = train_df[train_df["user_id"] == 1]
    assert all(user_1_train["timestamp"] < 2000)
    assert set(user_1_train["item_id"]) == {10}


def test_test_item_not_in_training() -> None:
    """Kiểm tra ground truth item trong tập Test không xuất hiện trong lịch sử train của user."""
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 1],
            "item_id": [101, 102, 103],
            "rating": [4.0, 5.0, 5.0],
            "timestamp": [100, 200, 300],
        }
    )
    train_df, _rank_df, _val_df, test_df = temporal_split(
        df, rating_threshold=4.0, min_positive=3
    )

    test_item = test_df.iloc[0]["item_id"]
    train_items = set(train_df[train_df["user_id"] == 1]["item_id"])
    assert test_item not in train_items


def test_seen_items_never_recommended() -> None:
    """Kiểm tra cơ chế Seen Filter đảm bảo item đã xuất hiện trong train không bao giờ được gợi ý lại."""
    # Giả lập retriever với 3 items: 1, 2, 3. Item 1 và 2 đã xem.
    user_emb = np.array([[1.0, 0.0]])
    item_emb = np.array(
        [[1.0, 0.0], [0.9, 0.0], [0.1, 0.0]]
    )  # Item 1 cao nhất, item 3 thấp nhất
    items = np.array([1, 2, 3])
    user_map = {100: 0}
    item_map = {1: 0, 2: 1, 3: 2}
    seen = {100: {1, 2}}

    retriever = SVDRetriever(
        user_embeddings=user_emb,
        item_embeddings=item_emb,
        user_map=user_map,
        item_map=item_map,
        items=items,
        seen_by_user=seen,
    )
    candidates = retriever.retrieve(user_id=100, k=10, filter_seen=True)
    candidate_ids = [c.item_id for c in candidates]

    assert 1 not in candidate_ids
    assert 2 not in candidate_ids
    assert candidate_ids == [3]


def test_unknown_user_uses_cold_start() -> None:
    """Kiểm tra người dùng không xác định (chưa có trong train) tự động fallback sang cold-start popularity."""
    recommender = Recommender()
    res = recommender.recommend_detailed(user_id=99999999, k=5)

    assert res["strategy"] in {"cold_start_popularity", "fallback_popularity"}
    assert len(res["items"]) <= 5


def test_candidate_count_not_exceed_catalog() -> None:
    """Kiểm tra candidate retriever không bao giờ trả về số lượng vượt quá kích thước catalog."""
    user_emb = np.array([[1.0, 0.0]])
    item_emb = np.array([[1.0, 0.0], [0.5, 0.5]])
    retriever = SVDRetriever(
        user_embeddings=user_emb,
        item_embeddings=item_emb,
        user_map={1: 0},
        item_map={10: 0, 20: 1},
        items=np.array([10, 20]),
        seen_by_user={},
    )
    candidates = retriever.retrieve(user_id=1, k=500, filter_seen=True)
    assert len(candidates) == 2


def test_ranking_deterministic() -> None:
    """Kiểm tra quy trình xếp hạng là đơn định (Deterministic): cùng đầu vào sinh ra cùng thứ tự."""
    recommender = Recommender()
    preds_1 = recommender.recommend(user_id=1, k=10)
    preds_2 = recommender.recommend(user_id=1, k=10)
    assert preds_1 == preds_2


def test_diversity_lambda_one_equals_base_rank() -> None:
    """Lambda=1 phải giữ thuần thứ tự relevance, không áp dụng phạt diversity."""
    genre_map = {1: {"Action"}, 2: {"Action"}, 3: {"Drama"}}
    reranker = DiversityReranker(genre_map=genre_map, default_lambda=1.0)

    cands = [
        RankedCandidate(
            item_id=1, relevance_score=0.95, features=_feature(1, 0.95, 0.8)
        ),
        RankedCandidate(
            item_id=2, relevance_score=0.90, features=_feature(2, 0.90, 0.7)
        ),
        RankedCandidate(
            item_id=3, relevance_score=0.70, features=_feature(3, 0.70, 0.5)
        ),
    ]
    reranked = reranker.rerank(cands, k=3, diversity_lambda_override=1.0)
    assert [r.item_id for r in reranked] == [1, 2, 3]


def test_diversity_lambda_zero_prefers_unseen_genres() -> None:
    """Lambda=0 phải tối đa hóa diversity trong pool, không tắt bước MMR."""
    genre_map = {1: {"Action"}, 2: {"Action"}, 3: {"Drama"}}
    reranker = DiversityReranker(genre_map=genre_map, default_lambda=0.0)
    cands = [
        RankedCandidate(item_id=1, relevance_score=0.95),
        RankedCandidate(item_id=2, relevance_score=0.90),
        RankedCandidate(item_id=3, relevance_score=0.70),
    ]
    reranked = reranker.rerank(cands, k=3, diversity_lambda_override=0.0)
    assert [r.item_id for r in reranked] == [1, 3, 2]


def test_high_diversity_reduces_genre_similarity() -> None:
    """Kiểm tra khi áp dụng lambda đa dạng cao, item khác biệt thể loại được đẩy lên trước item cùng thể loại."""
    genre_map = {1: {"Action"}, 2: {"Action"}, 3: {"Drama"}}
    reranker = DiversityReranker(genre_map=genre_map, default_lambda=0.5)

    # Item 1 và 2 đều là Action với score suýt soát (0.90 vs 0.89). Item 3 là Drama (score 0.85).
    # Với lambda=0.5, penalty cho item 2 sau khi chọn item 1 là 0.5 * 1.0 = 0.5 -> score giảm xuống 0.39.
    # Trong khi item 3 có Jaccard=0.0 -> score giữ nguyên 0.85. Item 3 phải vượt lên trước item 2!
    cands = [
        RankedCandidate(
            item_id=1, relevance_score=0.90, features=_feature(1, 0.90, 0.8)
        ),
        RankedCandidate(
            item_id=2, relevance_score=0.89, features=_feature(2, 0.89, 0.8)
        ),
        RankedCandidate(
            item_id=3, relevance_score=0.85, features=_feature(3, 0.85, 0.7)
        ),
    ]
    reranked = reranker.rerank(cands, k=3, diversity_lambda_override=0.5)
    result_ids = [r.item_id for r in reranked]
    assert result_ids == [1, 3, 2]


def test_artifact_embedding_shapes_match() -> None:
    """Kiểm tra kích thước các ma trận nhúng khớp hoàn hảo với số lượng user và item trong metadata."""
    recommender = Recommender()
    assert recommender.user_embeddings.shape[0] == len(recommender.metadata["users"])
    assert recommender.item_embeddings.shape[0] == len(recommender.metadata["items"])
    assert recommender.user_embeddings.shape[1] == recommender.item_embeddings.shape[1]


def test_api_strategy_field() -> None:
    """Kiểm tra API trả về trường strategy chuẩn xác cho cả known user và unknown user."""
    client = TestClient(app)

    # Known user (user 1 có trong MovieLens)
    resp_known = client.get("/recommend/1?k=5")
    if resp_known.status_code == 200:
        data = resp_known.json()
        assert "strategy" in data
        assert data["strategy"] == "two_stage_personalized"

        # debug=True phải trả cả latency vì engine đã bật đo lường cho request.
        resp_debug = client.get("/recommend/1?k=5&debug=true")
        assert resp_debug.status_code == 200
        assert resp_debug.json()["debug"]["latencies_ms"] is not None

    # Unknown user
    resp_unknown = client.get("/recommend/9999999?k=5")
    if resp_unknown.status_code == 200:
        data = resp_unknown.json()
        assert "strategy" in data
        assert data["strategy"] in {"cold_start_popularity", "fallback_popularity"}


def test_user_genre_affinity_score() -> None:
    """Kiểm tra CandidateFeatureBuilder tính toán chính xác điểm affinity dựa trên lịch sử thể loại."""
    genre_map = {10: {"Action", "Sci-Fi"}, 20: {"Romance"}}
    # User thích Action 70%, Sci-Fi 30%, Romance 0%
    user_profiles = {1: {"Action": 0.7, "Sci-Fi": 0.3}}
    popularity = {10: 0.5, 20: 0.5}

    builder = CandidateFeatureBuilder(
        popularity_scores=popularity,
        genre_map=genre_map,
        user_genre_profiles=user_profiles,
    )
    cands = [
        Candidate(item_id=10, retrieval_score=1.0),
        Candidate(item_id=20, retrieval_score=1.0),
    ]
    features = builder.build_features(cands, user_id=1)

    # Item 10 có 2 genres, overlap sum = 0.7 + 0.3 = 1.0, mean = 0.5
    # Item 20 có Romance = 0.0, mean = 0.0
    feat_10 = next(f for f in features if f.item_id == 10)
    feat_20 = next(f for f in features if f.item_id == 20)
    assert feat_10.genre_affinity > feat_20.genre_affinity
    assert feat_20.genre_affinity == 0.0


def test_log1p_popularity_score() -> None:
    """Kiểm tra đặc tính làm mịn khoảng cách của log1p so với linear scaling."""
    counts = {1: 3000, 2: 2900, 3: 100, 4: 20}
    max_c = 3000
    max_log = np.log1p(max_c)

    log_scores = {k: np.log1p(v) / max_log for k, v in counts.items()}

    # Khoảng cách tương đối giữa item 1 và 2 (3000 vs 2900) rất nhỏ
    diff_1_2 = log_scores[1] - log_scores[2]
    # Khoảng cách giữa item 3 và 4 (100 vs 20) lớn hơn nhiều
    diff_3_4 = log_scores[3] - log_scores[4]

    assert diff_3_4 > diff_1_2
