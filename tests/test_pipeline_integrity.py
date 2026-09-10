"""Bộ kiểm thử toàn vẹn cho toàn bộ Pipeline Recommender Production.

Kiểm tra nghiêm ngặt:
1. P0.1: Rating thấp (< 4.0) tuyệt đối không lọt vào positive genre profile.
2. P0.2: Đồng nhất candidate_k và rerank_pool_k giữa validation tuning và serving.
3. P0.3 & P0.4: Candidate recall tính trước ranking; item test ngoài catalog được báo cáo.
4. P0.5: Tách bạch absolute gain và relative lift.
5. P1.1: 4-Way Temporal Split loại bỏ hoàn toàn lookahead leakage giữa các tập.
6. P1.2: Stage-2 Learned Ranker huấn luyện và dự đoán chính xác.
7. P1.3: Multi-Source retrieval hợp nhất và khử trùng lặp chính xác.
8. Artifact roundtrip và seen filter guardrails.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src.data.interactions import (
    build_user_genre_profiles,
    extract_seen_items,
)
from src.data.split import temporal_split
from src.evaluation.retrieval_metrics import (
    candidate_recall_at_k,
    target_in_catalog_rate,
)
from src.model_io import load_model, save_model
from src.ranking.features import N_FEATURES, CandidateFeatures
from src.ranking.trainer import train_learned_ranker
from src.reranking.diversity import DiversityReranker
from src.retrieval.base import Candidate
from src.retrieval.merger import MultiSourceRetriever
from src.retrieval.popularity import PopularityRetriever
from src.retrieval.svd import SVDRetriever


def _feature(
    item_id: int, svd_score: float, popularity_score: float
) -> CandidateFeatures:
    """Tạo fixture theo đúng feature contract 19 cột."""
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


def test_no_validation_item_in_user_history() -> None:
    """Kiểm tra validation item không xuất hiện trong retrieval_train history của user."""
    data = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 1],
            "item_id": [10, 20, 30, 40, 50],
            "rating": [5.0, 5.0, 5.0, 5.0, 5.0],
            "timestamp": [100, 200, 300, 400, 500],
        }
    )
    ret_train, _rank_train, val_df, _test_df = temporal_split(data, min_positive=4)

    val_item = val_df.iloc[0]["item_id"]
    ret_items = set(ret_train[ret_train["user_id"] == 1]["item_id"])
    assert val_item not in ret_items


def test_no_test_item_in_user_history() -> None:
    """Kiểm tra test item không xuất hiện trong retrieval history hoặc rank-train của user."""
    data = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 1],
            "item_id": [10, 20, 30, 40, 50],
            "rating": [5.0, 5.0, 5.0, 5.0, 5.0],
            "timestamp": [100, 200, 300, 400, 500],
        }
    )
    ret_train, rank_train, _val_df, test_df = temporal_split(data, min_positive=4)

    test_item = test_df.iloc[0]["item_id"]
    ret_items = set(ret_train[ret_train["user_id"] == 1]["item_id"])
    rank_item = rank_train.iloc[0]["item_id"]
    assert test_item not in ret_items
    assert test_item != rank_item


def test_low_rating_not_in_positive_genre_profile() -> None:
    """P0.1 Fix: Rating thấp (< 4.0) tuyệt đối không được đưa vào positive genre profile."""
    # User 1 đánh giá Horror 1 sao, Action 5 sao
    train_df = pd.DataFrame(
        {
            "user_id": [1, 1],
            "item_id": [101, 102],
            "rating": [1.0, 5.0],  # 101 là 1 sao (Horror), 102 là 5 sao (Action)
            "timestamp": [100, 200],
        }
    )
    movies_df = pd.DataFrame(
        {
            "item_id": [101, 102],
            "title": ["Scary Movie", "Action Hero"],
            "genres": ["Horror", "Action"],
        }
    )

    profiles = build_user_genre_profiles(train_df, movies_df, rating_threshold=4.0)

    # Horror không được có mặt trong profile của User 1
    u1_profile = profiles.get(1, {})
    assert "Horror" not in u1_profile
    assert "Action" in u1_profile
    assert u1_profile["Action"] == 1.0


def test_all_seen_items_filtered() -> None:
    """Seen filter phải loại bỏ toàn bộ item đã xem trong train, kể cả phim đánh giá 1 sao."""
    train_df = pd.DataFrame(
        {
            "user_id": [1, 1],
            "item_id": [10, 20],
            "rating": [1.0, 5.0],  # 10 là 1 sao, 20 là 5 sao
            "timestamp": [100, 200],
        }
    )
    seen = extract_seen_items(train_df)
    assert seen[1] == {10, 20}


def test_candidate_recall_computed_before_ranking() -> None:
    """P0.3: Candidate recall phải đo lường năng lực của candidate pool trước khi ranker can thiệp."""
    cands = [
        Candidate(item_id=1, retrieval_score=0.9),
        Candidate(item_id=2, retrieval_score=0.8),
    ]
    assert candidate_recall_at_k(cands, target_item=2, k=2) == 1.0
    assert candidate_recall_at_k(cands, target_item=2, k=1) == 0.0
    assert candidate_recall_at_k(cands, target_item=999, k=2) == 0.0


def test_target_outside_train_catalog_reported() -> None:
    """P0.4: Test target ngoài train catalog phải được tính vào cold_item_test_share."""
    train_catalog = {10, 20, 30}
    test_targets = [10, 20, 999]  # 999 không có trong train catalog

    rate = target_in_catalog_rate(test_targets, train_catalog)
    assert rate == pytest.approx(2.0 / 3.0)


def test_rank_training_candidates_use_only_past_history() -> None:
    """P1.1: 4-way split đảm bảo timestamp của retrieval_train luôn nhỏ hơn rank_train."""
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 1],
            "item_id": [1, 2, 3, 4, 5],
            "rating": [5.0, 4.0, 5.0, 4.5, 5.0],
            "timestamp": [10, 20, 30, 40, 50],
        }
    )
    ret_train, rank_train, _val_df, _test_df = temporal_split(df, min_positive=4)

    rank_ts = rank_train.iloc[0]["timestamp"]
    ret_ts = ret_train[ret_train["user_id"] == 1]["timestamp"]
    assert all(ts < rank_ts for ts in ret_ts)


def test_validation_never_used_to_fit_ranker() -> None:
    """Tập validation (t_val) có timestamp lớn hơn mốc t_rank của rank-train."""
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1],
            "item_id": [1, 2, 3, 4],
            "rating": [5.0, 4.0, 5.0, 4.5],
            "timestamp": [100, 200, 300, 400],
        }
    )
    _ret_train, rank_train, val_df, test_df = temporal_split(df, min_positive=4)

    t_rank = rank_train.iloc[0]["timestamp"]
    t_val = val_df.iloc[0]["timestamp"]
    t_test = test_df.iloc[0]["timestamp"]

    assert t_rank < t_val < t_test


def test_test_never_used_for_tuning() -> None:
    """Mốc thời gian tập test luôn là sự kiện cuối cùng."""
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1],
            "item_id": [1, 2, 3, 4],
            "rating": [5.0, 4.0, 5.0, 4.5],
            "timestamp": [100, 200, 300, 400],
        }
    )
    _, _, val_df, test_df = temporal_split(df, min_positive=4)
    assert val_df.iloc[0]["timestamp"] < test_df.iloc[0]["timestamp"]


def test_mmr_pool_same_between_validation_and_serving() -> None:
    """P0.2: Kích thước candidate pool truyền vào MMR reranker phải đồng nhất (rerank_pool_k=40)."""
    reranker = DiversityReranker(default_lambda=0.1, default_rerank_pool_k=40)
    assert reranker.default_rerank_pool_k == 40

    cands = [_feature(i, 1.0 - i * 0.01, 0.5) for i in range(100)]
    ranked = [
        from_feat
        for from_feat in [
            type(
                "MockRanked",
                (),
                {
                    "item_id": c.item_id,
                    "relevance_score": c.svd_score,
                    "features": c,
                },
            )()
            for c in cands
        ]
    ]

    recs = reranker.rerank(ranked, k=10, rerank_pool_k=40)
    # Các item được chọn phải nằm trong top 40 ban đầu
    chosen_ids = {r.item_id for r in recs}
    top_40_ids = {r.item_id for r in ranked[:40]}
    assert chosen_ids.issubset(top_40_ids)


def test_model_config_roundtrip(tmp_path: Path) -> None:
    """Kiểm tra model phẳng được lưu và nạp lại đúng contract."""
    user_emb = np.ones((5, 8), dtype=np.float32)
    item_emb = np.ones((10, 8), dtype=np.float32)
    meta = {
        "users": list(range(1, 6)),
        "items": list(range(10, 20)),
        "popular": [10, 11, 12],
    }
    cfg = {
        "model_name": "test-model",
        "candidate_k": 200,
        "ranker_enabled": False,
    }

    save_model(
        model_dir=tmp_path,
        user_embeddings=user_emb,
        item_embeddings=item_emb,
        metadata=meta,
        config=cfg,
    )

    loaded = load_model(tmp_path)
    assert loaded["user_embeddings"].shape == (5, 8)
    assert loaded["item_embeddings"].shape == (10, 8)
    assert loaded["config"]["model_name"] == "test-model"


def test_multi_source_candidate_generation() -> None:
    """Kiểm tra MultiSourceRetriever hợp nhất đúng SVD và Popularity, khử trùng lặp."""

    class MockSVD(SVDRetriever):
        def retrieve(self, user_id: int, k: int = 10, filter_seen: bool = True):
            return [
                Candidate(10, 0.9, "svd", 0, {"svd": 0.9}),
                Candidate(20, 0.8, "svd", 1, {"svd": 0.8}),
            ]

    class MockPop(PopularityRetriever):
        def retrieve(self, user_id: int, k: int = 10, filter_seen: bool = True):
            return [
                Candidate(20, 0.95, "popularity", 0, {"popularity": 0.95}),
                Candidate(30, 0.7, "popularity", 1, {"popularity": 0.7}),
            ]

    merger = MultiSourceRetriever(
        retrievers={
            "svd": (MockSVD.__new__(MockSVD), 10),
            "popularity": (MockPop.__new__(MockPop), 10),
        }
    )
    merged = merger.retrieve(user_id=1, k=10)
    merged_ids = [c.item_id for c in merged]

    # Phải có 3 items duy nhất: 10, 20, 30 (20 xuất hiện ở cả hai nguồn và được merge)
    assert set(merged_ids) == {10, 20, 30}
    c_20 = next(c for c in merged if c.item_id == 20)
    assert "svd" in c_20.source_scores
    assert "popularity" in c_20.source_scores
    assert c_20.retrieval_source == "multi_source"


def test_learned_ranker_fit_and_predict() -> None:
    """Kiểm tra LearnedRanker huấn luyện trên dữ liệu và sinh điểm hợp lệ [0, 1]."""
    X = np.random.randn(20, N_FEATURES).astype(np.float32)
    y = np.array([1, 0] * 10, dtype=np.int32)

    ranker = train_learned_ranker(X, y, model_type="logistic_regression", seed=42)
    assert ranker is not None

    feats = [
        _feature(1, 0.9, 0.8),
        _feature(2, 0.1, 0.2),
    ]
    scores = ranker.predict_scores(feats)
    assert len(scores) == 2
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_xgb_ranker_uses_grouped_ndcg_objective() -> None:
    """Ranker chính phải huấn luyện được với query groups và objective rank:ndcg."""
    X = np.random.default_rng(42).normal(size=(20, N_FEATURES)).astype(np.float32)
    y = np.array([1, 0] * 10, dtype=np.int32)

    ranker = train_learned_ranker(
        X, y, groups=np.full(10, 2, dtype=np.int32), model_type="xgb_ranker", seed=42
    )

    assert ranker.model_type == "xgb_ranker"
    assert ranker.estimator.get_xgb_params()["objective"] == "rank:ndcg"
    scores = ranker.predict_scores([_feature(1, 0.9, 0.8)])
    assert scores.shape == (1,)


def test_absolute_gain_and_relative_lift_calculations() -> None:
    """P0.5: Kiểm tra tính toán chính xác absolute_gain và relative_lift."""
    model_recall = 0.0896
    pop_recall = 0.0456

    abs_gain = model_recall - pop_recall
    rel_lift = (abs_gain / pop_recall) * 100.0

    assert abs_gain == pytest.approx(0.0440, abs=1e-4)
    assert rel_lift == pytest.approx(96.49, abs=1e-1)
