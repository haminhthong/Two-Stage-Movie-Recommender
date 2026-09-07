"""Regression tests cho các lỗi logic P0 của candidate contract mới."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ranking.dataset import RankDatasetBuilder
from src.ranking.features import CandidateFeatureBuilder, N_FEATURES
from src.retrieval.base import Candidate, CandidateRetriever
from src.retrieval.merger import MultiSourceRetriever
from src.retrieval.svd import SVDRetriever


class _StaticRetriever(CandidateRetriever):
    """Retriever giả lập để kiểm tra dataset không inject target."""

    def retrieve(
        self,
        user_id: int,
        k: int = 200,
        filter_seen: bool = True,
        seen_items_override=None,
    ) -> list[Candidate]:
        return [Candidate(item_id=1, svd_score=0.8, svd_rank=1)]


def test_rank_dataset_skips_stage1_miss_without_target_injection() -> None:
    """Target không nằm trong pool thì query bị loại, không sinh nhãn dương giả."""
    builder = CandidateFeatureBuilder(popularity_scores={1: 1.0})
    dataset = RankDatasetBuilder(builder, candidate_k=200)
    rank_train = pd.DataFrame({"user_id": [7], "item_id": [99]})

    features, labels, groups = dataset.build_dataset(rank_train, _StaticRetriever())

    assert features.shape == (0, N_FEATURES)
    assert labels.size == 0
    assert groups.size == 0
    assert dataset.last_build_stats["target_retrieval_rate"] == 0.0


def test_multi_source_uses_rrf_and_preserves_source_signals() -> None:
    """RRF chỉ dùng rank, còn score từng source phải được giữ riêng."""

    class SourceRetriever(CandidateRetriever):
        def __init__(self, source: str, ids: list[int]) -> None:
            self.source = source
            self.ids = ids

        def retrieve(self, user_id, k=200, filter_seen=True, seen_items_override=None):
            return [
                Candidate(
                    item_id=item_id,
                    retrieval_score=1.0 / (index + 1),
                    retrieval_source=self.source,
                    retrieval_rank=index,
                    source_scores={self.source: 1.0 / (index + 1)},
                )
                for index, item_id in enumerate(self.ids[:k])
            ]

    retriever = MultiSourceRetriever(
        {
            "svd": (SourceRetriever("svd", [10, 20]), 2),
            "genre": (SourceRetriever("genre", [20, 30]), 2),
        },
        rrf_k=60,
    )
    candidates = retriever.retrieve(user_id=1, k=3)

    assert [candidate.item_id for candidate in candidates] == [20, 10, 30]
    merged = candidates[0]
    assert merged.source_count == 2
    assert merged.svd_score == 0.5
    assert merged.genre_score == 1.0
    assert merged.rrf_score == pytest.approx(1 / 62 + 1 / 61)


def test_seen_override_is_request_local() -> None:
    """Override recent session không được mutate seen state dùng chung."""
    retriever = SVDRetriever(
        user_embeddings=np.array([[1.0, 0.0]]),
        item_embeddings=np.array([[1.0, 0.0], [0.8, 0.0], [0.1, 0.0]]),
        user_map={1: 0},
        item_map={10: 0, 20: 1, 30: 2},
        items=np.array([10, 20, 30]),
        seen_by_user={1: {10}},
    )

    result = retriever.retrieve(1, k=3, seen_items_override={10, 20})

    assert [candidate.item_id for candidate in result] == [30]
    assert retriever.seen_by_user[1] == {10}
