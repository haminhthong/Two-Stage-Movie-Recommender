"""Bộ hợp nhất và khử trùng lặp ứng viên đa nguồn (Multi-Source Candidate Merger)."""

from __future__ import annotations

from typing import Sequence
from .base import Candidate, CandidateRetriever


class MultiSourceRetriever(CandidateRetriever):
    """Trích xuất và kết hợp ứng viên từ nhiều nguồn (SVD, Popularity, Genre)."""

    def __init__(
        self,
        retrievers: dict[str, tuple[CandidateRetriever, int]],
    ) -> None:
        """Khởi tạo MultiSourceRetriever.

        Args:
            retrievers: Dictionary ánh xạ source_name -> (retriever_instance, k_for_source).
                       Ví dụ: {"svd": (svd_retriever, 150), "popularity": (pop_retriever, 50), "genre": (genre_retriever, 50)}
        """
        self.retrievers = retrievers

    def retrieve(
        self,
        user_id: int,
        k: int = 200,
        filter_seen: bool = True,
    ) -> list[Candidate]:
        """Trích xuất từ từng nguồn, hợp nhất điểm số và khử trùng lặp."""
        combined_candidates: dict[int, dict] = {}

        # 1. Thu thập từ từng nguồn theo thứ tự ưu tiên
        for source_name, (retriever, source_k) in self.retrievers.items():
            source_cands = retriever.retrieve(user_id=user_id, k=source_k, filter_seen=filter_seen)
            for c in source_cands:
                item_id = c.item_id
                if item_id not in combined_candidates:
                    combined_candidates[item_id] = {
                        "primary_score": c.retrieval_score,
                        "primary_source": source_name,
                        "min_rank": c.retrieval_rank,
                        "source_scores": {source_name: c.retrieval_score},
                    }
                else:
                    # Cập nhật thêm điểm số của nguồn mới
                    combined_candidates[item_id]["source_scores"][source_name] = c.retrieval_score
                    if len(combined_candidates[item_id]["source_scores"]) > 1:
                        combined_candidates[item_id]["primary_source"] = "multi_source"

        if not combined_candidates:
            return []

        # 2. Xếp hạng sơ bộ theo ưu tiên nguồn và điểm số (SVD trước, rồi tới Genre, Popularity)
        def sort_key(item_tuple: tuple[int, dict]) -> tuple:
            _item_id, info = item_tuple
            src_scores = info["source_scores"]
            svd_sc = src_scores.get("svd", -float("inf"))
            genre_sc = src_scores.get("genre", -float("inf"))
            pop_sc = src_scores.get("popularity", -float("inf"))
            num_sources = len(src_scores)
            return (num_sources, svd_sc, genre_sc, pop_sc)

        sorted_items = sorted(combined_candidates.items(), key=sort_key, reverse=True)[:k]

        merged_list: list[Candidate] = []
        for rank, (item_id, info) in enumerate(sorted_items):
            merged_list.append(
                Candidate(
                    item_id=item_id,
                    retrieval_score=float(info["primary_score"]),
                    retrieval_source=str(info["primary_source"]),
                    retrieval_rank=rank,
                    source_scores=info["source_scores"],
                )
            )

        return merged_list
