"""Hợp nhất candidate đa nguồn bằng Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

from typing import Iterable

from .base import Candidate, CandidateRetriever


class MultiSourceRetriever(CandidateRetriever):
    """Union các nguồn retrieval rồi cắt về một pool theo RRF.

    Điểm SVD, popularity và genre có semantics/scale khác nhau nên tuyệt đối
    không được sort trực tiếp cùng nhau. RRF chỉ dùng thứ hạng của từng nguồn;
    score gốc được giữ riêng để Stage 2 học.
    """

    def __init__(
        self,
        retrievers: dict[str, tuple[CandidateRetriever, int]],
        rrf_k: int = 60,
    ) -> None:
        self.retrievers = retrievers
        self.rrf_k = max(1, int(rrf_k))

    @staticmethod
    def _retrieve_source(
        retriever: CandidateRetriever,
        user_id: int,
        source_k: int,
        filter_seen: bool,
        seen_items_override: Iterable[int] | None,
    ) -> list[Candidate]:
        """Gọi retriever mà không mutate state; giữ tương thích mock retriever cũ."""
        if seen_items_override is None:
            return retriever.retrieve(
                user_id=user_id,
                k=source_k,
                filter_seen=filter_seen,
            )

        return retriever.retrieve(
            user_id=user_id,
            k=source_k,
            filter_seen=filter_seen,
            seen_items_override=seen_items_override,
        )

    def retrieve(
        self,
        user_id: int,
        k: int = 200,
        filter_seen: bool = True,
        seen_items_override: Iterable[int] | None = None,
    ) -> list[Candidate]:
        """Lấy union ứng viên và xếp hạng theo RRF giảm dần."""
        if k <= 0:
            return []

        merged: dict[int, dict[str, object]] = {}
        for source_name, (retriever, source_k) in self.retrievers.items():
            candidates = self._retrieve_source(
                retriever,
                user_id=user_id,
                source_k=max(0, int(source_k)),
                filter_seen=filter_seen,
                seen_items_override=seen_items_override,
            )
            for zero_based_rank, candidate in enumerate(candidates):
                item_id = int(candidate.item_id)
                info = merged.setdefault(
                    item_id,
                    {"scores": {}, "ranks": {}, "rrf_score": 0.0},
                )
                scores = info["scores"]
                ranks = info["ranks"]
                assert isinstance(scores, dict)
                assert isinstance(ranks, dict)

                source_score = candidate.source_scores.get(
                    source_name,
                    float(candidate.retrieval_score),
                )
                source_rank = _source_rank(candidate, source_name, zero_based_rank)
                scores[source_name] = float(source_score)
                ranks[source_name] = source_rank
                info["rrf_score"] = float(info["rrf_score"]) + 1.0 / (
                    self.rrf_k + source_rank
                )

        ordered = sorted(
            merged.items(),
            key=lambda pair: (-float(pair[1]["rrf_score"]), pair[0]),
        )[:k]

        result: list[Candidate] = []
        for zero_based_rank, (item_id, info) in enumerate(ordered):
            scores = info["scores"]
            ranks = info["ranks"]
            assert isinstance(scores, dict)
            assert isinstance(ranks, dict)
            typed_scores = {str(key): float(value) for key, value in scores.items()}

            result.append(
                Candidate(
                    item_id=item_id,
                    retrieval_score=float(info["rrf_score"]),
                    retrieval_source=(
                        next(iter(typed_scores))
                        if len(typed_scores) == 1
                        else "multi_source"
                    ),
                    retrieval_rank=zero_based_rank,
                    source_scores=typed_scores,
                    svd_score=typed_scores.get("svd"),
                    svd_rank=_optional_int(ranks.get("svd")),
                    popularity_score=typed_scores.get("popularity"),
                    popularity_rank=_optional_int(ranks.get("popularity")),
                    genre_score=typed_scores.get("genre"),
                    genre_rank=_optional_int(ranks.get("genre")),
                    rrf_score=float(info["rrf_score"]),
                    source_count=len(typed_scores),
                )
            )

        return result


def _optional_int(value: object) -> int | None:
    """Chuyển rank tùy chọn về int để object Candidate dễ serialize."""
    return int(value) if value is not None else None


def _source_rank(candidate: Candidate, source_name: str, zero_based_rank: int) -> int:
    """Lấy rank 1-based, có fallback an toàn cho implementation cũ."""
    rank = {
        "svd": candidate.svd_rank,
        "popularity": candidate.popularity_rank,
        "genre": candidate.genre_rank,
    }.get(source_name)
    return int(rank) if rank is not None and rank > 0 else zero_based_rank + 1
