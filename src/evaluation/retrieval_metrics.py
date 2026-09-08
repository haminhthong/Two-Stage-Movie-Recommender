"""Chỉ số đánh giá tầng trích xuất ứng viên (Stage 1 Retrieval Metrics).

Bao gồm:
- Candidate Recall@K (50, 100, 200)
- Target in Catalog Rate (Kiểm tra xem phim test có nằm trong train catalog không)
- Cold Item Test Share (Tỷ lệ ground truth item là cold-start, Stage 1 không thể retrieve được).
"""

from __future__ import annotations

from collections.abc import Sequence

from ..retrieval.base import Candidate


def candidate_recall_at_k(
    candidates: Sequence[Candidate] | Sequence[int],
    target_item: int,
    k: int = 200,
) -> float:
    """Tính Candidate Recall@K cho một người dùng: item đúng có lọt vào top K candidates không.

    Args:
        candidates: Danh sách Candidate hoặc danh sách item_id.
        target_item: ID phim ground truth cần tìm.
        k: Ngưỡng cắt K (ví dụ: 50, 100, 200).

    Returns:
        float: 1.0 nếu target_item xuất hiện trong top K candidates, ngược lại 0.0.
    """
    if not candidates or k <= 0:
        return 0.0

    sliced = candidates[:k]
    if sliced and isinstance(sliced[0], Candidate):
        item_set = {c.item_id for c in sliced}  # type: ignore[union-attr]
    else:
        item_set = set(sliced)  # type: ignore[arg-type]

    return 1.0 if target_item in item_set else 0.0


def target_in_catalog_rate(
    test_targets: Sequence[int],
    train_catalog: set[int],
) -> float:
    """Đo lường tỷ lệ các item mục tiêu trong tập Test thực sự có mặt trong danh mục Train.

    Nếu item chưa từng xuất hiện trong Train, Stage 1 Collaborative không thể trích xuất nó.
    Đây là New-Item Cold-Start, không phải lỗi do thuật toán Ranking.
    """
    if not test_targets:
        return 0.0
    in_catalog_count = sum(1 for item in test_targets if item in train_catalog)
    return float(in_catalog_count / len(test_targets))


def cold_item_test_share(
    test_targets: Sequence[int],
    train_catalog: set[int],
) -> float:
    """Tỷ lệ các trường hợp test là cold-start item ngoài catalog train."""
    return 1.0 - target_in_catalog_rate(test_targets, train_catalog)
