"""Đo lường và phân tích độ trễ từng giai đoạn (Stage-Level Latency Profiling)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np


@dataclass(frozen=True)
class StageLatencyReport:
    """Báo cáo phân bổ độ trễ (p50, p95, mean) cho một giai đoạn."""

    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float


def summarize_latencies(latencies_ms: Sequence[float]) -> StageLatencyReport:
    """Tính toán p50, p95, p99, mean từ danh sách thời gian đo."""
    if not latencies_ms:
        return StageLatencyReport(0.0, 0.0, 0.0, 0.0)
    arr = np.asarray(latencies_ms)
    return StageLatencyReport(
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        mean_ms=float(np.mean(arr)),
    )
