"""Promotion explicit cho release candidate đã được review."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from ..utils import load_json, save_json
from .loader import validate_release_bundle


def promote_release(
    base_dir: str | Path,
    version: str,
    locked_test_metrics: dict[str, Any],
    *,
    min_dev_ndcg_gain: float = 0.0,
) -> Path:
    """Promote candidate thành production sau khi kiểm tra policy đã khai báo.

    Hàm này không tự chạy grid search. Test metrics phải được truyền từ một
    pipeline frozen và kết quả Dev model selection phải được ghi trong config.
    """
    root = Path(base_dir)
    release_dir = root / version
    validate_release_bundle(release_dir, expected_version=version)
    config = load_json(release_dir / "config.json")
    dev_metrics = config.get("dev_metrics", {})
    stage1_ndcg = float(dev_metrics.get("stage1_order_ndcg@10", 0.0))
    ranker_ndcg = float(dev_metrics.get("ranker_ndcg@10", 0.0))
    if (
        bool(config.get("ranker_enabled", False))
        and ranker_ndcg <= stage1_ndcg + min_dev_ndcg_gain
    ):
        raise ValueError(
            "Release bị từ chối: ranker chưa vượt retrieval order trên Dev."
        )

    save_json(release_dir / "locked_test_metrics.json", locked_test_metrics)
    save_json(
        root / "production.json",
        {
            "active_version": version,
            "updated_at": dt.datetime.now(dt.UTC).isoformat(),
            "schema_version": config.get("schema_version", 5),
        },
    )
    return release_dir
