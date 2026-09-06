"""Lưu trữ trọn bộ Model Artifacts theo phiên bản kèm Provenance Manifest."""

from __future__ import annotations

import datetime
from pathlib import Path
import shutil
from typing import Any
import joblib
import numpy as np

from ..utils import save_json


def save_versioned_bundle(
    base_dir: str | Path,
    version: str,
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    metadata: dict[str, Any],
    config_payload: dict[str, Any],
    ranker: Any | None = None,
    data_manifest: dict[str, Any] | None = None,
    split_manifest: dict[str, Any] | None = None,
    evaluation_report: dict[str, Any] | None = None,
) -> Path:
    """Lưu trữ trọn bộ Artifacts vào thư mục phiên bản models/<version>/ và tạo con trỏ production.json.

    Đồng thời duy trì tệp phẳng tại thư mục gốc models/ để đảm bảo tương thích ngược 100%.
    """
    root_path = Path(base_dir)
    version_dir = root_path / version

    # Tạo các thư mục con
    retrieval_dir = version_dir / "retrieval"
    ranking_dir = version_dir / "ranking"
    metadata_dir = version_dir / "metadata"

    for d in [retrieval_dir, ranking_dir, metadata_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Lưu retrieval embeddings
    np.save(retrieval_dir / "user_emb.npy", user_embeddings)
    np.save(retrieval_dir / "item_emb.npy", item_embeddings)

    # 2. Lưu metadata
    joblib.dump(metadata, metadata_dir / "meta.joblib")

    # 3. Lưu learned ranker (nếu có)
    if ranker is not None:
        joblib.dump(ranker, ranking_dir / "ranker.joblib")

    # 4. Lưu configs và manifests
    save_json(version_dir / "config.json", config_payload)
    if data_manifest is not None:
        save_json(version_dir / "data_manifest.json", data_manifest)
    if split_manifest is not None:
        save_json(version_dir / "split_manifest.json", split_manifest)
    if evaluation_report is not None:
        save_json(version_dir / "evaluation.json", evaluation_report)

    # 5. Cập nhật production.json trỏ tới phiên bản này
    prod_pointer = {
        "active_version": version,
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "schema_version": config_payload.get("schema_version", 4),
    }
    save_json(root_path / "production.json", prod_pointer)

    # 6. Tương thích ngược: Đồng bộ sang thư mục gốc models/
    np.save(root_path / "user_emb.npy", user_embeddings)
    np.save(root_path / "item_emb.npy", item_embeddings)
    joblib.dump(metadata, root_path / "meta.joblib")
    save_json(root_path / "config.json", config_payload)
    if ranker is not None:
        joblib.dump(ranker, root_path / "ranker.joblib")

    return version_dir
