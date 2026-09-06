"""Nạp Model Artifacts với cơ chế tự động tìm kiếm phiên bản Production."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import joblib
import numpy as np

from ..utils import load_json


def load_production_bundle(model_dir: str | Path = "models") -> dict[str, Any]:
    """Nạp trọn bộ Artifacts từ phiên bản active trong production.json hoặc thư mục gốc.

    Returns:
        dict: {
            "user_embeddings": np.ndarray,
            "item_embeddings": np.ndarray,
            "metadata": dict,
            "config": dict,
            "ranker": Any | None,
            "version": str,
        }
    """
    root_path = Path(model_dir)
    prod_file = root_path / "production.json"

    version_path = root_path
    version_name = "default"

    if prod_file.exists():
        try:
            prod_info = load_json(prod_file)
            active_ver = prod_info.get("active_version")
            if active_ver and (root_path / active_ver).exists():
                version_path = root_path / active_ver
                version_name = active_ver
        except Exception:
            version_path = root_path

    # Tìm user_emb
    if (version_path / "retrieval" / "user_emb.npy").exists():
        user_emb = np.load(version_path / "retrieval" / "user_emb.npy")
    elif (version_path / "user_emb.npy").exists():
        user_emb = np.load(version_path / "user_emb.npy")
    else:
        user_emb = np.load(root_path / "user_emb.npy")

    # Tìm item_emb
    if (version_path / "retrieval" / "item_emb.npy").exists():
        item_emb = np.load(version_path / "retrieval" / "item_emb.npy")
    elif (version_path / "item_emb.npy").exists():
        item_emb = np.load(version_path / "item_emb.npy")
    else:
        item_emb = np.load(root_path / "item_emb.npy")

    # Tìm metadata
    if (version_path / "metadata" / "meta.joblib").exists():
        meta = joblib.load(version_path / "metadata" / "meta.joblib")
    elif (version_path / "meta.joblib").exists():
        meta = joblib.load(version_path / "meta.joblib")
    else:
        meta = joblib.load(root_path / "meta.joblib")

    # Tìm config
    if (version_path / "config.json").exists():
        config = load_json(version_path / "config.json")
    else:
        config = load_json(root_path / "config.json")

    # Tìm ranker
    ranker = None
    if (version_path / "ranking" / "ranker.joblib").exists():
        ranker = joblib.load(version_path / "ranking" / "ranker.joblib")
    elif (version_path / "ranker.joblib").exists():
        ranker = joblib.load(version_path / "ranker.joblib")
    elif (root_path / "ranker.joblib").exists():
        ranker = joblib.load(root_path / "ranker.joblib")

    return {
        "user_embeddings": user_emb,
        "item_embeddings": item_emb,
        "metadata": meta,
        "config": config,
        "ranker": ranker,
        "version": version_name,
    }
