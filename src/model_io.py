"""Lưu và nạp model local cho training, evaluation và API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .ranking.features import N_FEATURES
from .utils import load_json, save_json


class ModelLoadError(RuntimeError):
    """Model local thiếu file hoặc không khớp contract cơ bản."""


def save_model(
    model_dir: str | Path,
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    metadata: dict[str, Any],
    config: dict[str, Any],
    ranker: Any | None = None,
) -> Path:
    """Ghi model vào một thư mục phẳng, không tạo pointer hay version phụ."""
    root = Path(model_dir)
    root.mkdir(parents=True, exist_ok=True)

    np.save(root / "user_embeddings.npy", user_embeddings)
    np.save(root / "item_embeddings.npy", item_embeddings)
    joblib.dump(metadata, root / "metadata.joblib")
    save_json(root / "config.json", config)
    ranker_path = root / "ranker.joblib"
    if ranker is not None:
        joblib.dump(ranker, ranker_path)
    elif ranker_path.exists():
        # Không để artifact ranker cũ tồn tại khi Dev chọn retrieval fallback.
        ranker_path.unlink()

    return root


def load_model(model_dir: str | Path = "models") -> dict[str, Any]:
    """Nạp model phẳng và kiểm tra các invariant cần cho serving."""
    root = Path(model_dir)
    required = (
        "user_embeddings.npy",
        "item_embeddings.npy",
        "metadata.joblib",
        "config.json",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ModelLoadError(f"Thiếu model files trong {root}: {', '.join(missing)}")

    user_embeddings = np.load(root / "user_embeddings.npy", allow_pickle=False)
    item_embeddings = np.load(root / "item_embeddings.npy", allow_pickle=False)
    metadata = joblib.load(root / "metadata.joblib")
    config = load_json(root / "config.json")

    if user_embeddings.ndim != 2 or item_embeddings.ndim != 2:
        raise ModelLoadError("Embedding phải là ma trận 2 chiều.")
    if user_embeddings.shape[1] != item_embeddings.shape[1]:
        raise ModelLoadError("User/item embedding khác embedding dimension.")
    if not isinstance(metadata, dict) or not isinstance(config, dict):
        raise ModelLoadError("metadata.joblib và config.json phải là dictionary.")
    users = metadata.get("users")
    items = metadata.get("items")
    if users is not None and len(users) != user_embeddings.shape[0]:
        raise ModelLoadError("Số user trong metadata không khớp user embeddings.")
    if items is not None and len(items) != item_embeddings.shape[0]:
        raise ModelLoadError("Số item trong metadata không khớp item embeddings.")

    ranker_path = root / "ranker.joblib"
    ranker_enabled = bool(config.get("ranker_enabled", False))
    ranker = joblib.load(ranker_path) if ranker_path.is_file() else None
    if ranker_enabled and ranker is None:
        raise ModelLoadError("config bật ranker nhưng thiếu ranker.joblib.")
    if ranker is not None:
        expected = getattr(ranker, "n_features_in_", N_FEATURES)
        if int(expected) != N_FEATURES:
            raise ModelLoadError(
                f"Ranker cần {expected} features, contract hiện tại có {N_FEATURES}."
            )

    return {
        "user_embeddings": user_embeddings,
        "item_embeddings": item_embeddings,
        "metadata": metadata,
        "config": config,
        "ranker": ranker,
        "model_dir": root,
    }
