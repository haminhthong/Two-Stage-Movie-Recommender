"""Ghi release bundle có version, manifest và provenance."""

from __future__ import annotations

import datetime as dt
import hashlib
import platform
import subprocess
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ..utils import save_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(root_path: Path) -> str:
    """Lấy commit hiện tại; bundle vẫn ghi được khi thư mục không phải Git repo."""
    git_root = next(
        (
            parent
            for parent in (root_path, *root_path.parents)
            if (parent / ".git").exists()
        ),
        root_path,
    )
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=git_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


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
    publish: bool = True,
) -> Path:
    """Ghi bundle nguyên tử theo version.

    ``publish=False`` dành cho Release Candidate. Tham số mặc định ``True``
    được giữ để code cũ gọi hàm này không bị đổi hành vi; pipeline train mới
    luôn truyền ``publish=False`` và promotion nằm ở bước riêng.
    """
    root_path = Path(base_dir)
    root_resolved = root_path.resolve()
    version_dir = (root_path / version).resolve()
    try:
        version_dir.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError("version phải nằm bên trong thư mục model.") from exc
    if version_dir == root_resolved:
        raise ValueError("version phải trỏ tới một release cụ thể.")
    if version_dir.exists() and any(version_dir.iterdir()):
        raise FileExistsError(f"Release đã tồn tại và không được ghi đè: {version_dir}")

    retrieval_dir = version_dir / "retrieval"
    ranking_dir = version_dir / "ranking"
    metadata_dir = version_dir / "metadata"
    for directory in (retrieval_dir, ranking_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    np.save(retrieval_dir / "user_emb.npy", user_embeddings)
    np.save(retrieval_dir / "item_emb.npy", item_embeddings)
    joblib.dump(metadata, metadata_dir / "meta.joblib")
    if ranker is not None:
        joblib.dump(ranker, ranking_dir / "ranker.joblib")

    save_json(version_dir / "config.json", config_payload)
    if data_manifest is not None:
        save_json(version_dir / "data_manifest.json", data_manifest)
    if split_manifest is not None:
        save_json(version_dir / "temporal_split_manifest.json", split_manifest)
    if evaluation_report is not None:
        save_json(version_dir / "dev_metrics.json", evaluation_report)

    save_json(
        version_dir / "retrieval_policy.json",
        dict(config_payload.get("candidate_contract", {})),
    )
    save_json(
        version_dir / "ranking_policy.json",
        {
            "ranker_type": config_payload.get("ranker_model_type", "xgb_ranker"),
            "feature_schema_version": config_payload.get(
                "feature_schema_version", "rank-features-v1"
            ),
            "enabled": bool(config_payload.get("ranker_enabled", True)),
        },
    )
    save_json(
        version_dir / "diversity_policy.json",
        {
            "lambda": config_payload.get("diversity_lambda", 0.0),
            "pool_k": config_payload.get("rerank_pool_k", 40),
        },
    )

    relative_files = [
        "retrieval/user_emb.npy",
        "retrieval/item_emb.npy",
        "metadata/meta.joblib",
        "config.json",
        "retrieval_policy.json",
        "ranking_policy.json",
        "diversity_policy.json",
    ]
    if data_manifest is not None:
        relative_files.append("data_manifest.json")
    if split_manifest is not None:
        relative_files.append("temporal_split_manifest.json")
    if evaluation_report is not None:
        relative_files.append("dev_metrics.json")
    if ranker is not None:
        relative_files.append("ranking/ranker.joblib")

    artifact_hashes = {
        relative_path: _sha256(version_dir / relative_path)
        for relative_path in relative_files
    }
    manifest = {
        "manifest_version": 1,
        "model_version": str(config_payload.get("version", version)),
        "release_path": version,
        "dataset_sha256": {
            "ratings": (data_manifest or {}).get("ratings_sha256", "unknown"),
            "movies": (data_manifest or {}).get("movies_sha256", "unknown"),
        },
        "feature_schema_version": config_payload.get(
            "feature_schema_version", "rank-features-v1"
        ),
        "retrieval_version": config_payload.get(
            "retrieval_version", "svd64-multisource-rrf-v1"
        ),
        "ranker_type": config_payload.get("ranker_model_type", "xgb_ranker"),
        "git_commit": _git_commit(root_path),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "artifact_sha256": artifact_hashes,
    }
    save_json(version_dir / "manifest.json", manifest)

    if publish:
        save_json(
            root_path / "production.json",
            {
                "active_version": version,
                "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "schema_version": config_payload.get("schema_version", 5),
            },
        )

    return version_dir
