"""Nạp production release theo manifest; không fallback sang artifact legacy."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ..utils import load_json


class ArtifactValidationError(RuntimeError):
    """Bundle không tồn tại, không đầy đủ hoặc không khớp manifest."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_release_path(root: Path, version: str) -> Path:
    """Đảm bảo production pointer không trỏ ra ngoài model root."""
    release = (root / version).resolve()
    root_resolved = root.resolve()
    try:
        release.relative_to(root_resolved)
    except ValueError as exc:
        raise ArtifactValidationError("active_version không hợp lệ.") from exc
    if release == root_resolved:
        raise ArtifactValidationError("active_version phải trỏ tới một release cụ thể.")
    return release


def validate_release_bundle(
    release_dir: str | Path,
    expected_version: str | None = None,
) -> dict[str, Any]:
    """Kiểm tra đủ file, schema và SHA256 trước khi load bất kỳ artifact nào."""
    release = Path(release_dir)
    if not release.is_dir():
        raise ArtifactValidationError(f"Không tìm thấy release: {release}")

    config_path = release / "config.json"
    if not config_path.is_file():
        raise ArtifactValidationError("Release thiếu config.json.")
    config = load_json(config_path)
    manifest_path = release / "manifest.json"
    if not manifest_path.is_file():
        # Chỉ cho phép artifact schema cũ đã được pointer trỏ trực tiếp; không
        # fallback root/mixed files. Mọi release schema 5 bắt buộc manifest.
        if int(config.get("schema_version", 0)) < 5:
            legacy_required = (
                "retrieval/user_emb.npy",
                "retrieval/item_emb.npy",
                "metadata/meta.joblib",
            )
            if not all((release / path).is_file() for path in legacy_required):
                raise ArtifactValidationError("Legacy release thiếu artifact bắt buộc.")
            return {
                "manifest_version": 0,
                "model_version": expected_version
                or str(config.get("version", "legacy")),
                "legacy": True,
            }
        raise ArtifactValidationError("Release schema 5 thiếu manifest.json.")

    manifest = load_json(manifest_path)
    manifest_version = str(manifest.get("model_version", ""))
    config_version = str(config.get("version", ""))
    if not config_version or manifest_version != config_version:
        raise ArtifactValidationError("version trong config/manifest không khớp.")
    if expected_version is not None and manifest_version not in {
        expected_version,
        release.name,
        config_version,
    }:
        raise ArtifactValidationError(
            "model_version trong manifest không khớp release."
        )
    if manifest.get("feature_schema_version") != config.get(
        "feature_schema_version", manifest.get("feature_schema_version")
    ):
        raise ArtifactValidationError(
            "Feature schema trong config/manifest không khớp."
        )

    required = {
        "retrieval/user_emb.npy",
        "retrieval/item_emb.npy",
        "metadata/meta.joblib",
        "config.json",
        "retrieval_policy.json",
        "ranking_policy.json",
        "diversity_policy.json",
    }
    hashes = manifest.get("artifact_sha256", {})
    if not isinstance(hashes, dict):
        raise ArtifactValidationError("artifact_sha256 trong manifest không hợp lệ.")

    for relative_path in required:
        path = release / relative_path
        if not path.is_file():
            raise ArtifactValidationError(
                f"Release thiếu artifact bắt buộc: {relative_path}"
            )
        expected_hash = hashes.get(relative_path)
        if not expected_hash or _sha256(path) != expected_hash:
            raise ArtifactValidationError(f"Hash artifact không khớp: {relative_path}")

    ranker_path = release / "ranking" / "ranker.joblib"
    ranker_enabled = bool(config.get("ranker_enabled", False))
    if ranker_enabled and (
        not ranker_path.is_file() or not hashes.get("ranking/ranker.joblib")
    ):
        raise ArtifactValidationError("config bật ranker nhưng bundle thiếu ranker.")
    if ranker_path.is_file() and hashes.get("ranking/ranker.joblib") != _sha256(
        ranker_path
    ):
        raise ArtifactValidationError("Hash ranker không khớp manifest.")

    return manifest


def load_production_bundle(model_dir: str | Path = "models") -> dict[str, Any]:
    """Nạp đúng release được production pointer chỉ định.

    Không fallback sang thư mục root hoặc artifact version khác. Nếu validation
    thất bại, service phải ở trạng thái NOT READY.
    """
    root = Path(model_dir)
    pointer_path = root / "production.json"
    if not pointer_path.is_file():
        raise ArtifactValidationError("Thiếu production.json; service chưa sẵn sàng.")

    pointer = load_json(pointer_path)
    active_version = pointer.get("active_version")
    if not isinstance(active_version, str) or not active_version.strip():
        raise ArtifactValidationError("production.json thiếu active_version.")
    release = _safe_release_path(root, active_version)
    manifest = validate_release_bundle(release, expected_version=active_version)

    user_emb = np.load(release / "retrieval" / "user_emb.npy", allow_pickle=False)
    item_emb = np.load(release / "retrieval" / "item_emb.npy", allow_pickle=False)
    metadata = joblib.load(release / "metadata" / "meta.joblib")
    config = load_json(release / "config.json")
    if str(config.get("version", "")) != str(manifest.get("model_version", "")):
        raise ArtifactValidationError("version trong config/manifest không khớp.")
    ranker_path = release / "ranking" / "ranker.joblib"
    # Không load artifact ranker khi release đã tắt ranker qua Dev model selection.
    # Điều này cũng cho phép service đọc legacy bundle mà không cài xgboost;
    # file thừa không được coi là active model.
    ranker = (
        joblib.load(ranker_path)
        if ranker_path.is_file() and bool(config.get("ranker_enabled", False))
        else None
    )

    if not isinstance(metadata, dict) or not isinstance(config, dict):
        raise ArtifactValidationError(
            "Metadata/config của release không phải dictionary."
        )
    return {
        "user_embeddings": user_emb,
        "item_embeddings": item_emb,
        "metadata": metadata,
        "config": config,
        "ranker": ranker,
        "version": active_version,
        "manifest": manifest,
        "release_dir": release,
    }
