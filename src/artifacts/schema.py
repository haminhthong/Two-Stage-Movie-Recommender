"""Schema định nghĩa cấu trúc lưu trữ và phiên bản mô hình (Artifact Schemas)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ProductionPointer:
    """Con trỏ xác định phiên bản mô hình đang kích hoạt cho môi trường Online."""

    active_version: str
    updated_at: str
    description: str
