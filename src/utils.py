"""Tiện ích dùng chung cho training, evaluation và model I/O."""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger("two_stage_recommender")


def setup_logging(default_level: str = "INFO") -> None:
    """Thiết lập logging đơn giản cho các script dòng lệnh."""
    level_name = os.getenv("LOG_LEVEL", default_level).upper()
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_seed(seed: int = 42) -> None:
    """Cố định seed cho Python và NumPy."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Tạo thư mục cha và lưu dictionary dưới dạng JSON UTF-8."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> dict[str, Any]:
    """Đọc một file JSON và báo lỗi rõ ràng nếu file không tồn tại."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file JSON tại: {file_path}")
    with file_path.open(encoding="utf-8") as file:
        return json.load(file)
