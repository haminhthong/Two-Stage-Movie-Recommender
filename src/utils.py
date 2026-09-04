"""Thư viện tiện ích dùng chung cho hệ thống gợi ý.

Bao gồm các hàm thiết lập ngẫu nhiên (seed), lưu/đọc JSON và cấu hình logging.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

# Khởi tạo logger dùng chung cho toàn bộ dự án
LOGGER = logging.getLogger("two_stage_recommender")


def setup_logging(default_level: str = "INFO") -> None:
    """Thiết lập cấu hình logging chuẩn cho toàn bộ ứng dụng.

    Args:
        default_level (str): Mức độ log mặc định ('DEBUG', 'INFO', 'WARNING', 'ERROR').
    """
    level_str = os.getenv("LOG_LEVEL", default_level).upper()
    level = getattr(logging, level_str, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_seed(seed: int = 42) -> None:
    """Cố định seed ngẫu nhiên cho Python, NumPy và PyTorch (nếu có) để đảm bảo tính tái lập.

    Args:
        seed (int): Giá trị seed ngẫu nhiên (Mặc định: 42).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
    except ImportError:
        LOGGER.debug("PyTorch không được cài đặt; chỉ đặt seed cho random và NumPy.")


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Lưu dữ liệu dạng dictionary vào file JSON với định dạng UTF-8.

    Args:
        path (str | Path): Đường dẫn file đầu ra.
        payload (dict[str, Any]): Dữ liệu dictionary cần lưu.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> dict[str, Any]:
    """Đọc dữ liệu từ file JSON.

    Args:
        path (str | Path): Đường dẫn file JSON cần đọc.

    Returns:
        dict[str, Any]: Dữ liệu dictionary đã đọc.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file JSON tại: {file_path}")
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)
