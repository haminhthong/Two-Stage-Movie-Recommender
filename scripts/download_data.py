"""Script tự động tải và giải nén bộ dữ liệu MovieLens 1M.

Bao gồm cơ chế kiểm tra an toàn lỗ hổng Zip Slip để đảm bảo các tệp được giải nén
chỉ nằm trong thư mục đích đã chỉ định.
"""

from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path

import requests
from src.utils import LOGGER, setup_logging

DATASET_URL: str = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    """Giải nén an toàn để phòng tránh lỗ hổng bảo mật Zip Slip.

    Zip Slip là lỗ hổng nguy hiểm cho phép kẻ tấn công ghi đè tệp ngoài thư mục đích
    bằng cách thêm các ký tự đường dẫn tương đối như `../outside.txt` trong archive ZIP.

    Args:
        archive (zipfile.ZipFile): Đối tượng ZIP file đang mở.
        destination (Path): Đường dẫn thư mục đích được phép giải nén.

    Raises:
        ValueError: Nếu phát hiện tệp có đường dẫn giải nén vượt ra ngoài thư mục đích.
    """
    dest_path = destination.resolve()
    for member in archive.infolist():
        target_path = (dest_path / member.filename).resolve()
        if dest_path not in target_path.parents and target_path != dest_path:
            raise ValueError(
                f"Phát hiện đường dẫn không an toàn (Zip Slip Attempt) trong tệp ZIP: {member.filename}"
            )
    archive.extractall(dest_path)


def download_movielens(output_dir: str | Path = "data/raw") -> None:
    """Tải tập dữ liệu MovieLens 1M từ nguồn chính thức GroupLens và giải nén an toàn.

    Args:
        output_dir (str | Path): Thư mục lưu trữ dữ liệu thô. (Mặc định: 'data/raw')
    """
    setup_logging()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Đang tải dữ liệu MovieLens 1M từ: %s", DATASET_URL)
    try:
        response = requests.get(DATASET_URL, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        LOGGER.error("Lỗi khi tải bộ dữ liệu từ URL: %s", exc)
        raise

    LOGGER.info("Giải nén tệp dữ liệu vào: %s", out_path.resolve())
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        _safe_extract(archive, out_path)

    LOGGER.info("Hoàn tất tải và giải nén dữ liệu MovieLens 1M!")


def main() -> None:
    """Hàm main thực thi script tải dữ liệu từ CLI."""
    download_movielens()


if __name__ == "__main__":
    main()
