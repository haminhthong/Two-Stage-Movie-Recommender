"""Bộ kiểm thử đơn vị (Unit Tests) và kiểm thử tích hợp (Integration Tests) cho hệ thống gợi ý."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from scripts.download_data import _safe_extract
from src.api import app
from src.evaluate import dcg, intra_list_diversity
from src.recommender import Recommender
from src.train import build_positive_interaction_matrix


def test_dcg_calculation() -> None:
    """Kiểm tra tính đúng đắn của chỉ số DCG theo vị trí rank."""
    assert dcg(0) == 1.0
    assert dcg(1) == pytest.approx(1.0 / np.log2(3))


def test_positive_matrix_does_not_store_zero_ratings() -> None:
    """Rating dưới ngưỡng không được tồn tại dưới dạng explicit zero trong CSR."""
    frame = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "item_id": [10, 20, 10],
            "rating": [5.0, 2.0, 4.0],
        }
    )
    matrix = build_positive_interaction_matrix(frame, {1: 0, 2: 1}, {10: 0, 20: 1})
    assert matrix.nnz == 2
    assert np.all(matrix.data == 1.0)


def test_intra_list_diversity_for_disjoint_genres() -> None:
    """Kiểm tra chỉ số ILD đạt giá trị cực đại (1.0) khi các thể loại hoàn toàn khác biệt."""
    genres_map = {1: {"Action"}, 2: {"Comedy"}}
    assert intra_list_diversity([1, 2], genres_map) == 1.0


def test_intra_list_diversity_for_identical_genres() -> None:
    """Kiểm tra chỉ số ILD đạt giá trị 0.0 khi các item cùng chung thể loại hoàn toàn."""
    genres_map = {1: {"Action", "Drama"}, 2: {"Action", "Drama"}}
    assert intra_list_diversity([1, 2], genres_map) == 0.0


def test_safe_extract_rejects_zip_slip(tmp_path: Path) -> None:
    """Kiểm tra cơ chế bảo mật ngăn chặn Zip Slip ném ngoại lệ ValueError."""
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("../outside.txt", "dữ liệu không an toàn")
    payload.seek(0)

    with (
        zipfile.ZipFile(payload) as archive,
        pytest.raises(ValueError, match="Zip Slip"),
    ):
        _safe_extract(archive, tmp_path)


def test_model_config_records_retrieval_contract() -> None:
    """Kiểm tra tệp cấu hình model config.json tuân thủ các quy tắc dự án."""
    config_path = Path(__file__).resolve().parents[1] / "models/config.json"
    if not config_path.exists():
        pytest.skip("Chưa có model artifact config.json để test.")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["schema_version"] in {1, 2}
    assert config["candidate_k"] >= config["top_k"]
    assert config["embedding_dimension"] > 0


def test_recommendation_metadata_is_client_friendly() -> None:
    """Kiểm tra format dữ liệu của hàm recommend_with_metadata đúng chuẩn UI client."""
    recommender = Recommender.__new__(Recommender)
    recommender.metadata = {
        "titles": {1: "Toy Story (1995)"},
        "genres": {1: {"Animation", "Comedy"}},
        "popularity_counts": {1: 123},
    }
    recommender.recommend = lambda *_args, **_kwargs: [1]

    result = recommender.recommend_with_metadata(42, k=1)

    assert result == [
        {
            "item_id": 1,
            "title": "Toy Story (1995)",
            "genres": ["Animation", "Comedy"],
            "interaction_count": 123,
        }
    ]


def test_cold_start_recommendation() -> None:
    """Kiểm tra gợi ý cho Cold-Start User theo thể loại yêu thích."""
    recommender = Recommender.__new__(Recommender)
    recommender.metadata = {
        "popular": [10, 20, 30],
        "genres": {10: {"Action"}, 20: {"Comedy"}, 30: {"Action", "Sci-Fi"}},
        "titles": {10: "Action Movie", 20: "Comedy Movie", 30: "Sci-Fi Movie"},
        "popularity_counts": {10: 500, 20: 300, 30: 200},
    }

    results = recommender.recommend_cold_start(preferred_genres=["Sci-Fi"], k=2)
    assert len(results) == 2
    assert results[0]["item_id"] == 30  # Chứa thể loại Sci-Fi


def test_api_health_endpoint() -> None:
    """Kiểm tra FastAPI client phản hồi 200 OK tại endpoint /health."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_ready" in data


def test_api_recommend_endpoint() -> None:
    """Kiểm tra FastAPI client phản hồi đúng tại endpoint /recommend/{user_id}."""
    client = TestClient(app)
    response = client.get("/recommend/1?k=5&include_metadata=true")
    if response.status_code == 503:
        pytest.skip("Model chưa sẵn sàng cho integration test API.")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == 1
    assert len(data["items"]) <= 5
