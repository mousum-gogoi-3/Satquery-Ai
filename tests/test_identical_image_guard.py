import io
from unittest.mock import patch, MagicMock
import numpy as np
import pytest
import tifffile
from fastapi.testclient import TestClient

from backend.main import app
from backend.config import settings
from backend.api.analysis import vqa_engine

client = TestClient(app)


def _create_synthetic_geotiff(filename: str, seed: int = 42, width: int = 128, height: int = 128) -> str:
    """Helper creating a test GeoTIFF with CRS metadata."""
    np.random.seed(seed)
    data = np.random.randint(50, 200, size=(3, height, width), dtype=np.uint8)
    file_path = settings.UPLOAD_DIR / filename
    
    geotags = {
        33550: (10.0, 10.0, 0.0),
        33922: (0.0, 0.0, 0.0, 500000.0, 4649700.0, 0.0),
        34735: (
            1, 1, 0, 7,
            1024, 0, 1, 1,
            1025, 0, 1, 1,
            3072, 0, 1, 32631
        ),
    }
    with tifffile.TiffWriter(str(file_path)) as tif:
        tif.write(data, extratags=[(k, 12, len(v), v, False) for k, v in geotags.items()])
    return filename


@pytest.fixture(scope="module")
def setup_test_rasters():
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    img_a_name = "test_guard_raster_a.tif"
    img_b_name = "test_guard_raster_b.tif"
    _create_synthetic_geotiff(img_a_name, seed=42)
    _create_synthetic_geotiff(img_b_name, seed=99)
    yield img_a_name, img_b_name
    for name in (img_a_name, img_b_name):
        p = settings.UPLOAD_DIR / name
        if p.exists():
            p.unlink()


def test_1_identical_query_what_changed(setup_test_rasters):
    """Test 1: Before = A, After = A | Query = 'What changed between these images?'"""
    img_a, _ = setup_test_rasters
    with patch.object(vqa_engine, "answer_question", wraps=vqa_engine.answer_question) as spy_vqa:
        resp = client.post("/api/v1/analysis/change", json={
            "file_id_before": img_a,
            "file_id_after": img_a,
            "query": "What changed between these images?"
        })
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["change_detected"] is False
        assert data["status"] in ("NO_CHANGE", "NO_SIGNIFICANT_CHANGE")
        assert "identical" in data["summary"].lower()
        assert spy_vqa.call_count == 0


def test_2_identical_query_what_had_changed_overtime(setup_test_rasters):
    """Test 2: Before = A, After = A | Query = 'What had changed between these two images overtime'"""
    img_a, _ = setup_test_rasters
    with patch.object(vqa_engine, "answer_question", wraps=vqa_engine.answer_question) as spy_vqa:
        resp = client.post("/api/v1/analysis/change", json={
            "file_id_before": img_a,
            "file_id_after": img_a,
            "query": "What had changed between these two images overtime"
        })
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["change_detected"] is False
        assert data["status"] in ("NO_CHANGE", "NO_SIGNIFICANT_CHANGE")
        assert "identical" in data["summary"].lower()
        assert spy_vqa.call_count == 0


def test_3_identical_query_what_was_changing(setup_test_rasters):
    """Test 3: Before = A, After = A | Query = 'What was changing in these images'"""
    img_a, _ = setup_test_rasters
    with patch.object(vqa_engine, "answer_question", wraps=vqa_engine.answer_question) as spy_vqa:
        # Test change endpoint
        resp = client.post("/api/v1/analysis/change", json={
            "file_id_before": img_a,
            "file_id_after": img_a,
            "query": "What was changing in these images"
        })
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["change_detected"] is False
        assert "identical" in data["summary"].lower()
        assert spy_vqa.call_count == 0

        # Test grounded endpoint with dual images
        resp_g = client.post("/api/v1/analysis/grounded", json={
            "file_id": img_a,
            "file_id_after": img_a,
            "query": "What was changing in these images"
        })
        assert resp_g.status_code == 200, resp_g.text
        data_g = resp_g.json()
        assert data_g["analysis_type"] == "temporal_change"
        assert "identical" in data_g["answer"].lower()
        assert spy_vqa.call_count == 0


def test_4_identical_query_explain_the_images(setup_test_rasters):
    """Test 4: Before = A, After = A | Query = 'Explain the images' -> MUST return identical result!"""
    img_a, _ = setup_test_rasters
    with patch.object(vqa_engine, "answer_question", wraps=vqa_engine.answer_question) as spy_vqa:
        resp = client.post("/api/v1/analysis/grounded", json={
            "file_id": img_a,
            "file_id_after": img_a,
            "query": "Explain the images"
        })
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["analysis_type"] == "temporal_change"
        assert "identical" in data["answer"].lower()
        assert spy_vqa.call_count == 0


def test_5_identical_query_where_are_buildings(setup_test_rasters):
    """Test 5: Before = A, After = A | Query = 'Where are the buildings?' -> MUST return identical result!"""
    img_a, _ = setup_test_rasters
    with patch.object(vqa_engine, "answer_question", wraps=vqa_engine.answer_question) as spy_vqa:
        resp = client.post("/api/v1/analysis/grounded", json={
            "file_id": img_a,
            "file_id_after": img_a,
            "query": "Where are the buildings?"
        })
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["analysis_type"] == "temporal_change"
        assert "identical" in data["answer"].lower()
        assert spy_vqa.call_count == 0


def test_6_different_images_what_was_changing_routes_to_temporal(setup_test_rasters):
    """Test 6: Before = A, After = B (different) | Query = 'What was changing in these images' -> TEMPORAL ANALYSIS, NOT IDENTICAL!"""
    img_a, img_b = setup_test_rasters
    resp = client.post("/api/v1/analysis/change", json={
        "file_id_before": img_a,
        "file_id_after": img_b,
        "query": "What was changing in these images"
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()
    # Must NOT be identical images
    assert data["reason"] != "IDENTICAL_IMAGES"
    assert "identical" not in data["summary"].lower()


def test_7_different_images_where_are_buildings_preserves_grounding(setup_test_rasters):
    """Test 7: Before = A, After = B (different) | Query = 'Where are the buildings?' -> preserves grounding routing!"""
    img_a, img_b = setup_test_rasters
    resp = client.post("/api/v1/analysis/grounded", json={
        "file_id": img_a,
        "file_id_after": img_b,
        "query": "Where are the buildings?"
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["analysis_type"] == "grounding"
    assert data["target_concept"] == "building"


def test_8_zero_vqa_calls_across_all_identical_queries(setup_test_rasters):
    """Test 8: Strict verification that VQA is never called when Before == After for any query."""
    img_a, _ = setup_test_rasters
    test_queries = [
        "What changed between these images?",
        "What had changed between these two images overtime",
        "What was changing in these images",
        "What changed?",
        "Did anything change?",
        "Are there any differences?",
        "Compare these images",
        "What is different?",
        "Explain the changes",
        "How did the area change?",
        "What happened over time?",
        "Tell me about these images",
        "Explain the image",
        "Where are the buildings?",
    ]

    with patch.object(vqa_engine, "answer_question") as mock_vqa:
        for q in test_queries:
            resp = client.post("/api/v1/analysis/grounded", json={
                "file_id": img_a,
                "file_id_after": img_a,
                "query": q
            })
            assert resp.status_code == 200, f"Query '{q}' failed: {resp.text}"
            data = resp.json()
            assert "identical" in data["answer"].lower(), f"Query '{q}' failed to report identical"
            assert data["analysis_type"] == "temporal_change"

        # Assert zero calls to VQA across all 14 queries!
        assert mock_vqa.call_count == 0, f"VQA was called {mock_vqa.call_count} times on identical images!"
