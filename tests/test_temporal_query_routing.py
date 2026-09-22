import io
import numpy as np
import pytest
from PIL import Image
import tifffile
from fastapi.testclient import TestClient

from backend.main import app
from backend.config import settings
from backend.models.router import SatQueryRouter, QueryType

client = TestClient(app)


def _create_synthetic_geotiff(filename: str, width: int = 128, height: int = 128, seed: int = 42) -> str:
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
    img_a_name = "test_routing_raster_a.tif"
    img_b_name = "test_routing_raster_b.tif"
    _create_synthetic_geotiff(img_a_name, seed=42)
    _create_synthetic_geotiff(img_b_name, seed=99)
    yield img_a_name
    for name in [img_a_name, img_b_name]:
        p = settings.UPLOAD_DIR / name
        if p.exists():
            p.unlink()


def test_router_unit_temporal_queries():
    """Verify that all temporal intent variations classify to TEMPORAL_CHANGE."""
    temporal_queries = [
        "What changed between these images?",
        "What had changed between these two images overtime",
        "What has changed over time?",
        "What changed over time?",
        "Did anything change over time?",
        "What is different between these images?",
        "What are the differences between the before and after images?",
        "Compare these two images.",
        "Compare the before and after images.",
        "Compare these images and tell me what changed.",
        "How did the scene change?",
        "How has this area changed?",
        "What changed between the before and after images?",
        "Tell me what changed over time between these images.",
        "Did construction occur between these dates?",
        "Where did vegetation decrease?",
    ]
    for q in temporal_queries:
        res = SatQueryRouter.classify_query(q, has_second_image=True)
        assert res.query_type == QueryType.TEMPORAL_CHANGE, f"Query '{q}' failed to route to TEMPORAL_CHANGE"


def test_router_unit_non_temporal_queries():
    """Verify non-temporal queries route to IMAGE_UNDERSTANDING or GROUNDING."""
    # Explanations without temporal intent
    res_explain = SatQueryRouter.classify_query("Explain the image", has_second_image=True)
    assert res_explain.query_type == QueryType.IMAGE_UNDERSTANDING

    res_describe = SatQueryRouter.classify_query("Describe this satellite scene", has_second_image=True)
    assert res_describe.query_type == QueryType.IMAGE_UNDERSTANDING

    # Single-image grounding
    res_ground = SatQueryRouter.classify_query("Where is the building?", has_second_image=False)
    assert res_ground.query_type == QueryType.GROUNDING
    assert res_ground.target_concept == "building"

    # Counting
    res_count = SatQueryRouter.classify_query("How many buildings are there?", has_second_image=False)
    assert res_count.query_type == QueryType.COUNTING


def test_1_identical_images_query1(setup_test_rasters):
    """Test 1: Query 1 'What changed between these images?' on identical images."""
    file_id = "test_routing_raster_a.tif"
    resp = client.post("/api/v1/analysis/change", json={
        "file_id_before": file_id,
        "file_id_after": file_id,
        "query": "What changed between these images?"
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["change_detected"] is False
    assert data["status"] in ("NO_CHANGE", "NO_SIGNIFICANT_CHANGE")
    assert "identical" in data["summary"].lower()


def test_2_identical_images_query2_the_reported_bug(setup_test_rasters):
    """
    Test 2: EXACT BUG REPORTED BY USER.
    'What had changed between these two images overtime'
    Must route to Change Analysis, NOT VQA, and report no change for identical images.
    """
    file_id = "test_routing_raster_a.tif"
    resp = client.post("/api/v1/analysis/change", json={
        "file_id_before": file_id,
        "file_id_after": file_id,
        "query": "What had changed between these two images overtime"
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["change_detected"] is False
    assert data["status"] in ("NO_CHANGE", "NO_SIGNIFICANT_CHANGE")
    assert "identical" in data["summary"].lower()


def test_3_identical_images_query3_over_time(setup_test_rasters):
    """Test 3: 'What has changed over time?' on identical images."""
    file_id = "test_routing_raster_a.tif"
    resp = client.post("/api/v1/analysis/change", json={
        "file_id_before": file_id,
        "file_id_after": file_id,
        "query": "What has changed over time?"
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["change_detected"] is False
    assert data["status"] in ("NO_CHANGE", "NO_SIGNIFICANT_CHANGE")


def test_4_identical_images_differences_query(setup_test_rasters):
    """Test 4: 'What are the differences between these images?'"""
    file_id = "test_routing_raster_a.tif"
    resp = client.post("/api/v1/analysis/change", json={
        "file_id_before": file_id,
        "file_id_after": file_id,
        "query": "What are the differences between these images?"
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["change_detected"] is False
    assert data["status"] in ("NO_CHANGE", "NO_SIGNIFICANT_CHANGE")


def test_5_identical_images_compare_query(setup_test_rasters):
    """Test 5: 'Compare these before and after images'"""
    file_id = "test_routing_raster_a.tif"
    resp = client.post("/api/v1/analysis/change", json={
        "file_id_before": file_id,
        "file_id_after": file_id,
        "query": "Compare these before and after images"
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["change_detected"] is False
    assert data["status"] in ("NO_CHANGE", "NO_SIGNIFICANT_CHANGE")


def test_6_one_image_explain_scene_understanding(setup_test_rasters):
    """Test 6: One image only + 'Explain the image' -> image_understanding."""
    file_id = "test_routing_raster_a.tif"
    resp = client.post("/api/v1/analysis/grounded", json={
        "file_id": file_id,
        "query": "Explain the image"
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["analysis_type"] == "image_understanding"
    assert data["model_info"]["router_decision"] == "image_understanding"


def test_7_one_image_grounding_query(setup_test_rasters):
    """Test 7: One image only + 'Where is the building?' -> grounding."""
    file_id = "test_routing_raster_a.tif"
    resp = client.post("/api/v1/analysis/grounded", json={
        "file_id": file_id,
        "query": "Where is the building?"
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["analysis_type"] == "grounding"
    assert data["target_concept"] == "building"


def test_8_two_images_explain_does_not_become_temporal(setup_test_rasters):
    """
    Test 8: Two DIFFERENT images uploaded, query = 'Explain the image'.
    Must NOT automatically become temporal analysis because there is no temporal intent.
    """
    file_id_a = "test_routing_raster_a.tif"
    file_id_b = "test_routing_raster_b.tif"
    resp = client.post("/api/v1/analysis/grounded", json={
        "file_id": file_id_a,
        "file_id_after": file_id_b,
        "query": "Explain the image"
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()
    # No temporal intent in query, so remains image_understanding
    assert data["analysis_type"] == "image_understanding"


def test_9_grounded_endpoint_with_dual_images_executes_change(setup_test_rasters):
    """
    Test 9: Call grounded endpoint with two images and reported temporal query.
    Must route to temporal_change and execute BiTemporalAnalysisEngine / guard.
    """
    file_id = "test_routing_raster_a.tif"
    resp = client.post("/api/v1/analysis/grounded", json={
        "file_id": file_id,
        "file_id_after": file_id,
        "query": "What had changed between these two images overtime"
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["analysis_type"] == "temporal_change"
    assert data["model_info"]["router_decision"] in ("identical_pair_guard", "temporal_change")
    assert data["model_info"]["model_name"] == "BiTemporalAnalysisEngine"
    assert "identical" in data["answer"].lower()


def test_10_vqa_guard_blocks_temporal_query_on_single_image(setup_test_rasters):
    """
    Test 10: Calling /vqa directly with a temporal comparison query must be blocked
    by the routing guard rather than letting VQA hallucinate changes.
    """
    file_id = "test_routing_raster_a.tif"
    resp = client.post("/api/v1/analysis/vqa", json={
        "file_id": file_id,
        "question": "What had changed between these two images overtime"
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["model_name"] == "routing_guard"
    assert data["image_used"] is False
    assert "two satellite observations" in data["answer"]


def test_11_grounded_guard_blocks_temporal_query_without_second_image(setup_test_rasters):
    """
    Test 11: Calling /grounded with a temporal query but NO second image.
    Must return temporal_change notice without running VQA.
    """
    file_id = "test_routing_raster_a.tif"
    resp = client.post("/api/v1/analysis/grounded", json={
        "file_id": file_id,
        "query": "What had changed between these two images overtime"
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["analysis_type"] == "temporal_change"
    assert "two satellite observations" in data["answer"]
