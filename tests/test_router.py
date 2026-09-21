import pytest
from pathlib import Path
from httpx import AsyncClient, ASGITransport

from backend.main import app
from backend.models.router import SatQueryRouter, QueryType


def test_router_general_image_understanding_queries():
    """Verify that general image understanding and scene description queries are routed correctly."""
    queries = [
        "Explain this image",
        "Describe this image",
        "What is in this image?",
        "What does this image show?",
        "Give me an overview of this scene",
        "Describe the satellite image",
        "What can you see?",
        "Analyze this image",
        "Tell me about this area",
        "Summary of this scene"
    ]
    for q in queries:
        result = SatQueryRouter.classify_query(q)
        assert result.query_type == QueryType.IMAGE_UNDERSTANDING, f"Failed for '{q}': got {result.query_type}"
        assert result.target_concept is None


def test_router_spatial_grounding_queries():
    """Verify that object location and spatial grounding queries are routed correctly with canonical concepts."""
    cases = [
        ("Where is the building?", "building"),
        ("Where are the buildings?", "building"),
        ("Locate the buildings.", "building"),
        ("Where is the road?", "road"),
        ("Where are the roads?", "road"),
        ("Locate the roads", "road"),
        ("Show me the water bodies.", "water"),
        ("Where is the water?", "water"),
        ("Find the vehicles.", "vehicle"),
        ("Where is the forest?", "vegetation"),
        ("Locate the airport.", "airport"),
        ("Where is the agricultural land?", "agriculture"),
    ]
    for q, expected_concept in cases:
        result = SatQueryRouter.classify_query(q)
        assert result.query_type == QueryType.GROUNDING, f"Failed for '{q}': got {result.query_type}"
        assert result.target_concept == expected_concept, f"Concept mismatch for '{q}': got {result.target_concept}, expected {expected_concept}"


def test_router_counting_queries():
    """Verify that counting queries are routed to COUNTING with proper target concepts."""
    cases = [
        ("How many buildings are there?", "building"),
        ("How many vehicles can you see?", "vehicle"),
        ("How many water bodies are present?", "water"),
        ("Count the buildings.", "building"),
        ("How many roads are visible?", "road"),
    ]
    for q, expected_concept in cases:
        result = SatQueryRouter.classify_query(q)
        assert result.query_type == QueryType.COUNTING, f"Failed for '{q}': got {result.query_type}"
        assert result.target_concept == expected_concept, f"Concept mismatch for '{q}': got {result.target_concept}, expected {expected_concept}"


def test_router_temporal_change_queries():
    """Verify that bi-temporal change queries are routed to TEMPORAL_CHANGE."""
    queries = [
        "What changed?",
        "What has changed?",
        "Where did vegetation decrease?",
        "What changed between these images?",
        "Show me the new buildings.",
        "Where did construction occur?",
        "Changes between observations"
    ]
    for q in queries:
        result = SatQueryRouter.classify_query(q)
        assert result.query_type == QueryType.TEMPORAL_CHANGE, f"Failed for '{q}': got {result.query_type}"


@pytest.mark.asyncio
async def test_api_explain_this_image_no_grounding(sample_geotiff_rgb: Path, tmp_data_dir):
    """
    Critical Test: Ensure 'Explain this image' via /api/v1/analysis/grounded
    produces a descriptive scene answer with ZERO bounding boxes and NO evidence image.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        with open(sample_geotiff_rgb, "rb") as f:
            upload_res = await client.post(
                "/api/v1/raster/upload",
                files={"file": ("explain_test.tif", f, "image/tiff")}
            )
        assert upload_res.status_code == 201
        file_id = upload_res.json()["file_id"]

        # Call grounded endpoint with "Explain this image"
        res = await client.post(
            "/api/v1/analysis/grounded",
            json={
                "file_id": file_id,
                "query": "Explain this image"
            }
        )
        assert res.status_code == 200
        data = res.json()

        # Assert correct classification and response behavior
        assert data["analysis_type"] == "image_understanding"
        assert data["evidence_regions"] == []
        assert data["evidence_image_path"] is None
        assert "EXPLAIN THIS IMAGE" not in data["answer"]
        assert len(data["answer"]) > 20


@pytest.mark.asyncio
async def test_api_where_is_building_uses_grounding(sample_geotiff_rgb: Path, tmp_data_dir):
    """
    Ensure 'Where is the building?' via /api/v1/analysis/grounded
    routes to grounding, identifies target concept 'building', and renders evidence.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        with open(sample_geotiff_rgb, "rb") as f:
            upload_res = await client.post(
                "/api/v1/raster/upload",
                files={"file": ("building_test.tif", f, "image/tiff")}
            )
        assert upload_res.status_code == 201
        file_id = upload_res.json()["file_id"]

        res = await client.post(
            "/api/v1/analysis/grounded",
            json={
                "file_id": file_id,
                "query": "Where is the building?"
            }
        )
        assert res.status_code == 200
        data = res.json()

        assert data["analysis_type"] == "grounding"
        assert data["target_concept"] == "building"
        assert data["evidence_image_path"] is not None
