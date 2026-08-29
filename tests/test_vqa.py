import pytest
from pathlib import Path
from httpx import AsyncClient, ASGITransport

from backend.main import app
from backend.config import settings


@pytest.mark.asyncio
async def test_vqa_valid_query(sample_geotiff_rgb: Path, tmp_data_dir):
    """Tests VQA endpoint with a valid uploaded raster file_id in mock mode."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        with open(sample_geotiff_rgb, "rb") as f:
            upload_res = await client.post(
                "/api/v1/raster/upload",
                files={"file": ("vqa_sample.tif", f, "image/tiff")}
            )
        assert upload_res.status_code == 201
        file_id = upload_res.json()["file_id"]

        vqa_res = await client.post(
            "/api/v1/analysis/vqa",
            json={
                "file_id": file_id,
                "question": "What type of land cover is visible in this satellite scene?"
            }
        )
        assert vqa_res.status_code == 200
        data = vqa_res.json()
        assert data["file_id"] == file_id
        assert data["question"] == "What type of land cover is visible in this satellite scene?"
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0
        assert data["confidence"] > 0.0
        assert data["processing_time_sec"] >= 0.0
        assert isinstance(data["model_name"], str)


@pytest.mark.asyncio
async def test_vqa_missing_file_id(tmp_data_dir):
    """Tests VQA endpoint with non-existent file_id returns HTTP 404."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        vqa_res = await client.post(
            "/api/v1/analysis/vqa",
            json={
                "file_id": "non_existent_id_12345",
                "question": "What is in this image?"
            }
        )
        assert vqa_res.status_code == 404


@pytest.mark.asyncio
async def test_vqa_empty_question(sample_geotiff_rgb: Path, tmp_data_dir):
    """Tests VQA endpoint with empty question returns HTTP 400."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        with open(sample_geotiff_rgb, "rb") as f:
            upload_res = await client.post(
                "/api/v1/raster/upload",
                files={"file": ("vqa_empty_q.tif", f, "image/tiff")}
            )
        assert upload_res.status_code == 201
        file_id = upload_res.json()["file_id"]

        vqa_res = await client.post(
            "/api/v1/analysis/vqa",
            json={
                "file_id": file_id,
                "question": "   "
            }
        )
        assert vqa_res.status_code == 400


@pytest.mark.asyncio
@pytest.mark.skipif(settings.USE_MOCK_VLM, reason="Real VLM mode disabled via SATQUERY_USE_MOCK_VLM=True")
async def test_vqa_real_qwen_inference(sample_geotiff_rgb: Path, tmp_data_dir):
    """Lightweight integration test for real Qwen VLM inference (runs only when USE_MOCK_VLM=False)."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        with open(sample_geotiff_rgb, "rb") as f:
            upload_res = await client.post(
                "/api/v1/raster/upload",
                files={"file": ("real_vqa.tif", f, "image/tiff")}
            )
        assert upload_res.status_code == 201
        file_id = upload_res.json()["file_id"]

        vqa_res = await client.post(
            "/api/v1/analysis/vqa",
            json={
                "file_id": file_id,
                "question": "What land cover types are visible?"
            }
        )
        assert vqa_res.status_code == 200
        assert len(vqa_res.json()["answer"]) > 0
