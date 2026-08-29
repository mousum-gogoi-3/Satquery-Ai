import pytest
from pathlib import Path
from httpx import AsyncClient, ASGITransport

from backend.main import app
from backend.config import settings


@pytest.mark.asyncio
async def test_caption_detail_levels_mock(sample_geotiff_rgb: Path, tmp_data_dir):
    """Tests brief, normal, and detailed caption modes in mock VLM mode."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        with open(sample_geotiff_rgb, "rb") as f:
            upload_res = await client.post(
                "/api/v1/raster/upload",
                files={"file": ("detail_levels.tif", f, "image/tiff")}
            )
        assert upload_res.status_code == 201
        file_id = upload_res.json()["file_id"]

        # Brief mode
        res_brief = await client.post("/api/v1/analysis/caption", json={"file_id": file_id, "detail_level": "brief"})
        assert res_brief.status_code == 200
        caption_brief = res_brief.json()["caption"]

        # Normal mode
        res_normal = await client.post("/api/v1/analysis/caption", json={"file_id": file_id, "detail_level": "normal"})
        assert res_normal.status_code == 200
        caption_normal = res_normal.json()["caption"]

        # Detailed mode
        res_detailed = await client.post("/api/v1/analysis/caption", json={"file_id": file_id, "detail_level": "detailed"})
        assert res_detailed.status_code == 200
        caption_detailed = res_detailed.json()["caption"]

        # Verify length progression: brief < normal < detailed
        assert len(caption_brief) < len(caption_normal) < len(caption_detailed)


@pytest.mark.asyncio
async def test_caption_missing_file_id(tmp_data_dir):
    """Tests Scene Captioning endpoint with non-existent file_id returns HTTP 404."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        caption_res = await client.post(
            "/api/v1/analysis/caption",
            json={
                "file_id": "missing_file_9999",
                "detail_level": "detailed"
            }
        )
        assert caption_res.status_code == 404


@pytest.mark.asyncio
@pytest.mark.skipif(settings.USE_MOCK_VLM, reason="Real VLM mode disabled via SATQUERY_USE_MOCK_VLM=True")
async def test_caption_real_qwen_inference(sample_geotiff_rgb: Path, tmp_data_dir):
    """Lightweight integration test for real Qwen scene captioning across detail levels."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        with open(sample_geotiff_rgb, "rb") as f:
            upload_res = await client.post(
                "/api/v1/raster/upload",
                files={"file": ("real_caption.tif", f, "image/tiff")}
            )
        assert upload_res.status_code == 201
        file_id = upload_res.json()["file_id"]

        # Real Qwen brief caption
        res_brief = await client.post("/api/v1/analysis/caption", json={"file_id": file_id, "detail_level": "brief"})
        assert res_brief.status_code == 200
        caption_brief = res_brief.json()["caption"]

        # Real Qwen detailed caption
        res_detailed = await client.post("/api/v1/analysis/caption", json={"file_id": file_id, "detail_level": "detailed"})
        assert res_detailed.status_code == 200
        caption_detailed = res_detailed.json()["caption"]

        assert len(caption_brief) > 0
        assert len(caption_detailed) > 0
        assert isinstance(res_detailed.json()["detected_concepts"], list)

