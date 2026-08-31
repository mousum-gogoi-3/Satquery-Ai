import pytest
from pathlib import Path
from httpx import AsyncClient, ASGITransport

from backend.main import app
from backend.config import settings


@pytest.mark.asyncio
async def test_health_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "OK"
        assert data["engine_ready"] is True


@pytest.mark.asyncio
async def test_upload_and_metadata_flow(sample_geotiff_rgb: Path, tmp_data_dir):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        with open(sample_geotiff_rgb, "rb") as f:
            response = await client.post(
                "/api/v1/raster/upload",
                files={"file": ("sample_rgb.tif", f, "image/tiff")}
            )

        assert response.status_code == 201
        upload_data = response.json()
        file_id = upload_data["file_id"]
        assert upload_data["metadata"]["width"] == 256
        assert upload_data["metadata"]["height"] == 256
        assert upload_data["metadata"]["band_count"] == 3
        assert len(upload_data["bands"]) == 3

        # Test metadata endpoint
        meta_res = await client.get(f"/api/v1/raster/{file_id}/metadata")
        assert meta_res.status_code == 200
        meta_data = meta_res.json()
        assert meta_data["file_id"] == file_id
        assert meta_data["metadata"]["width"] == 256

        # Test tiling endpoint
        tile_res = await client.post(
            "/api/v1/raster/tile",
            json={"file_id": file_id, "tile_size": 128, "overlap": 0}
        )
        assert tile_res.status_code == 200
        tile_data = tile_res.json()
        assert tile_data["total_tiles"] == 4


@pytest.mark.asyncio
async def test_upload_invalid_file_extension():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.post(
            "/api/v1/raster/upload",
            files={"file": ("script.py", b"print('invalid')", "text/x-python")}
        )
        assert response.status_code == 400


@pytest.mark.asyncio
async def test_upload_path_traversal_relative(sample_geotiff_rgb: Path, tmp_data_dir):
    """Tests that relative path traversal in filename is sanitized to base filename."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        with open(sample_geotiff_rgb, "rb") as f:
            response = await client.post(
                "/api/v1/raster/upload",
                files={"file": ("../../traversal_sample.tif", f, "image/tiff")}
            )
        assert response.status_code == 201
        data = response.json()
        assert data["filename"] == "traversal_sample.tif"
        saved_file = Path(data["file_path"])
        assert saved_file.parent.resolve() == settings.UPLOAD_DIR.resolve()
        assert saved_file.name.endswith("_traversal_sample.tif")


@pytest.mark.asyncio
async def test_upload_absolute_path_filename(sample_geotiff_rgb: Path, tmp_data_dir):
    """Tests that absolute path in upload filename is sanitized to base filename."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        with open(sample_geotiff_rgb, "rb") as f:
            response = await client.post(
                "/api/v1/raster/upload",
                files={"file": ("/etc/passwd_sample.tif", f, "image/tiff")}
            )
        assert response.status_code == 201
        data = response.json()
        assert data["filename"] == "passwd_sample.tif"
        saved_file = Path(data["file_path"])
        assert saved_file.parent.resolve() == settings.UPLOAD_DIR.resolve()


@pytest.mark.asyncio
async def test_upload_null_byte_filename(sample_geotiff_rgb: Path, tmp_data_dir):
    """Tests that null bytes in upload filename are rejected with HTTP 400."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        with open(sample_geotiff_rgb, "rb") as f:
            response = await client.post(
                "/api/v1/raster/upload",
                files={"file": ("null%00byte.tif", f, "image/tiff")}
            )
        assert response.status_code == 400


@pytest.mark.asyncio
async def test_upload_oversized_file(tmp_data_dir):
    """Tests that uploading a file exceeding MAX_FILE_SIZE_BYTES returns HTTP 413."""
    orig_max = settings.MAX_FILE_SIZE_BYTES
    settings.MAX_FILE_SIZE_BYTES = 100  # 100 bytes limit
    try:
        oversized_data = b"X" * 500
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            response = await client.post(
                "/api/v1/raster/upload",
                files={"file": ("large.tif", oversized_data, "image/tiff")}
            )
        assert response.status_code == 413
    finally:
        settings.MAX_FILE_SIZE_BYTES = orig_max


@pytest.mark.asyncio
async def test_upload_partial_file_cleanup(tmp_data_dir):
    """Tests that any partially written file is removed if upload fails mid-stream."""
    orig_max = settings.MAX_FILE_SIZE_BYTES
    settings.MAX_FILE_SIZE_BYTES = 50  # 50 bytes limit
    try:
        oversized_data = b"A" * 200
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            response = await client.post(
                "/api/v1/raster/upload",
                files={"file": ("partial_fail.tif", oversized_data, "image/tiff")}
            )
        assert response.status_code == 413
        # Verify no temporary files remain in UPLOAD_DIR
        uploaded_files = list(settings.UPLOAD_DIR.glob("*partial_fail.tif"))
        assert len(uploaded_files) == 0
    finally:
        settings.MAX_FILE_SIZE_BYTES = orig_max


@pytest.mark.asyncio
async def test_preview_path_traversal(tmp_data_dir):
    """Tests that path traversal attempts in preview endpoint are blocked or handled securely."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        res1 = await client.get("/api/v1/raster/preview/../../secret.png")
        assert res1.status_code in (400, 404)

        res2 = await client.get("/api/v1/raster/preview/null%00byte.png")
        assert res2.status_code == 400


@pytest.mark.asyncio
async def test_valid_upload_and_preview_behavior(sample_geotiff_rgb: Path, tmp_data_dir):
    """Tests complete flow of valid upload and thumbnail preview retrieval."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        with open(sample_geotiff_rgb, "rb") as f:
            upload_res = await client.post(
                "/api/v1/raster/upload",
                files={"file": ("valid_preview_test.tif", f, "image/tiff")}
            )
        assert upload_res.status_code == 201
        data = upload_res.json()
        assert data["thumbnail_url"] is not None

        # Fetch preview image via thumbnail_url
        preview_res = await client.get(data["thumbnail_url"])
        assert preview_res.status_code == 200
        assert preview_res.headers["content-type"] == "image/png"
        assert len(preview_res.content) > 0


@pytest.mark.asyncio
async def test_grounded_analysis_endpoint_valid(sample_geotiff_rgb: Path, tmp_data_dir):
    """Tests POST /api/v1/analysis/grounded with a valid uploaded raster file_id."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        with open(sample_geotiff_rgb, "rb") as f:
            upload_res = await client.post(
                "/api/v1/raster/upload",
                files={"file": ("grounding_sample.tif", f, "image/tiff")}
            )
        assert upload_res.status_code == 201
        file_id = upload_res.json()["file_id"]

        ground_res = await client.post(
            "/api/v1/analysis/grounded",
            json={
                "file_id": file_id,
                "query": "Where are the buildings and water bodies?"
            }
        )
        assert ground_res.status_code == 200
        data = ground_res.json()
        assert data["file_id"] == file_id
        assert data["query"] == "Where are the buildings and water bodies?"
        assert isinstance(data["answer"], str)
        assert isinstance(data["evidence_regions"], list)
        assert len(data["evidence_regions"]) >= 0
        assert data["overall_confidence"] >= 0.0
        assert data["evidence_image_path"] is not None
        assert data["source_image_dims"]["width"] == 256
        assert data["source_image_dims"]["height"] == 256
        assert data["processing_time_sec"] >= 0.0


@pytest.mark.asyncio
async def test_grounded_analysis_endpoint_missing_file_id(tmp_data_dir):
    """Tests POST /api/v1/analysis/grounded with non-existent file_id returns HTTP 404."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        res = await client.post(
            "/api/v1/analysis/grounded",
            json={"file_id": "non_existent_id_99999", "query": "Where are the buildings?"}
        )
        assert res.status_code == 404


@pytest.mark.asyncio
async def test_grounded_analysis_endpoint_empty_query(sample_geotiff_rgb: Path, tmp_data_dir):
    """Tests POST /api/v1/analysis/grounded with empty query returns HTTP 400."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        with open(sample_geotiff_rgb, "rb") as f:
            upload_res = await client.post(
                "/api/v1/raster/upload",
                files={"file": ("empty_q_sample.tif", f, "image/tiff")}
            )
        assert upload_res.status_code == 201
        file_id = upload_res.json()["file_id"]

        res = await client.post(
            "/api/v1/analysis/grounded",
            json={"file_id": file_id, "query": "   "}
        )
        assert res.status_code == 400


@pytest.mark.asyncio
async def test_grounded_analysis_endpoint_path_traversal_file_id(tmp_data_dir):
    """Tests POST /api/v1/analysis/grounded with invalid file_id returns HTTP 400."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        res = await client.post(
            "/api/v1/analysis/grounded",
            json={"file_id": "../traversal_id", "query": "Where is the vegetation?"}
        )
        assert res.status_code == 400


