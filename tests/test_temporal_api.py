import pytest
from pathlib import Path
import numpy as np
import tifffile
from httpx import AsyncClient, ASGITransport
from PIL import Image

from backend.main import app
from backend.config import settings
from backend.api.schemas import ChangeType, RegistrationQuality


@pytest.fixture
def api_temporal_pair_vegetation(tmp_path: Path):
    """Generates two 4-band GeoTIFFs showing vegetation loss."""
    path_b = tmp_path / "api_before_veg.tif"
    path_a = tmp_path / "api_after_veg.tif"

    h, w = 256, 256
    before = np.full((4, h, w), 600, dtype=np.uint16)
    # High NIR vegetation in before at [60..160, 60..160]
    before[0, 60:160, 60:160] = 300
    before[1, 60:160, 60:160] = 700
    before[2, 60:160, 60:160] = 300
    before[3, 60:160, 60:160] = 3500

    after = np.full((4, h, w), 600, dtype=np.uint16)
    # Deforested cleared soil in after at [60..160, 60..160]
    after[0, 60:160, 60:160] = 2500
    after[1, 60:160, 60:160] = 1800
    after[2, 60:160, 60:160] = 1500
    after[3, 60:160, 60:160] = 900

    extratags = [
        (33550, 'd', 3, (10.0, 10.0, 0.0)),
        (33922, 'd', 6, (0.0, 0.0, 0.0, 500000.0, 6000000.0, 0.0)),
        (34735, 'h', 8, (1, 1, 0, 1, 3072, 0, 1, 32633)),
    ]

    tifffile.imwrite(path_b, before, extratags=extratags, planarconfig='SEPARATE')
    tifffile.imwrite(path_a, after, extratags=extratags, planarconfig='SEPARATE')

    return path_b, path_a


@pytest.fixture
def api_temporal_pair_disjoint(tmp_path: Path):
    """Generates two non-overlapping GeoTIFFs."""
    path_b = tmp_path / "api_disjoint_b.tif"
    path_a = tmp_path / "api_disjoint_a.tif"

    h, w = 128, 128
    data = np.full((3, h, w), 120, dtype=np.uint8)

    tags_b = [
        (33550, 'd', 3, (10.0, 10.0, 0.0)),
        (33922, 'd', 6, (0.0, 0.0, 0.0, 500000.0, 6000000.0, 0.0)),
        (34735, 'h', 8, (1, 1, 0, 1, 3072, 0, 1, 32633)),
    ]
    tags_a = [
        (33550, 'd', 3, (10.0, 10.0, 0.0)),
        (33922, 'd', 6, (0.0, 0.0, 0.0, 800000.0, 8000000.0, 0.0)),
        (34735, 'h', 8, (1, 1, 0, 1, 3072, 0, 1, 32633)),
    ]

    tifffile.imwrite(path_b, data, extratags=tags_b, planarconfig='SEPARATE')
    tifffile.imwrite(path_a, data, extratags=tags_a, planarconfig='SEPARATE')

    return path_b, path_a


@pytest.mark.asyncio
async def test_temporal_api_success_flow(api_temporal_pair_vegetation, tmp_data_dir):
    """Tests POST /api/v1/analysis/change end-to-end flow with vegetation change."""
    path_b, path_a = api_temporal_pair_vegetation

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        # 1. Upload Before
        with open(path_b, "rb") as fb:
            res_b = await client.post("/api/v1/raster/upload", files={"file": ("before.tif", fb, "image/tiff")})
        assert res_b.status_code == 201
        id_b = res_b.json()["file_id"]

        # 2. Upload After
        with open(path_a, "rb") as fa:
            res_a = await client.post("/api/v1/raster/upload", files={"file": ("after.tif", fa, "image/tiff")})
        assert res_a.status_code == 201
        id_a = res_a.json()["file_id"]

        # 3. Request Bi-Temporal Change Analysis
        res_change = await client.post(
            "/api/v1/analysis/change",
            json={
                "file_id_before": id_b,
                "file_id_after": id_a,
                "query": "Where did vegetation decrease between these images?"
            }
        )

        assert res_change.status_code == 200
        data = res_change.json()
        assert data["file_id_before"] == id_b
        assert data["file_id_after"] == id_a
        assert data["status"] == "SUCCESS"
        assert len(data["change_regions"]) >= 1
        assert data["dominant_change_type"] == ChangeType.VEGETATION_LOSS.value
        assert data["total_changed_area_pixels"] > 0
        assert data["evidence_image_path"] is not None
        assert data["change_heatmap_path"] is not None
        assert data["registration_diagnostics"]["overlap_percentage"] >= 90.0

        # Verify evidence image exists on disk
        ev_file = settings.BASE_DIR / data["evidence_image_path"]
        assert ev_file.exists()


@pytest.mark.asyncio
async def test_temporal_api_no_change_flow(api_temporal_pair_vegetation, tmp_data_dir):
    """Tests POST /api/v1/analysis/change with identical before and after images."""
    path_b, _ = api_temporal_pair_vegetation

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        with open(path_b, "rb") as fb:
            res_b = await client.post("/api/v1/raster/upload", files={"file": ("same1.tif", fb, "image/tiff")})
        id_b = res_b.json()["file_id"]

        with open(path_b, "rb") as fb2:
            res_b2 = await client.post("/api/v1/raster/upload", files={"file": ("same2.tif", fb2, "image/tiff")})
        id_b2 = res_b2.json()["file_id"]

        res_change = await client.post(
            "/api/v1/analysis/change",
            json={
                "file_id_before": id_b,
                "file_id_after": id_b2,
                "query": "What changed?"
            }
        )

        assert res_change.status_code == 200
        data = res_change.json()
        assert data["status"] in ("NO_CHANGE", "NO_SIGNIFICANT_CHANGE")
        assert len(data["change_regions"]) == 0
        assert data["total_changed_area_pixels"] == 0


@pytest.mark.asyncio
async def test_temporal_api_disjoint_flow(api_temporal_pair_disjoint, tmp_data_dir):
    """Tests POST /api/v1/analysis/change with non-overlapping geographical rasters."""
    path_b, path_a = api_temporal_pair_disjoint

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        with open(path_b, "rb") as fb:
            res_b = await client.post("/api/v1/raster/upload", files={"file": ("disjoint_b.tif", fb, "image/tiff")})
        id_b = res_b.json()["file_id"]

        with open(path_a, "rb") as fa:
            res_a = await client.post("/api/v1/raster/upload", files={"file": ("disjoint_a.tif", fa, "image/tiff")})
        id_a = res_a.json()["file_id"]

        res_change = await client.post(
            "/api/v1/analysis/change",
            json={
                "file_id_before": id_b,
                "file_id_after": id_a,
                "query": "What changed?"
            }
        )

        assert res_change.status_code == 200
        data = res_change.json()
        assert data["status"] in ("INCOMPATIBLE_IMAGES", "NO_SPATIAL_OVERLAP")
        assert len(data["change_regions"]) == 0


@pytest.mark.asyncio
async def test_temporal_api_missing_file_id(tmp_data_dir):
    """Tests POST /api/v1/analysis/change with invalid or missing file_id returns 404."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        res = await client.post(
            "/api/v1/analysis/change",
            json={"file_id_before": "non_existent_1", "file_id_after": "non_existent_2", "query": "What changed?"}
        )
        assert res.status_code == 404


@pytest.mark.asyncio
async def test_temporal_api_empty_query(api_temporal_pair_vegetation, tmp_data_dir):
    """Tests POST /api/v1/analysis/change with empty query returns 400."""
    path_b, path_a = api_temporal_pair_vegetation
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        with open(path_b, "rb") as fb:
            res_b = await client.post("/api/v1/raster/upload", files={"file": ("q_b.tif", fb, "image/tiff")})
        id_b = res_b.json()["file_id"]

        with open(path_a, "rb") as fa:
            res_a = await client.post("/api/v1/raster/upload", files={"file": ("q_a.tif", fa, "image/tiff")})
        id_a = res_a.json()["file_id"]

        res = await client.post(
            "/api/v1/analysis/change",
            json={"file_id_before": id_b, "file_id_after": id_a, "query": "   "}
        )
        assert res.status_code == 400
