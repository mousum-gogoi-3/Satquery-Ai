import pytest
from pathlib import Path
import numpy as np
import tifffile
from PIL import Image
from httpx import AsyncClient, ASGITransport

from backend.main import app
from backend.config import settings
from backend.models.temporal_engine import BiTemporalAnalysisEngine
from backend.remote_sensing.pair_validator import TemporalPairValidator
from backend.api.schemas import PairClassification, ChangeType, RegistrationQuality


def create_test_geotiff(
    file_path: Path,
    data: np.ndarray,
    origin_x: float = 500000.0,
    origin_y: float = 6000000.0,
    res_x: float = 10.0,
    res_y: float = 10.0,
    epsg: int = 32633
) -> Path:
    """Helper to create test GeoTIFF with explicit tags."""
    tags = [
        (33550, 'd', 3, (res_x, res_y, 0.0)),
        (33922, 'd', 6, (0.0, 0.0, 0.0, origin_x, origin_y, 0.0)),
        (34735, 'h', 8, (1, 1, 0, 1, 3072, 0, 1, epsg)),
    ]
    kwargs = {"extratags": tags}
    if data.ndim == 3 and data.shape[0] > 1:
        kwargs["planarconfig"] = 'SEPARATE'
    tifffile.imwrite(file_path, data, **kwargs)
    return file_path


def test_validation_test1_exact_same_image(tmp_path: Path):
    """
    Section 12 Test 1 — Exact same image:
    before = image A, after = image A.
    Expected:
      - change_detected = False
      - status = NO_CHANGE
      - changed_pixels = 0
      - change_regions = []
      - evidence_image_path is None
      - change_heatmap_path is None
    """
    h, w = 128, 128
    data = np.full((3, h, w), 140, dtype=np.uint8)
    data[:, 30:70, 30:70] = 220
    img_path = tmp_path / "scene_a.tif"
    create_test_geotiff(img_path, data)

    engine = BiTemporalAnalysisEngine()
    res = engine.analyze_change(
        before_path=img_path,
        after_path=img_path,
        file_id_before="img_a",
        file_id_after="img_a",
        query="What changed?"
    )

    assert res.status == "NO_CHANGE"
    assert res.change_detected is False
    assert res.reason == "IDENTICAL_IMAGES"
    assert res.total_changed_area_pixels == 0
    assert res.total_changed_area_percentage == 0.0
    assert len(res.change_regions) == 0
    assert res.evidence_image_path is None
    assert res.change_heatmap_path is None
    assert res.pair_validation is not None
    assert res.pair_validation.classification == PairClassification.IDENTICAL_IMAGES
    assert res.pair_validation.exact_hash_match is True
    assert res.statistics.changed_pixels == 0


@pytest.mark.asyncio
async def test_validation_test2_same_image_different_uuids(tmp_path: Path, tmp_data_dir):
    """
    Section 12 Test 2 — Same image uploaded twice with different file IDs (UUIDs):
    Verify that differing UUIDs/filenames do NOT affect duplicate detection.
    """
    h, w = 128, 128
    data = np.full((4, h, w), 1000, dtype=np.uint16)
    data[:, 20:60, 20:60] = 2500
    img_path = tmp_path / "original_source.tif"
    create_test_geotiff(img_path, data)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        # First upload
        with open(img_path, "rb") as f1:
            r1 = await client.post("/api/v1/raster/upload", files={"file": ("first_upload.tif", f1, "image/tiff")})
        assert r1.status_code == 201
        uuid_1 = r1.json()["file_id"]

        # Second upload of the exact same content with different filename
        with open(img_path, "rb") as f2:
            r2 = await client.post("/api/v1/raster/upload", files={"file": ("second_upload.tif", f2, "image/tiff")})
        assert r2.status_code == 201
        uuid_2 = r2.json()["file_id"]

        assert uuid_1 != uuid_2, "UUIDs must be distinct"

        res = await client.post(
            "/api/v1/analysis/change",
            json={"file_id_before": uuid_1, "file_id_after": uuid_2, "query": "What changed?"}
        )
        assert res.status_code == 200
        data_res = res.json()

        assert data_res["status"] == "NO_CHANGE"
        assert data_res["change_detected"] is False
        assert data_res["reason"] == "IDENTICAL_IMAGES"
        assert data_res["total_changed_area_pixels"] == 0
        assert len(data_res["change_regions"]) == 0
        assert data_res["pair_validation"]["exact_hash_match"] is True
        assert data_res["evidence_image_path"] is None
        assert data_res["change_heatmap_path"] is None


def test_validation_test3_near_identical_image(tmp_path: Path):
    """
    Section 12 Test 3 — Near-identical image:
    Create an image with tiny numerical/compression/noise differences.
    Expected:
      - status = NO_SIGNIFICANT_CHANGE
      - change_detected = False
      - change_regions = []
      - changed_pixels = 0
    """
    h, w = 128, 128
    np.random.seed(42)
    base = np.full((3, h, w), 120, dtype=np.uint8)
    base[:, 30:70, 30:70] = 200

    # Near-identical with tiny numerical jitter (+-1 value on a few pixels, simulating compression)
    jitter = base.copy()
    noise = np.random.choice([-1, 0, 1], size=base.shape, p=[0.05, 0.90, 0.05])
    jitter = np.clip(jitter.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    pb = create_test_geotiff(tmp_path / "near_b.tif", base)
    pa = create_test_geotiff(tmp_path / "near_a.tif", jitter)

    engine = BiTemporalAnalysisEngine()
    res = engine.analyze_change(
        before_path=pb,
        after_path=pa,
        file_id_before="near_b",
        file_id_after="near_a",
        query="What changed?"
    )

    assert res.status == "NO_SIGNIFICANT_CHANGE"
    assert res.change_detected is False
    assert res.total_changed_area_pixels == 0
    assert len(res.change_regions) == 0
    assert res.evidence_image_path is None
    assert res.change_heatmap_path is None


def test_validation_test4_genuine_temporal_change(tmp_path: Path):
    """
    Section 12 Test 4 — Genuine temporal change:
    Same scene + synthetic building/land-cover change.
    Expected:
      - pair_validation classification = COMPATIBLE_TEMPORAL_PAIR
      - status = SUCCESS
      - change_detected = True
      - change_regions >= 1
      - total_changed_area_pixels > 0
    """
    h, w = 128, 128
    before = np.full((4, h, w), 600, dtype=np.uint16)
    before[0, 30:70, 30:70] = 300
    before[3, 30:70, 30:70] = 3200

    after = before.copy()
    # Synthetic clearing/building: Red goes up, NIR drops sharply
    after[0, 30:70, 30:70] = 2400
    after[3, 30:70, 30:70] = 700

    pb = create_test_geotiff(tmp_path / "genuine_b.tif", before)
    pa = create_test_geotiff(tmp_path / "genuine_a.tif", after)

    engine = BiTemporalAnalysisEngine()
    res = engine.analyze_change(
        before_path=pb,
        after_path=pa,
        file_id_before="gen_b",
        file_id_after="gen_a",
        query="What changed?"
    )

    assert res.status == "SUCCESS"
    assert res.change_detected is True
    assert res.pair_validation.classification == PairClassification.COMPATIBLE_TEMPORAL_PAIR
    assert res.total_changed_area_pixels > 0
    assert len(res.change_regions) >= 1
    assert res.evidence_image_path is not None
    assert res.change_heatmap_path is not None


def test_validation_test5_completely_different_images(tmp_path: Path):
    """
    Section 12 Test 5 — Completely different images:
    Two unrelated scenes without spatial overlap.
    Expected:
      - status = INCOMPATIBLE_IMAGES
      - change_detected = False
      - change_regions = []
      - changed_pixels = 0
      - No false evidence boxes or heatmaps
    """
    # Scene 1: Urban grid (non-geospatial PNG)
    img_b = np.zeros((200, 200, 3), dtype=np.uint8)
    for i in range(20, 180, 30):
        img_b[i:i+15, :, :] = 200
        img_b[:, i:i+15, :] = 200

    # Scene 2: Natural water body / smooth terrain (completely different)
    img_a = np.full((200, 200, 3), 40, dtype=np.uint8)
    img_a[:, :, 0] = 180  # blue
    img_a[:, :, 1] = 90

    pb = tmp_path / "unrelated_urban.png"
    pa = tmp_path / "unrelated_water.png"
    Image.fromarray(img_b).save(pb)
    Image.fromarray(img_a).save(pa)

    engine = BiTemporalAnalysisEngine()
    res = engine.analyze_change(
        before_path=pb,
        after_path=pa,
        file_id_before="urban",
        file_id_after="water",
        query="What changed?"
    )

    assert res.status == "INCOMPATIBLE_IMAGES"
    assert res.change_detected is False
    assert len(res.change_regions) == 0
    assert res.total_changed_area_pixels == 0
    assert res.pair_validation.classification == PairClassification.INCOMPATIBLE_IMAGES
    assert res.pair_validation.spatially_compatible is False
    assert res.evidence_image_path is None
    assert res.change_heatmap_path is None
    assert "different scenes" in res.summary.lower()


def test_validation_test6_different_dimensions_same_scene(tmp_path: Path):
    """
    Section 12 Test 6 — Different dimensions but same scene:
    Verify that valid temporal pairs with different dimensions align and analyze correctly.
    """
    before = np.full((4, 120, 140), 600, dtype=np.uint16)
    after = np.full((4, 150, 130), 600, dtype=np.uint16)

    # Change region
    before[0, 20:50, 20:50] = 200
    before[3, 20:50, 20:50] = 3000
    after[0, 20:50, 20:50] = 2200
    after[3, 20:50, 20:50] = 700

    pb = create_test_geotiff(tmp_path / "dim_b.tif", before)
    pa = create_test_geotiff(tmp_path / "dim_a.tif", after)

    engine = BiTemporalAnalysisEngine()
    res = engine.analyze_change(
        before_path=pb,
        after_path=pa,
        file_id_before="dim_b",
        file_id_after="dim_a",
        query="What changed?"
    )

    assert res.status == "SUCCESS"
    assert res.change_detected is True
    assert res.registration_diagnostics.original_dims_before == (140, 120)
    assert res.registration_diagnostics.original_dims_after == (130, 150)
    assert res.total_changed_area_pixels > 0


def test_validation_test7_geographically_disjoint_geotiffs(tmp_path: Path):
    """
    Section 12 Test 8 — Geographically disjoint GeoTIFFs:
    Two GeoTIFFs with completely non-overlapping bounding boxes (Location A vs Location B).
    Expected:
      - status = INCOMPATIBLE_IMAGES
      - change_detected = False
      - change_regions = []
    """
    data = np.full((3, 100, 100), 120, dtype=np.uint8)
    # Location A: UTM (500000, 6000000)
    pb = create_test_geotiff(tmp_path / "geo_loc_a.tif", data, origin_x=500000.0, origin_y=6000000.0)
    # Location B: Far away UTM (800000, 8000000)
    pa = create_test_geotiff(tmp_path / "geo_loc_b.tif", data, origin_x=800000.0, origin_y=8000000.0)

    engine = BiTemporalAnalysisEngine()
    res = engine.analyze_change(
        before_path=pb,
        after_path=pa,
        file_id_before="loc_a",
        file_id_after="loc_b",
        query="What changed?"
    )

    assert res.status == "INCOMPATIBLE_IMAGES"
    assert res.change_detected is False
    assert len(res.change_regions) == 0
    assert res.total_changed_area_pixels == 0
    assert res.pair_validation.classification == PairClassification.INCOMPATIBLE_IMAGES
    assert res.pair_validation.spatially_compatible is False
