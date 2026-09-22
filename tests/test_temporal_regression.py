import pytest
from pathlib import Path
import numpy as np
import tifffile

from backend.config import settings
from backend.models.temporal_engine import BiTemporalAnalysisEngine
from backend.api.schemas import ChangeType, RegistrationQuality
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def create_synthetic_geotiff(
    file_path: Path,
    data: np.ndarray,
    res_x: float = 10.0,
    res_y: float = 10.0,
    origin_x: float = 500000.0,
    origin_y: float = 6000000.0,
    epsg: int = 32633
) -> Path:
    """Helper to write a multi-band GeoTIFF with explicit metadata tags."""
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


def test_regression_1_obvious_synthetic_change(tmp_path: Path):
    """
    Test 1 — Obvious Synthetic Change:
    Two rasters where a known rectangular region changes significantly.
    Expected:
      - status != NO_SIGNIFICANT_CHANGE
      - total_changed_area_pixels > 0
      - change_regions >= 1
    """
    h, w = 128, 128
    before = np.full((4, h, w), 500, dtype=np.uint16)
    before[0, 30:70, 30:70] = 300
    before[3, 30:70, 30:70] = 3000

    after = np.full((4, h, w), 500, dtype=np.uint16)
    after[0, 30:70, 30:70] = 2500
    after[3, 30:70, 30:70] = 600

    pb = create_synthetic_geotiff(tmp_path / "reg1_before.tif", before)
    pa = create_synthetic_geotiff(tmp_path / "reg1_after.tif", after)

    engine = BiTemporalAnalysisEngine()
    res = engine.analyze_change(
        before_path=pb,
        after_path=pa,
        file_id_before="reg1_b",
        file_id_after="reg1_a",
        query="What changed?"
    )

    assert res.status == "SUCCESS"
    assert res.status != "NO_SIGNIFICANT_CHANGE"
    assert res.total_changed_area_pixels > 0
    assert len(res.change_regions) >= 1
    assert res.evidence_image_path is not None
    assert res.change_heatmap_path is not None


def test_regression_2_identical_images(tmp_path: Path):
    """
    Test 2 — Identical Images:
    Before and after rasters are identical.
    Expected:
      - status = NO_SIGNIFICANT_CHANGE
      - total_changed_area_pixels = 0
      - change_regions = []
    """
    h, w = 128, 128
    data = np.full((4, h, w), 1000, dtype=np.uint16)
    data[:, 20:60, 20:60] = 2500

    pb = create_synthetic_geotiff(tmp_path / "reg2_before.tif", data)
    pa = create_synthetic_geotiff(tmp_path / "reg2_after.tif", data.copy())

    engine = BiTemporalAnalysisEngine()
    res = engine.analyze_change(
        before_path=pb,
        after_path=pa,
        file_id_before="reg2_b",
        file_id_after="reg2_a",
        query="What changed?"
    )

    assert res.status in ("NO_CHANGE", "NO_SIGNIFICANT_CHANGE")
    assert res.total_changed_area_pixels == 0
    assert len(res.change_regions) == 0


def test_regression_3_subtle_meaningful_change(tmp_path: Path):
    """
    Test 3 — Subtle but Meaningful Change:
    Moderate physical change that should be detected with the default threshold.
    """
    h, w = 128, 128
    before = np.full((4, h, w), 800, dtype=np.uint16)
    before[0] = 600
    before[3] = 1800

    after = before.copy()
    after[0, 40:80, 40:80] = 1100
    after[3, 40:80, 40:80] = 1100

    pb = create_synthetic_geotiff(tmp_path / "reg3_before.tif", before)
    pa = create_synthetic_geotiff(tmp_path / "reg3_after.tif", after)

    engine = BiTemporalAnalysisEngine()
    res = engine.analyze_change(
        before_path=pb,
        after_path=pa,
        file_id_before="reg3_b",
        file_id_after="reg3_a",
        query="What changed?"
    )

    assert res.status == "SUCCESS"
    assert res.total_changed_area_pixels > 0
    assert len(res.change_regions) >= 1


def test_regression_4_high_confidence_threshold(tmp_path: Path):
    """
    Test 4 — High Threshold (confidence_threshold = 1.0):
    Tests that setting an extremely strict threshold (1.0) behaves consistently and transparently.
    """
    h, w = 128, 128
    before = np.full((4, h, w), 1000, dtype=np.uint16)
    after = before.copy()
    after[0, 30:50, 30:50] = 1400
    after[3, 30:50, 30:50] = 1400

    pb = create_synthetic_geotiff(tmp_path / "reg4_before.tif", before)
    pa = create_synthetic_geotiff(tmp_path / "reg4_after.tif", after)

    engine = BiTemporalAnalysisEngine()
    res = engine.analyze_change(
        before_path=pb,
        after_path=pa,
        file_id_before="reg4_b",
        file_id_after="reg4_a",
        query="What changed?",
        confidence_threshold=1.0
    )
    if len(res.change_regions) == 0:
        assert res.status == "NO_SIGNIFICANT_CHANGE"
        assert res.total_changed_area_pixels == 0
    else:
        assert all(r.confidence >= 1.0 for r in res.change_regions)

    res_balanced = engine.analyze_change(
        before_path=pb,
        after_path=pa,
        file_id_before="reg4_b",
        file_id_after="reg4_a",
        query="What changed?",
        confidence_threshold=0.35
    )
    assert res_balanced.status == "SUCCESS"
    assert res_balanced.total_changed_area_pixels > 0


def test_regression_5_no_target_type(tmp_path: Path):
    """
    Test 5 — No Target Type ("What changed?" with target_change_types = []):
    The detector evaluates and returns all detected change categories without restricting to vegetation loss.
    """
    h, w = 128, 128
    before = np.full((4, h, w), 1000, dtype=np.uint16)
    after = before.copy()
    after[0, 30:70, 30:70] = 200
    after[1, 30:70, 30:70] = 2500
    after[2, 30:70, 30:70] = 2800
    after[3, 30:70, 30:70] = 200

    pb = create_synthetic_geotiff(tmp_path / "reg5_before.tif", before)
    pa = create_synthetic_geotiff(tmp_path / "reg5_after.tif", after)

    engine = BiTemporalAnalysisEngine()
    res = engine.analyze_change(
        before_path=pb,
        after_path=pa,
        file_id_before="reg5_b",
        file_id_after="reg5_a",
        query="What changed?",
        target_change_types=[]
    )

    assert res.status == "SUCCESS"
    assert len(res.change_regions) >= 1
    assert res.dominant_change_type == ChangeType.WATER_EXPANSION


def test_regression_6_vegetation_loss(tmp_path: Path):
    """
    Test 6 — Vegetation Loss:
    Synthetic NDVI decrease correctly classifies as vegetation_loss.
    """
    h, w = 128, 128
    before = np.full((4, h, w), 500, dtype=np.uint16)
    before[0, 20:60, 20:60] = 200
    before[3, 20:60, 20:60] = 3200

    after = np.full((4, h, w), 500, dtype=np.uint16)
    after[0, 20:60, 20:60] = 2400
    after[3, 20:60, 20:60] = 800

    pb = create_synthetic_geotiff(tmp_path / "reg6_before.tif", before)
    pa = create_synthetic_geotiff(tmp_path / "reg6_after.tif", after)

    engine = BiTemporalAnalysisEngine()
    res = engine.analyze_change(
        before_path=pb,
        after_path=pa,
        file_id_before="reg6_b",
        file_id_after="reg6_a",
        query="Where did vegetation decrease?",
        target_change_types=[ChangeType.VEGETATION_LOSS]
    )

    assert res.status == "SUCCESS"
    assert res.dominant_change_type == ChangeType.VEGETATION_LOSS
    assert res.change_regions[0].spectral_evidence.delta_ndvi < -0.30


def test_regression_7_different_raster_dimensions(tmp_path: Path):
    """
    Test 7 — Different Raster Dimensions:
    Verifies multi-tier spatial registration aligns images with differing dimensions
    and still succeeds in detecting change.
    """
    before = np.full((4, 120, 140), 600, dtype=np.uint16)
    after = np.full((4, 150, 130), 600, dtype=np.uint16)

    before[0, 20:50, 20:50] = 200
    before[3, 20:50, 20:50] = 3000
    after[0, 20:50, 20:50] = 2200
    after[3, 20:50, 20:50] = 700

    pb = create_synthetic_geotiff(tmp_path / "reg7_before.tif", before)
    pa = create_synthetic_geotiff(tmp_path / "reg7_after.tif", after)

    engine = BiTemporalAnalysisEngine()
    res = engine.analyze_change(
        before_path=pb,
        after_path=pa,
        file_id_before="reg7_b",
        file_id_after="reg7_a",
        query="What changed?"
    )

    assert res.status == "SUCCESS"
    assert res.registration_diagnostics.registration_quality in (
        RegistrationQuality.HIGH, RegistrationQuality.MEDIUM
    )
    assert res.total_changed_area_pixels > 0


def test_regression_8_missing_optional_bands_rgb_only(tmp_path: Path):
    """
    Test 8 — Missing Optional Bands (RGB-Only Imagery):
    Verify 3-band RGB imagery does not crash because NIR/SWIR/SAR are unavailable,
    and successfully detects visual land-cover change.
    """
    h, w = 128, 128
    before = np.full((3, h, w), 120, dtype=np.uint8)
    after = np.full((3, h, w), 120, dtype=np.uint8)
    after[0, 30:70, 30:70] = 240
    after[1, 30:70, 30:70] = 40
    after[2, 30:70, 30:70] = 30

    pb = tmp_path / "reg8_rgb_before.tif"
    pa = tmp_path / "reg8_rgb_after.tif"
    tifffile.imwrite(pb, before, planarconfig='SEPARATE')
    tifffile.imwrite(pa, after, planarconfig='SEPARATE')

    engine = BiTemporalAnalysisEngine()
    res = engine.analyze_change(
        before_path=pb,
        after_path=pa,
        file_id_before="reg8_b",
        file_id_after="reg8_a",
        query="What changed?"
    )

    assert res.status == "SUCCESS"
    assert res.total_changed_area_pixels > 0
    assert len(res.change_regions) >= 1
    assert res.change_regions[0].spectral_evidence.delta_sar_vv is None
    assert res.change_regions[0].spectral_evidence.delta_rgb > 0.20


def test_regression_9_single_band_rasters(tmp_path: Path):
    """
    Test 9 — Single-band (Grayscale / 1-band Sentinel-2) Rasters:
    Verifies 1-band rasters detect intensity changes and compute ground area correctly in EPSG:4326.
    """
    h, w = 100, 100
    before = np.full((1, h, w), 2000, dtype=np.uint16)
    after = np.full((1, h, w), 2000, dtype=np.uint16)
    after[0, 20:60, 20:60] = 3500

    pb = create_synthetic_geotiff(
        tmp_path / "reg9_before.tif",
        before,
        res_x=9.4e-5,
        res_y=9.4e-5,
        origin_x=54.54,
        origin_y=24.36,
        epsg=4326
    )
    pa = create_synthetic_geotiff(
        tmp_path / "reg9_after.tif",
        after,
        res_x=9.4e-5,
        res_y=9.4e-5,
        origin_x=54.54,
        origin_y=24.36,
        epsg=4326
    )

    engine = BiTemporalAnalysisEngine()
    res = engine.analyze_change(
        before_path=pb,
        after_path=pa,
        file_id_before="reg9_b",
        file_id_after="reg9_a",
        query="What changed?",
        confidence_threshold=0.35
    )

    assert res.status == "SUCCESS"
    assert res.total_changed_area_pixels > 0
    assert len(res.change_regions) >= 1
    assert res.total_changed_area_sq_meters is not None
    assert res.total_changed_area_sq_meters > 50000.0
