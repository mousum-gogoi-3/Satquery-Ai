import pytest
import numpy as np
from backend.remote_sensing.change_detection import BiTemporalChangeDetector
from backend.config import settings


@pytest.fixture
def synthetic_vegetation_loss_pair():
    """
    Generates 4-band synthetic Before & After rasters (256x256):
    Before: High NIR/NDVI vegetation patch at [50..150, 50..150]
    After: Cleared dry soil (Low NIR/NDVI) at [50..150, 50..150]
    """
    h, w = 256, 256
    # Channels: 0=Red, 1=Green, 2=Blue, 3=NIR
    before = np.full((4, h, w), 500, dtype=np.uint16)
    # Vegetation patch in Before: Low Red (300), High NIR (3500) -> NDVI ~ 0.84
    before[0, 50:150, 50:150] = 300
    before[1, 50:150, 50:150] = 600
    before[2, 50:150, 50:150] = 300
    before[3, 50:150, 50:150] = 3500

    after = np.full((4, h, w), 500, dtype=np.uint16)
    # Cleared patch in After: High Red (2500), Low NIR (1000) -> NDVI ~ -0.42
    after[0, 50:150, 50:150] = 2500
    after[1, 50:150, 50:150] = 2000
    after[2, 50:150, 50:150] = 1800
    after[3, 50:150, 50:150] = 1000

    valid_mask = np.ones((h, w), dtype=bool)
    return before, after, valid_mask


@pytest.fixture
def synthetic_water_expansion_pair():
    """
    Generates 4-band synthetic Before & After rasters (256x256):
    Before: Dry land at [120..220, 100..200]
    After: Flooded water body (High NDWI, very low NIR absorption) at [120..220, 100..200]
    """
    h, w = 256, 256
    before = np.full((4, h, w), 1000, dtype=np.uint16)
    # Dry land in Before: Green 1200, NIR 2500 -> NDWI ~ -0.35
    before[0, 120:220, 100:200] = 1500
    before[1, 120:220, 100:200] = 1200
    before[2, 120:220, 100:200] = 1000
    before[3, 120:220, 100:200] = 2500

    after = np.full((4, h, w), 1000, dtype=np.uint16)
    # Flooded water in After: Green 2500, NIR 300 -> NDWI ~ +0.78
    after[0, 120:220, 100:200] = 400
    after[1, 120:220, 100:200] = 2500
    after[2, 120:220, 100:200] = 3000
    after[3, 120:220, 100:200] = 300

    valid_mask = np.ones((h, w), dtype=bool)
    return before, after, valid_mask


def test_ndvi_differencing_vegetation_loss(synthetic_vegetation_loss_pair):
    """Tests delta NDVI calculation and region extraction for vegetation loss."""
    before, after, valid_mask = synthetic_vegetation_loss_pair
    diffs = BiTemporalChangeDetector.compute_temporal_differences(before, after, valid_mask)
    mag_map = BiTemporalChangeDetector.compute_change_magnitude_map(diffs, valid_mask)
    regions = BiTemporalChangeDetector.extract_candidate_regions(mag_map, diffs, valid_mask)

    assert len(regions) == 1
    reg = regions[0]
    # Check that region bounding box tightly bounds the [50..150, 50..150] change area
    assert abs(reg["box"].xmin - 50) <= 2.0
    assert abs(reg["box"].ymin - 50) <= 2.0
    assert abs(reg["box"].xmax - 150) <= 2.0
    assert abs(reg["box"].ymax - 150) <= 2.0
    # Negative delta NDVI indicates vegetation loss
    assert reg["spectral_evidence"].delta_ndvi < -0.30
    assert reg["change_score"] > 0.40


def test_ndwi_differencing_water_expansion(synthetic_water_expansion_pair):
    """Tests delta NDWI calculation and region extraction for water expansion."""
    before, after, valid_mask = synthetic_water_expansion_pair
    diffs = BiTemporalChangeDetector.compute_temporal_differences(before, after, valid_mask)
    mag_map = BiTemporalChangeDetector.compute_change_magnitude_map(diffs, valid_mask)
    regions = BiTemporalChangeDetector.extract_candidate_regions(mag_map, diffs, valid_mask)

    assert len(regions) == 1
    reg = regions[0]
    assert abs(reg["box"].xmin - 100) <= 2.0
    assert abs(reg["box"].ymin - 120) <= 2.0
    assert abs(reg["box"].xmax - 200) <= 2.0
    assert abs(reg["box"].ymax - 220) <= 2.0
    # Positive delta NDWI indicates water expansion
    assert reg["spectral_evidence"].delta_ndwi > 0.30
    assert reg["change_score"] > 0.40


def test_identical_images_no_change():
    """Tests that identical before and after rasters produce zero change regions."""
    h, w = 256, 256
    before = np.full((4, h, w), 1200, dtype=np.uint16)
    after = np.full((4, h, w), 1200, dtype=np.uint16)
    valid_mask = np.ones((h, w), dtype=bool)

    diffs = BiTemporalChangeDetector.compute_temporal_differences(before, after, valid_mask)
    mag_map = BiTemporalChangeDetector.compute_change_magnitude_map(diffs, valid_mask)
    regions = BiTemporalChangeDetector.extract_candidate_regions(mag_map, diffs, valid_mask)

    assert len(regions) == 0
    assert mag_map.max() < settings.TEMPORAL_CHANGE_THRESHOLD


def test_morphological_noise_rejection():
    """Tests that isolated 2-pixel speckle noise is filtered out."""
    h, w = 256, 256
    before = np.full((4, h, w), 1000, dtype=np.uint16)
    after = np.full((4, h, w), 1000, dtype=np.uint16)
    # Inject isolated 2-pixel spike
    after[:, 10:12, 10:12] = 4000
    valid_mask = np.ones((h, w), dtype=bool)

    diffs = BiTemporalChangeDetector.compute_temporal_differences(before, after, valid_mask)
    mag_map = BiTemporalChangeDetector.compute_change_magnitude_map(diffs, valid_mask)
    regions = BiTemporalChangeDetector.extract_candidate_regions(mag_map, diffs, valid_mask)

    assert len(regions) == 0
