from typing import Tuple
from pathlib import Path
import pytest
import numpy as np
import tifffile
from PIL import Image

from backend.remote_sensing.alignment import ImageRegistrar
from backend.api.schemas import RegistrationQuality
from backend.config import settings


@pytest.fixture
def temporal_geotiff_pair_aligned(tmp_path: Path) -> Tuple[Path, Path]:
    """Generates two aligned 4-band GeoTIFFs with UTM coordinates."""
    path_before = tmp_path / "before_utm.tif"
    path_after = tmp_path / "after_utm.tif"

    w, h = 256, 256
    b_data = np.full((4, h, w), 1000, dtype=np.uint16)
    a_data = np.full((4, h, w), 1500, dtype=np.uint16)

    extratags = [
        (33550, 'd', 3, (10.0, 10.0, 0.0)),
        (33922, 'd', 6, (0.0, 0.0, 0.0, 500000.0, 6000000.0, 0.0)),
        (34735, 'h', 8, (1, 1, 0, 1, 3072, 0, 1, 32633)),
    ]

    tifffile.imwrite(path_before, b_data, extratags=extratags, planarconfig='SEPARATE')
    tifffile.imwrite(path_after, a_data, extratags=extratags, planarconfig='SEPARATE')

    return path_before, path_after


@pytest.fixture
def temporal_geotiff_pair_disjoint(tmp_path: Path) -> Tuple[Path, Path]:
    """Generates two spatially non-overlapping GeoTIFFs."""
    path_before = tmp_path / "before_disjoint.tif"
    path_after = tmp_path / "after_disjoint.tif"

    w, h = 128, 128
    data = np.full((3, h, w), 128, dtype=np.uint8)

    # Before raster: Location A (UTM 500000, 6000000)
    tags_before = [
        (33550, 'd', 3, (10.0, 10.0, 0.0)),
        (33922, 'd', 6, (0.0, 0.0, 0.0, 500000.0, 6000000.0, 0.0)),
        (34735, 'h', 8, (1, 1, 0, 1, 3072, 0, 1, 32633)),
    ]

    # After raster: Location B far away (UTM 700000, 7000000)
    tags_after = [
        (33550, 'd', 3, (10.0, 10.0, 0.0)),
        (33922, 'd', 6, (0.0, 0.0, 0.0, 700000.0, 7000000.0, 0.0)),
        (34735, 'h', 8, (1, 1, 0, 1, 3072, 0, 1, 32633)),
    ]

    tifffile.imwrite(path_before, data, extratags=tags_before, planarconfig='SEPARATE')
    tifffile.imwrite(path_after, data, extratags=tags_after, planarconfig='SEPARATE')

    return path_before, path_after


def test_alignment_aligned_geotiff_pair(temporal_geotiff_pair_aligned):
    """Verifies alignment of two overlapping GeoTIFFs produces correct diagnostics and array shapes."""
    before_path, after_path = temporal_geotiff_pair_aligned
    arr_b, arr_a, diag, valid_mask = ImageRegistrar.align_rasters(before_path, after_path)

    assert arr_b.shape == (4, 256, 256)
    assert arr_a.shape == (4, 256, 256)
    assert diag.overlap_percentage >= 95.0
    assert diag.registration_quality in (RegistrationQuality.HIGH, RegistrationQuality.MEDIUM)
    assert diag.aligned_dims == (256, 256)
    assert valid_mask.all()


def test_alignment_different_resolution_resampling(tmp_path: Path):
    """Verifies that an after raster with different dimensions is automatically resampled to match before."""
    before_path = tmp_path / "before_256.png"
    after_path = tmp_path / "after_512.png"

    # Before: 256x256
    img_b = Image.fromarray(np.full((256, 256, 3), 100, dtype=np.uint8))
    img_b.save(before_path)

    # After: 512x512
    img_a = Image.fromarray(np.full((512, 512, 3), 200, dtype=np.uint8))
    img_a.save(after_path)

    arr_b, arr_a, diag, valid_mask = ImageRegistrar.align_rasters(before_path, after_path)

    assert arr_b.shape == (3, 256, 256)
    assert arr_a.shape == (3, 256, 256)
    assert diag.original_dims_before == (256, 256)
    assert diag.original_dims_after == (512, 512)
    assert diag.aligned_dims == (256, 256)


def test_alignment_disjoint_geotiff_no_overlap(temporal_geotiff_pair_disjoint):
    """Verifies that disjoint spatial rasters return FAILED registration with 0.0% overlap."""
    before_path, after_path = temporal_geotiff_pair_disjoint
    arr_b, arr_a, diag, valid_mask = ImageRegistrar.align_rasters(before_path, after_path)

    assert diag.overlap_percentage == 0.0
    assert diag.registration_quality == RegistrationQuality.FAILED
    assert not valid_mask.any()


def test_alignment_missing_file_raises_error():
    """Verifies FileNotFoundError when either image path does not exist."""
    with pytest.raises(FileNotFoundError):
        ImageRegistrar.align_rasters(Path("non_existent_before.tif"), Path("non_existent_after.tif"))
