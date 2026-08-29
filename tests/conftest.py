import pytest
from pathlib import Path
import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.transform import from_origin
    HAS_RASTERIO = True
except (ImportError, Exception):
    HAS_RASTERIO = False

import tifffile

from backend.config import settings


@pytest.fixture
def tmp_data_dir(tmp_path: Path):
    """
    Creates temporary upload, processed, and results directories for test isolation.
    """
    upload_dir = tmp_path / "uploads"
    processed_dir = tmp_path / "processed"
    results_dir = tmp_path / "results"

    upload_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    orig_upload = settings.UPLOAD_DIR
    orig_processed = settings.PROCESSED_DIR
    orig_results = settings.RESULTS_DIR

    settings.UPLOAD_DIR = upload_dir
    settings.PROCESSED_DIR = processed_dir
    settings.RESULTS_DIR = results_dir

    yield tmp_path

    settings.UPLOAD_DIR = orig_upload
    settings.PROCESSED_DIR = orig_processed
    settings.RESULTS_DIR = orig_results


@pytest.fixture
def sample_geotiff_rgb(tmp_path: Path) -> Path:
    """
    Generates a 3-band (RGB) synthetic GeoTIFF (256x256).
    """
    file_path = tmp_path / "sample_rgb.tif"
    width, height = 256, 256

    r = np.linspace(0, 255, width * height, dtype=np.uint8).reshape((height, width))
    g = np.full((height, width), 128, dtype=np.uint8)
    b = np.linspace(255, 0, width * height, dtype=np.uint8).reshape((height, width))
    rgb_arr = np.stack([r, g, b], axis=0)  # (3, 256, 256)

    # GeoTIFF metadata tags: ModelPixelScaleTag(33550), ModelTiepointTag(33922), GeoKeyDirectoryTag(34735)
    extratags = [
        (33550, 'd', 3, (0.0001, 0.0001, 0.0)),
        (33922, 'd', 6, (0.0, 0.0, 0.0, 13.404954, 52.520008, 0.0)),
        (34735, 'h', 8, (1, 1, 0, 1, 2048, 0, 1, 4326)),
    ]

    tifffile.imwrite(file_path, rgb_arr, extratags=extratags, planarconfig='SEPARATE')
    return file_path


@pytest.fixture
def sample_geotiff_multispectral(tmp_path: Path) -> Path:
    """
    Generates a 4-band (Red, Green, Blue, NIR) synthetic GeoTIFF (512x512) with UTM CRS (EPSG:32633).
    """
    file_path = tmp_path / "sample_4band.tif"
    width, height = 512, 512

    data = np.random.randint(100, 4000, size=(4, height, width), dtype=np.uint16)
    extratags = [
        (33550, 'd', 3, (10.0, 10.0, 0.0)),
        (33922, 'd', 6, (0.0, 0.0, 0.0, 389800.0, 5819000.0, 0.0)),
        (34735, 'h', 8, (1, 1, 0, 1, 3072, 0, 1, 32633)),
    ]

    tifffile.imwrite(file_path, data, extratags=extratags, planarconfig='SEPARATE')
    return file_path


@pytest.fixture
def sample_geotiff_sar(tmp_path: Path) -> Path:
    """
    Generates a 2-band (VV, VH) synthetic Sentinel-1 SAR GeoTIFF (128x128).
    """
    file_path = tmp_path / "sample_sar.tif"
    width, height = 128, 128

    vv = np.random.uniform(0.01, 0.5, size=(height, width)).astype(np.float32)
    vh = np.random.uniform(0.001, 0.1, size=(height, width)).astype(np.float32)
    sar_data = np.stack([vv, vh], axis=0)

    tifffile.imwrite(file_path, sar_data, planarconfig='SEPARATE')
    return file_path


@pytest.fixture
def sample_png(tmp_path: Path) -> Path:
    """
    Generates a standard 3-channel PNG image file (200x200).
    """
    file_path = tmp_path / "sample_visual.png"
    arr = np.random.randint(0, 256, size=(200, 200, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    img.save(file_path)
    return file_path
