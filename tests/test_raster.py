from pathlib import Path
import numpy as np
from backend.remote_sensing.raster import RasterReader, RasterMetadata


def test_raster_reader_metadata_geotiff(sample_geotiff_multispectral: Path):
    meta: RasterMetadata = RasterReader.extract_metadata(sample_geotiff_multispectral)
    assert meta.width == 512
    assert meta.height == 512
    assert meta.band_count == 4
    assert meta.crs_epsg == 32633
    assert meta.is_geospatial is True
    assert meta.bounds is not None
    assert meta.resolution == (10.0, 10.0)


def test_raster_reader_read_bands(sample_geotiff_rgb: Path):
    band1 = RasterReader.read_band(sample_geotiff_rgb, 1)
    assert band1.shape == (256, 256)
    assert isinstance(band1, np.ndarray)

    all_bands = RasterReader.read_all_bands(sample_geotiff_rgb)
    assert all_bands.shape == (3, 256, 256)


def test_raster_reader_read_subwindow(sample_geotiff_rgb: Path):
    sub = RasterReader.read_subwindow(sample_geotiff_rgb, x_off=10, y_off=20, width=50, height=60)
    assert sub.shape == (3, 60, 50)


def test_raster_reader_png_fallback(sample_png: Path):
    meta = RasterReader.extract_metadata(sample_png)
    assert meta.width == 200
    assert meta.height == 200
    assert meta.band_count == 3
    assert meta.is_geospatial is False
