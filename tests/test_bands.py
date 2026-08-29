from pathlib import Path
import numpy as np
from backend.remote_sensing.raster import RasterReader
from backend.remote_sensing.bands import BandManager, BandType


def test_band_manager_inference_multispectral(sample_geotiff_multispectral: Path):
    meta = RasterReader.extract_metadata(sample_geotiff_multispectral)
    bands = BandManager.infer_band_configuration(meta)
    assert len(bands) == 4
    assert bands[0].band_type == BandType.RED
    assert bands[1].band_type == BandType.GREEN
    assert bands[2].band_type == BandType.BLUE
    assert bands[3].band_type == BandType.NIR


def test_band_manager_inference_sar(sample_geotiff_sar: Path):
    meta = RasterReader.extract_metadata(sample_geotiff_sar)
    bands = BandManager.infer_band_configuration(meta)
    assert len(bands) == 2
    assert bands[0].band_type == BandType.SAR_VV
    assert bands[1].band_type == BandType.SAR_VH


def test_band_manager_rgb_composite(sample_geotiff_rgb: Path):
    all_bands = RasterReader.read_all_bands(sample_geotiff_rgb)
    rgb = BandManager.create_rgb_composite(all_bands, (1, 2, 3))
    assert rgb.shape == (256, 256, 3)
