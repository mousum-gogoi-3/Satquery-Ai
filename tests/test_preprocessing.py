from pathlib import Path
import numpy as np
from backend.remote_sensing.preprocessing import Preprocessor


def test_normalize_to_uint8():
    arr = np.array([0, 10, 50, 100, 500, 1000], dtype=np.float32)
    norm = Preprocessor.normalize_to_uint8(arr, percentile_cut=True)
    assert norm.dtype == np.uint8
    assert norm.shape == arr.shape
    assert norm.min() == 0
    assert norm.max() == 255


def test_normalize_with_nodata():
    arr = np.array([-9999, 10, 20, 30, -9999], dtype=np.float32)
    norm = Preprocessor.normalize_to_uint8(arr, nodata=-9999)
    assert norm[0] == 0  # NoData masked to 0
    assert norm[4] == 0  # NoData masked to 0
    assert norm[1] == 0  # Min valid value (10) scales to 0
    assert norm[2] > 0   # Mid valid value (20) scales to > 0
    assert norm[3] == 255 # Max valid value (30) scales to 255


def test_generate_preview_thumbnail(sample_geotiff_multispectral: Path, tmp_path: Path):
    out_thumb = tmp_path / "thumb_preview.png"
    result_path = Preprocessor.generate_preview_thumbnail(
        sample_geotiff_multispectral, out_thumb, max_dim=128
    )
    assert result_path.exists()
    assert result_path.stat().st_size > 0
