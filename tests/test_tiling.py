from pathlib import Path
import numpy as np
from backend.remote_sensing.raster import RasterReader
from backend.remote_sensing.tiling import RasterTiler


def test_tiling_computation(sample_geotiff_multispectral: Path):
    meta = RasterReader.extract_metadata(sample_geotiff_multispectral)
    tiles = RasterTiler.compute_tiles(meta, tile_size=256, overlap=0)
    # 512x512 with 256 tile size = 2x2 = 4 tiles
    assert len(tiles) == 4
    assert tiles[0].width == 256
    assert tiles[0].height == 256
    assert tiles[0].bounds is not None


def test_tiling_extraction(sample_geotiff_rgb: Path):
    meta = RasterReader.extract_metadata(sample_geotiff_rgb)
    tiles = RasterTiler.compute_tiles(meta, tile_size=128, overlap=0)
    arr = RasterTiler.extract_tile_array(sample_geotiff_rgb, tiles[0])
    assert arr.shape == (3, 128, 128)


def test_save_tiles_to_disk(sample_geotiff_rgb: Path, tmp_path: Path):
    out_dir = tmp_path / "tiles"
    saved = RasterTiler.save_tiles_to_disk(sample_geotiff_rgb, out_dir, tile_size=128)
    assert len(saved) == 4
    for p in saved:
        assert p.exists()
