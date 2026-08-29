import math
from pathlib import Path
from typing import List, Optional, Tuple
from pydantic import BaseModel
import numpy as np

from PIL import Image

from backend.remote_sensing.raster import RasterMetadata, RasterReader
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class TileMetadata(BaseModel):
    tile_id: str
    row: int
    col: int
    x_offset: int
    y_offset: int
    width: int
    height: int
    bounds: Optional[Tuple[float, float, float, float]] = None


class RasterTiler:
    """
    Splits large satellite imagery into manageable sliding-window grid tiles.
    Essential for allowing VLMs and vision models to process high-resolution rasters without OOM.
    """

    @staticmethod
    def compute_tiles(
        metadata: RasterMetadata,
        tile_size: int = 512,
        overlap: int = 0
    ) -> List[TileMetadata]:
        if tile_size <= 0:
            raise ValueError("tile_size must be positive")
        if overlap < 0 or overlap >= tile_size:
            raise ValueError("overlap must be non-negative and less than tile_size")

        w, h = metadata.width, metadata.height
        step = tile_size - overlap

        num_cols = max(1, math.ceil((w - overlap) / step))
        num_rows = max(1, math.ceil((h - overlap) / step))

        tiles: List[TileMetadata] = []

        for r in range(num_rows):
            for c in range(num_cols):
                x_off = c * step
                y_off = r * step

                tile_w = min(tile_size, w - x_off)
                tile_h = min(tile_size, h - y_off)

                tile_id = f"tile_r{r}_c{c}_x{x_off}_y{y_off}"

                # Calculate spatial geo-bounds if transform exists
                bounds = None
                if metadata.transform and len(metadata.transform) >= 6:
                    a, b, c_tx, d, e, f_ty = metadata.transform[:6]
                    # Top-left corner
                    min_x = c_tx + x_off * a + y_off * b
                    max_y = f_ty + x_off * d + y_off * e
                    # Bottom-right corner
                    max_x = c_tx + (x_off + tile_w) * a + (y_off + tile_h) * b
                    min_y = f_ty + (x_off + tile_w) * d + (y_off + tile_h) * e
                    bounds = (min(min_x, max_x), min(min_y, max_y), max(min_x, max_x), max(min_y, max_y))

                tiles.append(
                    TileMetadata(
                        tile_id=tile_id,
                        row=r,
                        col=c,
                        x_offset=x_off,
                        y_offset=y_off,
                        width=tile_w,
                        height=tile_h,
                        bounds=bounds,
                    )
                )

        return tiles

    @staticmethod
    def extract_tile_array(file_path: Path, tile: TileMetadata) -> np.ndarray:
        """
        Extracts sub-window array for a specific tile.
        """
        return RasterReader.read_subwindow(
            file_path,
            x_off=tile.x_offset,
            y_off=tile.y_offset,
            width=tile.width,
            height=tile.height
        )

    @staticmethod
    def save_tiles_to_disk(
        file_path: Path,
        output_dir: Path,
        tile_size: int = 512,
        overlap: int = 0
    ) -> List[Path]:
        file_path = Path(file_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        meta = RasterReader.extract_metadata(file_path)
        tiles = RasterTiler.compute_tiles(meta, tile_size=tile_size, overlap=overlap)
        saved_paths: List[Path] = []

        from backend.remote_sensing.preprocessing import Preprocessor
        from backend.remote_sensing.bands import BandManager

        for tile in tiles:
            tile_arr = RasterTiler.extract_tile_array(file_path, tile)
            tile_file = output_dir / f"{tile.tile_id}.png"

            if tile_arr.ndim == 3 and tile_arr.shape[0] >= 3:
                rgb_arr = BandManager.create_rgb_composite(tile_arr, (1, 2, 3))
            elif tile_arr.ndim == 3:
                rgb_arr = np.stack([tile_arr[0]] * 3, axis=-1)
            else:
                rgb_arr = np.stack([tile_arr] * 3, axis=-1)

            norm_rgb = Preprocessor.normalize_to_uint8(rgb_arr, nodata=meta.nodata)
            img = Image.fromarray(norm_rgb)
            img.save(tile_file)
            saved_paths.append(tile_file)

        logger.info(f"Saved {len(saved_paths)} tiles to {output_dir}")
        return saved_paths
