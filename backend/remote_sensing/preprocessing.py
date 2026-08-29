from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import cv2
from PIL import Image

from backend.remote_sensing.raster import RasterReader
from backend.remote_sensing.bands import BandManager
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class Preprocessor:
    """
    Satellite imagery preprocessor offering 2%-98% percentile normalization,
    resizing, NoData value masking, and thumbnail generation.
    """

    @staticmethod
    def normalize_to_uint8(
        arr: np.ndarray,
        nodata: Optional[float] = None,
        percentile_cut: bool = True,
        p_low: float = 2.0,
        p_high: float = 98.0
    ) -> np.ndarray:
        """
        Normalizes any multi-band or single-band array to 8-bit uint8 (0-255).
        """
        arr_float = arr.astype(np.float32)

        if nodata is not None:
            mask = (arr_float == nodata) | np.isnan(arr_float)
        else:
            mask = np.isnan(arr_float)

        arr_float_masked = arr_float.copy()
        arr_float_masked[mask] = np.nan

        valid_vals = arr_float_masked[~mask]
        if valid_vals.size > 0:
            if percentile_cut and valid_vals.size > 10:
                vmin = float(np.percentile(valid_vals, p_low))
                vmax = float(np.percentile(valid_vals, p_high))
            else:
                vmin = float(np.min(valid_vals))
                vmax = float(np.max(valid_vals))
        else:
            vmin, vmax = 0.0, 1.0

        norm_arr = np.zeros_like(arr_float, dtype=np.uint8)

        if vmax > vmin:
            clipped = np.clip(np.nan_to_num(arr_float_masked, nan=vmin), vmin, vmax)
            norm_vals = ((clipped - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
            norm_arr = norm_vals

        norm_arr[mask] = 0
        return norm_arr

    @staticmethod
    def resize_image(arr: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resizes NumPy array of shape (height, width) or (height, width, channels) to target_size (width, height).
        """
        target_w, target_h = target_size
        return cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def generate_preview_thumbnail(
        file_path: Path,
        output_path: Path,
        max_dim: int = 1024
    ) -> Path:
        """
        Extracts multi-spectral/geospatial raster, applies normalization,
        and saves an RGB preview JPEG/PNG thumbnail.
        """
        file_path = Path(file_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        meta = RasterReader.extract_metadata(file_path)
        all_bands = RasterReader.read_all_bands(file_path)

        if all_bands.ndim == 3 and all_bands.shape[0] >= 3:
            rgb_arr = BandManager.create_rgb_composite(all_bands, (1, 2, 3))
        elif all_bands.ndim == 3 and all_bands.shape[0] < 3:
            # Grayscale or 2-band
            single_band = all_bands[0]
            rgb_arr = np.stack([single_band, single_band, single_band], axis=-1)
        else:
            rgb_arr = np.stack([all_bands, all_bands, all_bands], axis=-1)

        norm_rgb = Preprocessor.normalize_to_uint8(rgb_arr, nodata=meta.nodata, percentile_cut=True)

        h, w = norm_rgb.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            norm_rgb = Preprocessor.resize_image(norm_rgb, (new_w, new_h))

        img = Image.fromarray(norm_rgb)
        img.save(output_path, quality=90)
        logger.info(f"Generated preview thumbnail at {output_path}")
        return output_path
