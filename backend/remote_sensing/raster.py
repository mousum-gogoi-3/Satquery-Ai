from pathlib import Path
from typing import List, Optional, Tuple, Union
from pydantic import BaseModel, Field
import numpy as np

try:
    import rasterio
    from rasterio.windows import Window
    HAS_RASTERIO = True
except (ImportError, Exception):
    HAS_RASTERIO = False

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

from PIL import Image
import cv2

from backend.utils.logger import get_logger

logger = get_logger(__name__)


class RasterMetadata(BaseModel):
    width: int
    height: int
    band_count: int
    crs: Optional[str] = None
    crs_epsg: Optional[int] = None
    transform: Optional[List[float]] = None  # 6 element affine matrix
    resolution: Optional[Tuple[float, float]] = None  # (x_res, y_res)
    bounds: Optional[Tuple[float, float, float, float]] = None  # (left, bottom, right, top)
    nodata: Optional[Union[float, int]] = None
    driver: str = "UNKNOWN"
    datatype: str = "unknown"
    is_geospatial: bool = False


class RasterReader:
    """
    Modular raster reader for satellite imagery supporting multi-spectral GeoTIFFs,
    SAR rasters, and visual image formats (PNG, JPEG).
    Features a multi-backend fallback strategy (Rasterio -> Tifffile -> Pillow/OpenCV).
    """

    @staticmethod
    def extract_metadata(file_path: Path) -> RasterMetadata:
        file_path = Path(file_path)

        if HAS_RASTERIO:
            try:
                with rasterio.open(file_path) as src:
                    crs_str = str(src.crs) if src.crs else None
                    crs_epsg = src.crs.to_epsg() if src.crs and hasattr(src.crs, "to_epsg") else None
                    bounds_tuple = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top) if src.bounds else None
                    res_tuple = (abs(src.res[0]), abs(src.res[1])) if src.res else None
                    transform_list = list(src.transform)[:6] if src.transform else None
                    nodata_val = src.nodata

                    return RasterMetadata(
                        width=src.width,
                        height=src.height,
                        band_count=src.count,
                        crs=crs_str,
                        crs_epsg=crs_epsg,
                        transform=transform_list,
                        resolution=res_tuple,
                        bounds=bounds_tuple,
                        nodata=nodata_val,
                        driver=src.driver,
                        is_geospatial=bool(src.crs or (transform_list and transform_list != [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])),
                    )
            except Exception as e_rasterio:
                logger.debug(f"Rasterio metadata extraction failed for {file_path}: {e_rasterio}")

        if HAS_TIFFFILE and file_path.suffix.lower() in [".tif", ".tiff", ".geotiff"]:
            try:
                with tifffile.TiffFile(file_path) as tif:
                    series = tif.series[0]
                    shape = series.shape
                    dtype_str = str(series.dtype)

                    if len(shape) == 2:
                        h, w = shape
                        b = 1
                    elif len(shape) == 3:
                        if shape[0] < shape[2]:  # (bands, H, W)
                            b, h, w = shape
                        else:  # (H, W, bands)
                            h, w, b = shape
                    else:
                        h, w, b = 0, 0, 0

                    crs_epsg = None
                    transform_list = None
                    res_tuple = None
                    bounds_tuple = None
                    is_geo = False

                    page = tif.pages[0]
                    tags = page.tags

                    # ModelPixelScaleTag (33550)
                    if 33550 in tags:
                        scale = tags[33550].value
                        res_tuple = (float(scale[0]), float(scale[1]))
                        is_geo = True

                    # ModelTiepointTag (33922)
                    if 33922 in tags and res_tuple:
                        tiepoint = tags[33922].value
                        if len(tiepoint) >= 6:
                            px, py, pz, gx, gy, gz = tiepoint[:6]
                            sx, sy = res_tuple
                            c_tx = gx - px * sx
                            f_ty = gy + py * sy
                            transform_list = [sx, 0.0, c_tx, 0.0, -sy, f_ty]
                            min_x = c_tx
                            max_x = c_tx + w * sx
                            max_y = f_ty
                            min_y = f_ty - h * sy
                            bounds_tuple = (min_x, min(min_y, max_y), max_x, max(min_y, max_y))
                            is_geo = True

                    # GeoKeyDirectoryTag (34735) for EPSG code
                    if 34735 in tags:
                        geo_keys = tags[34735].value
                        num_keys = geo_keys[3] if len(geo_keys) >= 4 else 0
                        for idx in range(4, min(len(geo_keys) - 3, 4 + num_keys * 4), 4):
                            key_id, location, count, val = geo_keys[idx:idx+4]
                            if key_id in (3072, 2048) and location == 0:
                                crs_epsg = int(val)
                                is_geo = True
                                break

                    return RasterMetadata(
                        width=w,
                        height=h,
                        band_count=b,
                        crs=f"EPSG:{crs_epsg}" if crs_epsg else None,
                        crs_epsg=crs_epsg,
                        transform=transform_list,
                        resolution=res_tuple,
                        bounds=bounds_tuple,
                        nodata=None,
                        driver="GTiff",
                        datatype=dtype_str,
                        is_geospatial=is_geo,
                    )
            except Exception as e_tiff:
                logger.debug(f"Tifffile metadata extraction failed for {file_path}: {e_tiff}")

        # Fallback to Pillow
        with Image.open(file_path) as img:
            w, h = img.size
            mode_bands = {
                "1": 1, "L": 1, "P": 1, "RGB": 3, "RGBA": 4, "CMYK": 4, "YCbCr": 3, "I": 1, "F": 1
            }
            band_count = mode_bands.get(img.mode, len(img.getbands()))

            return RasterMetadata(
                width=w,
                height=h,
                band_count=band_count,
                driver=img.format or "IMAGE",
                datatype=img.mode,
                is_geospatial=False,
            )

    @staticmethod
    def read_all_bands(file_path: Path) -> np.ndarray:
        """
        Reads all bands into a NumPy array of shape (bands, height, width).
        """
        file_path = Path(file_path)

        if HAS_RASTERIO:
            try:
                with rasterio.open(file_path) as src:
                    return src.read()
            except Exception:
                pass

        if HAS_TIFFFILE and file_path.suffix.lower() in [".tif", ".tiff", ".geotiff"]:
            try:
                arr = tifffile.imread(file_path)
                if arr.ndim == 2:
                    return arr[np.newaxis, :, :]
                elif arr.ndim == 3:
                    if arr.shape[0] < arr.shape[2]:  # (bands, H, W)
                        return arr
                    else:  # (H, W, bands)
                        return np.moveaxis(arr, -1, 0)
            except Exception:
                pass

        with Image.open(file_path) as img:
            arr = np.array(img)
            if arr.ndim == 2:
                return arr[np.newaxis, :, :]
            elif arr.ndim == 3:
                return np.moveaxis(arr, -1, 0)
            else:
                raise ValueError(f"Unexpected image array shape {arr.shape}")

    @staticmethod
    def read_band(file_path: Path, band_index: int = 1) -> np.ndarray:
        """
        Reads a single band (1-indexed). Returns 2D array of shape (height, width).
        """
        file_path = Path(file_path)

        if HAS_RASTERIO:
            try:
                with rasterio.open(file_path) as src:
                    if 1 <= band_index <= src.count:
                        return src.read(band_index)
            except Exception:
                pass

        all_bands = RasterReader.read_all_bands(file_path)
        if band_index < 1 or band_index > all_bands.shape[0]:
            raise ValueError(f"Band index {band_index} out of range (1 to {all_bands.shape[0]})")
        return all_bands[band_index - 1]

    @staticmethod
    def read_subwindow(
        file_path: Path, x_off: int, y_off: int, width: int, height: int
    ) -> np.ndarray:
        """
        Reads a spatial sub-window of the raster of shape (bands, height, width).
        """
        file_path = Path(file_path)

        if HAS_RASTERIO:
            try:
                with rasterio.open(file_path) as src:
                    win = Window(x_off, y_off, width, height)
                    return src.read(window=win)
            except Exception:
                pass

        all_bands = RasterReader.read_all_bands(file_path)
        return all_bands[:, y_off : y_off + height, x_off : x_off + width]
