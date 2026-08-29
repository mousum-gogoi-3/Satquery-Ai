import os
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field

from PIL import Image

try:
    import rasterio
    HAS_RASTERIO = True
except (ImportError, Exception):
    HAS_RASTERIO = False

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

from backend.config import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class ValidationResult(BaseModel):
    valid: bool
    format: str = "UNKNOWN"
    file_size_bytes: int = 0
    width: Optional[int] = None
    height: Optional[int] = None
    bands: Optional[int] = None
    datatype: Optional[str] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class FileValidator:
    """
    Validates satellite imagery file types, sizes, dimensions, band counts, and raster integrity.
    Supports Rasterio, Tifffile, and Pillow fallback chains.
    """

    @staticmethod
    def validate(file_path: Path) -> ValidationResult:
        file_path = Path(file_path)
        errors: List[str] = []
        warnings: List[str] = []

        if not file_path.exists() or not file_path.is_file():
            return ValidationResult(
                valid=False,
                errors=[f"File not found: {file_path}"]
            )

        file_size = file_path.stat().st_size
        ext = file_path.suffix.lower()

        # 1. Extension Validation
        if ext not in settings.ALLOWED_EXTENSIONS:
            errors.append(
                f"Unsupported extension '{ext}'. Allowed extensions: {sorted(list(settings.ALLOWED_EXTENSIONS))}"
            )

        # 2. File Size Validation
        if file_size > settings.MAX_FILE_SIZE_BYTES:
            errors.append(
                f"File size ({file_size / (1024 * 1024):.2f} MB) exceeds maximum allowed size ({settings.MAX_FILE_SIZE_BYTES / (1024 * 1024):.2f} MB)"
            )
        elif file_size == 0:
            errors.append("File is empty (0 bytes)")

        if errors:
            return ValidationResult(
                valid=False,
                file_size_bytes=file_size,
                errors=errors,
                warnings=warnings,
            )

        # 3. Raster Integrity & Dimension Check
        format_detected = "UNKNOWN"
        width, height, band_count, datatype = None, None, None, None
        read_success = False

        if HAS_RASTERIO:
            try:
                with rasterio.open(file_path) as dataset:
                    format_detected = dataset.driver
                    width = dataset.width
                    height = dataset.height
                    band_count = dataset.count
                    datatype = str(dataset.dtypes[0]) if dataset.dtypes else "unknown"
                    read_success = True
            except Exception as e_rasterio:
                logger.debug(f"Rasterio read attempt failed for {file_path}: {e_rasterio}")

        if not read_success and HAS_TIFFFILE and ext in [".tif", ".tiff", ".geotiff"]:
            try:
                with tifffile.TiffFile(file_path) as tif:
                    format_detected = "GTiff"
                    series = tif.series[0]
                    shape = series.shape
                    datatype = str(series.dtype)

                    if len(shape) == 2:
                        height, width = shape
                        band_count = 1
                    elif len(shape) == 3:
                        if shape[0] < shape[2]:  # (bands, H, W)
                            band_count, height, width = shape
                        else:  # (H, W, bands)
                            height, width, band_count = shape
                    read_success = True
            except Exception as e_tiff:
                logger.debug(f"Tifffile read attempt failed for {file_path}: {e_tiff}")

        if not read_success:
            try:
                with Image.open(file_path) as img:
                    format_detected = img.format or "IMAGE"
                    width, height = img.size
                    mode_bands = {
                        "1": 1, "L": 1, "P": 1, "RGB": 3, "RGBA": 4, "CMYK": 4, "YCbCr": 3, "I": 1, "F": 1
                    }
                    band_count = mode_bands.get(img.mode, len(img.getbands()))
                    datatype = img.mode
                    read_success = True
            except Exception as e_pil:
                errors.append(f"Corrupted or unreadable image file: {str(e_pil)}")

        if read_success and width and height and band_count:
            if width <= 0 or height <= 0:
                errors.append("Invalid raster dimensions (width or height <= 0)")
            if width > settings.MAX_RASTER_DIMENSION or height > settings.MAX_RASTER_DIMENSION:
                warnings.append(f"High-resolution image ({width}x{height}) exceeds standard view limit; tiling recommended.")
            if band_count <= 0:
                errors.append("Raster contains 0 bands")
            elif band_count > settings.MAX_BANDS:
                errors.append(f"Band count ({band_count}) exceeds max supported limit ({settings.MAX_BANDS})")

        is_valid = len(errors) == 0
        return ValidationResult(
            valid=is_valid,
            format=format_detected,
            file_size_bytes=file_size,
            width=width,
            height=height,
            bands=band_count,
            datatype=datatype,
            errors=errors,
            warnings=warnings,
        )
