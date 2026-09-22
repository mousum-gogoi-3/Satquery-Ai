import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
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
from backend.remote_sensing.raster import RasterReader, RasterMetadata

logger = get_logger(__name__)


class SatelliteValidationResult(BaseModel):
    """
    Structured validation diagnostics for satellite and remote-sensing input verification.
    """
    valid: bool
    classification: str = Field(
        ...,
        description="VALID_SATELLITE_RASTER | NON_SATELLITE_IMAGE | CORRUPTED_FILE | DISALLOWED_FILE_TYPE"
    )
    format: str = "UNKNOWN"
    width: Optional[int] = None
    height: Optional[int] = None
    band_count: Optional[int] = None
    dtype: Optional[str] = None
    has_crs: bool = False
    has_geotransform: bool = False
    crs: Optional[str] = None
    spatial_bounds: Optional[Tuple[float, float, float, float]] = None
    error_code: Optional[str] = None
    reason: Optional[str] = None
    message: Optional[str] = None


class SatelliteImageValidator:
    """
    Authoritative validator ensuring SatQuery AI strictly accepts and processes only
    legitimate satellite and remote-sensing raster imagery (e.g., GeoTIFF, multispectral,
    SAR, georeferenced satellite products), while strictly rejecting ordinary photographs,
    screenshots, movie posters, artwork, memes, and ordinary visual images (JPEG, PNG,
    WebP, BMP, and renamed JPEGs).
    """

    # In-memory validation cache: (resolved_path_str, mtime) -> SatelliteValidationResult
    _cache: Dict[Tuple[str, float], SatelliteValidationResult] = {}

    @classmethod
    def validate_satellite_raster(cls, file_path: Path) -> SatelliteValidationResult:
        """
        Performs multi-signal remote-sensing verification on an uploaded or referenced raster file.
        Uses in-memory caching keyed by file path and last modification timestamp.
        """
        file_path = Path(file_path).resolve()

        if not file_path.exists() or not file_path.is_file():
            return SatelliteValidationResult(
                valid=False,
                classification="CORRUPTED_FILE",
                format="UNKNOWN",
                error_code="FILE_NOT_FOUND",
                reason=f"File does not exist: {file_path}",
                message="Invalid image. File could not be found on server."
            )

        mtime = file_path.stat().st_mtime
        cache_key = (str(file_path), mtime)
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        result = cls._inspect_raster(file_path)
        cls._cache[cache_key] = result
        return result

    @classmethod
    def _inspect_raster(cls, file_path: Path) -> SatelliteValidationResult:
        ext = file_path.suffix.lower()

        # 1. Read metadata using RasterReader
        try:
            meta: RasterMetadata = RasterReader.extract_metadata(file_path)
        except Exception as e:
            logger.warning(f"Satellite validation failed to extract metadata for {file_path.name}: {e}")
            return SatelliteValidationResult(
                valid=False,
                classification="CORRUPTED_FILE",
                format="UNKNOWN",
                error_code="CORRUPTED_FILE",
                reason=f"Failed to read file as a valid raster: {str(e)}",
                message="Invalid image. File is corrupted or unreadable."
            )

        format_detected = meta.driver or "UNKNOWN"
        width = meta.width
        height = meta.height
        band_count = meta.band_count
        dtype_str = meta.datatype.lower() if meta.datatype else "unknown"

        if (dtype_str == "unknown" or not dtype_str) and ext in [".tif", ".tiff", ".geotiff"] and HAS_TIFFFILE:
            try:
                with tifffile.TiffFile(file_path) as tif:
                    if tif.series:
                        dtype_str = str(tif.series[0].dtype).lower()
            except Exception:
                pass

        # 2. Check for explicit geospatial referencing
        has_crs = bool(meta.crs or meta.crs_epsg)
        has_geotransform = bool(
            meta.transform and meta.transform != [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        )
        is_geo = bool(meta.is_geospatial or has_crs or has_geotransform)

        # Inspect GeoTIFF tags directly if it is a TIFF file
        geotiff_tags_present = False
        if ext in [".tif", ".tiff", ".geotiff"] and HAS_TIFFFILE:
            try:
                with tifffile.TiffFile(file_path) as tif:
                    if tif.pages:
                        page = tif.pages[0]
                        tags = page.tags
                        # ModelPixelScaleTag (33550), ModelTiepointTag (33922),
                        # ModelTransformationTag (34264), GeoKeyDirectoryTag (34735)
                        if any(t in tags for t in (33550, 33922, 34264, 34735)):
                            geotiff_tags_present = True
                            is_geo = True
            except Exception as e_tag:
                logger.debug(f"TIFF tag inspection error for {file_path.name}: {e_tag}")

        has_geospatial_evidence = bool(is_geo or geotiff_tags_present)

        # 3. RULE 1: Ordinary consumer image formats (PNG, JPEG, WebP, BMP, GIF)
        # IMPORTANT: Consumer visual formats (screenshots, movie posters, photos, wallpapers)
        # often have 3 bands (RGB) or 4 bands (RGBA with Alpha channel).
        # Band count ALONE is NEVER evidence of satellite imagery for consumer formats.
        # Without verified geospatial metadata (CRS/geotransform), they are STRICTLY REJECTED.
        is_consumer_format = (
            ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"] or
            format_detected.upper() in ["PNG", "JPEG", "WEBP", "BMP", "GIF"]
        )

        if is_consumer_format:
            if not has_geospatial_evidence:
                reason = (
                    f"Ordinary visual image detected ({format_detected}, {width}x{height}, {band_count} band(s), {dtype_str}) "
                    f"lacking remote-sensing geospatial metadata (no CRS, geotransform, or satellite tags). "
                    f"Consumer PNG/JPEG/photo/screenshot files are not accepted."
                )
                logger.warning(f"Satellite validation REJECTED consumer visual file '{file_path.name}': {reason}")
                return SatelliteValidationResult(
                    valid=False,
                    classification="NON_SATELLITE_IMAGE",
                    format=format_detected,
                    width=width,
                    height=height,
                    band_count=band_count,
                    dtype=dtype_str,
                    has_crs=False,
                    has_geotransform=False,
                    crs=None,
                    spatial_bounds=None,
                    error_code="INVALID_SATELLITE_IMAGE",
                    reason=reason,
                    message="Invalid image. SatQuery AI only accepts satellite or remote-sensing imagery such as GeoTIFF/TIFF and supported satellite raster data."
                )

        # 4. RULE 2: Valid Geospatial Remote-Sensing Raster (GeoTIFF, etc.)
        # If the raster contains verified geospatial coordinates, CRS, or GeoTIFF tags,
        # it is a verified remote-sensing candidate (RGB GeoTIFF, multispectral GeoTIFF, or single-band GeoTIFF).
        if has_geospatial_evidence:
            logger.info(
                f"Satellite validation accepted geospatial raster '{file_path.name}' "
                f"(CRS={meta.crs}, driver={format_detected}, dims={width}x{height}, bands={band_count})"
            )
            return SatelliteValidationResult(
                valid=True,
                classification="VALID_SATELLITE_RASTER",
                format=format_detected,
                width=width,
                height=height,
                band_count=band_count,
                dtype=dtype_str,
                has_crs=has_crs,
                has_geotransform=has_geotransform,
                crs=meta.crs,
                spatial_bounds=meta.bounds,
                reason="Raster contains valid geospatial referencing (CRS / geotransform / GeoTIFF tags)."
            )

        # 5. RULE 3: TIFF files without embedded geospatial tags
        # Could be:
        # A) Renamed photo (photo.jpg -> photo.tif) or scanned document/graphic (uint8, 1, 3, or 4 bands) -> REJECT
        # B) Raw satellite sensor data (e.g. Sentinel-1 SAR float32, Sentinel-2 16-bit raw reflectance)
        is_satellite_dtype = any(
            t in dtype_str for t in ["float32", "float64", "uint16", "int16"]
        )

        if is_satellite_dtype and ext in [".tif", ".tiff", ".geotiff"]:
            logger.info(
                f"Satellite validation accepted radiometric remote-sensing raster '{file_path.name}' "
                f"(dtype={dtype_str}, bands={band_count})"
            )
            return SatelliteValidationResult(
                valid=True,
                classification="VALID_SATELLITE_RASTER",
                format=format_detected,
                width=width,
                height=height,
                band_count=band_count,
                dtype=dtype_str,
                has_crs=False,
                has_geotransform=False,
                crs=None,
                spatial_bounds=None,
                reason=f"Remote-sensing radiometry confirmed ({dtype_str} bit-depth, {band_count} band(s))."
            )

        # 6. Default Rejection for all unreferenced 8-bit images (including renamed JPEGs and plain TIFFs)
        reason = (
            f"Image does not contain sufficient remote-sensing/geospatial characteristics. "
            f"Detected unreferenced {format_detected} image ({width}x{height}, {band_count} band(s), {dtype_str}) "
            f"lacking CRS, geospatial geotransform, and satellite radiometric structure."
        )
        logger.warning(f"Satellite validation REJECTED unreferenced raster '{file_path.name}': {reason}")
        return SatelliteValidationResult(
            valid=False,
            classification="NON_SATELLITE_IMAGE",
            format=format_detected,
            width=width,
            height=height,
            band_count=band_count,
            dtype=dtype_str,
            has_crs=False,
            has_geotransform=False,
            crs=None,
            spatial_bounds=None,
            error_code="INVALID_SATELLITE_IMAGE",
            reason=reason,
            message="Invalid image. SatQuery AI only accepts satellite or remote-sensing imagery such as GeoTIFF/TIFF and supported satellite raster data."
        )

    @classmethod
    def clear_cache(cls) -> None:
        """Clears the validation cache."""
        cls._cache.clear()
