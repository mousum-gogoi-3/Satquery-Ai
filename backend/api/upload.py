import uuid
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks, status
from fastapi.responses import FileResponse

from backend.config import settings
from backend.utils.logger import get_logger
from backend.remote_sensing.validator import FileValidator
from backend.remote_sensing.satellite_validator import SatelliteImageValidator
from backend.remote_sensing.raster import RasterReader
from backend.remote_sensing.bands import BandManager
from backend.remote_sensing.preprocessing import Preprocessor
from backend.remote_sensing.tiling import RasterTiler
from backend.api.schemas import (
    UploadResponse,
    MetadataResponse,
    TilingRequest,
    TilingResponse,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/raster", tags=["Raster & Ingestion Engine"])


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_raster(file: UploadFile = File(...)):
    """
    Ingests satellite imagery (GeoTIFF, TIFF, PNG, JPEG), validates raster integrity,
    extracts geospatial metadata & band channels, and generates a preview thumbnail.
    """
    raw_filename = file.filename or "uploaded_image.tif"

    if "\x00" in raw_filename or "\0" in raw_filename or "%00" in raw_filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename contains invalid null characters"
        )

    filename = Path(raw_filename).name
    if not filename or filename in (".", ".."):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename"
        )

    ext = Path(filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported extension '{ext}'. Allowed extensions: {sorted(list(settings.ALLOWED_EXTENSIONS))}"
        )

    file_id = str(uuid.uuid4())
    saved_filename = f"{file_id}_{filename}"
    file_path = settings.UPLOAD_DIR / saved_filename

    # Prevent path traversal outside UPLOAD_DIR
    upload_dir_resolved = settings.UPLOAD_DIR.resolve()
    try:
        file_path.resolve().relative_to(upload_dir_resolved)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Path traversal attempt detected"
        )

    # Save uploaded file using streaming chunked writes and enforce max file size limit
    CHUNK_SIZE = 1024 * 1024  # 1 MB chunks
    written = 0
    try:
        with open(file_path, "wb") as f:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                written += len(chunk)
                if written > settings.MAX_FILE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File size exceeds maximum allowed limit of {settings.MAX_FILE_SIZE_BYTES} bytes."
                    )
                f.write(chunk)
    except HTTPException:
        if file_path.exists():
            file_path.unlink()
        raise
    except Exception as e:
        logger.error(f"Failed to write uploaded file: {e}")
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file on server"
        )

    # Process and validate file with cleanup on failure
    try:
        # 1. File security & basic readability validation
        validation = FileValidator.validate(file_path)
        if not validation.valid:
            if file_path.exists():
                file_path.unlink()
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "message": "File validation failed",
                    "errors": validation.errors,
                    "warnings": validation.warnings,
                }
            )

        # 2. Authoritative Satellite / Remote-Sensing Suitability Validation
        sat_validation = SatelliteImageValidator.validate_satellite_raster(file_path)
        if not sat_validation.valid:
            if file_path.exists():
                file_path.unlink()
            logger.warning(
                f"Rejected upload '{filename}' ({file_id}): {sat_validation.reason}"
            )
            classification_str = (
                sat_validation.classification.value
                if hasattr(sat_validation.classification, "value")
                else str(sat_validation.classification)
            )
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "valid": False,
                    "error_code": sat_validation.error_code or "INVALID_SATELLITE_IMAGE",
                    "classification": classification_str,
                    "message": sat_validation.message or "Invalid image. SatQuery AI only accepts satellite or remote-sensing imagery.",
                    "reason": sat_validation.reason,
                    "diagnostics": sat_validation.model_dump(),
                }
            )

        # Extract metadata & band information
        metadata = RasterReader.extract_metadata(file_path)
        bands = BandManager.infer_band_configuration(metadata)

        # Generate preview thumbnail
        thumbnail_filename = f"{file_id}_thumb.png"
        thumbnail_path = settings.PROCESSED_DIR / thumbnail_filename
        try:
            Preprocessor.generate_preview_thumbnail(file_path, thumbnail_path)
            thumbnail_url = f"/api/v1/raster/preview/{thumbnail_filename}"
        except Exception as e_thumb:
            logger.warning(f"Failed to generate thumbnail for {file_id}: {e_thumb}")
            thumbnail_url = None

        logger.info(f"Successfully processed upload file_id={file_id}, dimensions={metadata.width}x{metadata.height}")

        return UploadResponse(
            file_id=file_id,
            filename=filename,
            file_path=str(file_path),
            validation=validation,
            satellite_validation=sat_validation,
            metadata=metadata,
            bands=bands,
            thumbnail_url=thumbnail_url,
        )
    except HTTPException:
        if file_path.exists():
            file_path.unlink()
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing raster: {e}")
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing uploaded raster file"
        )


@router.get("/{file_id}/metadata", response_model=MetadataResponse)
async def get_raster_metadata(file_id: str):
    """
    Retrieves detailed metadata and band configuration for an uploaded raster.
    """
    matching_files = list(settings.UPLOAD_DIR.glob(f"{file_id}_*"))
    if not matching_files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Raster file with ID '{file_id}' not found"
        )

    file_path = matching_files[0]
    filename = file_path.name.replace(f"{file_id}_", "")

    metadata = RasterReader.extract_metadata(file_path)
    bands = BandManager.infer_band_configuration(metadata)

    return MetadataResponse(
        file_id=file_id,
        filename=filename,
        metadata=metadata,
        bands=bands,
    )


@router.post("/tile", response_model=TilingResponse)
async def compute_raster_tiles(request: TilingRequest):
    """
    Computes grid tile partition bounds for a high-resolution raster.
    """
    matching_files = list(settings.UPLOAD_DIR.glob(f"{request.file_id}_*"))
    if not matching_files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Raster file with ID '{request.file_id}' not found"
        )

    file_path = matching_files[0]
    metadata = RasterReader.extract_metadata(file_path)
    tiles = RasterTiler.compute_tiles(
        metadata,
        tile_size=request.tile_size,
        overlap=request.overlap
    )

    return TilingResponse(
        file_id=request.file_id,
        total_tiles=len(tiles),
        tile_size=request.tile_size,
        overlap=request.overlap,
        tiles=tiles,
    )


@router.get("/preview/{filename:path}")
async def get_raster_preview(filename: str):
    """
    Serves generated preview thumbnails securely.
    """
    if "\x00" in filename or "\0" in filename or "%00" in filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename contains invalid null characters"
        )

    sanitized_filename = Path(filename).name
    if not sanitized_filename or sanitized_filename in (".", ".."):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid preview filename"
        )

    file_path = (settings.PROCESSED_DIR / sanitized_filename).resolve()
    processed_dir_resolved = settings.PROCESSED_DIR.resolve()

    try:
        file_path.relative_to(processed_dir_resolved)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Path traversal attempt detected"
        )

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Preview image not found"
        )

    return FileResponse(file_path)

