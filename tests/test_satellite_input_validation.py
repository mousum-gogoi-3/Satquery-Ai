import pytest
from pathlib import Path
import numpy as np
from PIL import Image
import tifffile
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch

from backend.main import app
from backend.config import settings
from backend.remote_sensing.satellite_validator import (
    SatelliteImageValidator,
    SatelliteValidationResult
)


def create_mock_geotiff(
    path: Path,
    data: np.ndarray,
    origin_x: float = 500000.0,
    origin_y: float = 6000000.0,
    res_x: float = 10.0,
    res_y: float = 10.0,
    epsg: int = 32633
) -> Path:
    """Helper to create GeoTIFF with explicit geospatial tags."""
    tags = [
        (33550, 'd', 3, (res_x, res_y, 0.0)),
        (33922, 'd', 6, (0.0, 0.0, 0.0, origin_x, origin_y, 0.0)),
        (34735, 'h', 8, (1, 1, 0, 1, 3072, 0, 1, epsg)),
    ]
    kwargs = {"extratags": tags}
    if data.ndim == 3 and data.shape[0] > 1:
        kwargs["planarconfig"] = 'SEPARATE'
    tifffile.imwrite(path, data, **kwargs)
    return path


def create_phone_photo_jpeg(path: Path) -> Path:
    """Simulates an ordinary RGB phone camera photograph saved as JPEG."""
    arr = np.random.randint(20, 240, size=(300, 400, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    # Add simulated camera EXIF (Make: 'Apple', Model: 'iPhone 15')
    exif = img.getexif()
    exif[271] = "Apple"
    exif[272] = "iPhone 15 Pro"
    img.save(path, "JPEG", exif=exif)
    return path


def create_phone_photo_png(path: Path) -> Path:
    """Simulates an ordinary RGB visual screenshot or PNG image."""
    arr = np.random.randint(0, 256, size=(250, 350, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    img.save(path, "PNG")
    return path


def create_movie_poster_png(path: Path) -> Path:
    """Simulates an ordinary movie poster image (800x1200, 3 bands RGB)."""
    arr = np.random.randint(10, 240, size=(1200, 800, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    img.save(path, "PNG")
    return path


def create_screenshot_rgba_png(path: Path, width: int = 415, height: int = 417) -> Path:
    """
    Simulates the exact user scenario:
    A desktop screenshot PNG with an Alpha transparency channel (4 bands RGBA).
    Dimensions: 415x417, 4 bands, uint8.
    """
    arr = np.random.randint(0, 256, size=(height, width, 4), dtype=np.uint8)
    arr[:, :, 3] = 255  # Fully opaque alpha
    img = Image.fromarray(arr, mode="RGBA")
    img.save(path, "PNG")
    return path


def create_plain_unreferenced_tiff(path: Path) -> Path:
    """Simulates a plain 8-bit unreferenced TIFF (like a scanned document or converted photo)."""
    arr = np.random.randint(10, 250, size=(3, 200, 200), dtype=np.uint8)
    tifffile.imwrite(path, arr, planarconfig='SEPARATE')
    return path


# ==============================================================================
# Test 1 — Valid GeoTIFF satellite image -> ACCEPT
# ==============================================================================
@pytest.mark.asyncio
async def test_1_valid_geotiff_accepted(tmp_path: Path, tmp_data_dir):
    """Verifies that a legitimate satellite GeoTIFF is accepted with HTTP 201."""
    data = np.full((3, 128, 128), 120, dtype=np.uint8)
    geo_path = create_mock_geotiff(tmp_path / "valid_sat.tif", data)

    res_val = SatelliteImageValidator.validate_satellite_raster(geo_path)
    assert res_val.valid is True
    assert res_val.classification == "VALID_SATELLITE_RASTER"
    assert res_val.has_crs is True

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        with open(geo_path, "rb") as f:
            res = await client.post("/api/v1/raster/upload", files={"file": ("valid_sat.tif", f, "image/tiff")})
        assert res.status_code == 201
        data_json = res.json()
        assert "file_id" in data_json
        assert data_json["satellite_validation"]["valid"] is True


# ==============================================================================
# Test 2 — Valid multispectral GeoTIFF -> ACCEPT
# ==============================================================================
@pytest.mark.asyncio
async def test_2_valid_multispectral_geotiff_accepted(tmp_path: Path, tmp_data_dir):
    """Verifies that 4-band and higher multispectral GeoTIFFs are accepted."""
    multi_data = np.random.randint(100, 3000, size=(4, 128, 128), dtype=np.uint16)
    multi_path = create_mock_geotiff(tmp_path / "sentinel2_4band.tif", multi_data)

    res_val = SatelliteImageValidator.validate_satellite_raster(multi_path)
    assert res_val.valid is True
    assert res_val.band_count == 4

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        with open(multi_path, "rb") as f:
            res = await client.post("/api/v1/raster/upload", files={"file": ("sentinel2_4band.tif", f, "image/tiff")})
        assert res.status_code == 201


# ==============================================================================
# Test 3 — Valid single-band satellite GeoTIFF -> ACCEPT
# ==============================================================================
@pytest.mark.asyncio
async def test_3_valid_single_band_satellite_geotiff_accepted(tmp_path: Path, tmp_data_dir):
    """Verifies that single-band remote-sensing GeoTIFFs (panchromatic / B8) are accepted."""
    pan_data = np.full((128, 128), 150, dtype=np.uint16)
    pan_path = create_mock_geotiff(tmp_path / "panchromatic_b8.tif", pan_data)

    res_val = SatelliteImageValidator.validate_satellite_raster(pan_path)
    assert res_val.valid is True
    assert res_val.band_count == 1

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        with open(pan_path, "rb") as f:
            res = await client.post("/api/v1/raster/upload", files={"file": ("panchromatic_b8.tif", f, "image/tiff")})
        assert res.status_code == 201


# ==============================================================================
# Test 4 — Normal photograph JPEG -> REJECT
# ==============================================================================
@pytest.mark.asyncio
async def test_4_normal_jpeg_photo_rejected(tmp_path: Path, tmp_data_dir):
    """Verifies that an ordinary phone/selfie JPEG photo is strictly rejected with HTTP 422."""
    jpeg_path = create_phone_photo_jpeg(tmp_path / "selfie.jpg")

    res_val = SatelliteImageValidator.validate_satellite_raster(jpeg_path)
    assert res_val.valid is False
    assert res_val.classification == "NON_SATELLITE_IMAGE"
    assert res_val.error_code == "INVALID_SATELLITE_IMAGE"

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        with open(jpeg_path, "rb") as f:
            res = await client.post("/api/v1/raster/upload", files={"file": ("selfie.jpg", f, "image/jpeg")})
        assert res.status_code == 422
        body = res.json()["detail"]
        assert body["valid"] is False
        assert body["error_code"] == "INVALID_SATELLITE_IMAGE"
        assert "only accepts satellite" in body["message"].lower()


# ==============================================================================
# Test 5 — Normal photograph PNG -> REJECT
# ==============================================================================
@pytest.mark.asyncio
async def test_5_normal_png_photo_rejected(tmp_path: Path, tmp_data_dir):
    """Verifies that an ordinary PNG photograph is strictly rejected with HTTP 422."""
    png_path = create_phone_photo_png(tmp_path / "phone_photo.png")

    res_val = SatelliteImageValidator.validate_satellite_raster(png_path)
    assert res_val.valid is False
    assert res_val.classification == "NON_SATELLITE_IMAGE"
    assert res_val.error_code == "INVALID_SATELLITE_IMAGE"

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        with open(png_path, "rb") as f:
            res = await client.post("/api/v1/raster/upload", files={"file": ("phone_photo.png", f, "image/png")})
        assert res.status_code == 422
        body = res.json()["detail"]
        assert body["valid"] is False
        assert body["error_code"] == "INVALID_SATELLITE_IMAGE"


# ==============================================================================
# Test 6 — Movie poster PNG -> REJECT
# ==============================================================================
@pytest.mark.asyncio
async def test_6_movie_poster_png_rejected(tmp_path: Path, tmp_data_dir):
    """Verifies that an arbitrary movie poster PNG is rejected."""
    poster_path = create_movie_poster_png(tmp_path / "movie_poster.png")

    res_val = SatelliteImageValidator.validate_satellite_raster(poster_path)
    assert res_val.valid is False
    assert res_val.classification == "NON_SATELLITE_IMAGE"
    assert res_val.error_code == "INVALID_SATELLITE_IMAGE"

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        with open(poster_path, "rb") as f:
            res = await client.post("/api/v1/raster/upload", files={"file": ("movie_poster.png", f, "image/png")})
        assert res.status_code == 422
        body = res.json()["detail"]
        assert body["valid"] is False
        assert body["error_code"] == "INVALID_SATELLITE_IMAGE"


# ==============================================================================
# Test 7 — Screenshot PNG (User reported 4-band RGBA 415x417 screenshot) -> REJECT
# ==============================================================================
@pytest.mark.asyncio
async def test_7_screenshot_rgba_png_rejected(tmp_path: Path, tmp_data_dir):
    """
    Tests the exact reported bug:
    A desktop screenshot PNG with 4 bands (RGBA) and dimensions 415x417.
    The validator MUST NOT trust '4 bands' as satellite imagery.
    It MUST reject the screenshot with HTTP 422 and NEVER invoke VQA.
    """
    screenshot_path = create_screenshot_rgba_png(
        tmp_path / "Screenshot 2026-07-28 233643.png",
        width=415,
        height=417
    )

    # 1. Unit level validation check
    res_val = SatelliteImageValidator.validate_satellite_raster(screenshot_path)
    assert res_val.valid is False
    assert res_val.classification == "NON_SATELLITE_IMAGE"
    assert res_val.error_code == "INVALID_SATELLITE_IMAGE"
    assert res_val.band_count == 4
    assert res_val.width == 415
    assert res_val.height == 417

    # 2. Upload endpoint check
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        with open(screenshot_path, "rb") as f:
            res = await client.post(
                "/api/v1/raster/upload",
                files={"file": ("Screenshot 2026-07-28 233643.png", f, "image/png")}
            )
        assert res.status_code == 422
        body = res.json()["detail"]
        assert body["valid"] is False
        assert body["error_code"] == "INVALID_SATELLITE_IMAGE"
        assert "only accepts satellite" in body["message"].lower()


# ==============================================================================
# Test 8 — JPEG renamed to .tif -> REJECT
# ==============================================================================
@pytest.mark.asyncio
async def test_8_renamed_jpeg_to_tiff_rejected(tmp_path: Path, tmp_data_dir):
    """
    Mandatory test: A normal JPEG photo renamed to .tif must NOT bypass validation.
    The validator must inspect internal file structure, discover no CRS/GeoTIFF tags, and reject.
    """
    real_jpeg = create_phone_photo_jpeg(tmp_path / "normal_photo.jpg")
    renamed_tif = tmp_path / "normal_photo.tif"
    renamed_tif.write_bytes(real_jpeg.read_bytes())

    res_val = SatelliteImageValidator.validate_satellite_raster(renamed_tif)
    assert res_val.valid is False
    assert res_val.classification == "NON_SATELLITE_IMAGE"
    assert res_val.error_code == "INVALID_SATELLITE_IMAGE"

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        with open(renamed_tif, "rb") as f:
            res = await client.post("/api/v1/raster/upload", files={"file": ("normal_photo.tif", f, "image/tiff")})
        assert res.status_code == 422
        body = res.json()["detail"]
        assert body["valid"] is False
        assert body["error_code"] == "INVALID_SATELLITE_IMAGE"


# ==============================================================================
# Test 9 — Random TIFF with no meaningful geospatial metadata -> REJECT
# ==============================================================================
@pytest.mark.asyncio
async def test_9_random_non_satellite_tiff_rejected(tmp_path: Path, tmp_data_dir):
    """Verifies that a plain 8-bit unreferenced TIFF is rejected."""
    plain_tif = create_plain_unreferenced_tiff(tmp_path / "document_scan.tif")

    res_val = SatelliteImageValidator.validate_satellite_raster(plain_tif)
    assert res_val.valid is False
    assert res_val.classification == "NON_SATELLITE_IMAGE"
    assert res_val.error_code == "INVALID_SATELLITE_IMAGE"

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        with open(plain_tif, "rb") as f:
            res = await client.post("/api/v1/raster/upload", files={"file": ("document_scan.tif", f, "image/tiff")})
        assert res.status_code == 422
        body = res.json()["detail"]
        assert body["valid"] is False
        assert body["error_code"] == "INVALID_SATELLITE_IMAGE"


# ==============================================================================
# Test 10 — Invalid image sent directly to VQA endpoint -> REJECT BEFORE MODEL INVOCATION
# ==============================================================================
@pytest.mark.asyncio
async def test_10_vqa_cannot_process_invalid_image(tmp_path: Path, tmp_data_dir):
    """Verifies that an invalid image file placed on disk is rejected by /analysis/vqa before VQA engine runs."""
    fake_file_id = "test-invalid-vqa-uuid"
    bad_img_path = settings.UPLOAD_DIR / f"{fake_file_id}_poster.png"
    create_movie_poster_png(bad_img_path)

    with patch("backend.models.vqa.VQAEngine.answer_question") as mock_vqa:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            res = await client.post(
                "/api/v1/analysis/vqa",
                json={"file_id": fake_file_id, "question": "Explain this image"}
            )
        assert res.status_code == 422
        assert res.json()["detail"]["error_code"] == "INVALID_SATELLITE_IMAGE"
        # Verify VQA model was NEVER invoked
        mock_vqa.assert_not_called()


# ==============================================================================
# Test 11 — Invalid image sent directly to caption endpoint -> REJECT BEFORE MODEL INVOCATION
# ==============================================================================
@pytest.mark.asyncio
async def test_11_caption_cannot_process_invalid_image(tmp_path: Path, tmp_data_dir):
    """Verifies that an invalid image file placed on disk is rejected by /analysis/caption before caption engine runs."""
    fake_file_id = "test-invalid-cap-uuid"
    bad_img_path = settings.UPLOAD_DIR / f"{fake_file_id}_screenshot.png"
    create_screenshot_rgba_png(bad_img_path)

    with patch("backend.models.caption.CaptionEngine.generate_caption") as mock_cap:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            res = await client.post(
                "/api/v1/analysis/caption",
                json={"file_id": fake_file_id, "detail_level": "standard"}
            )
        assert res.status_code == 422
        assert res.json()["detail"]["error_code"] == "INVALID_SATELLITE_IMAGE"
        mock_cap.assert_not_called()


# ==============================================================================
# Test 12 — Invalid image sent directly to grounding endpoint -> REJECT BEFORE MODEL INVOCATION
# ==============================================================================
@pytest.mark.asyncio
async def test_12_grounding_cannot_process_invalid_image(tmp_path: Path, tmp_data_dir):
    """Verifies that an invalid image is rejected by /analysis/grounded before grounding engine runs."""
    fake_file_id = "test-invalid-ground-uuid"
    bad_img_path = settings.UPLOAD_DIR / f"{fake_file_id}_unrelated.tif"
    create_plain_unreferenced_tiff(bad_img_path)

    with patch("backend.models.grounding.GroundingEngine.detect_evidence") as mock_ground:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            res = await client.post(
                "/api/v1/analysis/grounded",
                json={"file_id": fake_file_id, "query": "Where are the cars?"}
            )
        assert res.status_code == 422
        assert res.json()["detail"]["error_code"] == "INVALID_SATELLITE_IMAGE"
        mock_ground.assert_not_called()


# ==============================================================================
# Test 13 — Invalid image used as before-image in change analysis -> REJECT
# ==============================================================================
@pytest.mark.asyncio
async def test_13_change_analysis_rejects_invalid_before_image(tmp_path: Path, tmp_data_dir):
    """Verifies that /analysis/change rejects requests if before_image is a non-satellite image."""
    invalid_before_id = "test-invalid-before-uuid"
    valid_after_id = "test-valid-after-uuid"

    create_screenshot_rgba_png(settings.UPLOAD_DIR / f"{invalid_before_id}_shot.png")
    valid_data = np.full((3, 100, 100), 100, dtype=np.uint8)
    create_mock_geotiff(settings.UPLOAD_DIR / f"{valid_after_id}_after.tif", valid_data)

    with patch("backend.models.temporal_engine.BiTemporalAnalysisEngine.analyze_change") as mock_change:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            res = await client.post(
                "/api/v1/analysis/change",
                json={
                    "file_id_before": invalid_before_id,
                    "file_id_after": valid_after_id,
                    "query": "What changed?"
                }
            )
        assert res.status_code == 422
        assert res.json()["detail"]["error_code"] == "INVALID_SATELLITE_IMAGE"
        mock_change.assert_not_called()


# ==============================================================================
# Test 14 — Invalid image used as after-image in change analysis -> REJECT
# ==============================================================================
@pytest.mark.asyncio
async def test_14_change_analysis_rejects_invalid_after_image(tmp_path: Path, tmp_data_dir):
    """Verifies that /analysis/change rejects requests if after_image is a non-satellite image."""
    valid_before_id = "test-valid-before-uuid"
    invalid_after_id = "test-invalid-after-uuid"

    valid_data = np.full((3, 100, 100), 100, dtype=np.uint8)
    create_mock_geotiff(settings.UPLOAD_DIR / f"{valid_before_id}_before.tif", valid_data)
    create_movie_poster_png(settings.UPLOAD_DIR / f"{invalid_after_id}_poster.png")

    with patch("backend.models.temporal_engine.BiTemporalAnalysisEngine.analyze_change") as mock_change:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            res = await client.post(
                "/api/v1/analysis/change",
                json={
                    "file_id_before": valid_before_id,
                    "file_id_after": invalid_after_id,
                    "query": "What changed?"
                }
            )
        assert res.status_code == 422
        assert res.json()["detail"]["error_code"] == "INVALID_SATELLITE_IMAGE"
        mock_change.assert_not_called()
