from pathlib import Path
from backend.config import settings


def test_settings_initialization():
    assert settings.APP_NAME == "SatQuery AI"
    assert settings.DEFAULT_TILE_SIZE == 512
    assert ".tif" in settings.ALLOWED_EXTENSIONS
    assert ".geotiff" in settings.ALLOWED_EXTENSIONS
    assert ".png" in settings.ALLOWED_EXTENSIONS


def test_directory_creation():
    settings.ensure_directories()
    assert settings.DATA_DIR.exists()
    assert settings.UPLOAD_DIR.exists()
    assert settings.PROCESSED_DIR.exists()
    assert settings.RESULTS_DIR.exists()
