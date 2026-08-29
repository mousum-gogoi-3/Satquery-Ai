import os
from pathlib import Path
from typing import Set, List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralized configuration management for SatQuery AI.
    Environment variables override defaults seamlessly.
    """
    model_config = SettingsConfigDict(
        env_prefix="SATQUERY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Project Paths
    BASE_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    DATA_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data")
    UPLOAD_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "uploads")
    PROCESSED_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "processed")
    RESULTS_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "results")

    # Upload Rules
    MAX_FILE_SIZE_BYTES: int = 500 * 1024 * 1024  # 500 MB limit
    ALLOWED_EXTENSIONS: Set[str] = {".tif", ".tiff", ".geotiff", ".png", ".jpg", ".jpeg", ".jp2"}
    ALLOWED_MIME_TYPES: Set[str] = {
        "image/tiff",
        "image/geotiff",
        "image/png",
        "image/jpeg",
        "image/jp2",
        "application/octet-stream",
    }

    # Raster Processing & Tiling Defaults
    DEFAULT_TILE_SIZE: int = 512
    DEFAULT_TILE_OVERLAP: int = 0
    MAX_RASTER_DIMENSION: int = 50000  # Max width/height to prevent OOM
    MAX_BANDS: int = 256

    # Application Settings
    APP_NAME: str = "SatQuery AI"
    API_PREFIX: str = "/api/v1"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    def ensure_directories(self) -> None:
        """Create necessary data directories if they do not exist."""
        for path in [self.DATA_DIR, self.UPLOAD_DIR, self.PROCESSED_DIR, self.RESULTS_DIR]:
            path.mkdir(parents=True, exist_ok=True)


# Global settings singleton instance
settings = Settings()
settings.ensure_directories()
