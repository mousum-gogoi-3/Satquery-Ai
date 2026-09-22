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

    # Vision-Language Model (VLM) Settings - Phase 2
    VLM_MODEL_NAME: str = "Qwen/Qwen2-VL-2B-Instruct"
    VLM_DEVICE: str = "auto"  # "cuda", "cpu", or "auto"
    USE_MOCK_VLM: bool = False  # Real Qwen2-VL VLM by default on RTX 5060; overridable via SATQUERY_USE_MOCK_VLM
    MAX_VLM_TOKENS: int = 512

    # Spatial Grounding & Evidence Settings - Phase 3
    GROUNDING_MODEL_NAME: str = "google/owlv2-base-patch16-ensemble"
    GROUNDING_CONF_THRESHOLD: float = 0.005
    GROUNDING_DEVICE: str = "auto"  # "cuda", "cpu", or "auto"
    USE_MOCK_GROUNDING: bool = False  # Production API uses real OWLv2 by default
    GROUNDING_TILE_SIZE: int = 256
    GROUNDING_TILE_OVERLAP: float = 0.25
    GROUNDING_NMS_IOU_THRESHOLD: float = 0.35
    GROUNDING_MAX_EVIDENCE_REGIONS: int = 50

    # Bi-Temporal Change Detection Settings - Phase 4
    TEMPORAL_CHANGE_THRESHOLD: float = 0.22
    MIN_CHANGE_REGION_AREA: int = 40
    MAX_CHANGE_REGION_AREA_PCT: float = 0.40
    NDVI_CHANGE_THRESHOLD: float = 0.18
    NDWI_CHANGE_THRESHOLD: float = 0.15
    NDBI_CHANGE_THRESHOLD: float = 0.15
    TEMPORAL_MAX_CHANGE_REGIONS: int = 10
    TEMPORAL_CONFIDENCE_THRESHOLD: float = 0.35
    TEMPORAL_ADAPTIVE_THRESHOLD_SIGMA: float = 3.0
    TEMPORAL_MIN_MAGNITUDE_FLOOR: float = 0.04
    TEMPORAL_CHANGE_THRESHOLD_MIN: float = 0.06
    TEMPORAL_CHANGE_THRESHOLD_MAX: float = 0.35
    REGISTRATION_FEATURE_MAX_KEYPOINTS: int = 1000
    REGISTRATION_MIN_MATCHES: int = 8
    REGISTRATION_MIN_OVERLAP_PCT: float = 0.05

    # Input-Pair Validation Settings - Phase 4 Bugfix
    TEMPORAL_IDENTICAL_IMAGE_THRESHOLD: float = 0.999
    TEMPORAL_NEAR_DUPLICATE_THRESHOLD: float = 0.980
    TEMPORAL_NEAR_DUPLICATE_DIFF_THRESHOLD: float = 0.015
    TEMPORAL_PAIR_SIMILARITY_THRESHOLD: float = 0.40
    TEMPORAL_MIN_GEOSPATIAL_OVERLAP_PCT: float = 10.0
    TEMPORAL_MIN_FEATURE_INLIERS: int = 6

    def ensure_directories(self) -> None:
        """Create necessary data directories if they do not exist."""
        for path in [self.DATA_DIR, self.UPLOAD_DIR, self.PROCESSED_DIR, self.RESULTS_DIR]:
            path.mkdir(parents=True, exist_ok=True)


# Global settings singleton instance
settings = Settings()
settings.ensure_directories()
