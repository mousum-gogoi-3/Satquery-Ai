from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field

from backend.remote_sensing.validator import ValidationResult
from backend.remote_sensing.raster import RasterMetadata
from backend.remote_sensing.bands import BandInfo
from backend.remote_sensing.tiling import TileMetadata


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    debug: bool
    engine_ready: bool
    allowed_formats: List[str]


class UploadResponse(BaseModel):
    file_id: str
    filename: str
    file_path: str
    validation: ValidationResult
    metadata: Optional[RasterMetadata] = None
    bands: List[BandInfo] = Field(default_factory=list)
    thumbnail_url: Optional[str] = None


class MetadataResponse(BaseModel):
    file_id: str
    filename: str
    metadata: RasterMetadata
    bands: List[BandInfo]


class TilingRequest(BaseModel):
    file_id: str
    tile_size: int = Field(default=512, ge=64, le=4096)
    overlap: int = Field(default=0, ge=0, le=512)


class TilingResponse(BaseModel):
    file_id: str
    total_tiles: int
    tile_size: int
    overlap: int
    tiles: List[TileMetadata]


# Single-Image AI Schemas - Phase 2
class VQARequest(BaseModel):
    file_id: str
    question: str = Field(..., min_length=1, description="Natural-language question about the satellite scene.")
    tile_id: Optional[str] = None


class VQAResponse(BaseModel):
    file_id: str
    question: str
    answer: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    processing_time_sec: float
    model_name: str


class CaptionRequest(BaseModel):
    file_id: str
    detail_level: str = Field(default="detailed", description="Detail level: 'brief', 'normal', or 'detailed'")



class CaptionResponse(BaseModel):
    file_id: str
    caption: str
    detected_concepts: List[str] = Field(
        default_factory=list,
        description="Land-cover and scene concepts inferred by the VLM pipeline (unverified by spatial grounding/evidence)."
    )
    processing_time_sec: float
    model_name: str

