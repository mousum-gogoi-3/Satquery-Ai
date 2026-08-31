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


# Grounding & Visual Evidence Schemas - Phase 3
class BoundingBox(BaseModel):
    xmin: float = Field(..., description="Minimum x coordinate (pixels)")
    ymin: float = Field(..., description="Minimum y coordinate (pixels)")
    xmax: float = Field(..., description="Maximum x coordinate (pixels)")
    ymax: float = Field(..., description="Maximum y coordinate (pixels)")


class EvidenceRegion(BaseModel):
    box: BoundingBox
    label: str = Field(..., description="Detected land-cover, feature, or target label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    explanation: Optional[str] = Field(default=None, description="Optional reasoning or spectral explanation")
    owlv2_score: Optional[float] = Field(default=None, description="Raw zero-shot OWLv2 semantic score")
    spectral_score: Optional[float] = Field(default=None, description="Multispectral index match score (NDVI/NDWI/NDBI)")
    spatial_score: Optional[float] = Field(default=None, description="Spatial morphology / connected component score")
    shape_score: Optional[float] = Field(default=None, description="Shape, texture, or linearity score")
    final_score: Optional[float] = Field(default=None, description="Final candidate fusion score")
    source_tile: Optional[str] = Field(default=None, description="Source tile origin or detection scale")
    merged_detection_count: Optional[int] = Field(default=1, description="Number of candidate detections merged into this region")


class ImageDimensions(BaseModel):
    width: int = Field(..., ge=1, description="Image width in pixels")
    height: int = Field(..., ge=1, description="Image height in pixels")


class GroundedAnalysisRequest(BaseModel):
    file_id: str
    query: str = Field(..., min_length=1, description="Target query or concept to ground visually (e.g. 'Where are the buildings?').")
    confidence_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Detection confidence cut-off threshold.")


class GroundedAnalysisResponse(BaseModel):
    file_id: str
    query: str
    answer: str = Field(..., description="Textual VQA answer from VLM engine")
    evidence_regions: List[EvidenceRegion] = Field(default_factory=list, description="Spatial bounding boxes of detected visual evidence")
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    evidence_image_path: Optional[str] = Field(default=None, description="Relative path or URI to the rendered visual evidence image")
    source_image_dims: ImageDimensions
    model_info: Dict[str, str] = Field(default_factory=dict, description="Metadata on models used for VQA and spatial grounding")
    processing_time_sec: float


