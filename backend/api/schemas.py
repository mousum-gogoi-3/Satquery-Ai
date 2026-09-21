from enum import Enum
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
    coord_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Detailed coordinate transformation and debug metadata")


class ImageDimensions(BaseModel):
    width: int = Field(..., ge=1, description="Image width in pixels")
    height: int = Field(..., ge=1, description="Image height in pixels")


class GroundedAnalysisRequest(BaseModel):
    file_id: str
    query: str = Field(..., min_length=1, description="Target query or concept to ground visually (e.g. 'Where are the buildings?').")
    confidence_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Detection confidence cut-off threshold.")
    max_regions: Optional[int] = Field(default=None, ge=1, le=500, description="Maximum number of evidence regions to return.")
    debug_mode: bool = Field(default=False, description="Enable debug diagnostics and candidate inspection.")


class GroundedAnalysisResponse(BaseModel):
    file_id: str
    query: str
    answer: str = Field(..., description="Textual VQA answer from VLM engine")
    evidence_regions: List[EvidenceRegion] = Field(default_factory=list, description="Spatial bounding boxes of detected visual evidence")
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    evidence_image_path: Optional[str] = Field(default=None, description="Relative path or URI to the rendered visual evidence image")
    source_image_dims: ImageDimensions
    model_info: Dict[str, Any] = Field(default_factory=dict, description="Metadata on models used for VQA and spatial grounding")
    processing_time_sec: float


# Bi-Temporal Change Detection Schemas - Phase 4
class ChangeType(str, Enum):
    VEGETATION_LOSS = "vegetation_loss"
    VEGETATION_GAIN = "vegetation_gain"
    WATER_EXPANSION = "water_expansion"
    WATER_REDUCTION = "water_reduction"
    NEW_BUILT_UP = "new_built_up"
    BUILT_UP_LOSS = "built_up_loss"
    ROAD_INFRASTRUCTURE_CHANGE = "road_infrastructure_change"
    SURFACE_CHANGE = "surface_change"
    GENERAL_LAND_COVER_CHANGE = "general_land_cover_change"
    BARE_SOIL_CHANGE = "bare_soil_change"
    UNKNOWN_CHANGE = "unknown_change"


class RegistrationQuality(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    FAILED = "FAILED"


class RegistrationDiagnostics(BaseModel):
    method_applied: str = Field(..., description="Alignment method applied: e.g. 'geospatial_reprojection', 'feature_homography', 'pixel_grid_resample'")
    overlap_percentage: float = Field(..., ge=0.0, le=100.0, description="Estimated spatial overlap percentage between images")
    overlap_bounds: Optional[Tuple[float, float, float, float]] = Field(default=None, description="Geographic or pixel bounds of overlapping intersection")
    original_dims_before: Tuple[int, int] = Field(..., description="(width, height) of before image")
    original_dims_after: Tuple[int, int] = Field(..., description="(width, height) of after image")
    aligned_dims: Tuple[int, int] = Field(..., description="(width, height) of aligned raster grid")
    resolution_before: Optional[Tuple[float, float]] = Field(default=None, description="(res_x, res_y) in ground units")
    resolution_after: Optional[Tuple[float, float]] = Field(default=None, description="(res_x, res_y) in ground units")
    crs_before: Optional[str] = Field(default=None, description="CRS of before raster")
    crs_after: Optional[str] = Field(default=None, description="CRS of after raster")
    registration_quality: RegistrationQuality = Field(default=RegistrationQuality.HIGH, description="Estimated quality of image registration")


class SpectralEvidence(BaseModel):
    delta_ndvi: float = Field(..., description="Average change in Normalized Difference Vegetation Index (NDVI_after - NDVI_before)")
    delta_ndwi: float = Field(..., description="Average change in Normalized Difference Water Index (NDWI_after - NDWI_before)")
    delta_ndbi: float = Field(..., description="Average change in Built-up / Brightness Index (NDBI_after - NDBI_before)")
    delta_rgb: float = Field(..., description="Average Euclidean normalized RGB / color reflectance difference")
    delta_sar_vv: Optional[float] = Field(default=None, description="Change in SAR VV polarization backscatter if available")
    delta_sar_vh: Optional[float] = Field(default=None, description="Change in SAR VH polarization backscatter if available")


class ChangeRegion(BaseModel):
    region_id: str = Field(..., description="Unique identifier for the detected change region")
    change_type: ChangeType = Field(..., description="Classified temporal change category")
    bounding_box: BoundingBox = Field(..., description="Spatial bounding box on the aligned image canvas")
    area_pixels: int = Field(..., ge=1, description="Area of changed region in pixels")
    area_percentage: float = Field(..., ge=0.0, le=100.0, description="Percentage of scene canvas occupied by this change")
    area_sq_meters: Optional[float] = Field(default=None, description="Estimated ground surface area in square meters (if resolution known)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall classification confidence score")
    change_score: float = Field(..., ge=0.0, le=1.0, description="Normalized physical change magnitude score")
    spectral_evidence: SpectralEvidence = Field(..., description="Multispectral differential evidence metrics")
    semantic_evidence: Optional[Dict[str, Any]] = Field(default=None, description="OWLv2 or semantic differential object grounding scores")
    explanation: str = Field(..., description="Human-readable physical reasoning for this detected change")


class TemporalAnalysisRequest(BaseModel):
    file_id_before: str = Field(..., min_length=1, description="File ID of the earlier 'Before' satellite raster")
    file_id_after: str = Field(..., min_length=1, description="File ID of the later 'After' satellite raster")
    query: str = Field(default="What changed?", min_length=1, description="Natural-language question regarding changes between the two rasters")
    confidence_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Optional detection confidence cut-off threshold")
    target_change_types: Optional[List[ChangeType]] = Field(default=None, description="Optional target change category filter")
    debug_mode: bool = Field(default=False, description="Enable debug overlays and diagnostic masks")


class TemporalAnalysisResponse(BaseModel):
    file_id_before: str
    file_id_after: str
    query: str
    status: str = Field(..., description="Status string: 'SUCCESS', 'NO_SIGNIFICANT_CHANGE', 'NO_SPATIAL_OVERLAP', or 'ERROR'")
    summary: str = Field(..., description="Natural-language synthesis of temporal changes detected")
    dominant_change_type: Optional[ChangeType] = Field(default=None, description="Primary detected change type in the scene")
    total_changed_area_pixels: int = Field(default=0, description="Total changed surface in pixels across all detected regions")
    total_changed_area_percentage: float = Field(default=0.0, description="Total changed percentage of the scene canvas")
    total_changed_area_sq_meters: Optional[float] = Field(default=None, description="Total changed surface in square meters")
    change_regions: List[ChangeRegion] = Field(default_factory=list, description="Extracted individual change regions with bounding boxes and evidence")
    registration_diagnostics: RegistrationDiagnostics = Field(..., description="Diagnostics on spatial alignment and overlap")
    evidence_image_path: Optional[str] = Field(default=None, description="Relative path to multi-panel visual evidence image")
    change_heatmap_path: Optional[str] = Field(default=None, description="Relative path to change magnitude heatmap image")
    processing_time_sec: float



