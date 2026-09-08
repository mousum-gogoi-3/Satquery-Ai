import time
from pathlib import Path
from fastapi import APIRouter, HTTPException, status
import numpy as np

from backend.config import settings
from backend.utils.logger import get_logger
from backend.api.schemas import (
    VQARequest, VQAResponse,
    CaptionRequest, CaptionResponse,
    GroundedAnalysisRequest, GroundedAnalysisResponse,
    TemporalAnalysisRequest, TemporalAnalysisResponse
)
from backend.models.vqa import VQAEngine
from backend.models.caption import CaptionEngine
from backend.models.grounding import GroundingEngine
from backend.models.temporal_engine import BiTemporalAnalysisEngine
from backend.utils.rendering import render_visual_evidence

logger = get_logger(__name__)
router = APIRouter(prefix="/analysis", tags=["Remote-Sensing AI Analysis"])

# Lazy engine singletons
vqa_engine = VQAEngine()
caption_engine = CaptionEngine()
grounding_engine = GroundingEngine()
temporal_engine = BiTemporalAnalysisEngine()


def _resolve_uploaded_file(file_id: str) -> Path:
    """Helper to locate uploaded raster file by file_id in UPLOAD_DIR."""
    if "\x00" in file_id or "%00" in file_id or ".." in file_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file_id parameter"
        )
    matching_files = list(settings.UPLOAD_DIR.glob(f"{file_id}_*"))
    if not matching_files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Raster file with ID '{file_id}' not found"
        )
    return matching_files[0]


@router.post("/vqa", response_model=VQAResponse, status_code=status.HTTP_200_OK)
async def perform_vqa(request: VQARequest):
    """
    Performs Visual Question Answering (VQA) on an uploaded satellite raster.
    """
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question text cannot be empty"
        )

    file_path = _resolve_uploaded_file(request.file_id)

    start_time = time.perf_counter()
    try:
        answer, confidence = vqa_engine.answer_question(file_path, request.question)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Raster file with ID '{request.file_id}' not found"
        )
    except Exception as e:
        logger.error(f"VQA execution failed for file_id={request.file_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during VLM Visual Question Answering analysis."
        )

    elapsed = time.perf_counter() - start_time
    logger.info(f"Completed VQA for file_id={request.file_id} in {elapsed:.4f}s")

    return VQAResponse(
        file_id=request.file_id,
        question=request.question,
        answer=answer,
        confidence=confidence,
        processing_time_sec=round(elapsed, 4),
        model_name=vqa_engine.model_name,
    )


@router.post("/caption", response_model=CaptionResponse, status_code=status.HTTP_200_OK)
async def generate_scene_caption(request: CaptionRequest):
    """
    Generates a natural-language description and infers land-cover concepts for a satellite raster scene.
    """
    file_path = _resolve_uploaded_file(request.file_id)

    start_time = time.perf_counter()
    try:
        caption, concepts = caption_engine.generate_caption(file_path, detail_level=request.detail_level)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Raster file with ID '{request.file_id}' not found"
        )
    except Exception as e:
        logger.error(f"Caption execution failed for file_id={request.file_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during VLM scene captioning analysis."
        )

    elapsed = time.perf_counter() - start_time
    logger.info(f"Completed Caption generation for file_id={request.file_id} in {elapsed:.4f}s")

    return CaptionResponse(
        file_id=request.file_id,
        caption=caption,
        detected_concepts=concepts,
        processing_time_sec=round(elapsed, 4),
        model_name=caption_engine.model_name,
    )


@router.post("/grounded", response_model=GroundedAnalysisResponse, status_code=status.HTTP_200_OK)
async def perform_grounded_analysis(request: GroundedAnalysisRequest):
    """
    Performs spatial grounding analysis on a satellite scene, returning both semantic text answers
    and visual evidence bounding boxes rendered on an output visualization image.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query text cannot be empty"
        )

    file_path = _resolve_uploaded_file(request.file_id)

    start_time = time.perf_counter()
    try:
        # 1. Semantic reasoning via VQA Engine
        answer, vqa_conf = vqa_engine.answer_question(file_path, request.query)

        # 2. Spatial Grounding evidence detection via Grounding Engine
        evidence_regions, img_dims, model_info = grounding_engine.detect_evidence(
            file_path=file_path,
            query=request.query,
            confidence_threshold=request.confidence_threshold
        )

        # Combine model metadata
        model_info["vlm_model"] = vqa_engine.model_name

        # 3. Render visual evidence visualization image (preserves source file completely)
        rendered_image_path = render_visual_evidence(
            source_image=file_path,
            regions=evidence_regions,
            filename_prefix=f"evidence_{request.file_id}"
        )

        # Relative path for API response
        try:
            rel_path = str(rendered_image_path.relative_to(settings.BASE_DIR)).replace("\\", "/")
        except ValueError:
            rel_path = str(rendered_image_path)

        overall_conf = round(
            float(np.mean([r.confidence for r in evidence_regions])) if evidence_regions else vqa_conf,
            4
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Raster file with ID '{request.file_id}' not found"
        )
    except Exception as e:
        logger.error(f"Grounded analysis failed for file_id={request.file_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during spatial grounding analysis: {str(e)}"
        )

    elapsed = time.perf_counter() - start_time
    logger.info(f"Completed Grounded Analysis for file_id={request.file_id} in {elapsed:.4f}s")

    return GroundedAnalysisResponse(
        file_id=request.file_id,
        query=request.query,
        answer=answer,
        evidence_regions=evidence_regions,
        overall_confidence=overall_conf,
        evidence_image_path=rel_path,
        source_image_dims=img_dims,
        model_info=model_info,
        processing_time_sec=round(elapsed, 4)
    )


@router.post("/change", response_model=TemporalAnalysisResponse, status_code=status.HTTP_200_OK)
async def perform_temporal_change_analysis(request: TemporalAnalysisRequest):
    """
    Performs bi-temporal change detection and spatial grounding analysis between two satellite images.
    Accepts:
      - file_id_before: Earlier satellite observation
      - file_id_after: Later satellite observation
      - query: Natural-language query describing desired change analysis
    Returns:
      - Full spatial registration diagnostics
      - Detected change regions with bounding boxes, spectral deltas, and confidence
      - Quantified change statistics and dominant change category
      - Multi-panel visual evidence and heatmap image paths
      - Natural-language explanation
    """
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query text cannot be empty"
        )

    file_path_before = _resolve_uploaded_file(request.file_id_before)
    file_path_after = _resolve_uploaded_file(request.file_id_after)

    try:
        response = temporal_engine.analyze_change(
            before_path=file_path_before,
            after_path=file_path_after,
            file_id_before=request.file_id_before,
            file_id_after=request.file_id_after,
            query=request.query,
            confidence_threshold=request.confidence_threshold,
            target_change_types=request.target_change_types,
            debug_mode=request.debug_mode
        )
        return response
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Temporal change analysis failed for before={request.file_id_before}, after={request.file_id_after}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during bi-temporal change analysis: {str(e)}"
        )

