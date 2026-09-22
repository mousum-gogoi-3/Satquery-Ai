import time
from pathlib import Path
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
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
from backend.models.router import SatQueryRouter, QueryType
from backend.remote_sensing.raster import RasterReader
from backend.remote_sensing.satellite_validator import SatelliteImageValidator
from backend.remote_sensing.pair_validator import TemporalPairValidator, PairClassification
from backend.api.schemas import ImageDimensions
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
    if not file_id or not str(file_id).strip() or "\x00" in file_id or "%00" in file_id or ".." in file_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file_id parameter"
        )
    fid = str(file_id).strip()
    matching_files = list(settings.UPLOAD_DIR.glob(f"{fid}_*"))
    if not matching_files:
        matching_files = list(settings.UPLOAD_DIR.glob(f"{fid}.*"))
    if not matching_files and (settings.UPLOAD_DIR / fid).is_file():
        matching_files = [settings.UPLOAD_DIR / fid]
    if not matching_files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Raster file with ID '{file_id}' not found"
        )
    file_path = matching_files[0]

    # Enforce authoritative satellite validation before allowing any AI analysis
    sat_validation = SatelliteImageValidator.validate_satellite_raster(file_path)
    if not sat_validation.valid:
        logger.warning(
            f"Analysis endpoint blocked invalid raster '{file_path.name}' (file_id='{file_id}'): {sat_validation.reason}"
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

    return file_path


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

    # Critical Architecture Guard: If two images are provided and are identical, return immediately
    if request.file_id_after and str(request.file_id_after).strip():
        file_path_after = _resolve_uploaded_file(request.file_id_after)
        pair_diag = TemporalPairValidator.validate_pair(file_path, file_path_after)
        if pair_diag.classification in (PairClassification.IDENTICAL_IMAGES, PairClassification.NEAR_DUPLICATE):
            logger.info(
                f"VQA endpoint intercepted query on identical pair: '{request.question}'"
            )
            return VQAResponse(
                file_id=request.file_id,
                question=request.question,
                answer="The uploaded Before and After images are identical. No temporal changes detected.",
                confidence=1.0,
                processing_time_sec=0.0,
                model_name="identical_pair_guard",
                image_used=False,
            )

    # Guard against single-image VQA hallucinating temporal changes
    if SatQueryRouter.is_temporal_change_query(request.question):
        logger.info(
            f"VQA endpoint blocked temporal comparison query on single image: '{request.question}'"
        )
        return VQAResponse(
            file_id=request.file_id,
            question=request.question,
            answer=(
                "Temporal change comparison requires two satellite observations (Before and After images). "
                "To analyze changes over time, please upload both images and run Change Analysis."
            ),
            confidence=1.0,
            processing_time_sec=0.0,
            model_name="routing_guard",
            image_used=False,
        )

    logger.info(
        f"VQA endpoint called | file_id='{request.file_id}' | question='{request.question}' | resolved_file='{file_path.name}'"
    )

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
            detail=f"Visual Question Answering analysis failed: {e}"
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
        image_used=True,
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
@router.post("/query", response_model=GroundedAnalysisResponse, status_code=status.HTTP_200_OK)
async def perform_grounded_analysis(request: GroundedAnalysisRequest):
    """
    Intelligently routes natural-language user queries to the appropriate capability:
      1. General Image Understanding (Captioning/VQA) - Scene descriptions, explain queries
      2. Spatial Grounding - Object localization & bounding box evidence image
      3. Counting Questions - Object detection, verification, and quantification
      4. Temporal Change Detection
    """
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query text cannot be empty"
        )

    file_path = _resolve_uploaded_file(request.file_id)
    start_time = time.perf_counter()

    second_id = request.file_id_after or request.second_file_id
    has_second_image = bool(second_id and str(second_id).strip())

    # ------------------------------------------------------------------
    # CRITICAL ARCHITECTURAL GUARD:
    # When two images are present, check if Before and After are identical
    # BEFORE query-intent routing.
    # ------------------------------------------------------------------
    if has_second_image:
        file_path_after = _resolve_uploaded_file(second_id)
        pair_diag = TemporalPairValidator.validate_pair(file_path, file_path_after)
        if pair_diag.classification in (PairClassification.IDENTICAL_IMAGES, PairClassification.NEAR_DUPLICATE):
            logger.info(
                f"[SatQuery Architecture Guard] Before ({request.file_id}) and After ({second_id}) are IDENTICAL. "
                f"Returning identical-image result immediately without downstream VQA/grounding model invocation."
            )
            meta = RasterReader.extract_metadata(file_path)
            img_dims = ImageDimensions(width=meta.width, height=meta.height)
            elapsed = time.perf_counter() - start_time
            return GroundedAnalysisResponse(
                file_id=request.file_id,
                query=request.query,
                answer="The uploaded Before and After images are identical. No temporal changes detected.",
                analysis_type=QueryType.TEMPORAL_CHANGE.value,
                target_concept=None,
                count=0,
                evidence_regions=[],
                overall_confidence=1.0,
                evidence_image_path=None,
                source_image_dims=img_dims,
                model_info={
                    "router_decision": "identical_pair_guard",
                    "model_name": "BiTemporalAnalysisEngine",
                    "change_detected": False,
                    "dominant_change_type": None,
                    "status": "NO_CHANGE",
                    "pair_classification": pair_diag.classification.value,
                },
                processing_time_sec=round(elapsed, 4)
            )

    # Route query with awareness of whether two images are available
    routing = SatQueryRouter.classify_query(request.query, has_second_image=has_second_image)
    query_type = routing.query_type
    target_concept = routing.target_concept

    try:
        meta = RasterReader.extract_metadata(file_path)
        img_dims = ImageDimensions(width=meta.width, height=meta.height)

        # -------------------------------------------------------------
        # CASE 1: GENERAL IMAGE UNDERSTANDING (Captioning / VQA)
        # -------------------------------------------------------------
        if query_type == QueryType.IMAGE_UNDERSTANDING:
            # Defensive guard: never allow temporal comparison query to run scene-level VQA
            if SatQueryRouter.is_temporal_change_query(request.query, has_second_image=has_second_image):
                query_type = QueryType.TEMPORAL_CHANGE
            else:
                logger.info(f"[SatQuery Router] Query: '{request.query}' | Type: image_understanding | Model: VQAEngine")
                
                answer, conf = vqa_engine.answer_question(file_path, request.query)
                model_name = vqa_engine.model_name

                elapsed = time.perf_counter() - start_time
                return GroundedAnalysisResponse(
                    file_id=request.file_id,
                    query=request.query,
                    answer=answer,
                    analysis_type=QueryType.IMAGE_UNDERSTANDING.value,
                    target_concept=None,
                    count=None,
                    evidence_regions=[],
                    overall_confidence=conf,
                    evidence_image_path=None,  # No evidence image for general descriptions
                    source_image_dims=img_dims,
                    model_info={
                        "router_decision": "image_understanding",
                        "model_name": model_name,
                        "mode": "scene_level_vlm",
                        "image_used": True,
                    },
                    processing_time_sec=round(elapsed, 4)
                )

        # -------------------------------------------------------------
        # CASE 2: COUNTING QUESTIONS
        # -------------------------------------------------------------
        elif query_type == QueryType.COUNTING:
            logger.info(f"[SatQuery Router] Query: '{request.query}' | Type: counting | Target: {target_concept} | Model: OWLv2 + Fusion")
            
            grounding_query = target_concept if target_concept else request.query
            evidence_regions, img_dims, model_info = grounding_engine.detect_evidence(
                file_path=file_path,
                query=grounding_query,
                confidence_threshold=request.confidence_threshold,
                max_regions=request.max_regions,
                debug_mode=request.debug_mode
            )

            count = len(evidence_regions)
            target_display = target_concept or "feature"
            if count == 0:
                answer = f"No distinct {target_display} features were confidently detected in this satellite image."
            elif count == 1:
                answer = f"The satellite image contains approximately 1 detected {target_display} region."
            else:
                plural = target_display if target_display.endswith("s") else f"{target_display}s"
                answer = f"The satellite image contains approximately {count} detected {plural} across the scene."

            rendered_image_path = render_visual_evidence(
                source_image=file_path,
                regions=evidence_regions,
                filename_prefix=f"count_{request.file_id}"
            )

            try:
                rel_path = str(rendered_image_path.relative_to(settings.BASE_DIR)).replace("\\", "/")
            except ValueError:
                rel_path = str(rendered_image_path)

            overall_conf = round(
                float(np.mean([r.confidence for r in evidence_regions])) if evidence_regions else 0.85,
                4
            )
            model_info["router_decision"] = "counting"
            model_info["target_concept"] = target_concept

            elapsed = time.perf_counter() - start_time
            return GroundedAnalysisResponse(
                file_id=request.file_id,
                query=request.query,
                answer=answer,
                analysis_type=QueryType.COUNTING.value,
                target_concept=target_concept,
                count=count,
                evidence_regions=evidence_regions,
                overall_confidence=overall_conf,
                evidence_image_path=rel_path,
                source_image_dims=img_dims,
                model_info=model_info,
                processing_time_sec=round(elapsed, 4)
            )

        # -------------------------------------------------------------
        # CASE 3: TEMPORAL CHANGE
        # -------------------------------------------------------------
        elif query_type == QueryType.TEMPORAL_CHANGE:
            logger.info(f"[SatQuery Router] Query: '{request.query}' | Type: temporal_change | Model: BiTemporalAnalysisEngine")

            if has_second_image:
                file_path_after = _resolve_uploaded_file(second_id)
                temp_resp = temporal_engine.analyze_change(
                    before_path=file_path,
                    after_path=file_path_after,
                    file_id_before=request.file_id,
                    file_id_after=second_id,
                    query=request.query,
                    confidence_threshold=request.confidence_threshold,
                    debug_mode=request.debug_mode,
                )
                elapsed = time.perf_counter() - start_time
                return GroundedAnalysisResponse(
                    file_id=request.file_id,
                    query=request.query,
                    answer=temp_resp.summary or "Bi-temporal change analysis complete.",
                    analysis_type=QueryType.TEMPORAL_CHANGE.value,
                    target_concept=target_concept,
                    count=len(temp_resp.change_regions),
                    evidence_regions=temp_resp.change_regions,
                    overall_confidence=1.0 if not temp_resp.change_detected else 0.95,
                    evidence_image_path=temp_resp.evidence_image_path,
                    source_image_dims=img_dims,
                    model_info={
                        "router_decision": "temporal_change",
                        "model_name": "BiTemporalAnalysisEngine",
                        "change_detected": temp_resp.change_detected,
                        "dominant_change_type": temp_resp.dominant_change_type,
                        "status": temp_resp.status,
                        "change_heatmap_path": temp_resp.change_heatmap_path,
                    },
                    processing_time_sec=round(elapsed, 4)
                )

            # Single image provided: guide the user without hallucinating
            answer = (
                "Temporal change detection requires two satellite observations (before and after images). "
                "To analyze changes like vegetation loss, water expansion, or new construction, please provide both temporal images."
            )
            elapsed = time.perf_counter() - start_time
            return GroundedAnalysisResponse(
                file_id=request.file_id,
                query=request.query,
                answer=answer,
                analysis_type=QueryType.TEMPORAL_CHANGE.value,
                target_concept=target_concept,
                count=None,
                evidence_regions=[],
                overall_confidence=0.90,
                evidence_image_path=None,
                source_image_dims=img_dims,
                model_info={
                    "router_decision": "temporal_change",
                    "notice": "Requires bi-temporal image pair"
                },
                processing_time_sec=round(elapsed, 4)
            )

        # -------------------------------------------------------------
        # CASE 4: SPATIAL GROUNDING (Default Grounding path)
        # -------------------------------------------------------------
        else:
            logger.info(f"[SatQuery Router] Query: '{request.query}' | Type: grounding | Target: {target_concept} | Model: OWLv2 + Fusion")

            # 1. Semantic reasoning via VQA Engine
            answer, vqa_conf = vqa_engine.answer_question(file_path, request.query)

            # 2. Spatial Grounding evidence detection using clean target concept
            grounding_query = target_concept if target_concept else request.query
            evidence_regions, img_dims, model_info = grounding_engine.detect_evidence(
                file_path=file_path,
                query=grounding_query,
                confidence_threshold=request.confidence_threshold,
                max_regions=request.max_regions,
                debug_mode=request.debug_mode
            )

            model_info["vlm_model"] = vqa_engine.model_name
            model_info["router_decision"] = "grounding"
            model_info["target_concept"] = target_concept

            # 3. Render visual evidence visualization image
            rendered_image_path = render_visual_evidence(
                source_image=file_path,
                regions=evidence_regions,
                filename_prefix=f"evidence_{request.file_id}"
            )

            try:
                rel_path = str(rendered_image_path.relative_to(settings.BASE_DIR)).replace("\\", "/")
            except ValueError:
                rel_path = str(rendered_image_path)

            overall_conf = round(
                float(np.mean([r.confidence for r in evidence_regions])) if evidence_regions else vqa_conf,
                4
            )

            elapsed = time.perf_counter() - start_time
            return GroundedAnalysisResponse(
                file_id=request.file_id,
                query=request.query,
                answer=answer,
                analysis_type=QueryType.GROUNDING.value,
                target_concept=target_concept,
                count=len(evidence_regions),
                evidence_regions=evidence_regions,
                overall_confidence=overall_conf,
                evidence_image_path=rel_path,
                source_image_dims=img_dims,
                model_info=model_info,
                processing_time_sec=round(elapsed, 4)
            )

    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Raster file with ID '{request.file_id}' not found"
        )
    except Exception as e:
        logger.error(f"Analysis query failed for file_id={request.file_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during AI analysis: {str(e)}"
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


@router.get("/evidence/{filename:path}")
async def get_evidence_image(filename: str):
    """
    Serves generated visual evidence images safely.
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
            detail="Invalid evidence filename"
        )

    file_path = (settings.RESULTS_DIR / sanitized_filename).resolve()
    results_dir_resolved = settings.RESULTS_DIR.resolve()

    try:
        file_path.relative_to(results_dir_resolved)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Path traversal attempt detected"
        )

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evidence image not found"
        )

    return FileResponse(file_path)


