import time
from pathlib import Path
from fastapi import APIRouter, HTTPException, status

from backend.config import settings
from backend.utils.logger import get_logger
from backend.api.schemas import VQARequest, VQAResponse, CaptionRequest, CaptionResponse
from backend.models.vqa import VQAEngine
from backend.models.caption import CaptionEngine

logger = get_logger(__name__)
router = APIRouter(prefix="/analysis", tags=["Single-Image AI Analysis"])

# Lazy engine singletons
vqa_engine = VQAEngine()
caption_engine = CaptionEngine()


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
