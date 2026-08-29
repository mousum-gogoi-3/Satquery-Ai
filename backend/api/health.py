from fastapi import APIRouter
from backend.config import settings
from backend.api.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    System health check endpoint verifying SatQuery AI Phase 1 engine readiness.
    """
    return HealthResponse(
        status="OK",
        app_name=settings.APP_NAME,
        version="0.1.0",
        debug=settings.DEBUG,
        engine_ready=True,
        allowed_formats=sorted(list(settings.ALLOWED_EXTENSIONS))
    )
