from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.utils.logger import get_logger
from backend.api import health, upload, analysis

logger = get_logger("SatQueryAI.Main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown lifecycle manager.
    """
    logger.info(f"Starting {settings.APP_NAME} Backend Engine...")
    settings.ensure_directories()
    logger.info(f"Upload directory: {settings.UPLOAD_DIR}")
    logger.info(f"Processed directory: {settings.PROCESSED_DIR}")
    yield
    logger.info(f"Shutting down {settings.APP_NAME} Backend Engine.")


app = FastAPI(
    title=settings.APP_NAME,
    description="Multimodal Remote-Sensing Vision-Language Agent Backend Engine (Phase 2)",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Router Registrations
app.include_router(health.router, prefix=settings.API_PREFIX)
app.include_router(upload.router, prefix=settings.API_PREFIX)
app.include_router(analysis.router, prefix=settings.API_PREFIX)


@app.get("/")
async def root():
    return {
        "app": settings.APP_NAME,
        "phase": "Phase 2 - Single-Image AI",
        "docs": "/docs",
        "health": f"{settings.API_PREFIX}/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
