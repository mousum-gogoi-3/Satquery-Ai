# SatQuery AI — Phase 1: Data & Image Engine

SatQuery AI (SIH Problem Statement: 26167) is an Agentic Multimodal Remote-Sensing Vision-Language Agent system.

This repository contains **Phase 1: Data & Image Engine** — the foundational local-first data ingestion, raster metadata extraction, multi-spectral band management, percentile preprocessor, high-resolution sliding-window tiler, and FastAPI web backend.

## Phase 1 Capabilities

- **Raster Validation**: File format validation (GeoTIFF, TIFF, PNG, JPEG, JP2), size limits, band bounds, and corruption checks.
- **Geospatial & Image Metadata Extraction**: Automatic extraction of dimensions, band count, CRS (EPSG), affine transform matrices, spatial resolution, nodata values, and geospatial bounds.
- **Multi-Band Management**: Automated role inferencing for single-band panchromatic, Sentinel-1 dual-pol SAR (VV/VH), RGB, 4-band RGB-NIR, and multi-spectral rasters.
- **Percentile Preprocessing**: 2%-98% percentile normalization scaling, NoData masking, thumbnail preview rendering, and image resampling.
- **High-Resolution Raster Tiling**: Sliding-window grid partitioning with sub-window array extraction for handling high-resolution satellite scenes without out-of-memory errors.
- **FastAPI Endpoints**:
  - `/api/v1/health`
  - `/api/v1/raster/upload`
  - `/api/v1/raster/{file_id}/metadata`
  - `/api/v1/raster/tile`
  - `/api/v1/raster/preview/{filename}`

## Quickstart & Running Tests

### 1. Environment Setup (Python 3.12 Required)
```bash
# Create virtual environment using Python 3.12
py -3.12 -m venv .venv

# Activate environment (Windows PowerShell)
.venv\Scripts\activate

# Install Phase 1 local dependencies
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
.venv\Scripts\pytest -v
```

### 3. Start Backend Server
```bash
.venv\Scripts\uvicorn backend.main:app --reload --port 8000
```
Interactive Swagger documentation is available at `http://127.0.0.1:8000/docs`.

