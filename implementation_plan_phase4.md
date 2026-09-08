# Phase 4 Implementation Plan: Bi-Temporal Change Analysis

## 1. Current Architecture Assessment

SatQuery AI currently possesses a robust, validated, and frozen Phase 1–3 foundation:
- **Phase 1 (Data & Raster Ingestion)**: [`RasterReader`](file:///d:/Satquery%20Ai/backend/remote_sensing/raster.py), [`BandManager`](file:///d:/Satquery%20Ai/backend/remote_sensing/bands.py), [`Preprocessor`](file:///d:/Satquery%20Ai/backend/remote_sensing/preprocessing.py), [`FileValidator`](file:///d:/Satquery%20Ai/backend/remote_sensing/validator.py), [`TilingManager`](file:///d:/Satquery%20Ai/backend/remote_sensing/tiling.py).
- **Phase 2 (Single-Image AI)**: [`BaseVLMEngine`](file:///d:/Satquery%20Ai/backend/models/base.py), [`VQAEngine`](file:///d:/Satquery%20Ai/backend/models/vqa.py), [`CaptionEngine`](file:///d:/Satquery%20Ai/backend/models/caption.py).
- **Phase 3 (Spatial Grounding & Multi-Criteria Evidence Fusion)**: [`GroundingEngine`](file:///d:/Satquery%20Ai/backend/models/grounding.py), [`RemoteSensingEvidenceFusionEngine`](file:///d:/Satquery%20Ai/backend/models/fusion.py), [`render_visual_evidence`](file:///d:/Satquery%20Ai/backend/utils/rendering.py).

All 61 existing test cases (59 passed, 2 skipped) pass with 100% integrity. Phase 4 will strictly integrate with and extend these capabilities without breaking or rewriting existing modules.

---

## 2. Proposed Architecture & End-to-End Pipeline

```
BEFORE IMAGE (A)  +  AFTER IMAGE (B)
        │                   │
        ▼                   ▼
1. Temporal Validation & Ingestion (CRS, resolution, bounds compatibility)
        │
        ▼
2. Multi-Tier Image Registration (CRS Reprojection -> Resolution Normalization -> Grid Intersection -> Feature Fallback)
        │
        ▼
3. Cloud, Shadow & NoData Masking
        │
        ▼
4. Multispectral & Structural Differencing (ΔNDVI, ΔNDWI, ΔNDBI, ΔRGB, ΔSAR)
        │
        ▼
5. Normalized Change Magnitude Map Synthesis M(x, y)
        │
        ▼
6. Morphological Filtering & Region Extraction (Opening/Closing, Connected Components, Area Filtering, Cluster Consolidation)
        │
        ▼
7. Query Interpretation & Multi-Criteria Change Classification
        │
        ▼
8. Phase 3 OWLv2 Grounding / Semantic Evidence Fusion Integration
        │
        ▼
9. Multi-Panel Visual Evidence Rendering (Before | After | Change Map | Grounded Change Regions)
        │
        ▼
10. Quantified Natural-Language Change Explanation
```

---

## 3. Files to Create

1. [`backend/remote_sensing/alignment.py`](file:///d:/Satquery%20Ai/backend/remote_sensing/alignment.py) — Multi-tier image registration (CRS/reprojection, resampling, grid intersection, ORB/RANSAC fallback) and diagnostics generation.
2. [`backend/remote_sensing/change_detection.py`](file:///d:/Satquery%20Ai/backend/remote_sensing/change_detection.py) — Multispectral differencing ($\Delta\text{NDVI}, \Delta\text{NDWI}, \Delta\text{NDBI}$), normalized change magnitude map calculation, morphological filtering, connected component extraction, region merging.
3. [`backend/models/temporal_classifier.py`](file:///d:/Satquery%20Ai/backend/models/temporal_classifier.py) — Query interpretation layer, rule-based physical classifier, OWLv2 bi-temporal differential grounding ($\Delta S_{\text{semantic}}$), confidence scoring.
4. [`backend/models/temporal_engine.py`](file:///d:/Satquery%20Ai/backend/models/temporal_engine.py) — Bi-temporal orchestration engine coordinating alignment, change detection, classification, and natural language summary generation.
5. [`backend/utils/temporal_rendering.py`](file:///d:/Satquery%20Ai/backend/utils/temporal_rendering.py) — Multi-panel (Before | After | Change Heatmap | Grounded Regions) visual evidence renderer with category-specific color coding and debug overlays.
6. [`tests/test_temporal_alignment.py`](file:///d:/Satquery%20Ai/tests/test_temporal_alignment.py) — Tests for CRS reprojection, resolution normalization, grid alignment, feature-based registration, overlap calculation, and disjoint spatial detection.
7. [`tests/test_temporal_change.py`](file:///d:/Satquery%20Ai/tests/test_temporal_change.py) — Tests for spectral differencing, change thresholding, noise removal, connected component extraction, and classification rules.
8. [`tests/test_temporal_api.py`](file:///d:/Satquery%20Ai/tests/test_temporal_api.py) — End-to-end API integration tests for `POST /api/v1/analysis/change`.
9. [`tests/test_temporal_evaluation.py`](file:///d:/Satquery%20Ai/tests/test_temporal_evaluation.py) — Quantitative scientific evaluation calculating TP, FP, FN, Precision, Recall, F1, and IoU on known synthetic temporal test pairs.

---

## 4. Files to Modify

1. [`backend/config.py`](file:///d:/Satquery%20Ai/backend/config.py) — Add configurable Phase 4 temporal thresholds (`TEMPORAL_CHANGE_THRESHOLD`, `MIN_CHANGE_REGION_AREA`, `NDVI_CHANGE_THRESHOLD`, `NDWI_CHANGE_THRESHOLD`, `NDBI_CHANGE_THRESHOLD`, `TEMPORAL_MAX_CHANGE_REGIONS`, etc.).
2. [`backend/api/schemas.py`](file:///d:/Satquery%20Ai/backend/api/schemas.py) — Add `ChangeType` Enum, `RegistrationDiagnostics`, `SpectralEvidence`, `ChangeRegion`, `TemporalAnalysisRequest`, `TemporalAnalysisResponse`.
3. [`backend/api/analysis.py`](file:///d:/Satquery%20Ai/backend/api/analysis.py) — Add `POST /api/v1/analysis/change` endpoint.

---

## 5. API Changes

- **Endpoint**: `POST /api/v1/analysis/change`
- **Request**:
  - `file_id_before`: str
  - `file_id_after`: str
  - `query`: str (default `"What changed?"`)
  - `confidence_threshold`: Optional[float]
  - `target_change_types`: Optional[List[ChangeType]]
  - `debug_mode`: bool
- **Response**:
  - `file_id_before`, `file_id_after`, `query`
  - `status`: `"SUCCESS"` | `"NO_SIGNIFICANT_CHANGE"` | `"NO_SPATIAL_OVERLAP"` | `"ERROR"`
  - `summary`: str (natural language summary)
  - `dominant_change_type`: Optional[ChangeType]
  - `total_changed_area_pixels`: int
  - `total_changed_area_percentage`: float
  - `total_changed_area_sq_meters`: Optional[float]
  - `change_regions`: List[ChangeRegion]
  - `registration_diagnostics`: RegistrationDiagnostics
  - `evidence_image_path`: Optional[str]
  - `change_heatmap_path`: Optional[str]
  - `processing_time_sec`: float

---

## 6. Schema Changes

- `ChangeType`: Enum (`vegetation_loss`, `vegetation_gain`, `water_expansion`, `water_reduction`, `new_built_up`, `built_up_loss`, `road_infrastructure_change`, `unknown_change`).
- `RegistrationQuality`: Enum (`HIGH`, `MEDIUM`, `LOW`, `FAILED`).
- `RegistrationDiagnostics`: method, overlap %, bounds, original dims, aligned dims, resolutions, CRS, quality.
- `SpectralEvidence`: $\Delta\text{NDVI}$, $\Delta\text{NDWI}$, $\Delta\text{NDBI}$, $\Delta\text{RGB}$, $\Delta\text{SAR}$.
- `ChangeRegion`: region_id, change_type, bounding_box, area_pixels, area_percentage, area_sq_meters, confidence, change_score, spectral_evidence, semantic_evidence, explanation.
- `TemporalAnalysisRequest`, `TemporalAnalysisResponse`.

---

## 7. Algorithms & Differencing Methodology

- $\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red} + 1e-6}$, $\Delta\text{NDVI} = \text{NDVI}_B - \text{NDVI}_A$
- $\text{NDWI} = \frac{\text{Green} - \text{NIR}}{\text{Green} + \text{NIR} + 1e-6}$, $\Delta\text{NDWI} = \text{NDWI}_B - \text{NDWI}_A$
- $\text{NDBI} = \frac{\text{Red} - \text{NIR}}{\text{Red} + \text{NIR} + 1e-6}$, $\Delta\text{NDBI} = \text{NDBI}_B - \text{NDBI}_A$
- $\Delta\text{RGB} = \frac{1}{\sqrt{3}} \sqrt{(\Delta R)^2 + (\Delta G)^2 + (\Delta B)^2}$
- Normalized Change Magnitude:
  $$M(x, y) = 0.40 \cdot \max(|\Delta\text{NDVI}|, |\Delta\text{NDWI}|, |\Delta\text{NDBI}|) + 0.35 \cdot \Delta\text{RGB} + 0.25 \cdot \Delta\text{Edge}$$
- Morphological opening ($3\times 3$) + closing ($5\times 5$) + connected components + cluster enclosure.

---

## 8. Registration Strategy

1. **Tier 1 (Geospatial Reprojection)**: If both rasters possess CRS/transform, reproject `After` to `Before`'s CRS using rasterio/warp.
2. **Tier 2 (Resolution Normalization)**: Resample to uniform pixel dimensions.
3. **Tier 3 (Grid Intersection)**: Crop to common overlapping footprint. If overlap is empty, return `NO_SPATIAL_OVERLAP`.
4. **Tier 4 (Feature-Based Fallback)**: For standard images or unreferenced rasters, run ORB keypoint matching + RANSAC Affine/Homography estimation.

---

## 9. Spectral Change Methodology

- Distinguish true temporal change from normal sensor noise using threshold $\tau$ (default 0.22).
- Zero-out NoData pixels and saturated cloud artifacts.
- Multi-index correlation to avoid single-band false alarms.

---

## 10. Change Classification & Grounding Integration

- Decision-tree classifier combining $\Delta\text{NDVI}$, $\Delta\text{NDWI}$, $\Delta\text{NDBI}$, structural edges, and shape linearity.
- For target queries ("Where are the new buildings?"), run OWLv2 on candidate crops in Before vs After:
  $$\Delta S_{\text{semantic}} = S_{\text{after}} - S_{\text{before}}$$
- Final Confidence:
  $$\text{Conf} = 0.40 \cdot \text{SpectralScore} + 0.40 \cdot \Delta S_{\text{semantic}} + 0.20 \cdot \text{StructuralScore}$$

---

## 11. Visual Evidence Rendering

- 4-panel composite: Before | After | Change Magnitude Heatmap | Grounded Change Regions.
- Semantic color palette (Red for vegetation loss, Green for vegetation gain, Blue for water expansion, Orange for new construction).
- Label pills with % confidence and area.
- Debug mode overlaying rejected regions and spectral masks.

---

## 12. Testing Plan

- `tests/test_temporal_alignment.py` (CRS reprojection, resolution normalization, overlap detection, non-overlapping error handling).
- `tests/test_temporal_change.py` (spectral differencing, thresholding, connected components, noise removal, classification).
- `tests/test_temporal_api.py` (full API endpoint testing).
- `tests/test_temporal_evaluation.py` (quantitative precision/recall/F1/IoU).

---

## 13. GPU & Memory Considerations

- Safe execution on RTX 5060 8GB VRAM.
- NumPy/OpenCV/Rasterio array operations run entirely on CPU.
- OWLv2 model is evaluated only on cropped candidate regions when semantic refinement is needed.

---

## 14. Backward Compatibility

- Existing endpoints (`/api/v1/raster/*`, `/api/v1/analysis/vqa`, `/api/v1/analysis/caption`, `/api/v1/analysis/grounded`) remain completely functional and unaffected.
- Source raster files are never modified in-place.

---

## 15. Scientific Evaluation Methodology

- 3 Synthetic Temporal Test Pairs:
  1. Vegetation loss (deforestation / clearing)
  2. Water expansion (flooding / reservoir)
  3. New construction (building emergence)
- Metrics: True Positives (TP), False Positives (FP), False Negatives (FN), Precision, Recall, F1 Score, Mean IoU ($\ge 0.50$).

---

## 16. Risks and Fallback Strategies

- **Misalignment / Different Image Sizes**: Handled by Multi-Tier Registration & Resampling.
- **Illumination Shifts / Clouds**: Handled by multi-criteria spectral thresholds and cloud/NoData masking.
- **Disjoint Rasters**: Handled gracefully by returning `NO_SPATIAL_OVERLAP` status.
- **Identical Rasters**: Handled cleanly with `NO_SIGNIFICANT_CHANGE`.

---

## 17. Implementation Order

1. Step 1: Configuration & Schemas
2. Step 2: Temporal Image Alignment & Registration
3. Step 3: Multispectral Differencing & Region Extraction
4. Step 4: Change Classification & Semantic Grounding
5. Step 5: Visual Evidence Rendering
6. Step 6: API Router Integration
7. Step 7: Scientific Evaluation Suite
8. Step 8: Full Regression Suite Verification
