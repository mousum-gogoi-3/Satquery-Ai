import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from PIL import Image

from backend.config import settings
from backend.remote_sensing.raster import RasterReader
from backend.remote_sensing.alignment import ImageRegistrar
from backend.remote_sensing.change_detection import BiTemporalChangeDetector
from backend.remote_sensing.pair_validator import TemporalPairValidator
from backend.models.temporal_classifier import TemporalChangeClassifier
from backend.models.base import load_raster_as_pil_image
from backend.utils.temporal_rendering import render_temporal_evidence
from backend.api.schemas import (
    TemporalAnalysisRequest, TemporalAnalysisResponse,
    ChangeRegion, ChangeType, RegistrationQuality,
    RegistrationDiagnostics, PairClassification,
    PairValidationDiagnostics, TemporalStatistics
)
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class BiTemporalAnalysisEngine:
    """
    End-to-End Orchestrator for Bi-Temporal Satellite Imagery Change Analysis.
    Combines:
      1. Multi-tier Image Registration & Reprojection
      2. Multispectral Differential Indexing (ΔNDVI, ΔNDWI, ΔNDBI, ΔRGB)
      3. Morphological Change Region Extraction & Cluster Consolidation
      4. Query-Aware Physical Classification & OWLv2 Semantic Grounding
      5. Multi-Panel Visual Evidence Rendering
      6. Natural-Language Synthesis
    """

    def __init__(self):
        pass

    def analyze_change(
        self,
        before_path: Path,
        after_path: Path,
        file_id_before: str,
        file_id_after: str,
        query: str = "What changed?",
        confidence_threshold: Optional[float] = None,
        target_change_types: Optional[List[ChangeType]] = None,
        debug_mode: bool = False
    ) -> TemporalAnalysisResponse:
        """
        Executes bi-temporal change detection pipeline for a pair of satellite rasters and user query.
        """
        start_time = time.perf_counter()
        before_path = Path(before_path)
        after_path = Path(after_path)

        # ------------------------------------------------------------------
        # STAGE 0: Input-Pair Validation (Pre-Registration Filter)
        # ------------------------------------------------------------------
        pair_diag = TemporalPairValidator.validate_pair(before_path, after_path)
        meta_b = RasterReader.extract_metadata(before_path)
        meta_a = RasterReader.extract_metadata(after_path)

        logger.info(
            f"\nPair Validation Report:\n"
            f"- exact_hash_match: {pair_diag.exact_hash_match}\n"
            f"- similarity_score: {pair_diag.similarity_score}\n"
            f"- spatial_overlap_pct: {pair_diag.spatial_overlap_pct}\n"
            f"- crs_compatible: {pair_diag.crs_compatible}\n"
            f"- resolution_compatible: {pair_diag.resolution_compatible}\n"
            f"- pair_classification: {pair_diag.classification.value}\n"
            f"- reason: {pair_diag.reason}"
        )

        # Case 0A: EXACT DUPLICATE -> NO_CHANGE
        if pair_diag.classification == PairClassification.IDENTICAL_IMAGES:
            elapsed = time.perf_counter() - start_time
            diag_identical = RegistrationDiagnostics(
                method_applied="exact_duplicate_check",
                overlap_percentage=100.0,
                overlap_bounds=None,
                original_dims_before=(meta_b.width, meta_b.height),
                original_dims_after=(meta_a.width, meta_a.height),
                aligned_dims=(meta_b.width, meta_b.height),
                resolution_before=meta_b.resolution,
                resolution_after=meta_a.resolution,
                crs_before=meta_b.crs,
                crs_after=meta_a.crs,
                registration_quality=RegistrationQuality.HIGH
            )
            return TemporalAnalysisResponse(
                file_id_before=file_id_before,
                file_id_after=file_id_after,
                query=query,
                status="NO_CHANGE",
                change_detected=False,
                reason="IDENTICAL_IMAGES",
                pair_validation=pair_diag,
                statistics=TemporalStatistics(changed_pixels=0, change_percentage=0.0, changed_sq_meters=None),
                summary="The uploaded Before and After images are identical. No temporal changes detected.",
                dominant_change_type=None,
                total_changed_area_pixels=0,
                total_changed_area_percentage=0.0,
                total_changed_area_sq_meters=None,
                change_regions=[],
                registration_diagnostics=diag_identical,
                evidence_image_path=None,
                change_heatmap_path=None,
                processing_time_sec=round(elapsed, 4)
            )

        # Case 0B: NEAR DUPLICATE -> NO_SIGNIFICANT_CHANGE
        if pair_diag.classification == PairClassification.NEAR_DUPLICATE:
            elapsed = time.perf_counter() - start_time
            diag_near = RegistrationDiagnostics(
                method_applied="near_duplicate_check",
                overlap_percentage=100.0,
                overlap_bounds=None,
                original_dims_before=(meta_b.width, meta_b.height),
                original_dims_after=(meta_a.width, meta_a.height),
                aligned_dims=(meta_b.width, meta_b.height),
                resolution_before=meta_b.resolution,
                resolution_after=meta_a.resolution,
                crs_before=meta_b.crs,
                crs_after=meta_a.crs,
                registration_quality=RegistrationQuality.HIGH
            )
            return TemporalAnalysisResponse(
                file_id_before=file_id_before,
                file_id_after=file_id_after,
                query=query,
                status="NO_SIGNIFICANT_CHANGE",
                change_detected=False,
                reason="NEAR_DUPLICATE",
                pair_validation=pair_diag,
                statistics=TemporalStatistics(changed_pixels=0, change_percentage=0.0, changed_sq_meters=None),
                summary="No significant physical or spectral land-cover changes detected between the two observation dates.",
                dominant_change_type=None,
                total_changed_area_pixels=0,
                total_changed_area_percentage=0.0,
                total_changed_area_sq_meters=None,
                change_regions=[],
                registration_diagnostics=diag_near,
                evidence_image_path=None,
                change_heatmap_path=None,
                processing_time_sec=round(elapsed, 4)
            )

        # Case 0C: INCOMPATIBLE IMAGES -> INCOMPATIBLE_IMAGES
        if pair_diag.classification == PairClassification.INCOMPATIBLE_IMAGES:
            elapsed = time.perf_counter() - start_time
            diag_incompat = RegistrationDiagnostics(
                method_applied="input_pair_validation",
                overlap_percentage=pair_diag.spatial_overlap_pct or 0.0,
                overlap_bounds=None,
                original_dims_before=(meta_b.width, meta_b.height),
                original_dims_after=(meta_a.width, meta_a.height),
                aligned_dims=(meta_b.width, meta_b.height),
                resolution_before=meta_b.resolution,
                resolution_after=meta_a.resolution,
                crs_before=meta_b.crs,
                crs_after=meta_a.crs,
                registration_quality=RegistrationQuality.FAILED
            )
            reason_msg = pair_diag.reason or "The uploaded images appear to represent different scenes and cannot be reliably compared as a temporal pair."
            return TemporalAnalysisResponse(
                file_id_before=file_id_before,
                file_id_after=file_id_after,
                query=query,
                status="INCOMPATIBLE_IMAGES",
                change_detected=False,
                reason=reason_msg,
                pair_validation=pair_diag,
                statistics=TemporalStatistics(changed_pixels=0, change_percentage=0.0, changed_sq_meters=None),
                summary=reason_msg,
                dominant_change_type=None,
                total_changed_area_pixels=0,
                total_changed_area_percentage=0.0,
                total_changed_area_sq_meters=None,
                change_regions=[],
                registration_diagnostics=diag_incompat,
                evidence_image_path=None,
                change_heatmap_path=None,
                processing_time_sec=round(elapsed, 4)
            )

        # ------------------------------------------------------------------
        # STAGE 1: Multi-Tier Image Registration & Spatial Alignment
        # ------------------------------------------------------------------
        aligned_b, aligned_a, diagnostics, valid_mask = ImageRegistrar.align_rasters(before_path, after_path)

        # Verify aligned array dimensions
        if aligned_b.shape != aligned_a.shape:
            raise ValueError(f"Aligned raster shape mismatch: before={aligned_b.shape}, after={aligned_a.shape}")

        # 2. Check for Disjoint / Non-Overlapping Images
        if diagnostics.registration_quality == RegistrationQuality.FAILED or diagnostics.overlap_percentage < (settings.REGISTRATION_MIN_OVERLAP_PCT * 100.0):
            elapsed = time.perf_counter() - start_time
            msg = "The provided Before and After images do not overlap geographically. Cannot compute bi-temporal change analysis."
            return TemporalAnalysisResponse(
                file_id_before=file_id_before,
                file_id_after=file_id_after,
                query=query,
                status="INCOMPATIBLE_IMAGES",
                change_detected=False,
                reason=msg,
                pair_validation=pair_diag,
                statistics=TemporalStatistics(changed_pixels=0, change_percentage=0.0, changed_sq_meters=None),
                summary=msg,
                dominant_change_type=None,
                total_changed_area_pixels=0,
                total_changed_area_percentage=0.0,
                total_changed_area_sq_meters=None,
                change_regions=[],
                registration_diagnostics=diagnostics,
                evidence_image_path=None,
                change_heatmap_path=None,
                processing_time_sec=round(elapsed, 4)
            )

        # 3. Compute Multispectral & Structural Temporal Differences
        diffs = BiTemporalChangeDetector.compute_temporal_differences(aligned_b, aligned_a, valid_mask)

        # 4. Synthesize Normalized Multi-Signal Change Magnitude Map M(x, y)
        mag_map = BiTemporalChangeDetector.compute_change_magnitude_map(diffs, valid_mask)

        # 5. Robust Data-Adaptive Thresholding & Candidate Region Extraction
        adaptive_tau = BiTemporalChangeDetector.compute_adaptive_threshold(mag_map, valid_mask)
        candidates = BiTemporalChangeDetector.extract_candidate_regions(
            magnitude_map=mag_map,
            diffs=diffs,
            valid_mask=valid_mask,
            threshold=adaptive_tau
        )

        # Load PIL images for differential semantic grounding if needed
        pil_b = None
        pil_a = None
        try:
            pil_b = load_raster_as_pil_image(before_path)
            pil_a = load_raster_as_pil_image(after_path)
        except Exception:
            pass

        # 6. Parse Query Intent & Semantic Prompts
        query_intent = TemporalChangeClassifier.parse_query_intent(query)
        semantic_prompt = query_intent["semantic_prompt"]
        normalized_targets = TemporalChangeClassifier.normalize_target_change_types(target_change_types)
        effective_targets = normalized_targets or query_intent["target_types"]

        # 7. Classify Extracted Regions with Physical Rules & Multi-Signal Confidence
        classified_regions: List[ChangeRegion] = []
        for cand in candidates:
            reg = TemporalChangeClassifier.classify_region(
                candidate=cand,
                pil_before=pil_b,
                pil_after=pil_a,
                semantic_prompt=semantic_prompt,
                resolution_m=diagnostics.resolution_before,
                crs=diagnostics.crs_before
            )
            classified_regions.append(reg)

        # 8. Filter by Region Classification Confidence
        # Expected range: 0.0 to 1.0 (default: 0.35)
        raw_conf_th = confidence_threshold if confidence_threshold is not None else settings.TEMPORAL_CONFIDENCE_THRESHOLD
        # Handle percentage input e.g. 50 -> 0.50, but preserve 1.0 as 1.0
        if raw_conf_th > 1.0 and raw_conf_th <= 100.0:
            raw_conf_th = raw_conf_th / 100.0
        effective_conf_th = float(np.clip(raw_conf_th, 0.0, 1.0))

        conf_filtered_regions = [r for r in classified_regions if r.confidence >= effective_conf_th]

        # 9. Query-Aware Prioritization & Filtering
        filtered_regions: List[ChangeRegion] = []
        if effective_targets:
            filtered_regions = [r for r in conf_filtered_regions if r.change_type in effective_targets]
            # If target-specific query has matches, focus on them; otherwise fallback to all confidence-passed changes
            if not filtered_regions:
                filtered_regions = conf_filtered_regions
        else:
            filtered_regions = conf_filtered_regions

        # Diagnostic Logging for Debug Mode
        if debug_mode:
            valid_px = int(np.sum(valid_mask))
            d_rgb_mean = float(diffs['delta_rgb'].mean())
            d_rgb_max = float(diffs['delta_rgb'].max())
            d_ndvi_mean = float(diffs['delta_ndvi'].mean())
            d_ndwi_mean = float(diffs['delta_ndwi'].mean())
            d_ndbi_mean = float(diffs['delta_ndbi'].mean())
            sar_vv_str = f"{float(diffs['delta_sar_vv'].mean()):.4f}" if diffs['delta_sar_vv'] is not None else "N/A"
            sar_vh_str = f"{float(diffs['delta_sar_vh'].mean()):.4f}" if diffs['delta_sar_vh'] is not None else "N/A"
            logger.info(
                f"\nTemporal Change Analysis Debug\n"
                f"------------------------\n"
                f"Before shape: {aligned_b.shape} (min={aligned_b.min()}, max={aligned_b.max()}, mean={aligned_b.mean():.2f})\n"
                f"After shape: {aligned_a.shape} (min={aligned_a.min()}, max={aligned_a.max()}, mean={aligned_a.mean():.2f})\n"
                f"Aligned shape: {aligned_b.shape}\n"
                f"Overlap: {diagnostics.overlap_percentage}%\n"
                f"Valid pixels: {valid_px}\n"
                f"RGB difference mean: {d_rgb_mean:.4f} (max={d_rgb_max:.4f})\n"
                f"NDVI difference mean: {d_ndvi_mean:.4f}\n"
                f"NDWI difference mean: {d_ndwi_mean:.4f}\n"
                f"NDBI difference mean: {d_ndbi_mean:.4f}\n"
                f"SAR VV difference mean: {sar_vv_str}\n"
                f"SAR VH difference mean: {sar_vh_str}\n"
                f"Adaptive threshold: {adaptive_tau:.4f}\n"
                f"Confidence threshold: {effective_conf_th:.2f}\n"
                f"Candidate changed regions: {len(candidates)}\n"
                f"Confidence-filtered regions: {len(conf_filtered_regions)}\n"
                f"Final regions: {len(filtered_regions)}\n"
            )

        # 10. Handle NO SIGNIFICANT CHANGE
        if not filtered_regions:
            ev_path = None
            hm_path = None
            if debug_mode:
                p_ev, p_hm = render_temporal_evidence(
                    bands_before=aligned_b,
                    bands_after=aligned_a,
                    magnitude_map=mag_map,
                    regions=[],
                    filename_prefix=f"temporal_{file_id_before}_{file_id_after}",
                    debug_mode=debug_mode
                )
                ev_path = self._to_rel_path(p_ev)
                hm_path = self._to_rel_path(p_hm)

            elapsed = time.perf_counter() - start_time
            return TemporalAnalysisResponse(
                file_id_before=file_id_before,
                file_id_after=file_id_after,
                query=query,
                status="NO_SIGNIFICANT_CHANGE",
                change_detected=False,
                reason="NO_SIGNIFICANT_CHANGE",
                pair_validation=pair_diag,
                statistics=TemporalStatistics(changed_pixels=0, change_percentage=0.0, changed_sq_meters=None),
                summary="No significant physical or spectral land-cover changes detected between the two observation dates.",
                dominant_change_type=None,
                total_changed_area_pixels=0,
                total_changed_area_percentage=0.0,
                total_changed_area_sq_meters=None,
                change_regions=[],
                registration_diagnostics=diagnostics,
                evidence_image_path=ev_path,
                change_heatmap_path=hm_path,
                processing_time_sec=round(elapsed, 4)
            )

        # 11. Compute Summary Metrics & Dominant Change Type using valid overlap area
        total_pixels = sum(r.area_pixels for r in filtered_regions)
        valid_pixel_count = max(1, int(np.sum(valid_mask)))
        total_pct = round(min(100.0, (total_pixels / float(valid_pixel_count)) * 100.0), 2)
        total_sq_m = sum(r.area_sq_meters for r in filtered_regions if r.area_sq_meters is not None) or None

        # Dominant change type by total pixel area
        type_areas: Dict[ChangeType, int] = {}
        for r in filtered_regions:
            type_areas[r.change_type] = type_areas.get(r.change_type, 0) + r.area_pixels
        dominant_type = max(type_areas.items(), key=lambda item: item[1])[0]

        # 12. Synthesize Natural-Language Summary
        summary = self._synthesize_summary(query, filtered_regions, dominant_type, total_pct, total_sq_m)

        # 13. Render Visual Evidence Images
        ev_path, hm_path = render_temporal_evidence(
            bands_before=aligned_b,
            bands_after=aligned_a,
            magnitude_map=mag_map,
            regions=filtered_regions,
            filename_prefix=f"temporal_{file_id_before}_{file_id_after}",
            debug_mode=debug_mode
        )

        rel_ev_path = self._to_rel_path(ev_path)
        rel_hm_path = self._to_rel_path(hm_path)
        elapsed = time.perf_counter() - start_time

        return TemporalAnalysisResponse(
            file_id_before=file_id_before,
            file_id_after=file_id_after,
            query=query,
            status="SUCCESS",
            change_detected=True,
            reason=None,
            pair_validation=pair_diag,
            statistics=TemporalStatistics(
                changed_pixels=total_pixels,
                change_percentage=total_pct,
                changed_sq_meters=round(total_sq_m, 2) if total_sq_m else None
            ),
            summary=summary,
            dominant_change_type=dominant_type,
            total_changed_area_pixels=total_pixels,
            total_changed_area_percentage=total_pct,
            total_changed_area_sq_meters=round(total_sq_m, 2) if total_sq_m else None,
            change_regions=filtered_regions,
            registration_diagnostics=diagnostics,
            evidence_image_path=rel_ev_path,
            change_heatmap_path=rel_hm_path,
            processing_time_sec=round(elapsed, 4)
        )

    def _synthesize_summary(
        self,
        query: str,
        regions: List[ChangeRegion],
        dominant_type: ChangeType,
        total_pct: float,
        total_sq_m: Optional[float]
    ) -> str:
        """Synthesizes an insightful natural-language explanation of detected changes."""
        reg_count = len(regions)
        area_str = f"covering {total_pct}% of the scene"
        if total_sq_m:
            if total_sq_m >= 10000.0:
                area_str += f" ({total_sq_m / 10000.0:.2f} hectares)"
            else:
                area_str += f" ({total_sq_m:.1f} m²)"

        type_names = {
            ChangeType.VEGETATION_LOSS: "vegetation loss / deforestation",
            ChangeType.VEGETATION_GAIN: "vegetation expansion / greening",
            ChangeType.WATER_EXPANSION: "water body expansion / flooding",
            ChangeType.WATER_REDUCTION: "water body reduction / drying",
            ChangeType.NEW_BUILT_UP: "new construction / built-up expansion",
            ChangeType.BUILT_UP_LOSS: "demolition / structural loss",
            ChangeType.ROAD_INFRASTRUCTURE_CHANGE: "road and infrastructure modification",
            ChangeType.SURFACE_CHANGE: "surface reflectance / land-cover transition",
            ChangeType.GENERAL_LAND_COVER_CHANGE: "general land-cover change",
            ChangeType.BARE_SOIL_CHANGE: "bare soil / clearing",
            ChangeType.UNKNOWN_CHANGE: "surface spectral change",
        }

        dom_name = type_names.get(dominant_type, dominant_type.value)

        lines = [
            f"Detected {reg_count} significant change region(s) {area_str}.",
            f"The primary detected transition is {dom_name}."
        ]

        if reg_count == 1:
            lines.append(f"Region details: {regions[0].explanation} (Confidence: {regions[0].confidence * 100:.0f}%).")
        else:
            top_exps = "; ".join([f"{r.region_id}: {r.explanation}" for r in regions[:2]])
            lines.append(f"Key highlights: {top_exps}.")

        return " ".join(lines)

    @staticmethod
    def _to_rel_path(p: Path) -> str:
        """Converts absolute Path to workspace-relative forward-slash path."""
        try:
            return str(p.relative_to(settings.BASE_DIR)).replace("\\", "/")
        except ValueError:
            return str(p).replace("\\", "/")
