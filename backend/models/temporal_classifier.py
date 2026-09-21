import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from PIL import Image

from backend.config import settings
from backend.api.schemas import BoundingBox, ChangeType, ChangeRegion, SpectralEvidence
from backend.models.grounding import OWLv2ModelManager
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class TemporalChangeClassifier:
    """
    Physical Decision-Tree Change Classifier & Semantic Grounding Fusion.
    Classifies candidate spatial change regions into canonical ChangeTypes
    and optionally performs differential OWLv2 semantic verification.
    """

    @classmethod
    def parse_query_intent(cls, query: str) -> Dict[str, Any]:
        """
        Parses natural-language user query to identify target change category filters and semantic prompts.
        """
        q = query.lower().strip()
        intent = {
            "target_types": None,
            "semantic_prompt": None,
            "category": "general"
        }

        # Ignore generic queries like "What changed?", "What has changed?", "Tell me changes"
        generic_patterns = ["what changed", "what has changed", "tell me what changed", "show changes", "any changes"]
        if any(p in q for p in generic_patterns) and not any(w in q for w in ["vegetation", "water", "tree", "forest", "build", "road", "flood", "soil"]):
            return intent

        if any(w in q for w in ["vegetation decrease", "vegetation loss", "deforest", "deforestation", "tree loss", "cleared", "forest decrease"]):
            intent["target_types"] = [ChangeType.VEGETATION_LOSS]
            intent["semantic_prompt"] = "forest trees"
            intent["category"] = "vegetation_loss"
        elif any(w in q for w in ["vegetation increase", "vegetation gain", "reforest", "reforestation", "greener", "tree increase", "new vegetation"]):
            intent["target_types"] = [ChangeType.VEGETATION_GAIN]
            intent["semantic_prompt"] = "forest trees"
            intent["category"] = "vegetation_gain"
        elif any(w in q for w in ["water expand", "flooding", "flood", "lake expand", "water increase", "reservoir expand"]):
            intent["target_types"] = [ChangeType.WATER_EXPANSION]
            intent["semantic_prompt"] = "water body"
            intent["category"] = "water_expansion"
        elif any(w in q for w in ["water shrink", "water decrease", "water reduction", "drought", "lake shrink", "drying"]):
            intent["target_types"] = [ChangeType.WATER_REDUCTION]
            intent["semantic_prompt"] = "water body"
            intent["category"] = "water_reduction"
        elif any(w in q for w in ["water change", "water body change"]):
            intent["target_types"] = [ChangeType.WATER_EXPANSION, ChangeType.WATER_REDUCTION]
            intent["semantic_prompt"] = "water body"
            intent["category"] = "water_change"
        elif any(w in q for w in ["new building", "construction", "new structure", "built-up expand", "urban expand", "urban expansion"]):
            intent["target_types"] = [ChangeType.NEW_BUILT_UP]
            intent["semantic_prompt"] = "building structure"
            intent["category"] = "new_built_up"
        elif any(w in q for w in ["demolish", "building removed", "structure loss", "building loss"]):
            intent["target_types"] = [ChangeType.BUILT_UP_LOSS]
            intent["semantic_prompt"] = "building structure"
            intent["category"] = "built_up_loss"
        elif any(w in q for w in ["built-up change", "building change", "urban change"]):
            intent["target_types"] = [ChangeType.NEW_BUILT_UP, ChangeType.BUILT_UP_LOSS]
            intent["semantic_prompt"] = "building structure"
            intent["category"] = "built_up_change"
        elif any(w in q for w in ["road", "highway", "infrastructure"]):
            intent["target_types"] = [ChangeType.ROAD_INFRASTRUCTURE_CHANGE]
            intent["semantic_prompt"] = "road highway"
            intent["category"] = "road_infrastructure_change"
        elif any(w in q for w in ["bare soil", "soil change", "ground clearing"]):
            intent["target_types"] = [ChangeType.BARE_SOIL_CHANGE, ChangeType.SURFACE_CHANGE]
            intent["semantic_prompt"] = "bare soil ground"
            intent["category"] = "bare_soil_change"
        elif any(w in q for w in ["surface change", "land cover change", "general change"]):
            intent["target_types"] = [ChangeType.SURFACE_CHANGE, ChangeType.GENERAL_LAND_COVER_CHANGE]
            intent["semantic_prompt"] = "ground surface"
            intent["category"] = "surface_change"

        return intent

    @classmethod
    def normalize_target_change_types(cls, targets: Optional[List[Any]]) -> Optional[List[ChangeType]]:
        """
        Normalizes target change type filters from strings or enums into canonical ChangeTypes.
        Supports aliases like deforestation -> vegetation_loss, flooding -> water_expansion, etc.
        """
        if not targets:
            return None

        alias_map = {
            "deforestation": [ChangeType.VEGETATION_LOSS],
            "vegetation_loss": [ChangeType.VEGETATION_LOSS],
            "reforestation": [ChangeType.VEGETATION_GAIN],
            "vegetation_gain": [ChangeType.VEGETATION_GAIN],
            "flooding": [ChangeType.WATER_EXPANSION],
            "water_expansion": [ChangeType.WATER_EXPANSION],
            "water_reduction": [ChangeType.WATER_REDUCTION],
            "water_change": [ChangeType.WATER_EXPANSION, ChangeType.WATER_REDUCTION],
            "urban_expansion": [ChangeType.NEW_BUILT_UP],
            "new_built_up": [ChangeType.NEW_BUILT_UP],
            "built_up_loss": [ChangeType.BUILT_UP_LOSS],
            "built_up_change": [ChangeType.NEW_BUILT_UP, ChangeType.BUILT_UP_LOSS],
            "road_infrastructure_change": [ChangeType.ROAD_INFRASTRUCTURE_CHANGE],
            "bare_soil_change": [ChangeType.BARE_SOIL_CHANGE, ChangeType.SURFACE_CHANGE],
            "surface_change": [ChangeType.SURFACE_CHANGE],
            "general_land_cover_change": [ChangeType.GENERAL_LAND_COVER_CHANGE, ChangeType.SURFACE_CHANGE],
            "unknown_change": [ChangeType.UNKNOWN_CHANGE],
        }

        normalized: List[ChangeType] = []
        for t in targets:
            key = t.value if isinstance(t, ChangeType) else str(t).lower()
            if key in alias_map:
                for ct in alias_map[key]:
                    if ct not in normalized:
                        normalized.append(ct)
            else:
                try:
                    ct = ChangeType(key)
                    if ct not in normalized:
                        normalized.append(ct)
                except ValueError:
                    pass

        return normalized if normalized else None

    @classmethod
    def classify_region(
        cls,
        candidate: Dict[str, Any],
        pil_before: Optional[Image.Image] = None,
        pil_after: Optional[Image.Image] = None,
        semantic_prompt: Optional[str] = None,
        resolution_m: Optional[Tuple[float, float]] = None,
        crs: Optional[str] = None,
        center_lat: Optional[float] = None
    ) -> ChangeRegion:
        """
        Applies physical remote sensing rules and differential semantic evaluation to classify a change region.
        Accurately scores multi-signal confidence and calculates ground surface area in square meters.
        """
        import math
        box: BoundingBox = candidate["box"]
        spec: SpectralEvidence = candidate["spectral_evidence"]
        change_score = float(candidate["change_score"])
        area_pixels = int(candidate["area_pixels"])
        area_pct = float(candidate["area_percentage"])

        d_ndvi = spec.delta_ndvi
        d_ndwi = spec.delta_ndwi
        d_ndbi = spec.delta_ndbi
        d_rgb = spec.delta_rgb

        ndvi_b = float(candidate.get("ndvi_before", 0.0))
        ndvi_a = float(candidate.get("ndvi_after", 0.0))
        ndwi_b = float(candidate.get("ndwi_before", 0.0))
        ndwi_a = float(candidate.get("ndwi_after", 0.0))
        ndbi_b = float(candidate.get("ndbi_before", 0.0))
        ndbi_a = float(candidate.get("ndbi_after", 0.0))
        d_edge = float(candidate.get("delta_edge", 0.0))

        bw = box.xmax - box.xmin
        bh = box.ymax - box.ymin
        aspect_ratio = max(bw, bh) / max(1.0, min(bw, bh))

        # 1. Differential OWLv2 Semantic Grounding (if PIL images provided and prompt specified)
        sem_evidence = None
        sem_delta = 0.0
        if pil_before is not None and pil_after is not None and semantic_prompt and not settings.USE_MOCK_GROUNDING:
            sem_evidence, sem_delta = cls._evaluate_semantic_delta(
                box, pil_before, pil_after, semantic_prompt
            )

        # 2. Physics-Based Remote Sensing Decision Tree
        th_ndvi = settings.NDVI_CHANGE_THRESHOLD
        th_ndwi = settings.NDWI_CHANGE_THRESHOLD
        th_ndbi = settings.NDBI_CHANGE_THRESHOLD

        has_spectral = not (d_ndvi == 0.0 and d_ndwi == 0.0 and d_ndbi == 0.0 and ndvi_b == 0.0 and ndvi_a == 0.0)

        # Physical Water Body Detection: High NDWI, NDWI strictly dominates NDBI and NDVI, Low NDBI (<0.15)
        is_water_after = (ndwi_a >= 0.15 and ndwi_a > (ndbi_a + 0.10) and ndwi_a > ndvi_a and ndvi_a < 0.10)
        is_water_before = (ndwi_b >= 0.15 and ndwi_b > (ndbi_b + 0.10) and ndwi_b > ndvi_b and ndvi_b < 0.10)

        logger.debug(f"[CLASSIFIER] d_ndvi={d_ndvi:+.2f}, d_ndwi={d_ndwi:+.2f}, d_ndbi={d_ndbi:+.2f}, d_rgb={d_rgb:.2f} | ndvi={ndvi_b:.2f}->{ndvi_a:.2f}, ndwi={ndwi_b:.2f}->{ndwi_a:.2f}, ndbi={ndbi_b:.2f}->{ndbi_a:.2f}")

        # 1. Water Inundation / Flooding: Area became water in After
        if has_spectral and d_ndwi >= th_ndwi and is_water_after:
            change_type = ChangeType.WATER_EXPANSION
            confidence = min(1.0, max(0.65, d_ndwi * 0.80 + change_score * 0.35 + (0.10 if d_edge > 0.1 else 0.0)))
            explanation = f"Water body expansion / inundation detected (ΔNDWI={d_ndwi:+.2f}, NDWI {ndwi_b:.2f} -> {ndwi_a:.2f})"

        # 2. Water Drying / Shoreline Reduction: Area was water in Before and dried up
        elif has_spectral and d_ndwi <= -th_ndwi and is_water_before:
            change_type = ChangeType.WATER_REDUCTION
            confidence = min(1.0, max(0.65, abs(d_ndwi) * 0.80 + change_score * 0.35 + (0.10 if d_edge > 0.1 else 0.0)))
            explanation = f"Water body reduction / drying detected (ΔNDWI={d_ndwi:+.2f}, NDWI {ndwi_b:.2f} -> {ndwi_a:.2f})"

        # 3. Vegetation Gain / Greening: After has high vegetation NDVI
        elif has_spectral and d_ndvi >= th_ndvi and ndvi_a >= 0.20 and ndvi_a > ndwi_a:
            change_type = ChangeType.VEGETATION_GAIN
            confidence = min(1.0, max(0.65, d_ndvi * 0.85 + change_score * 0.35 + (0.08 if d_rgb > 0.15 else 0.0)))
            explanation = f"Vegetation gain / greening detected (ΔNDVI={d_ndvi:+.2f}, NDVI {ndvi_b:.2f} -> {ndvi_a:.2f})"

        # 4. Vegetation Loss: Before had high vegetation NDVI and lost it
        elif has_spectral and d_ndvi <= -th_ndvi and ndvi_b >= 0.20 and not is_water_after:
            change_type = ChangeType.VEGETATION_LOSS
            confidence = min(1.0, max(0.65, abs(d_ndvi) * 0.85 + change_score * 0.35 + (0.08 if d_rgb > 0.15 else 0.0)))
            explanation = f"Vegetation loss / land clearing detected (ΔNDVI={d_ndvi:+.2f}, NDVI {ndvi_b:.2f} -> {ndvi_a:.2f})"

        # 5. New Built-up Construction: High visible / NDBI reflection on non-water surface
        elif has_spectral and (d_ndbi >= th_ndbi or d_rgb >= 0.20) and ndbi_a > ndvi_a and ndvi_a < 0.25 and not is_water_after:
            change_type = ChangeType.NEW_BUILT_UP
            confidence = min(1.0, max(0.65, (d_ndbi * 0.55 + d_rgb * 0.35 + max(0.0, sem_delta) * 0.30 + change_score * 0.25)))
            explanation = f"New built-up / construction detected (ΔNDBI={d_ndbi:+.2f}, ΔRGB={d_rgb:.2f}, Edge={d_edge:.2f})"

        # 6. Built-up Loss / Demolition
        elif has_spectral and (d_ndbi <= -th_ndbi or (d_rgb >= 0.25 and sem_delta < -0.10)) and ndbi_b > ndvi_b and ndvi_b < 0.25:
            change_type = ChangeType.BUILT_UP_LOSS
            confidence = min(1.0, max(0.60, abs(d_ndbi) * 0.60 + d_rgb * 0.40))
            explanation = f"Built-up / structure loss detected (ΔNDBI={d_ndbi:+.2f}, ΔRGB={d_rgb:.2f})"

        # 7. Linear Road / Infrastructure
        elif aspect_ratio >= 2.5 and d_edge > 0.18 and ndvi_a < 0.25 and not is_water_after:
            change_type = ChangeType.ROAD_INFRASTRUCTURE_CHANGE
            confidence = min(1.0, max(0.60, d_edge * 0.60 + d_rgb * 0.40))
            explanation = f"Linear road / infrastructure modification detected (Aspect ratio={aspect_ratio:.1f}, ΔEdge={d_edge:.2f})"

        elif not has_spectral:
            # Single-band / Grayscale imagery change
            change_type = ChangeType.SURFACE_CHANGE
            # Confidence derived from contrast and edge strength
            confidence = min(1.0, max(0.50, d_rgb * 2.2 + d_edge * 0.6 + change_score * 0.6))
            explanation = f"Significant surface reflectance change detected (ΔIntensity={d_rgb:.2f}, ΔEdge={d_edge:.2f}, Score={change_score:.2f})"

        else:
            change_type = ChangeType.SURFACE_CHANGE
            confidence = min(1.0, max(0.50, change_score * 1.5))
            explanation = f"General surface reflectance change (ΔRGB={d_rgb:.2f}, ChangeScore={change_score:.2f})"

        # Calculate square meters accurately considering CRS units
        area_sq_m = None
        if resolution_m and resolution_m[0] > 0 and resolution_m[1] > 0:
            rx, ry = abs(resolution_m[0]), abs(resolution_m[1])
            is_geographic = (crs and "4326" in crs) or (rx < 0.05 and ry < 0.05)
            if is_geographic:
                lat = center_lat if center_lat is not None else 25.0
                lat_rad = math.radians(lat)
                m_per_deg_lat = 111320.0
                m_per_deg_lon = 111320.0 * math.cos(lat_rad)
                pixel_area_m2 = (rx * m_per_deg_lon) * (ry * m_per_deg_lat)
            else:
                pixel_area_m2 = rx * ry
            area_sq_m = round(float(area_pixels * pixel_area_m2), 2)

        return ChangeRegion(
            region_id=candidate.get("region_id", "change_1"),
            change_type=change_type,
            bounding_box=box,
            area_pixels=area_pixels,
            area_percentage=area_pct,
            area_sq_meters=area_sq_m,
            confidence=round(float(confidence), 4),
            change_score=round(float(change_score), 4),
            spectral_evidence=spec,
            semantic_evidence=sem_evidence,
            explanation=explanation
        )

    @classmethod
    def _evaluate_semantic_delta(
        cls,
        box: BoundingBox,
        pil_b: Image.Image,
        pil_a: Image.Image,
        prompt: str
    ) -> Tuple[Dict[str, Any], float]:
        """Runs OWLv2 zero-shot detection on cropped patches of Before and After images."""
        try:
            w_b, h_b = pil_b.size
            crop_b = pil_b.crop((int(box.xmin), int(box.ymin), int(box.xmax), int(box.ymax)))
            crop_a = pil_a.crop((int(box.xmin), int(box.ymin), int(box.xmax), int(box.ymax)))

            OWLv2ModelManager.load_model()
            dets_b = OWLv2ModelManager.run_detection(crop_b, [prompt], score_threshold=0.001)
            dets_a = OWLv2ModelManager.run_detection(crop_a, [prompt], score_threshold=0.001)

            max_b = max((d[2] for d in dets_b), default=0.0)
            max_a = max((d[2] for d in dets_a), default=0.0)

            delta = float(max_a - max_b)
            evidence = {
                "prompt": prompt,
                "before_score": round(float(max_b), 4),
                "after_score": round(float(max_a), 4),
                "semantic_delta": round(delta, 4)
            }
            return evidence, delta
        except Exception as e:
            logger.debug(f"OWLv2 semantic delta evaluation skipped: {e}")
            return {}, 0.0
