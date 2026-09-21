import cv2
import math
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from PIL import Image

from backend.config import settings
from backend.api.schemas import BoundingBox, EvidenceRegion, ImageDimensions
from backend.remote_sensing.raster import RasterReader
from backend.remote_sensing.bands import BandManager
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def validate_and_clip_coords(
    xmin: float, ymin: float, xmax: float, ymax: float,
    img_w: int, img_h: int,
    min_size: float = 2.0
) -> Optional[BoundingBox]:
    """Clamps box coordinates strictly within image bounds [0, img_w] x [0, img_h]."""
    if img_w <= 0 or img_h <= 0:
        return None
    if xmax < xmin:
        xmin, xmax = xmax, xmin
    if ymax < ymin:
        ymin, ymax = ymax, ymin

    clamped_xmin = max(0.0, min(float(xmin), float(img_w)))
    clamped_ymin = max(0.0, min(float(ymin), float(img_h)))
    clamped_xmax = max(0.0, min(float(xmax), float(img_w)))
    clamped_ymax = max(0.0, min(float(ymax), float(img_h)))

    if (clamped_xmax - clamped_xmin) < min_size or (clamped_ymax - clamped_ymin) < min_size:
        return None

    return BoundingBox(
        xmin=round(clamped_xmin, 2),
        ymin=round(clamped_ymin, 2),
        xmax=round(clamped_xmax, 2),
        ymax=round(clamped_ymax, 2)
    )


def compute_box_iou(b1: BoundingBox, b2: BoundingBox) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes."""
    x1 = max(float(b1.xmin), float(b2.xmin))
    y1 = max(float(b1.ymin), float(b2.ymin))
    x2 = min(float(b1.xmax), float(b2.xmax))
    y2 = min(float(b1.ymax), float(b2.ymax))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (float(b1.xmax) - float(b1.xmin)) * (float(b1.ymax) - float(b1.ymin))
    area2 = (float(b2.xmax) - float(b2.xmin)) * (float(b2.ymax) - float(b2.ymin))
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def compute_containment_ratio(b1: BoundingBox, b2: BoundingBox) -> float:
    """Computes intersection over minimum area (containment ratio)."""
    x1 = max(float(b1.xmin), float(b2.xmin))
    y1 = max(float(b1.ymin), float(b2.ymin))
    x2 = min(float(b1.xmax), float(b2.xmax))
    y2 = min(float(b1.ymax), float(b2.ymax))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (float(b1.xmax) - float(b1.xmin)) * (float(b1.ymax) - float(b1.ymin))
    area2 = (float(b2.xmax) - float(b2.xmin)) * (float(b2.ymax) - float(b2.ymin))
    min_area = min(area1, area2)

    if min_area <= 0.0:
        return 0.0
    return inter_area / min_area


class RemoteSensingEvidenceFusionEngine:
    """
    Multi-criteria Evidence Fusion Engine for overhead satellite and aerial imagery.
    Combines:
      1. OWLv2 zero-shot semantic candidate grounding
      2. Physical multispectral index masks (NDVI, NDWI, NDBI) when genuine NIR is present
         OR visual color/contrast masks for standard RGB/RGBA imagery
      3. Second-stage spatial clustering & deduplication
      4. Transparent candidate fusion scoring & dtype-aware physical cross-category elimination
    """

    WEIGHT_SEMANTIC = 0.35
    WEIGHT_SPECTRAL = 0.35
    WEIGHT_SPATIAL  = 0.15
    WEIGHT_SHAPE    = 0.15

    FUSION_SCORE_THRESHOLD = 0.28  # Calibrated acceptance threshold

    @classmethod
    def _normalize_bands(cls, bands: np.ndarray) -> np.ndarray:
        """Normalizes band array to float32 in [0.0, 1.0] range based on dtype and value range."""
        b_float = bands.astype(np.float32)
        if bands.dtype == np.uint8:
            return np.clip(b_float / 255.0, 0.0, 1.0)
        elif bands.dtype == np.uint16:
            # Check if 12-bit (0-4095) or standard Sentinel-2 surface reflectance (0-10000) or full 16-bit
            max_v = float(np.percentile(b_float, 99.5)) if b_float.size > 100 else float(b_float.max())
            divisor = max(10000.0, max_v) if max_v > 4095.0 else max(4095.0, max_v)
            divisor = max(1.0, divisor)
            return np.clip(b_float / divisor, 0.0, 1.0)
        else:
            max_v = float(b_float.max())
            if max_v > 1.0:
                return np.clip(b_float / max_v, 0.0, 1.0)
            return np.clip(b_float, 0.0, 1.0)

    @classmethod
    def extract_spectral_masks(
        cls,
        file_path: Path,
        img_w: int,
        img_h: int,
        preloaded_bands: Optional[np.ndarray] = None,
        preloaded_meta: Optional[Any] = None
    ) -> Dict[str, np.ndarray]:
        """
        Extracts uint8 binary masks (0 or 255).
        If authentic physical NIR band is present: calculates physical NDVI, NDWI, built-up indices.
        If RGB or RGBA: calculates visual color/contrast indices without fabricating NIR data.
        """
        masks = {
            "building": np.zeros((img_h, img_w), dtype=np.uint8),
            "water": np.zeros((img_h, img_w), dtype=np.uint8),
            "vegetation": np.zeros((img_h, img_w), dtype=np.uint8),
            "road": np.zeros((img_h, img_w), dtype=np.uint8),
        }

        try:
            meta = preloaded_meta or RasterReader.extract_metadata(file_path)
            bands = preloaded_bands if preloaded_bands is not None else RasterReader.read_all_bands(file_path)

            b_count = bands.shape[0]
            if b_count < 1:
                return masks

            has_nir = BandManager.has_physical_nir(meta)
            norm_bands = cls._normalize_bands(bands)

            if b_count >= 3:
                red = cv2.resize(norm_bands[0], (img_w, img_h))
                green = cv2.resize(norm_bands[1], (img_w, img_h))
                blue = cv2.resize(norm_bands[2], (img_w, img_h))
            else:
                single = cv2.resize(norm_bands[0], (img_w, img_h))
                red, green, blue = single, single, single

            if has_nir and b_count >= 4:
                # Authentic Near-Infrared (NIR) Band Available (Multispectral Satellite)
                nir = cv2.resize(norm_bands[3], (img_w, img_h))

                # 1. Vegetation Mask — High NDVI (>0.20) & Elevated NIR
                ndvi = (nir - red) / (nir + red + 1e-6)
                veg_cond = (ndvi > 0.20) & (nir > 0.20)
                masks["vegetation"] = (veg_cond * 255).astype(np.uint8)

                # 2. Water Mask — High NDWI (>0.12) & Low NIR (absorption)
                ndwi = (green - nir) / (green + nir + 1e-6)
                water_cond = (ndwi > 0.12) & (nir < 0.20)
                masks["water"] = (water_cond * 255).astype(np.uint8)

                # 3. Built-up / Buildings Mask — Visible Reflection, Low NDVI, Low NDWI
                bldg_cond = (ndvi < 0.20) & (ndwi < 0.05) & (red > 0.20) & (blue > 0.20) & (nir < 0.40)
                bldg_binary = (bldg_cond * 255).astype(np.uint8)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                masks["building"] = cv2.morphologyEx(bldg_binary, cv2.MORPH_CLOSE, kernel)

                # 4. Road Mask — Linear Canny edges with moderate reflectance & low NDVI
                gray = (0.299 * red + 0.587 * green + 0.114 * blue)
                gray_u8 = (gray * 255).astype(np.uint8)
                edges = cv2.Canny(gray_u8, 50, 150)
                road_cond = (edges > 0) & (ndvi < 0.20) & (ndwi < 0.08) & (red > 0.15)
                masks["road"] = (road_cond * 255).astype(np.uint8)

                # 5. Agriculture Mask — Elevated NDVI (>0.15) with low NDWI
                ag_cond = (ndvi > 0.15) & (ndwi < 0.10) & (nir > 0.18)
                masks["agriculture"] = (ag_cond * 255).astype(np.uint8)

                # 6. Urban Mask — Mirrors built-up/building footprint
                masks["urban"] = masks["building"]

                # 7. Bare Soil Mask — Low NDVI, Low NDWI, high reflectance in red
                soil_cond = (ndvi < 0.15) & (ndwi < 0.05) & (red > 0.20)
                masks["bare_soil"] = (soil_cond * 255).astype(np.uint8)

            else:
                # Standard RGB / RGBA Imagery (Visual Color-Space Heuristics)
                # 1. Vegetation — Green channel noticeably exceeds Red and Blue
                veg_cond = (green > red * 1.08) & (green > blue * 1.08) & (green > 0.15)
                masks["vegetation"] = (veg_cond * 255).astype(np.uint8)

                # 2. Water — Blue/Green dominant with absorption or deep darkness
                water_cond = ((blue > red * 1.15) & (green > red * 1.05) & (blue < 0.55)) | (
                    (red < 0.12) & (green < 0.15) & (blue < 0.25)
                )
                masks["water"] = (water_cond * 255).astype(np.uint8)

                # 3. Built-up / Buildings — Strong localized edge gradients and contrast
                gray = (0.299 * red + 0.587 * green + 0.114 * blue)
                gray_u8 = (gray * 255).astype(np.uint8)
                edges = cv2.Canny(gray_u8, 60, 160)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                masks["building"] = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

                # 4. Road — Canny edge network
                masks["road"] = edges

                # 5. Urban — Mirrors building edges
                masks["urban"] = masks["building"]

        except Exception as e:
            logger.error(f"Error computing spectral/visual masks: {e}")

        return masks

    @classmethod
    def generate_spectral_segmentation_candidates(
        cls,
        masks: Dict[str, np.ndarray],
        category: str,
        img_w: int,
        img_h: int,
        has_nir: bool = False
    ) -> List[Tuple[BoundingBox, float, str]]:
        """Extracts candidate bounding boxes from connected components of spectral masks."""
        if not has_nir:
            # Plain RGB / Grayscale Canny edges are not authentic spectral signatures
            return []

        cat_key = category.lower()
        mask = masks.get(cat_key)
        if mask is None or not np.any(mask):
            return []

        candidates = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = (img_w * img_h) * 0.001
        max_area = (img_w * img_h) * 0.40

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area <= area <= max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                box = validate_and_clip_coords(x, y, x + w, y + h, img_w, img_h)
                if box:
                    candidates.append((box, 0.020, "spectral_connected_component"))

        return candidates

    @classmethod
    def evaluate_candidate_scores(
        cls,
        file_path: Path,
        box: BoundingBox,
        category: str,
        owlv2_raw_score: float,
        source_tile: str,
        masks: Dict[str, np.ndarray],
        img_w: int,
        img_h: int,
        preloaded_bands: Optional[np.ndarray] = None,
        preloaded_meta: Optional[Any] = None
    ) -> Tuple[float, Dict[str, float], str]:
        """
        Calculates candidate fusion score and applies dtype-aware physical cross-category elimination.
        Uses preloaded raster arrays to avoid disk I/O bottlenecks.
        Uses adaptive weighting when multispectral data is not applicable.
        """
        cat_lower = category.lower()
        bw = float(box.xmax - box.xmin)
        bh = float(box.ymax - box.ymin)
        area = bw * bh
        area_ratio = area / float(img_w * img_h)

        # Container box penalty (>45% canvas)
        if area_ratio > 0.45:
            return 0.0, {
                "owlv2_score": owlv2_raw_score,
                "spectral_score": 0.0,
                "spatial_score": 0.0,
                "shape_score": 0.0,
                "final_score": 0.0
            }, f"Rejected: Container box (>45% area: {area_ratio*100:.1f}%)"

        # Minimum physical size filter for large infrastructure / regional objects
        if any(term in cat_lower for term in ["building", "house", "structure", "airport", "agricultural", "agriculture", "field", "urban"]):
            if bw < 14.0 or bh < 14.0 or area < 160.0:
                return 0.0, {
                    "owlv2_score": owlv2_raw_score,
                    "spectral_score": 0.0,
                    "spatial_score": 0.0,
                    "shape_score": 0.0,
                    "final_score": 0.0
                }, f"Rejected: Noise fragment too small ({bw:.1f}x{bh:.1f} < 14px)"

        # Calibrated monotonic semantic score preserving relative model certainty
        # 0.025 -> 0.208, 0.035 -> 0.292, 0.10 -> 0.833, >=0.12 -> 1.0
        s_semantic = float(np.clip(owlv2_raw_score / 0.12, 0.0, 1.0))

        rx0, ry0 = max(0, min(img_w - 1, int(box.xmin))), max(0, min(img_h - 1, int(box.ymin)))
        rx1, ry1 = max(rx0 + 1, min(img_w, int(box.xmax))), max(ry0 + 1, min(img_h, int(box.ymax)))

        # Check metadata for genuine NIR
        meta = preloaded_meta or RasterReader.extract_metadata(file_path)
        has_nir = BandManager.has_physical_nir(meta)

        mask = masks.get(cat_lower)
        s_spectral = 0.0
        if mask is not None and mask.size > 0 and rx1 > rx0 and ry1 > ry0:
            box_mask_slice = mask[ry0:ry1, rx0:rx1]
            if box_mask_slice.size > 0:
                if has_nir:
                    mean_val = float(box_mask_slice.mean())
                    s_spectral = (mean_val / 255.0) if not math.isnan(mean_val) else 0.0
                else:
                    # In visual/edge mode, edge pixels scale with perimeter, NOT area
                    perimeter = 2.0 * (bw + bh)
                    edge_pixel_count = float(np.count_nonzero(box_mask_slice))
                    s_spectral = min(1.0, edge_pixel_count / max(1.0, perimeter * 0.75))

        # Dtype-aware physical cross-category elimination ONLY when authentic NIR exists
        if has_nir and preloaded_bands is not None and preloaded_bands.shape[0] >= 4 and rx1 > rx0 and ry1 > ry0:
            try:
                norm_bands = cls._normalize_bands(preloaded_bands)
                # Resample box slice safely
                bh_orig, bw_orig = preloaded_bands.shape[1], preloaded_bands.shape[2]
                sx0 = int(rx0 * (bw_orig / float(img_w)))
                sy0 = int(ry0 * (bh_orig / float(img_h)))
                sx1 = max(sx0 + 1, int(rx1 * (bw_orig / float(img_w))))
                sy1 = max(sy0 + 1, int(ry1 * (bh_orig / float(img_h))))

                red_s = float(norm_bands[0, sy0:sy1, sx0:sx1].mean())
                green_s = float(norm_bands[1, sy0:sy1, sx0:sx1].mean())
                nir_s = float(norm_bands[3, sy0:sy1, sx0:sx1].mean())

                ndvi_s = float((nir_s - red_s) / (nir_s + red_s + 1e-6))
                ndwi_s = float((green_s - nir_s) / (green_s + nir_s + 1e-6))

                if any(term in cat_lower for term in ["building", "structure", "house", "urban"]):
                    if nir_s < 0.12 and ndwi_s > 0.18:
                        return 0.0, {"owlv2_score": owlv2_raw_score, "spectral_score": 0.0, "spatial_score": 0.0, "shape_score": 0.0, "final_score": 0.0}, "Physical Elimination: Building candidate over deep water"
                elif any(term in cat_lower for term in ["agriculture", "agricultural", "field", "farmland"]):
                    if ndwi_s > 0.25:
                        return 0.0, {"owlv2_score": owlv2_raw_score, "spectral_score": 0.0, "spatial_score": 0.0, "shape_score": 0.0, "final_score": 0.0}, "Physical Elimination: Agriculture candidate over deep water"
                elif "water" in cat_lower:
                    if nir_s > 0.35 and ndwi_s < 0.0:
                        return 0.0, {"owlv2_score": owlv2_raw_score, "spectral_score": 0.0, "spatial_score": 0.0, "shape_score": 0.0, "final_score": 0.0}, "Physical Elimination: Water candidate over reflective dry land"
                elif "vegetation" in cat_lower or "forest" in cat_lower or "tree" in cat_lower:
                    if ndvi_s < 0.10 and nir_s < 0.15:
                        return 0.0, {"owlv2_score": owlv2_raw_score, "spectral_score": 0.0, "spatial_score": 0.0, "shape_score": 0.0, "final_score": 0.0}, "Physical Elimination: Vegetation candidate over non-vegetated terrain"
                elif "road" in cat_lower:
                    if ndwi_s > 0.25:
                        return 0.0, {"owlv2_score": owlv2_raw_score, "spectral_score": 0.0, "spatial_score": 0.0, "shape_score": 0.0, "final_score": 0.0}, "Physical Elimination: Road candidate over deep water"
            except Exception as e:
                logger.debug(f"Physical spectral elimination check skipped: {e}")

        s_spatial = s_spectral if (mask is not None and has_nir) else 0.50

        aspect_ratio = max(bw, bh) / max(1.0, min(bw, bh))

        if "road" in cat_lower or "highway" in cat_lower:
            s_shape = min(1.0, aspect_ratio / 2.5) if aspect_ratio >= 1.8 else 0.55
        elif any(term in cat_lower for term in ["building", "house", "urban"]):
            s_shape = 0.90 if 0.4 <= (bw / max(1.0, bh)) <= 2.5 else 0.50
        elif any(term in cat_lower for term in ["agriculture", "agricultural", "field", "cropland", "farmland"]):
            s_shape = 0.85 if 0.3 <= (bw / max(1.0, bh)) <= 3.0 else 0.60
        elif any(term in cat_lower for term in ["bare_soil", "bare soil", "barren"]):
            s_shape = 0.75
        elif "vehicle" in cat_lower or "car" in cat_lower:
            s_shape = 0.85 if 0.3 <= (bw / max(1.0, bh)) <= 3.0 and area_ratio < 0.05 else 0.45
        elif "ship" in cat_lower or "boat" in cat_lower:
            s_shape = 0.85 if aspect_ratio >= 1.5 else 0.55
        else:
            s_shape = 0.75

        # Adaptive Evidence Fusion Weighting
        is_spectral_mode = has_nir and (cat_lower in ("building", "water", "vegetation", "road", "structure", "agriculture", "urban", "bare_soil"))

        if is_spectral_mode:
            # Multispectral Satellite Mode
            w_sem = cls.WEIGHT_SEMANTIC
            w_spec = cls.WEIGHT_SPECTRAL
            w_spat = cls.WEIGHT_SPATIAL
            w_shape = cls.WEIGHT_SHAPE
        else:
            # Visual RGB / Grayscale or Open-Vocabulary Category Mode
            # Vision model OWLv2 holds primary grounding authority
            if mask is None or mask.size == 0 or not np.any(mask):
                w_sem = 0.75
                w_spec = 0.00
                w_spat = 0.12
                w_shape = 0.13
            else:
                w_sem = 0.65
                w_spec = 0.10
                w_spat = 0.12
                w_shape = 0.13

        final_score = (
            w_sem * s_semantic +
            w_spec * s_spectral +
            w_spat * s_spatial +
            w_shape * s_shape
        )

        comp_scores = {
            "owlv2_score": round(owlv2_raw_score, 4),
            "spectral_score": round(s_spectral, 4),
            "spatial_score": round(s_spatial, 4),
            "shape_score": round(s_shape, 4),
            "final_score": round(final_score, 4)
        }

        explanation = f"Fusion score={final_score:.3f} (Sem={s_semantic:.2f}, Spec={s_spectral:.2f}, Shape={s_shape:.2f})"
        return final_score, comp_scores, explanation

    @classmethod
    def cluster_and_consolidate_regions(
        cls,
        regions: List[EvidenceRegion],
        category: str,
        masks: Dict[str, np.ndarray],
        img_w: int,
        img_h: int,
        max_regions: Optional[int] = None
    ) -> List[EvidenceRegion]:
        """
        Second-stage spatial clustering and deduplication.
        Uses non-transitive greedy centroid clustering to merge duplicate detections
        of the same object across overlapping tiles while strictly preserving distinct objects.
        Guarantees that winning candidate box, score, tile, and metadata remain coherent.
        """
        if not regions:
            return []

        limit = max_regions or settings.GROUNDING_MAX_EVIDENCE_REGIONS
        cat_lower = category.lower()
        is_continuous_layer = cat_lower in ("water", "vegetation")

        # Sort descending by final score / confidence
        sorted_regs = sorted(
            regions,
            key=lambda r: r.final_score if r.final_score is not None else r.confidence,
            reverse=True
        )

        used = [False] * len(sorted_regs)
        consolidated: List[EvidenceRegion] = []

        for i, leader in enumerate(sorted_regs):
            if used[i]:
                continue
            used[i] = True
            cluster = [leader]

            b1 = leader.box
            c1x = (b1.xmin + b1.xmax) / 2.0
            c1y = (b1.ymin + b1.ymax) / 2.0
            w1 = b1.xmax - b1.xmin
            h1 = b1.ymax - b1.ymin

            for j in range(i + 1, len(sorted_regs)):
                if used[j]:
                    continue
                cand = sorted_regs[j]
                b2 = cand.box
                c2x = (b2.xmin + b2.xmax) / 2.0
                c2y = (b2.ymin + b2.ymax) / 2.0
                w2 = b2.xmax - b2.xmin
                h2 = b2.ymax - b2.ymin

                iou = compute_box_iou(b1, b2)
                containment = compute_containment_ratio(b1, b2)
                center_dist = math.hypot(c1x - c2x, c1y - c2y)
                min_dim = min(w1, h1, w2, h2)

                should_merge = False
                if is_continuous_layer:
                    if iou >= 0.25 or (containment >= 0.60 and center_dist < max(w1, h1)):
                        should_merge = True
                else:
                    # Discrete objects: merge if high IoU OR high containment with very close centers
                    if iou >= 0.45 or (containment >= 0.75 and center_dist <= min_dim * 0.75):
                        should_merge = True

                if should_merge:
                    used[j] = True
                    cluster.append(cand)
                    if not is_continuous_layer and len(cluster) >= 12:
                        break

            if is_continuous_layer:
                u_xmin = min(r.box.xmin for r in cluster)
                u_ymin = min(r.box.ymin for r in cluster)
                u_xmax = max(r.box.xmax for r in cluster)
                u_ymax = max(r.box.ymax for r in cluster)
                final_box = validate_and_clip_coords(u_xmin, u_ymin, u_xmax, u_ymax, img_w, img_h)
            else:
                final_box = leader.box

            if not final_box:
                continue

            total_merged = sum((r.merged_detection_count or 1 for r in cluster))
            lead_final = leader.final_score if leader.final_score is not None else leader.confidence

            consolidated.append(EvidenceRegion(
                box=final_box,
                label=category,
                confidence=round(lead_final, 4),
                explanation=f"Detected {category} (Conf={lead_final:.3f}, Merged={total_merged})",
                owlv2_score=leader.owlv2_score,
                spectral_score=leader.spectral_score,
                spatial_score=leader.spatial_score,
                shape_score=leader.shape_score,
                final_score=round(lead_final, 4),
                source_tile=leader.source_tile,
                merged_detection_count=total_merged,
                coord_metadata=leader.coord_metadata
            ))

            if len(consolidated) >= limit:
                break

        return consolidated
