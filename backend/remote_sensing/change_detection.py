import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from backend.config import settings
from backend.api.schemas import BoundingBox, SpectralEvidence
from backend.models.fusion import validate_and_clip_coords, compute_box_iou, compute_containment_ratio
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class BiTemporalChangeDetector:
    """
    Multispectral Bi-Temporal Change Detection and Spatial Region Extraction Engine.
    Computes:
      - Physical spectral index differentials (ΔNDVI, ΔNDWI, ΔNDBI, ΔSAR)
      - Perceptual color / reflectance shift (ΔRGB)
      - Structural edge differential (ΔEdge)
      - Fused change magnitude map M(x, y)
      - Connected components & morphological noise filtering
      - Non-overlapping region cluster consolidation
    """

    @classmethod
    def compute_spectral_indices(cls, bands: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculates multispectral and physical surface indices from band array shape (bands, H, W).
        """
        b_count, h, w = bands.shape
        indices: Dict[str, np.ndarray] = {
            "ndvi": np.zeros((h, w), dtype=np.float32),
            "ndwi": np.zeros((h, w), dtype=np.float32),
            "ndbi": np.zeros((h, w), dtype=np.float32),
            "rgb_norm": np.zeros((h, w, 3), dtype=np.float32),
            "gray": np.zeros((h, w), dtype=np.float32),
            "sar_vv": None,
            "sar_vh": None,
        }

        if b_count == 0:
            return indices

        # Extract Red, Green, Blue, NIR channels
        if b_count >= 3:
            r = bands[0].astype(np.float32)
            g = bands[1].astype(np.float32)
            b = bands[2].astype(np.float32)
            nir = bands[3].astype(np.float32) if b_count >= 4 else r
        elif b_count == 2:
            # SAR VV & VH
            indices["sar_vv"] = bands[0].astype(np.float32)
            indices["sar_vh"] = bands[1].astype(np.float32)
            r = g = b = nir = bands[0].astype(np.float32)
        else:
            r = g = b = nir = bands[0].astype(np.float32)

        # 1. NDVI = (NIR - Red) / (NIR + Red + 1e-6)
        if b_count >= 4:
            indices["ndvi"] = (nir - r) / (nir + r + 1e-6)
        else:
            # Proxy from visible contrast if NIR not present
            indices["ndvi"] = np.clip((g - r) / (g + r + 1e-6), -1.0, 1.0)

        # 2. NDWI = (Green - NIR) / (Green + NIR + 1e-6)
        if b_count >= 4:
            indices["ndwi"] = (g - nir) / (g + nir + 1e-6)
        else:
            indices["ndwi"] = np.clip((b - r) / (b + r + 1e-6), -1.0, 1.0)

        # 3. NDBI = (Red - NIR) / (Red + NIR + 1e-6) (or SWIR proxy)
        if b_count >= 4:
            indices["ndbi"] = (r - nir) / (r + nir + 1e-6)
        else:
            indices["ndbi"] = np.clip((r - g) / (r + g + 1e-6), -1.0, 1.0)

        # 4. Normalized RGB [0.0, 1.0]
        r_norm = cls._normalize_channel(r)
        g_norm = cls._normalize_channel(g)
        b_norm = cls._normalize_channel(b)
        indices["rgb_norm"] = np.stack([r_norm, g_norm, b_norm], axis=-1)

        # 5. Grayscale & Structural Edges
        gray = 0.299 * r_norm + 0.587 * g_norm + 0.114 * b_norm
        indices["gray"] = gray

        return indices

    @classmethod
    def compute_temporal_differences(
        cls,
        bands_before: np.ndarray,
        bands_after: np.ndarray,
        valid_mask: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Computes pixel-wise differential maps between Before and After images using
        robust joint radiometric scaling.
        """
        idx_b = cls.compute_spectral_indices(bands_before)
        idx_a = cls.compute_spectral_indices(bands_after)

        h, w = valid_mask.shape
        b_count = min(bands_before.shape[0], bands_after.shape[0])

        # Temporal Spectral Deltas
        delta_ndvi = (idx_a["ndvi"] - idx_b["ndvi"]) * valid_mask
        delta_ndwi = (idx_a["ndwi"] - idx_b["ndwi"]) * valid_mask
        delta_ndbi = (idx_a["ndbi"] - idx_b["ndbi"]) * valid_mask

        # Joint Radiometric Scaling for Consistent RGB Distance
        rgb_diff_sq = np.zeros((h, w), dtype=np.float32)

        if b_count >= 3:
            for c in range(3):
                cb = bands_before[c].astype(np.float32)
                ca = bands_after[c].astype(np.float32)
                # Robust joint percentile normalization to prevent single outlier pixels from squashing dynamic range
                p_lo = min(float(np.percentile(cb, 1.0)), float(np.percentile(ca, 1.0)))
                p_hi = max(float(np.percentile(cb, 99.0)), float(np.percentile(ca, 99.0)))
                joint_range = max(1e-6, p_hi - p_lo)
                norm_cb = np.clip((cb - p_lo) / joint_range, 0.0, 1.0)
                norm_ca = np.clip((ca - p_lo) / joint_range, 0.0, 1.0)
                rgb_diff_sq += (norm_ca - norm_cb) ** 2
            delta_rgb = (np.sqrt(rgb_diff_sq) / np.sqrt(3.0)) * valid_mask
        else:
            cb = bands_before[0].astype(np.float32)
            ca = bands_after[0].astype(np.float32)
            p_lo = min(float(np.percentile(cb, 1.0)), float(np.percentile(ca, 1.0)))
            p_hi = max(float(np.percentile(cb, 99.0)), float(np.percentile(ca, 99.0)))
            joint_range = max(1e-6, p_hi - p_lo)
            norm_cb = np.clip((cb - p_lo) / joint_range, 0.0, 1.0)
            norm_ca = np.clip((ca - p_lo) / joint_range, 0.0, 1.0)
            delta_rgb = np.abs(norm_ca - norm_cb) * valid_mask

        # Structural Edge Differences (Sobel on joint grayscale)
        gray_b = (bands_before[0].astype(np.float32) / max(1e-6, float(bands_before[0].max()))) * 255.0
        gray_a = (bands_after[0].astype(np.float32) / max(1e-6, float(bands_after[0].max()))) * 255.0

        sobel_b_x = cv2.Sobel(gray_b, cv2.CV_32F, 1, 0, ksize=3)
        sobel_b_y = cv2.Sobel(gray_b, cv2.CV_32F, 0, 1, ksize=3)
        edge_b = np.sqrt(sobel_b_x ** 2 + sobel_b_y ** 2)

        sobel_a_x = cv2.Sobel(gray_a, cv2.CV_32F, 1, 0, ksize=3)
        sobel_a_y = cv2.Sobel(gray_a, cv2.CV_32F, 0, 1, ksize=3)
        edge_a = np.sqrt(sobel_a_x ** 2 + sobel_a_y ** 2)

        edge_max = max(1.0, float(max(edge_b.max(), edge_a.max())))
        delta_edge = (np.abs(edge_a - edge_b) / edge_max) * valid_mask

        # SAR Deltas if available
        delta_sar_vv = None
        delta_sar_vh = None
        if idx_b["sar_vv"] is not None and idx_a["sar_vv"] is not None:
            delta_sar_vv = (idx_a["sar_vv"] - idx_b["sar_vv"]) * valid_mask
        if idx_b["sar_vh"] is not None and idx_a["sar_vh"] is not None:
            delta_sar_vh = (idx_a["sar_vh"] - idx_b["sar_vh"]) * valid_mask

        return {
            "delta_ndvi": delta_ndvi,
            "delta_ndwi": delta_ndwi,
            "delta_ndbi": delta_ndbi,
            "delta_rgb": delta_rgb,
            "delta_edge": delta_edge,
            "delta_sar_vv": delta_sar_vv,
            "delta_sar_vh": delta_sar_vh,
            "ndvi_before": idx_b["ndvi"],
            "ndvi_after": idx_a["ndvi"],
            "ndwi_before": idx_b["ndwi"],
            "ndwi_after": idx_a["ndwi"],
            "ndbi_before": idx_b["ndbi"],
            "ndbi_after": idx_a["ndbi"],
            "rgb_before": idx_b["rgb_norm"],
            "rgb_after": idx_a["rgb_norm"],
            "band_count": b_count,
            "has_multispectral": (b_count >= 4),
            "has_sar": (delta_sar_vv is not None and delta_sar_vh is not None),
        }

    @classmethod
    def compute_change_magnitude_map(
        cls,
        diffs: Dict[str, np.ndarray],
        valid_mask: np.ndarray
    ) -> np.ndarray:
        """
        Fuses available spectral, color, structural, or SAR differentials into a
        normalized magnitude map M(x, y) in [0.0, 1.0] using data-driven signal weighting.
        """
        b_count = diffs.get("band_count", 4)
        has_sar = diffs.get("has_sar", False)
        has_multi = diffs.get("has_multispectral", False)

        if has_sar and diffs.get("delta_sar_vv") is not None and diffs.get("delta_sar_vh") is not None:
            # SAR dual-pol mode
            sar_max = np.maximum(np.abs(diffs["delta_sar_vv"]), np.abs(diffs["delta_sar_vh"]))
            magnitude = 0.50 * sar_max + 0.30 * diffs["delta_rgb"] + 0.20 * diffs["delta_edge"]
        elif has_multi:
            # Full multispectral mode (RGB + NIR/SWIR)
            spectral_max = np.maximum(
                np.maximum(np.abs(diffs["delta_ndvi"]), np.abs(diffs["delta_ndwi"])),
                np.abs(diffs["delta_ndbi"])
            )
            magnitude = (
                0.40 * spectral_max +
                0.35 * diffs["delta_rgb"] +
                0.25 * diffs["delta_edge"]
            )
        elif b_count >= 3:
            # RGB-only mode (visible spectrum + structural edge)
            vis_veg_diff = np.abs(diffs["delta_ndvi"])  # Proxy from visible contrast
            magnitude = (
                0.50 * diffs["delta_rgb"] +
                0.30 * diffs["delta_edge"] +
                0.20 * vis_veg_diff
            )
        else:
            # Single-band / Grayscale mode (intensity + structural edge)
            magnitude = (
                0.65 * diffs["delta_rgb"] +
                0.35 * diffs["delta_edge"]
            )

        # Zero out invalid/NoData pixels
        magnitude[~valid_mask] = 0.0
        return np.clip(magnitude, 0.0, 1.0)

    @classmethod
    def compute_adaptive_threshold(
        cls,
        magnitude_map: np.ndarray,
        valid_mask: np.ndarray,
        sensitivity: Optional[float] = None
    ) -> float:
        """
        Computes a robust, data-adaptive change segmentation threshold based on
        valid-pixel distribution statistics (median + k * MAD).
        Ensures identical or noise-only images return a high threshold above max magnitude,
        while scenes with genuine physical shifts get an optimal separating threshold.
        """
        if not np.any(valid_mask):
            return settings.TEMPORAL_CHANGE_THRESHOLD

        valid_vals = magnitude_map[valid_mask]
        max_val = float(np.max(valid_vals))

        # If overall change in the entire scene is below minimum noise floor, no significant change
        if max_val < settings.TEMPORAL_MIN_MAGNITUDE_FLOOR:
            return max(settings.TEMPORAL_CHANGE_THRESHOLD, max_val + 0.1)

        med = float(np.median(valid_vals))
        mad = float(np.median(np.abs(valid_vals - med)))

        # Robust baseline: median + k * MAD
        k = settings.TEMPORAL_ADAPTIVE_THRESHOLD_SIGMA
        adaptive_tau = med + (k * mad)

        # Also consider high percentile for scenes with small concentrated change
        p_threshold = float(np.percentile(valid_vals, 97.5))
        chosen_tau = max(adaptive_tau, p_threshold * 0.75)

        # Clamp to configured bounds
        tau_min = settings.TEMPORAL_CHANGE_THRESHOLD_MIN
        tau_max = settings.TEMPORAL_CHANGE_THRESHOLD_MAX
        final_tau = float(np.clip(chosen_tau, tau_min, tau_max))

        if sensitivity is not None and 0.0 < sensitivity <= 1.0:
            final_tau = final_tau * (1.0 / max(0.5, min(2.0, sensitivity * 2.0)))
            final_tau = float(np.clip(final_tau, tau_min * 0.7, tau_max))

        return final_tau

    @classmethod
    def extract_candidate_regions(
        cls,
        magnitude_map: np.ndarray,
        diffs: Dict[str, np.ndarray],
        valid_mask: np.ndarray,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Extracts clean, non-overlapping candidate change regions via:
          1. Multi-criteria adaptive or explicit thresholding
          2. Morphological opening (noise removal) & closing (hole filling)
          3. Connected components analysis
          4. Minimum/maximum area bounding box constraints
          5. Spatial cluster consolidation
        """
        h, w = magnitude_map.shape
        valid_pixel_count = max(1, int(np.sum(valid_mask)))

        # Use adaptive data-driven threshold if none provided
        tau = threshold if threshold is not None else cls.compute_adaptive_threshold(magnitude_map, valid_mask)

        # Early exit if scene has no pixels reaching threshold
        if not np.any(valid_mask) or float(np.max(magnitude_map[valid_mask])) < tau:
            return []

        # 1. Binary threshold
        binary_mask = (magnitude_map >= tau).astype(np.uint8) * 255
        binary_mask[~valid_mask] = 0

        if not np.any(binary_mask):
            return []

        # 2. Morphological filtering
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        clean_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_close)

        if not np.any(clean_mask):
            return []

        # 3. Connected Components
        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)

        min_area = settings.MIN_CHANGE_REGION_AREA
        max_area = int(valid_pixel_count * settings.MAX_CHANGE_REGION_AREA_PCT)

        raw_candidates: List[Dict[str, Any]] = []

        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < min_area or area > max_area:
                continue

            x = int(stats[label_id, cv2.CC_STAT_LEFT])
            y = int(stats[label_id, cv2.CC_STAT_TOP])
            bw = int(stats[label_id, cv2.CC_STAT_WIDTH])
            bh = int(stats[label_id, cv2.CC_STAT_HEIGHT])

            box = validate_and_clip_coords(x, y, x + bw, y + bh, w, h)
            if not box:
                continue

            # Extract component slice mask
            comp_mask = (labels_im == label_id)
            comp_score = float(magnitude_map[comp_mask].mean()) if np.any(comp_mask) else 0.0

            raw_candidates.append({
                "box": box,
                "area_pixels": area,
                "area_percentage": round((area / float(valid_pixel_count)) * 100.0, 2),
                "change_score": round(comp_score, 4),
                "label_id": label_id,
            })

        if not raw_candidates:
            return []

        # 4. Spatial Cluster Consolidation
        consolidated = cls._consolidate_candidate_boxes(raw_candidates, magnitude_map, diffs, w, h, valid_pixel_count)

        # 5. Extract Detailed Spectral Evidence for each Consolidated Region
        final_regions: List[Dict[str, Any]] = []
        for i, reg in enumerate(consolidated):
            box = reg["box"]
            bx0, by0 = max(0, int(box.xmin)), max(0, int(box.ymin))
            bx1, by1 = min(w, int(box.xmax)), min(h, int(box.ymax))

            if bx1 <= bx0 or by1 <= by0:
                continue

            # Regional mean metrics
            sub_d_ndvi = float(diffs["delta_ndvi"][by0:by1, bx0:bx1].mean())
            sub_d_ndwi = float(diffs["delta_ndwi"][by0:by1, bx0:bx1].mean())
            sub_d_ndbi = float(diffs["delta_ndbi"][by0:by1, bx0:bx1].mean())
            sub_d_rgb  = float(diffs["delta_rgb"][by0:by1, bx0:bx1].mean())
            sub_d_edge = float(diffs["delta_edge"][by0:by1, bx0:bx1].mean())

            sub_ndvi_b = float(diffs["ndvi_before"][by0:by1, bx0:bx1].mean())
            sub_ndvi_a = float(diffs["ndvi_after"][by0:by1, bx0:bx1].mean())
            sub_ndwi_b = float(diffs["ndwi_before"][by0:by1, bx0:bx1].mean())
            sub_ndwi_a = float(diffs["ndwi_after"][by0:by1, bx0:bx1].mean())
            sub_ndbi_b = float(diffs["ndbi_before"][by0:by1, bx0:bx1].mean())
            sub_ndbi_a = float(diffs["ndbi_after"][by0:by1, bx0:bx1].mean())

            spectral_evidence = SpectralEvidence(
                delta_ndvi=round(sub_d_ndvi, 4),
                delta_ndwi=round(sub_d_ndwi, 4),
                delta_ndbi=round(sub_d_ndbi, 4),
                delta_rgb=round(sub_d_rgb, 4),
                delta_sar_vv=round(float(diffs["delta_sar_vv"][by0:by1, bx0:bx1].mean()), 4) if diffs["delta_sar_vv"] is not None else None,
                delta_sar_vh=round(float(diffs["delta_sar_vh"][by0:by1, bx0:bx1].mean()), 4) if diffs["delta_sar_vh"] is not None else None,
            )

            final_regions.append({
                "region_id": f"change_{i + 1}",
                "box": box,
                "area_pixels": reg["area_pixels"],
                "area_percentage": reg["area_percentage"],
                "change_score": reg["change_score"],
                "spectral_evidence": spectral_evidence,
                "ndvi_before": sub_ndvi_b,
                "ndvi_after": sub_ndvi_a,
                "ndwi_before": sub_ndwi_b,
                "ndwi_after": sub_ndwi_a,
                "ndbi_before": sub_ndbi_b,
                "ndbi_after": sub_ndbi_a,
                "delta_edge": sub_d_edge,
            })

        final_regions.sort(key=lambda r: r["change_score"], reverse=True)
        return final_regions[:settings.TEMPORAL_MAX_CHANGE_REGIONS]

    @classmethod
    def _consolidate_candidate_boxes(
        cls,
        candidates: List[Dict[str, Any]],
        magnitude_map: np.ndarray,
        diffs: Dict[str, np.ndarray],
        img_w: int,
        img_h: int,
        valid_pixel_count: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Merges overlapping or adjacent candidate boxes into coherent single bounding boxes."""
        if not candidates:
            return []

        denom = valid_pixel_count if (valid_pixel_count is not None and valid_pixel_count > 0) else (img_w * img_h)
        n = len(candidates)
        parent = list(range(n))

        def find(i: int) -> int:
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]

        def union(i: int, j: int):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j

        for i in range(n):
            for j in range(i + 1, n):
                b1 = candidates[i]["box"]
                b2 = candidates[j]["box"]
                iou = compute_box_iou(b1, b2)
                containment = compute_containment_ratio(b1, b2)
                if iou > 0.15 or containment > 0.60:
                    union(i, j)

        clusters: Dict[int, List[Dict[str, Any]]] = {}
        for i in range(n):
            root = find(i)
            clusters.setdefault(root, []).append(candidates[i])

        merged: List[Dict[str, Any]] = []
        for root_id, clist in clusters.items():
            u_xmin = min(c["box"].xmin for c in clist)
            u_ymin = min(c["box"].ymin for c in clist)
            u_xmax = max(c["box"].xmax for c in clist)
            u_ymax = max(c["box"].ymax for c in clist)

            enclosing_box = validate_and_clip_coords(u_xmin, u_ymin, u_xmax, u_ymax, img_w, img_h)
            if not enclosing_box:
                continue

            total_area = sum(c["area_pixels"] for c in clist)
            max_score = max(c["change_score"] for c in clist)

            merged.append({
                "box": enclosing_box,
                "area_pixels": total_area,
                "area_percentage": round((total_area / float(denom)) * 100.0, 2),
                "change_score": max_score,
            })

        return merged

    @staticmethod
    def _normalize_channel(arr: np.ndarray) -> np.ndarray:
        """Helper to normalize a single 2D channel to [0.0, 1.0]."""
        p_min = float(np.percentile(arr, 2.0))
        p_max = float(np.percentile(arr, 98.0))
        if p_max > p_min:
            return np.clip((arr - p_min) / (p_max - p_min), 0.0, 1.0)
        else:
            amin = float(np.min(arr))
            amax = float(np.max(arr))
            if amax > amin:
                return np.clip((arr - amin) / (amax - amin), 0.0, 1.0)
            else:
                return np.zeros_like(arr, dtype=np.float32)

