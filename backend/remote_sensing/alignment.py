import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

try:
    import rasterio
    from rasterio.warp import reproject, Resampling, calculate_default_transform
    from rasterio.crs import CRS
    HAS_RASTERIO = True
except (ImportError, Exception):
    HAS_RASTERIO = False

from backend.config import settings
from backend.remote_sensing.raster import RasterReader, RasterMetadata
from backend.api.schemas import RegistrationDiagnostics, RegistrationQuality
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class ImageRegistrar:
    """
    Multi-Tier Image Registration and Spatial Alignment Engine.
    Tier 1: Geospatial CRS Reprojection (Rasterio warp)
    Tier 2: Resolution Normalization & Pixel Grid Resampling
    Tier 3: Bounding Box Overlap & Intersection Truncation
    Tier 4: Feature-Based Alignment Fallback (ORB / RANSAC Homography)
    """

    @staticmethod
    def validate_temporal_pair(
        before_path: Path,
        after_path: Path
    ) -> Tuple[RasterMetadata, RasterMetadata]:
        """
        Extracts metadata and validates that both files exist and are readable rasters.
        """
        before_path = Path(before_path)
        after_path = Path(after_path)

        if not before_path.exists():
            raise FileNotFoundError(f"Before raster not found at '{before_path}'")
        if not after_path.exists():
            raise FileNotFoundError(f"After raster not found at '{after_path}'")

        meta_before = RasterReader.extract_metadata(before_path)
        meta_after = RasterReader.extract_metadata(after_path)

        return meta_before, meta_after

    @classmethod
    def check_geospatial_overlap(
        cls,
        meta_before: RasterMetadata,
        meta_after: RasterMetadata
    ) -> Tuple[bool, float, Optional[Tuple[float, float, float, float]]]:
        """
        Computes spatial bounding box intersection if geospatial metadata is present.
        Returns Tuple of (has_overlap, overlap_percentage, intersection_bounds).
        """
        if not meta_before.is_geospatial or not meta_after.is_geospatial:
            # Fallback to visual assumption (100% canvas comparison unless proven disjoint)
            return True, 100.0, None

        b_before = meta_before.bounds
        b_after = meta_after.bounds

        if not b_before or not b_after:
            return True, 100.0, None

        # Check if CRS is different and rasterio is available to transform bounds
        if meta_before.crs and meta_after.crs and meta_before.crs != meta_after.crs and HAS_RASTERIO:
            try:
                from rasterio.warp import transform_bounds
                b_after_transformed = transform_bounds(
                    meta_after.crs,
                    meta_before.crs,
                    b_after[0], b_after[1], b_after[2], b_after[3]
                )
                b_after = b_after_transformed
            except Exception as e:
                logger.warning(f"Could not transform bounds between CRS '{meta_after.crs}' and '{meta_before.crs}': {e}")

        # Compute intersection of [min_x, min_y, max_x, max_y]
        inter_min_x = max(b_before[0], b_after[0])
        inter_min_y = max(b_before[1], b_after[1])
        inter_max_x = min(b_before[2], b_after[2])
        inter_max_y = min(b_before[3], b_after[3])

        if inter_max_x <= inter_min_x or inter_max_y <= inter_min_y:
            # Disjoint bounds
            return False, 0.0, None

        inter_area = (inter_max_x - inter_min_x) * (inter_max_y - inter_min_y)
        area_before = (b_before[2] - b_before[0]) * (b_before[3] - b_before[1])
        overlap_pct = (inter_area / max(1e-6, area_before)) * 100.0
        overlap_pct = max(0.0, min(100.0, round(float(overlap_pct), 2)))

        inter_bounds = (inter_min_x, inter_min_y, inter_max_x, inter_max_y)
        has_overlap = overlap_pct >= (settings.REGISTRATION_MIN_OVERLAP_PCT * 100.0)

        return has_overlap, overlap_pct, inter_bounds

    @classmethod
    def align_rasters(
        cls,
        before_path: Path,
        after_path: Path
    ) -> Tuple[np.ndarray, np.ndarray, RegistrationDiagnostics, np.ndarray]:
        """
        Performs multi-tier alignment and returns aligned arrays for before and after rasters:
          - aligned_before: np.ndarray shape (bands, H, W)
          - aligned_after: np.ndarray shape (bands, H, W)
          - diagnostics: RegistrationDiagnostics
          - valid_mask: np.ndarray shape (H, W) boolean mask of valid (non-NoData/non-cloud) pixels
        """
        before_path = Path(before_path)
        after_path = Path(after_path)

        meta_before, meta_after = cls.validate_temporal_pair(before_path, after_path)
        bands_before = RasterReader.read_all_bands(before_path)
        bands_after = RasterReader.read_all_bands(after_path)

        orig_dims_b = (meta_before.width, meta_before.height)
        orig_dims_a = (meta_after.width, meta_after.height)

        # 1. Check Geospatial Overlap
        has_overlap, overlap_pct, inter_bounds = cls.check_geospatial_overlap(meta_before, meta_after)

        if not has_overlap:
            diag_fail = RegistrationDiagnostics(
                method_applied="geospatial_bounds_check",
                overlap_percentage=0.0,
                overlap_bounds=None,
                original_dims_before=orig_dims_b,
                original_dims_after=orig_dims_a,
                aligned_dims=orig_dims_b,
                resolution_before=meta_before.resolution,
                resolution_after=meta_after.resolution,
                crs_before=meta_before.crs,
                crs_after=meta_after.crs,
                registration_quality=RegistrationQuality.FAILED
            )
            # Create empty arrays & mask indicating no overlap
            return bands_before, bands_after, diag_fail, np.zeros((meta_before.height, meta_before.width), dtype=bool)

        # 2. Tier 1: Geospatial Reprojection (if both have CRS and Rasterio is available)
        if meta_before.is_geospatial and meta_after.is_geospatial and HAS_RASTERIO and meta_before.crs:
            try:
                aligned_after_geo = np.zeros_like(bands_before, dtype=bands_after.dtype) if bands_before.shape[0] == bands_after.shape[0] else np.zeros((bands_after.shape[0], meta_before.height, meta_before.width), dtype=bands_after.dtype)

                with rasterio.open(before_path) as src_before, rasterio.open(after_path) as src_after:
                    reproject(
                        source=rasterio.band(src_after, list(range(1, src_after.count + 1))),
                        destination=aligned_after_geo,
                        src_transform=src_after.transform,
                        src_crs=src_after.crs,
                        dst_transform=src_before.transform,
                        dst_crs=src_before.crs,
                        resampling=Resampling.bilinear
                    )

                # Ensure dimensions match before exactly
                target_h, target_w = meta_before.height, meta_before.width
                aligned_before = bands_before
                aligned_after = aligned_after_geo

                valid_mask = np.ones((target_h, target_w), dtype=bool)
                if meta_before.nodata is not None:
                    valid_mask &= (aligned_before[0] != meta_before.nodata)
                if meta_after.nodata is not None:
                    valid_mask &= (aligned_after[0] != meta_after.nodata)

                diag = RegistrationDiagnostics(
                    method_applied="geospatial_reprojection",
                    overlap_percentage=overlap_pct,
                    overlap_bounds=inter_bounds,
                    original_dims_before=orig_dims_b,
                    original_dims_after=orig_dims_a,
                    aligned_dims=(target_w, target_h),
                    resolution_before=meta_before.resolution,
                    resolution_after=meta_after.resolution,
                    crs_before=meta_before.crs,
                    crs_after=meta_after.crs,
                    registration_quality=RegistrationQuality.HIGH
                )
                logger.info(f"Successfully aligned rasters via geospatial reprojection (Overlap={overlap_pct}%).")
                return aligned_before, aligned_after, diag, valid_mask
            except Exception as e:
                logger.warning(f"Geospatial reprojection failed: {e}. Falling back to resolution & feature alignment.")

        # 3. Tier 2 & Tier 4: Resolution Normalization and Feature-Based Alignment Fallback
        target_h, target_w = meta_before.height, meta_before.width
        aligned_before = bands_before.copy()

        # Resample after bands to target dimensions first
        b_count_after = bands_after.shape[0]
        resampled_after = np.zeros((b_count_after, target_h, target_w), dtype=bands_after.dtype)
        for i in range(b_count_after):
            resampled_after[i] = cv2.resize(
                bands_after[i].astype(np.float32),
                (target_w, target_h),
                interpolation=cv2.INTER_AREA if (meta_after.width > target_w) else cv2.INTER_LINEAR
            ).astype(bands_after.dtype)

        # Attempt Feature-Based Registration (ORB + RANSAC) on 8-bit visual composites
        reg_method = "pixel_grid_resample"
        reg_quality = RegistrationQuality.MEDIUM

        try:
            # Create grayscale 8-bit reference images
            b_gray = cls._to_gray_uint8(aligned_before)
            a_gray = cls._to_gray_uint8(resampled_after)

            orb = cv2.ORB_create(nfeatures=settings.REGISTRATION_FEATURE_MAX_KEYPOINTS)
            kp_b, des_b = orb.detectAndCompute(b_gray, None)
            kp_a, des_a = orb.detectAndCompute(a_gray, None)

            if des_b is not None and des_a is not None and len(kp_b) >= 8 and len(kp_a) >= 8:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                raw_matches = bf.knnMatch(des_a, des_b, k=2)

                good_matches = []
                for m_pair in raw_matches:
                    if len(m_pair) == 2:
                        m, n = m_pair
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)

                if len(good_matches) >= settings.REGISTRATION_MIN_MATCHES:
                    src_pts = np.float32([kp_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                    # Estimate affine or homography transformation
                    matrix, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)

                    if matrix is not None and inliers is not None and inliers.sum() >= 6:
                        warped_after = np.zeros_like(resampled_after)
                        for b_idx in range(b_count_after):
                            warped_after[b_idx] = cv2.warpAffine(
                                resampled_after[b_idx],
                                matrix,
                                (target_w, target_h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0
                            )
                        resampled_after = warped_after
                        reg_method = "feature_affine_ransac"
                        reg_quality = RegistrationQuality.HIGH
                        logger.info(f"Applied feature-based affine registration ({inliers.sum()} inliers).")
        except Exception as e_feat:
            logger.debug(f"Feature registration fallback skipped: {e_feat}")

        aligned_after = resampled_after

        # Compute valid pixel mask
        valid_mask = np.ones((target_h, target_w), dtype=bool)
        if meta_before.nodata is not None:
            valid_mask &= (aligned_before[0] != meta_before.nodata)
        if meta_after.nodata is not None:
            valid_mask &= (aligned_after[0] != meta_after.nodata)

        # Also mask out zero-padded border areas from affine warping if any
        if reg_method == "feature_affine_ransac":
            valid_mask &= (np.any(aligned_after > 0, axis=0))

        diag = RegistrationDiagnostics(
            method_applied=reg_method,
            overlap_percentage=overlap_pct,
            overlap_bounds=inter_bounds,
            original_dims_before=orig_dims_b,
            original_dims_after=orig_dims_a,
            aligned_dims=(target_w, target_h),
            resolution_before=meta_before.resolution,
            resolution_after=meta_after.resolution,
            crs_before=meta_before.crs,
            crs_after=meta_after.crs,
            registration_quality=reg_quality
        )

        return aligned_before, aligned_after, diag, valid_mask

    @staticmethod
    def _to_gray_uint8(bands_arr: np.ndarray) -> np.ndarray:
        """Converts multi-band or single-band array into an 8-bit grayscale image."""
        if bands_arr.ndim == 2:
            arr = bands_arr.astype(np.float32)
        elif bands_arr.shape[0] >= 3:
            # Standard RGB composite to gray
            r, g, b = bands_arr[0].astype(np.float32), bands_arr[1].astype(np.float32), bands_arr[2].astype(np.float32)
            arr = 0.299 * r + 0.587 * g + 0.114 * b
        else:
            arr = bands_arr[0].astype(np.float32)

        p_min, p_max = float(np.percentile(arr, 2.0)), float(np.percentile(arr, 98.0))
        if p_max > p_min:
            norm = np.clip((arr - p_min) / (p_max - p_min) * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            norm = np.zeros_like(arr, dtype=np.uint8)
        return norm
