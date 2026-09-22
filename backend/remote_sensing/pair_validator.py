import hashlib
from enum import Enum
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import numpy as np
import cv2
from pydantic import BaseModel, Field

try:
    import rasterio
    HAS_RASTERIO = True
except (ImportError, Exception):
    HAS_RASTERIO = False

from backend.config import settings
from backend.remote_sensing.raster import RasterReader, RasterMetadata
from backend.remote_sensing.alignment import ImageRegistrar
from backend.api.schemas import PairClassification, PairValidationDiagnostics
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class TemporalPairValidator:
    """
    Input-Pair Validation Engine for Bi-Temporal Satellite Imagery Analysis.
    Validates image pairs BEFORE registration and change detection:
      1. Exact cryptographic content hash check (SHA-256) -> IDENTICAL_IMAGES
      2. Pixel array exact equality check -> IDENTICAL_IMAGES
      3. Geospatial metadata compatibility & footprint overlap -> INCOMPATIBLE_IMAGES if disjoint
      4. Near-duplicate structural similarity & difference metrics -> NEAR_DUPLICATE
      5. Visual scene geometry & feature consensus (ORB/RANSAC) -> INCOMPATIBLE_IMAGES if unrelated
    """

    @staticmethod
    def compute_file_sha256(file_path: Path) -> str:
        """Computes SHA-256 hexadecimal digest of raw file bytes."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Computes Structural Similarity Index (SSIM) between two 8-bit grayscale images.
        Vectorized implementation using OpenCV filter2D with a Gaussian kernel.
        """
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_LINEAR)

        im1 = img1.astype(np.float32)
        im2 = img2.astype(np.float32)

        C1 = (0.01 * 255.0) ** 2
        C2 = (0.03 * 255.0) ** 2

        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(im1, -1, window)
        mu2 = cv2.filter2D(im2, -1, window)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.filter2D(im1 * im1, -1, window) - mu1_sq
        sigma2_sq = cv2.filter2D(im2 * im2, -1, window) - mu2_sq
        sigma12 = cv2.filter2D(im1 * im2, -1, window) - mu1_mu2

        denom = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        denom = np.maximum(denom, 1e-6)
        ssim_map = ((2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)) / denom

        return float(np.clip(ssim_map.mean(), -1.0, 1.0))

    @classmethod
    def check_visual_feature_consensus(
        cls,
        gray_b: np.ndarray,
        gray_a: np.ndarray
    ) -> Tuple[int, int, float]:
        """
        Computes geometric feature correspondence between two 8-bit grayscale images using ORB and RANSAC.
        Returns:
            Tuple of (good_matches_count, inlier_count, inlier_ratio)
        """
        orb = cv2.ORB_create(nfeatures=settings.REGISTRATION_FEATURE_MAX_KEYPOINTS)
        kp_b, des_b = orb.detectAndCompute(gray_b, None)
        kp_a, des_a = orb.detectAndCompute(gray_a, None)

        if des_b is None or des_a is None or len(kp_b) < 6 or len(kp_a) < 6:
            return 0, 0, 0.0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        try:
            raw_matches = bf.knnMatch(des_a, des_b, k=2)
        except Exception:
            return 0, 0, 0.0

        good_matches = []
        for pair in raw_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 4:
            return len(good_matches), 0, 0.0

        src_pts = np.float32([kp_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        try:
            matrix, inliers = cv2.estimateAffinePartial2D(
                src_pts,
                dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=5.0
            )
            inlier_count = int(inliers.sum()) if inliers is not None else 0
            inlier_ratio = float(inlier_count / max(1, len(good_matches)))
            return len(good_matches), inlier_count, inlier_ratio
        except Exception:
            return len(good_matches), 0, 0.0

    @classmethod
    def validate_pair(
        cls,
        before_path: Path,
        after_path: Path
    ) -> PairValidationDiagnostics:
        """
        Validates the input temporal pair and returns comprehensive diagnostics and classification.
        """
        before_path = Path(before_path)
        after_path = Path(after_path)

        if not before_path.exists():
            raise FileNotFoundError(f"Before image not found at '{before_path}'")
        if not after_path.exists():
            raise FileNotFoundError(f"After image not found at '{after_path}'")

        # ------------------------------------------------------------------
        # 1. Exact Cryptographic Check (SHA-256)
        # ------------------------------------------------------------------
        sha_b = cls.compute_file_sha256(before_path)
        sha_a = cls.compute_file_sha256(after_path)

        if sha_b == sha_a:
            logger.info(f"Pair validation: Exact SHA-256 match ({sha_b[:12]}...). Classifying as IDENTICAL_IMAGES.")
            return PairValidationDiagnostics(
                classification=PairClassification.IDENTICAL_IMAGES,
                similarity_score=1.0,
                spatially_compatible=True,
                exact_hash_match=True,
                spatial_overlap_pct=100.0,
                crs_compatible=True,
                resolution_compatible=True,
                structural_similarity=1.0,
                feature_inliers=None,
                reason="IDENTICAL_IMAGES"
            )

        # ------------------------------------------------------------------
        # 2. Extract Metadata & Band Arrays
        # ------------------------------------------------------------------
        meta_b = RasterReader.extract_metadata(before_path)
        meta_a = RasterReader.extract_metadata(after_path)

        # ------------------------------------------------------------------
        # 2. Geospatial Compatibility Evaluation
        # ------------------------------------------------------------------
        crs_compat: Optional[bool] = None
        res_compat: Optional[bool] = None
        overlap_pct: Optional[float] = None

        if meta_b.is_geospatial and meta_a.is_geospatial:
            has_overlap, overlap_pct, _ = ImageRegistrar.check_geospatial_overlap(meta_b, meta_a)
            crs_compat = bool(meta_b.crs == meta_a.crs or (HAS_RASTERIO and meta_b.crs and meta_a.crs))

            # Check resolution compatibility (ratio within 50x)
            if meta_b.resolution and meta_a.resolution:
                r_b = max(meta_b.resolution)
                r_a = max(meta_a.resolution)
                ratio = max(r_b, r_a) / max(1e-6, min(r_b, r_a))
                res_compat = (ratio <= 50.0)
            else:
                res_compat = True

            min_required_overlap = settings.TEMPORAL_MIN_GEOSPATIAL_OVERLAP_PCT
            if not has_overlap or overlap_pct < min_required_overlap:
                reason = (
                    f"The two images represent geographically disjoint scenes "
                    f"(overlap {overlap_pct:.1f}% is below required {min_required_overlap}% threshold) "
                    f"and cannot be reliably compared as a temporal pair."
                )
                logger.info(f"Pair validation: Disjoint geospatial footprints ({overlap_pct}%). Classifying as INCOMPATIBLE_IMAGES.")
                return PairValidationDiagnostics(
                    classification=PairClassification.INCOMPATIBLE_IMAGES,
                    similarity_score=0.0,
                    spatially_compatible=False,
                    exact_hash_match=False,
                    spatial_overlap_pct=overlap_pct,
                    crs_compatible=crs_compat,
                    resolution_compatible=res_compat,
                    structural_similarity=0.0,
                    feature_inliers=0,
                    reason=reason
                )

        # ------------------------------------------------------------------
        # 3. Exact Pixel Array Equality Match (Spatially Compatible)
        # ------------------------------------------------------------------
        bands_b = RasterReader.read_all_bands(before_path)
        bands_a = RasterReader.read_all_bands(after_path)

        if bands_b.shape == bands_a.shape and np.array_equal(bands_b, bands_a):
            logger.info("Pair validation: Exact pixel array equality match. Classifying as IDENTICAL_IMAGES.")
            return PairValidationDiagnostics(
                classification=PairClassification.IDENTICAL_IMAGES,
                similarity_score=1.0,
                spatially_compatible=True,
                exact_hash_match=False,
                spatial_overlap_pct=overlap_pct if overlap_pct is not None else 100.0,
                crs_compatible=crs_compat if crs_compat is not None else True,
                resolution_compatible=res_compat if res_compat is not None else True,
                structural_similarity=1.0,
                feature_inliers=None,
                reason="IDENTICAL_IMAGES"
            )

        # ------------------------------------------------------------------
        # 4. Normalized Grayscale Conversion & Near-Duplicate Check
        # ------------------------------------------------------------------
        target_w, target_h = meta_b.width, meta_b.height
        gray_b = ImageRegistrar._to_gray_uint8(bands_b)

        # Resample after raster to match before canvas for similarity comparison
        if bands_a.shape[1:] != (target_h, target_w):
            b_count_a = bands_a.shape[0]
            resampled_a = np.zeros((b_count_a, target_h, target_w), dtype=bands_a.dtype)
            for i in range(b_count_a):
                resampled_a[i] = cv2.resize(
                    bands_a[i].astype(np.float32),
                    (target_w, target_h),
                    interpolation=cv2.INTER_AREA if (meta_a.width > target_w) else cv2.INTER_LINEAR
                ).astype(bands_a.dtype)
            gray_a = ImageRegistrar._to_gray_uint8(resampled_a)
        else:
            gray_a = ImageRegistrar._to_gray_uint8(bands_a)

        ssim_val = cls.compute_ssim(gray_b, gray_a)
        mean_abs_diff = float(np.mean(np.abs(gray_b.astype(np.float32) - gray_a.astype(np.float32)))) / 255.0
        max_abs_diff = float(np.max(np.abs(gray_b.astype(np.float32) - gray_a.astype(np.float32)))) / 255.0

        # Check for near-identical scene (tiny compression / quantization / noise)
        if (
            ssim_val >= settings.TEMPORAL_IDENTICAL_IMAGE_THRESHOLD and
            mean_abs_diff <= 0.002
        ):
            logger.info(f"Pair validation: Near-perfect SSIM ({ssim_val:.4f}) and tiny delta ({mean_abs_diff:.5f}). Classifying as IDENTICAL_IMAGES.")
            return PairValidationDiagnostics(
                classification=PairClassification.IDENTICAL_IMAGES,
                similarity_score=round(ssim_val, 4),
                spatially_compatible=True,
                exact_hash_match=False,
                spatial_overlap_pct=overlap_pct or 100.0,
                crs_compatible=crs_compat if crs_compat is not None else True,
                resolution_compatible=res_compat if res_compat is not None else True,
                structural_similarity=round(ssim_val, 4),
                feature_inliers=None,
                reason="IDENTICAL_IMAGES"
            )

        if (
            ssim_val >= settings.TEMPORAL_NEAR_DUPLICATE_THRESHOLD and
            mean_abs_diff <= settings.TEMPORAL_NEAR_DUPLICATE_DIFF_THRESHOLD
        ):
            logger.info(f"Pair validation: Near-duplicate detected (SSIM={ssim_val:.4f}, mean_delta={mean_abs_diff:.4f}). Classifying as NEAR_DUPLICATE.")
            return PairValidationDiagnostics(
                classification=PairClassification.NEAR_DUPLICATE,
                similarity_score=round(ssim_val, 4),
                spatially_compatible=True,
                exact_hash_match=False,
                spatial_overlap_pct=overlap_pct or 100.0,
                crs_compatible=crs_compat if crs_compat is not None else True,
                resolution_compatible=res_compat if res_compat is not None else True,
                structural_similarity=round(ssim_val, 4),
                feature_inliers=None,
                reason="NEAR_DUPLICATE"
            )

        # ------------------------------------------------------------------
        # 5. Visual Compatibility Check (For Non-Geospatial or Unregistered Pairs)
        # ------------------------------------------------------------------
        good_matches, inlier_count, inlier_ratio = cls.check_visual_feature_consensus(gray_b, gray_a)

        # If rasters are not georeferenced, we rely on visual feature consensus & SSIM
        is_both_geospatial = bool(meta_b.is_geospatial and meta_a.is_geospatial)

        if not is_both_geospatial:
            # When both images are non-geospatial:
            # If feature matching reveals 0 or near-0 inliers AND SSIM is very low AND no significant background is shared, they are unrelated scenes
            has_feature_consensus = (inlier_count >= settings.TEMPORAL_MIN_FEATURE_INLIERS)
            has_structural_consensus = (ssim_val >= settings.TEMPORAL_PAIR_SIMILARITY_THRESHOLD)

            # Check fraction of scene that has identical or low pixel difference (stable background)
            pixel_diff = np.abs(gray_b.astype(np.float32) - gray_a.astype(np.float32))
            frac_unchanged = float(np.mean(pixel_diff <= 25.0))
            has_background_consistency = (frac_unchanged >= 0.35)

            if not has_feature_consensus and not has_structural_consensus and not has_background_consistency:
                reason = "The uploaded images appear to represent different scenes and cannot be reliably compared as a temporal pair."
                logger.info(
                    f"Pair validation: Incompatible non-geospatial scenes "
                    f"(SSIM={ssim_val:.4f} < {settings.TEMPORAL_PAIR_SIMILARITY_THRESHOLD}, Inliers={inlier_count} < {settings.TEMPORAL_MIN_FEATURE_INLIERS}, "
                    f"FracUnchanged={frac_unchanged:.2f} < 0.35). "
                    f"Classifying as INCOMPATIBLE_IMAGES."
                )
                return PairValidationDiagnostics(
                    classification=PairClassification.INCOMPATIBLE_IMAGES,
                    similarity_score=round(max(0.0, ssim_val), 4),
                    spatially_compatible=False,
                    exact_hash_match=False,
                    spatial_overlap_pct=None,
                    crs_compatible=None,
                    resolution_compatible=None,
                    structural_similarity=round(ssim_val, 4),
                    feature_inliers=inlier_count,
                    reason=reason
                )

        # ------------------------------------------------------------------
        # 6. Compatible Temporal Pair
        # ------------------------------------------------------------------
        logger.info(
            f"Pair validation: COMPATIBLE_TEMPORAL_PAIR confirmed "
            f"(SSIM={ssim_val:.4f}, Inliers={inlier_count}, Overlap={overlap_pct}%)."
        )
        return PairValidationDiagnostics(
            classification=PairClassification.COMPATIBLE_TEMPORAL_PAIR,
            similarity_score=round(max(0.0, min(1.0, ssim_val)), 4),
            spatially_compatible=True,
            exact_hash_match=False,
            spatial_overlap_pct=overlap_pct,
            crs_compatible=crs_compat if crs_compat is not None else True,
            resolution_compatible=res_compat if res_compat is not None else True,
            structural_similarity=round(ssim_val, 4),
            feature_inliers=inlier_count,
            reason="Compatible temporal pair ready for change detection."
        )
