from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
import numpy as np

from backend.remote_sensing.raster import RasterMetadata, RasterReader
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class BandType(str, Enum):
    RED = "Red"
    GREEN = "Green"
    BLUE = "Blue"
    NIR = "Near-Infrared (NIR)"
    SWIR1 = "Short-Wave Infrared 1 (SWIR1)"
    SWIR2 = "Short-Wave Infrared 2 (SWIR2)"
    SAR_VV = "SAR (VV)"
    SAR_VH = "SAR (VH)"
    SAR_HH = "SAR (HH)"
    SAR_HV = "SAR (HV)"
    PANCHROMATIC = "Panchromatic"
    ALPHA = "Alpha Channel"
    GRAYSCALE = "Grayscale"
    UNKNOWN = "Unknown Band"


class BandInfo(BaseModel):
    band_index: int  # 1-indexed
    name: str
    band_type: BandType
    wavelength_nm: Optional[float] = None
    description: Optional[str] = None


class BandManager:
    """
    Identifies raster band roles (RGB, NIR, SWIR, SAR VV/VH) and generates visual composites.
    """

    @staticmethod
    def infer_band_configuration(metadata: RasterMetadata) -> List[BandInfo]:
        band_count = metadata.band_count
        bands_info: List[BandInfo] = []

        if band_count == 1:
            bands_info.append(
                BandInfo(
                    band_index=1,
                    name="Band 1",
                    band_type=BandType.GRAYSCALE,
                    description="Single-band panchromatic, elevation, or intensity raster"
                )
            )
        elif band_count == 2:
            # Common for Sentinel-1 dual-pol SAR (VV, VH)
            bands_info.append(
                BandInfo(
                    band_index=1,
                    name="Band 1 (VV)",
                    band_type=BandType.SAR_VV,
                    description="SAR Vertical-Vertical polarization backscatter"
                )
            )
            bands_info.append(
                BandInfo(
                    band_index=2,
                    name="Band 2 (VH)",
                    band_type=BandType.SAR_VH,
                    description="SAR Vertical-Horizontal polarization backscatter"
                )
            )
        elif band_count == 3:
            # Standard RGB imagery
            bands_info.append(BandInfo(band_index=1, name="Red", band_type=BandType.RED, wavelength_nm=665.0))
            bands_info.append(BandInfo(band_index=2, name="Green", band_type=BandType.GREEN, wavelength_nm=560.0))
            bands_info.append(BandInfo(band_index=3, name="Blue", band_type=BandType.BLUE, wavelength_nm=490.0))
        elif band_count == 4:
            # Check if this is an RGBA visual image (PNG, JPEG, visual TIFF with Alpha)
            # or a genuine 4-band satellite product with NIR (PlanetScope, Sentinel-2 sub-pack)
            is_rgba = (
                metadata.datatype in ("RGBA", "rgba") or
                metadata.driver in ("PNG", "JPEG", "WEBP") or
                (not metadata.is_geospatial and metadata.driver in ("IMAGE", "TIFF", "GTiff") and metadata.band_count == 4)
            )

            if is_rgba:
                bands_info.append(BandInfo(band_index=1, name="Red", band_type=BandType.RED, wavelength_nm=665.0))
                bands_info.append(BandInfo(band_index=2, name="Green", band_type=BandType.GREEN, wavelength_nm=560.0))
                bands_info.append(BandInfo(band_index=3, name="Blue", band_type=BandType.BLUE, wavelength_nm=490.0))
                bands_info.append(BandInfo(band_index=4, name="Alpha", band_type=BandType.ALPHA, description="Transparency / Alpha Mask"))
            else:
                bands_info.append(BandInfo(band_index=1, name="Red", band_type=BandType.RED, wavelength_nm=665.0))
                bands_info.append(BandInfo(band_index=2, name="Green", band_type=BandType.GREEN, wavelength_nm=560.0))
                bands_info.append(BandInfo(band_index=3, name="Blue", band_type=BandType.BLUE, wavelength_nm=490.0))
                bands_info.append(BandInfo(band_index=4, name="NIR", band_type=BandType.NIR, wavelength_nm=842.0))
        else:
            # Generic multi-spectral band assignment
            for i in range(1, band_count + 1):
                bands_info.append(
                    BandInfo(
                        band_index=i,
                        name=f"Band {i}",
                        band_type=BandType.UNKNOWN,
                        description=f"Multi-spectral band {i}"
                    )
                )

        return bands_info

    @classmethod
    def has_physical_nir(cls, metadata: RasterMetadata) -> bool:
        """Determines if the raster contains an authentic physical Near-Infrared (NIR) band."""
        configs = cls.infer_band_configuration(metadata)
        return any(b.band_type == BandType.NIR for b in configs)

    @classmethod
    def get_imagery_type(cls, metadata: RasterMetadata) -> str:
        """
        Classifies imagery into standard categories:
        'RGB', 'RGBA', 'GRAYSCALE', 'SAR', or 'MULTISPECTRAL'.
        """
        b_count = metadata.band_count
        if b_count == 1:
            return "GRAYSCALE"
        elif b_count == 2:
            return "SAR"
        elif b_count == 3:
            return "RGB"
        elif b_count == 4:
            if cls.has_physical_nir(metadata):
                return "MULTISPECTRAL"
            return "RGBA"
        elif b_count >= 5:
            return "MULTISPECTRAL"
        return "RGB"

    @staticmethod
    def create_rgb_composite(
        all_bands_array: np.ndarray,
        rgb_indices: Tuple[int, int, int] = (1, 2, 3)
    ) -> np.ndarray:
        """
        Creates an 8-bit RGB image array of shape (height, width, 3) from input band array (bands, height, width).
        `rgb_indices` are 1-indexed band numbers.
        """
        b_count, h, w = all_bands_array.shape
        r_idx, g_idx, b_idx = [i - 1 for i in rgb_indices]

        # Clamp indices to available bands
        r_idx = min(r_idx, b_count - 1)
        g_idx = min(g_idx, b_count - 1)
        b_idx = min(b_idx, b_count - 1)

        r_band = all_bands_array[r_idx]
        g_band = all_bands_array[g_idx]
        b_band = all_bands_array[b_idx]

        rgb_stack = np.stack([r_band, g_band, b_band], axis=-1)
        return rgb_stack
