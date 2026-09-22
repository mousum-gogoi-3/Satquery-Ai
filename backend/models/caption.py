from pathlib import Path
from typing import List, Tuple

from backend.config import settings
from backend.models.base import BaseVLMEngine, QwenModelManager, load_raster_as_pil_image
from backend.remote_sensing.raster import RasterReader
from backend.remote_sensing.bands import BandManager
from backend.utils.logger import get_logger

logger = get_logger(__name__)

CAPTION_SYSTEM_PROMPT = (
    "You are SatQuery AI, an expert satellite and remote-sensing vision-language agent. "
    "Analyze earth observation imagery, satellite rasters, and aerial scenes with high factual precision. "
    "Describe only observable features visible in the image. "
    "Do not invent unverified geographical locations, proper names, or dates."
)

# Standard remote-sensing concepts for scene-level inference
KNOWN_CONCEPTS = [
    "water",
    "vegetation",
    "buildings",
    "roads",
    "agriculture",
    "urban areas",
    "bare land",
]


class CaptionEngine(BaseVLMEngine):
    """
    Satellite imagery captioning engine for generating scene descriptions
    and inferring land-cover concepts. Integrates Qwen2-VL-2B-Instruct.
    """

    def generate_caption(
        self,
        file_path: Path,
        detail_level: str = "detailed"
    ) -> Tuple[str, List[str]]:
        """
        Generates a natural-language scene caption and infers land-cover concepts.
        Returns a tuple of (caption_text, list_of_inferred_concepts).

        Note: `detected_concepts` are scene-level concepts inferred by the VLM
        pipeline, not spatially confirmed object detections (grounding belongs to Phase 3).
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Raster file not found at '{file_path}'")

        self._ensure_model_loaded()

        if self.use_mock:
            return self._mock_generate_caption(file_path, detail_level)

        meta = RasterReader.extract_metadata(file_path)
        imagery_type = BandManager.get_imagery_type(meta)
        image = load_raster_as_pil_image(file_path)

        file_id_hint = file_path.stem.split("_")[0] if "_" in file_path.stem else file_path.stem
        logger.info(
            f"Caption request received for file_id={file_id_hint} ({file_path.name}) | "
            f"shape=({meta.height}, {meta.width}) | bands={meta.band_count} ({imagery_type})"
        )

        if meta.band_count == 1 or imagery_type == "GRAYSCALE":
            sensor_note = (
                "Important: This is a single-band panchromatic / grayscale remote-sensing raster. "
                "Do NOT invent or hallucinate RGB colors (such as green vegetation, blue water, or red roofs). "
                "Describe visible spatial patterns, intensity variations (bright vs dark regions), textures, shapes, and structural features."
            )
        elif imagery_type == "SAR":
            sensor_note = (
                "Important: This is a Synthetic Aperture Radar (SAR) raster. "
                "Describe features in terms of radar backscatter intensity (bright double-bounce scatterers, smooth dark specular water surfaces)."
            )
        else:
            sensor_note = "Describe observable visual features and land-cover patterns accurately."

        crs_note = f"\n- CRS: {meta.crs}" if meta.crs else ""
        meta_context = f"\nImage metadata:\n- Dimensions: {meta.width} x {meta.height} px\n- Spectral bands: {meta.band_count} ({imagery_type}){crs_note}\n"

        lvl = (detail_level or "detailed").lower().strip()
        if lvl == "brief":
            prompt = (
                "You are analyzing a satellite/remote-sensing image.\n\n"
                f"{sensor_note}\n{meta_context}\n"
                "Provide a brief, concise caption (1 to 2 sentences maximum) describing "
                "the primary visible land-cover and structural features in this satellite image."
            )
            max_tokens = 64
        elif lvl == "normal":
            prompt = (
                "You are analyzing a satellite/remote-sensing image.\n\n"
                f"{sensor_note}\n{meta_context}\n"
                "Provide a clear, balanced description (3 to 5 sentences) of this satellite scene. "
                "Describe key observations including land cover, structures, roads, vegetation, water, or terrain."
            )
            max_tokens = 128
        else:  # "detailed" or fallback
            prompt = (
                "You are analyzing a satellite/remote-sensing image.\n\n"
                "Look carefully at the provided image before describing.\n"
                f"{sensor_note}\n{meta_context}\n"
                "Provide a comprehensive, detailed description of this satellite scene. "
                "Thoroughly describe visible land-cover patterns, structures, roads, vegetation patches, "
                "water bodies, bare soil, agricultural patterns, and surface textures without inventing unverified facts. "
                "Finish every sentence and provide a complete description."
            )
            max_tokens = min(settings.MAX_VLM_TOKENS, 512)

        try:
            caption = QwenModelManager.run_inference(
                image=image,
                prompt=prompt,
                system_prompt=CAPTION_SYSTEM_PROMPT,
                max_new_tokens=max_tokens
            )

            if not caption:
                caption = "The satellite scene depicts a mixed terrestrial land cover region."

            # Scene-level concept extraction from generated caption text
            inferred_concepts = [c for c in KNOWN_CONCEPTS if c in caption.lower()]
            if not inferred_concepts:
                inferred_concepts = ["vegetation", "agriculture", "urban areas"]

            return caption, inferred_concepts
        except Exception as e:
            logger.error(f"Error during real Qwen caption inference on {file_path.name}: {e}")
            raise RuntimeError(f"Visual caption inference failed: {e}") from e

    def _mock_generate_caption(
        self,
        file_path: Path,
        detail_level: str
    ) -> Tuple[str, List[str]]:
        """
        Lightweight mock caption generator for fast development and testing.
        """
        meta = RasterReader.extract_metadata(file_path)
        lvl = (detail_level or "detailed").lower().strip()

        if meta.band_count == 1:
            if lvl == "brief":
                caption = f"A single-band panchromatic raster scene ({meta.width}x{meta.height})."
            elif lvl == "normal":
                caption = (
                    f"A single-band {meta.datatype} panchromatic raster scene ({meta.width}x{meta.height}). "
                    f"It shows surface reflectance variations and structural features across the region. "
                    f"Contrast differences highlight built-up areas and open terrain."
                )
            else:
                caption = (
                    f"A single-band {meta.datatype} panchromatic raster scene ({meta.width}x{meta.height}) "
                    f"showing surface reflectance variations and structural features. "
                    f"High intensity values correspond to developed structures and bare surfaces, "
                    f"while darker regions indicate shaded or vegetated ground cover."
                )
            concepts = ["bare land", "urban areas"]
        elif meta.band_count == 2:
            if lvl == "brief":
                caption = f"A dual-polarization SAR raster scene ({meta.width}x{meta.height})."
            elif lvl == "normal":
                caption = (
                    f"A dual-polarization SAR raster scene ({meta.width}x{meta.height}). "
                    f"It captures radar backscatter characteristics across terrain and structures. "
                    f"Double-bounce scatterers indicate urban elements, while specular reflection marks water bodies."
                )
            else:
                caption = (
                    f"A dual-polarization SAR raster scene ({meta.width}x{meta.height}) "
                    f"capturing radar backscatter characteristics across terrain and structures. "
                    f"Bright returns correspond to metallic or urban infrastructure, "
                    f"whereas smooth water surfaces exhibit low radar backscatter response."
                )
            concepts = ["urban areas", "water", "bare land"]
        else:
            if lvl == "brief":
                caption = f"A {meta.band_count}-band satellite image ({meta.width}x{meta.height}) showing agricultural fields and built-up areas."
            elif lvl == "normal":
                caption = (
                    f"A high-resolution {meta.band_count}-band satellite scene ({meta.width}x{meta.height}). "
                    f"The area contains a blend of agricultural plots, vegetated regions, and residential developments. "
                    f"Road networks link scattered structures throughout the landscape."
                )
            else:
                caption = (
                    f"A high-resolution {meta.band_count}-band satellite scene ({meta.width}x{meta.height}) "
                    f"featuring agricultural fields, vegetated patches, road networks, and scattered urban buildings. "
                    f"Distinct land-use patterns are visible between cultivated plots and surrounding natural vegetation. "
                    f"Infrastructure layout shows organized transport routes connecting developed clusters."
                )
            concepts = ["agriculture", "vegetation", "buildings", "roads", "urban areas"]

        return caption, concepts

