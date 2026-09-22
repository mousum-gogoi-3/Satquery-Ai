from pathlib import Path
from typing import Tuple

from backend.config import settings
from backend.models.base import BaseVLMEngine, QwenModelManager, load_raster_as_pil_image
from backend.remote_sensing.bands import BandManager
from backend.remote_sensing.raster import RasterReader
from backend.utils.logger import get_logger

logger = get_logger(__name__)

VQA_SYSTEM_PROMPT = (
    "You are SatQuery AI, an expert satellite and remote-sensing vision-language agent. "
    "Analyze earth observation imagery, satellite rasters, and aerial scenes with high factual precision. "
    "Answer questions strictly using observable features in the image. "
    "Do not invent unverified geographical locations, proper names, or dates. "
    "If the image is grayscale or single-band, do not invent RGB colors."
)


class VQAEngine(BaseVLMEngine):
    """
    Visual Question Answering (VQA) engine for satellite imagery.
    Integrates Qwen2-VL-2B-Instruct for real VLM inference and provides
    a lightweight mock fallback for test execution.
    """

    def answer_question(self, file_path: Path, question: str) -> Tuple[str, float]:
        """
        Answers a natural-language question about an ingested satellite scene.
        Returns a tuple of (answer_text, confidence_score).
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Raster file not found at '{file_path}'")

        self._ensure_model_loaded()

        if self.use_mock:
            return self._mock_answer_question(file_path, question)

        meta = RasterReader.extract_metadata(file_path)
        imagery_type = BandManager.get_imagery_type(meta)
        image = load_raster_as_pil_image(file_path)

        file_id_hint = file_path.stem.split("_")[0] if "_" in file_path.stem else file_path.stem
        logger.info(
            f"VQA request received\n"
            f"file_id = {file_id_hint}\n"
            f"question = {question}\n"
            f"resolved file = {file_path.name}\n"
            f"raster shape = ({meta.height}, {meta.width})\n"
            f"band count = {meta.band_count} ({imagery_type})\n"
            f"preprocessed image shape = {image.size} ({image.mode})\n"
            f"model input image = actual image\n"
            f"model = {self.model_name}"
        )

        # Anti-color hallucination & imagery specific notes
        if meta.band_count == 1 or imagery_type == "GRAYSCALE":
            sensor_note = (
                "For single-band imagery, describe brightness, texture, spatial patterns, and structures without inventing colors."
            )
        elif imagery_type == "SAR":
            sensor_note = (
                "For SAR imagery, describe radar backscatter intensity: bright double-bounce scatterers or rough terrain, and smooth dark specular water surfaces without inventing optical colors."
            )
        elif imagery_type == "MULTISPECTRAL":
            sensor_note = (
                f"This is a multispectral satellite scene ({meta.band_count} bands) displayed as a visual composite."
            )
        else:
            sensor_note = "This is an optical satellite / aerial image."

        crs_note = f"\n- CRS / Projection: {meta.crs}" if meta.crs else ""

        prompt = (
            f"Analyze the provided remote-sensing image carefully.\n\n"
            f"Answer the user's question using only visual evidence from the image.\n\n"
            f"For \"Explain this image\", provide a concise but complete description of the major visible features.\n\n"
            f"Cover relevant visible:\n"
            f"- land cover\n"
            f"- buildings or structures\n"
            f"- roads\n"
            f"- vegetation\n"
            f"- water\n"
            f"- bare surfaces\n"
            f"- spatial patterns\n\n"
            f"{sensor_note}\n\n"
            f"Image metadata:\n"
            f"- Dimensions: {meta.width} x {meta.height} px\n"
            f"- Spectral bands: {meta.band_count} band(s) ({imagery_type}){crs_note}\n\n"
            f"Do not repeat information unnecessarily.\n"
            f"Finish every sentence and provide a complete answer.\n\n"
            f"User question:\n"
            f"{question}"
        )

        try:
            answer = QwenModelManager.run_inference(
                image=image,
                prompt=prompt,
                system_prompt=VQA_SYSTEM_PROMPT,
                max_new_tokens=settings.MAX_VLM_TOKENS
            )

            if not answer:
                answer = "The satellite scene does not contain sufficient clear visual evidence to answer this question."

            return answer, 0.95
        except Exception as e:
            logger.error(f"Error during real Qwen VQA inference on {file_path.name}: {e}")
            raise RuntimeError(f"Visual inference failed for question '{question}': {e}") from e

    def _mock_answer_question(self, file_path: Path, question: str) -> Tuple[str, float]:
        """
        Lightweight mock VQA generator for testing and fast development.
        Utilizes image metadata and query keywords.
        """
        meta = RasterReader.extract_metadata(file_path)
        q_lower = question.lower().strip()

        if "land cover" in q_lower or "land use" in q_lower or "type of land" in q_lower:
            answer = "The scene contains a mix of agricultural land, vegetated patches, and built-up structures."
        elif "building" in q_lower or "urban" in q_lower or "structure" in q_lower or "road" in q_lower:
            answer = "Constructed urban features and buildings are visible across the central portion of the image."
        elif "water" in q_lower or "river" in q_lower or "lake" in q_lower or "sea" in q_lower:
            answer = "Water bodies are identified in the scene with distinct low-reflectance spectral characteristics."
        elif "resolution" in q_lower or "size" in q_lower or "dimension" in q_lower or "band" in q_lower:
            answer = f"The satellite scene has dimensions of {meta.width}x{meta.height} pixels across {meta.band_count} spectral band(s)."
        elif any(w in q_lower for w in ["describe", "explain", "what is", "what does", "summary", "overview", "analyze", "tell me", "can you see"]):
            answer = (
                f"The satellite image ({meta.width}x{meta.height} px, {meta.band_count} bands) depicts an active terrestrial landscape "
                f"with visible clusters of developed structures, connecting transport corridors, and interspersed vegetated and open terrain."
            )
        else:
            answer = f"Visual analysis of the satellite raster ({meta.width}x{meta.height}) indicates observable surface features corresponding to '{question}'."


        return answer, 0.90
