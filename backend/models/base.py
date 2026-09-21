import abc
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
import numpy as np
from PIL import Image

from backend.config import settings
from backend.remote_sensing.raster import RasterReader
from backend.remote_sensing.bands import BandManager
from backend.remote_sensing.preprocessing import Preprocessor
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def load_raster_as_pil_image(
    file_path: Path,
    max_dim: int = 1536,
    return_scale: bool = False
) -> Union[Image.Image, Tuple[Image.Image, Tuple[int, int], Tuple[float, float]]]:
    """
    Utility function reusing Phase 1 RasterReader and Preprocessor logic
    to convert raw GeoTIFF / multi-spectral arrays into normalized 8-bit RGB PIL Images.
    Supports PNG, JPEG, GeoTIFF, TIFF, SAR, 4-band RGB-NIR, multi-spectral rasters.
    Ensures safe maximum image dimensions to prevent GPU/Memory OOM.
    If return_scale is True, returns (pil_img, (orig_w, orig_h), (scale_x, scale_y)).
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Raster file not found: '{file_path}'")

    meta = RasterReader.extract_metadata(file_path)
    all_bands = RasterReader.read_all_bands(file_path)

    orig_h = meta.height
    orig_w = meta.width

    if all_bands.ndim == 3 and all_bands.shape[0] >= 3:
        rgb_arr = BandManager.create_rgb_composite(all_bands, (1, 2, 3))
    elif all_bands.ndim == 3 and all_bands.shape[0] < 3:
        single_band = all_bands[0]
        rgb_arr = np.stack([single_band, single_band, single_band], axis=-1)
    else:
        rgb_arr = np.stack([all_bands, all_bands, all_bands], axis=-1)

    norm_rgb = Preprocessor.normalize_to_uint8(
        rgb_arr,
        nodata=meta.nodata,
        percentile_cut=True
    )

    # Safe scaling for high-resolution satellite imagery
    h, w = norm_rgb.shape[:2]
    scale_x = 1.0
    scale_y = 1.0

    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        norm_rgb = Preprocessor.resize_image(norm_rgb, (new_w, new_h))
        scale_x = float(orig_w) / float(new_w)
        scale_y = float(orig_h) / float(new_h)

    pil_img = Image.fromarray(norm_rgb).convert("RGB")

    if return_scale:
        return pil_img, (orig_w, orig_h), (scale_x, scale_y)
    return pil_img


class QwenModelManager:
    """
    Centralized, thread-safe lazy loader for the Qwen2-VL-2B-Instruct model container.
    Ensures the VLM is loaded only once on first inference request and reused thereafter.
    """

    _instance_lock = threading.Lock()
    _is_loaded = False
    _model = None
    _processor = None
    _device = None
    _model_name = None

    @classmethod
    def get_device(cls, device_setting: str) -> str:
        """Determines target device ('cuda' or 'cpu')."""
        if device_setting == "cuda":
            return "cuda"
        elif device_setting == "cpu":
            return "cpu"
        else:  # "auto"
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"

    @classmethod
    def load_model(cls, model_name: str, device_setting: str) -> bool:
        """
        Lazy-loads the Qwen2-VL model and processor onto target device.
        Returns True if successfully loaded, False otherwise.
        """
        if cls._is_loaded:
            return True

        with cls._instance_lock:
            if cls._is_loaded:
                return True

            target_device = cls.get_device(device_setting)
            logger.info(f"Lazy loading Qwen2-VL model '{model_name}' on device '{target_device}'...")

            try:
                import torch
                from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

                dtype = torch.bfloat16 if (target_device == "cuda" and torch.cuda.is_bf16_supported()) else (
                    torch.float16 if target_device == "cuda" else torch.float32
                )

                cls._processor = AutoProcessor.from_pretrained(model_name)
                cls._model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    device_map=target_device if target_device == "cuda" else None
                )
                if target_device == "cpu":
                    cls._model.to("cpu")

                cls._device = target_device
                cls._model_name = model_name
                cls._is_loaded = True
                logger.info(f"Successfully loaded Qwen2-VL model '{model_name}' on {target_device}.")
                return True
            except Exception as e:
                logger.error(f"Failed to load Qwen2-VL model '{model_name}': {e}")
                return False

    @classmethod
    def run_inference(
        cls,
        image: Image.Image,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 256
    ) -> str:
        """
        Executes Qwen2-VL inference for an input image and prompt.
        Handles CUDA OOM gracefully.
        """
        if not cls._is_loaded or cls._model is None or cls._processor is None:
            raise RuntimeError("Qwen2-VL model is not loaded. Call load_model() first.")

        import torch
        from qwen_vl_utils import process_vision_info

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        })

        try:
            text_input = cls._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = cls._processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            if cls._device == "cuda":
                inputs = inputs.to("cuda")

            with torch.inference_mode():
                generated_ids = cls._model.generate(**inputs, max_new_tokens=max_new_tokens)

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = cls._processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            return output_text.strip()
        except getattr(torch.cuda, "OutOfMemoryError", Exception) as oom:
            if "out of memory" in str(oom).lower():
                logger.error("CUDA Out of Memory during VLM inference. Flushing CUDA cache.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise RuntimeError("GPU memory limit exceeded during VLM inference.") from oom
            raise


class BaseVLMEngine(abc.ABC):
    """
    Abstract base class for Vision-Language Models (VLM) in SatQuery AI.
    Supports lazy model loading, hardware auto-detection (CPU/CUDA),
    and a lightweight mock/fallback mode for fast execution and testing.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        use_mock: Optional[bool] = None
    ):
        self.model_name = model_name or settings.VLM_MODEL_NAME
        self.device_setting = device or settings.VLM_DEVICE
        self.use_mock = settings.USE_MOCK_VLM if use_mock is None else use_mock

    @property
    def is_mock(self) -> bool:
        return self.use_mock

    def _ensure_model_loaded(self) -> None:
        """
        Lazy-loads the Vision-Language Model on first inference request.
        Falls back seamlessly to mock mode if loading fails or `use_mock=True`.
        """
        if self.use_mock:
            return

        success = QwenModelManager.load_model(self.model_name, self.device_setting)
        if not success:
            logger.warning(
                f"Could not load real VLM model '{self.model_name}'. "
                f"Falling back to lightweight mock engine."
            )
            self.use_mock = True

