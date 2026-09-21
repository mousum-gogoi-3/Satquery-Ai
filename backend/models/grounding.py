import re
import threading
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Union
import numpy as np
from PIL import Image

from backend.config import settings
from backend.models.base import BaseVLMEngine, load_raster_as_pil_image
from backend.remote_sensing.raster import RasterReader
from backend.remote_sensing.bands import BandManager
from backend.api.schemas import BoundingBox, EvidenceRegion, ImageDimensions
from backend.models.fusion import RemoteSensingEvidenceFusionEngine, validate_and_clip_coords
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def validate_and_clip_box(
    box_coords: Tuple[float, float, float, float],
    img_width: int,
    img_height: int,
    min_size: float = 2.0
) -> Optional[BoundingBox]:
    """Validates bounding box coordinates and strictly clamps them to image bounds [0, width] x [0, height]."""
    return validate_and_clip_coords(box_coords[0], box_coords[1], box_coords[2], box_coords[3], img_width, img_height)


def compute_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes."""
    x1 = max(float(box1.xmin), float(box2.xmin))
    y1 = max(float(box1.ymin), float(box2.ymin))
    x2 = min(float(box1.xmax), float(box2.xmax))
    y2 = min(float(box1.ymax), float(box2.ymax))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (float(box1.xmax) - float(box1.xmin)) * (float(box1.ymax) - float(box1.ymin))
    area2 = (float(box2.xmax) - float(box2.xmin)) * (float(box2.ymax) - float(box2.ymin))
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def apply_nms(
    regions: List[EvidenceRegion],
    iou_threshold: float = 0.35
) -> List[EvidenceRegion]:
    """
    Applies per-label Non-Maximum Suppression (NMS) to deduplicate overlapping boxes.
    """
    if not regions:
        return []

    by_label: Dict[str, List[EvidenceRegion]] = {}
    for r in regions:
        by_label.setdefault(r.label.lower(), []).append(r)

    keep_regions: List[EvidenceRegion] = []

    for label, reg_list in by_label.items():
        # Sort descending by confidence score / final_score
        reg_list.sort(key=lambda r: r.final_score if r.final_score is not None else r.confidence, reverse=True)

        selected: List[EvidenceRegion] = []
        for current in reg_list:
            should_keep = True
            for sel in selected:
                if compute_iou(current.box, sel.box) > iou_threshold:
                    should_keep = False
                    break
            if should_keep:
                selected.append(current)

        keep_regions.extend(selected)

    keep_regions.sort(key=lambda r: r.final_score if r.final_score is not None else r.confidence, reverse=True)
    return keep_regions


def generate_tile_crops(
    image: Image.Image,
    tile_size: int = 256,
    overlap_pct: float = 0.25
) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
    """Splits an image canvas into overlapping sub-tiles."""
    img_w, img_h = image.size
    tile_size = max(64, min(tile_size, min(img_w, img_h)))
    overlap_pct = max(0.0, min(overlap_pct, 0.5))

    stride = max(32, int(tile_size * (1.0 - overlap_pct)))

    x_starts = list(range(0, max(1, img_w - tile_size + 1), stride))
    y_starts = list(range(0, max(1, img_h - tile_size + 1), stride))

    if x_starts[-1] + tile_size < img_w:
        x_starts.append(img_w - tile_size)
    if y_starts[-1] + tile_size < img_h:
        y_starts.append(img_h - tile_size)

    x_starts = sorted(list(set(x_starts)))
    y_starts = sorted(list(set(y_starts)))

    tiles = []
    for y0 in y_starts:
        for x0 in x_starts:
            x1 = min(img_w, x0 + tile_size)
            y1 = min(img_h, y0 + tile_size)
            crop = image.crop((x0, y0, x1, y1))
            tiles.append((crop, (x0, y0, x1, y1)))

    return tiles


class OWLv2ModelManager:
    """Thread-safe lazy loader for OWLv2 zero-shot object detection model."""
    _instance_lock = threading.Lock()
    _is_loaded = False
    _model = None
    _processor = None
    _device = None
    _model_name = None

    @classmethod
    def get_device(cls, device_setting: str) -> str:
        if device_setting == "cuda":
            return "cuda"
        elif device_setting == "cpu":
            return "cpu"
        else:
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"

    @classmethod
    def load_model(cls, model_name: Optional[str] = None, device_setting: Optional[str] = None) -> bool:
        model_name = model_name or settings.GROUNDING_MODEL_NAME
        device_setting = device_setting or settings.GROUNDING_DEVICE

        if cls._is_loaded:
            return True

        with cls._instance_lock:
            if cls._is_loaded:
                return True

            target_device = cls.get_device(device_setting)
            logger.info(f"Lazy loading OWLv2 grounding model '{model_name}' on device '{target_device}'...")

            try:
                import torch
                from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

                checkpoint_path = Path(model_name) if model_name else None
                base_hf_name = settings.GROUNDING_MODEL_NAME

                if checkpoint_path and checkpoint_path.exists() and checkpoint_path.suffix.lower() == ".pt":
                    logger.info(f"Loading custom checkpoint weights from '{checkpoint_path}'...")
                    cls._processor = AutoProcessor.from_pretrained(base_hf_name)
                    cls._model = AutoModelForZeroShotObjectDetection.from_pretrained(base_hf_name)
                    ckpt = torch.load(checkpoint_path, map_location=target_device)
                    if "model_state_dict" in ckpt:
                        cls._model.load_state_dict(ckpt["model_state_dict"])
                    else:
                        cls._model.load_state_dict(ckpt)
                else:
                    cls._processor = AutoProcessor.from_pretrained(model_name)
                    cls._model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
                
                if target_device == "cuda":
                    cls._model.to("cuda")
                else:
                    cls._model.to("cpu")

                cls._model.eval()
                cls._device = target_device
                cls._model_name = model_name
                cls._is_loaded = True
                logger.info(f"Successfully loaded OWLv2 grounding model '{model_name}' on {target_device}.")
                return True
            except Exception as e:
                logger.error(f"Failed to load OWLv2 model '{model_name}': {e}")
                return False

    @classmethod
    def run_detection(
        cls,
        image: Image.Image,
        text_queries: List[str],
        score_threshold: float = 0.003
    ) -> List[Tuple[Tuple[float, float, float, float], str, float]]:
        """Executes zero-shot object detection using OWLv2 on target text queries."""
        if not cls._is_loaded or cls._model is None or cls._processor is None:
            raise RuntimeError("OWLv2 grounding model is not loaded.")

        import torch

        formatted_queries = [f"a photo of {q}" for q in text_queries]
        inputs = cls._processor(text=[formatted_queries], images=image, return_tensors="pt")
        
        if cls._device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = cls._model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        if cls._device == "cuda":
            target_sizes = target_sizes.to("cuda")

        results = cls._processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=score_threshold,
            text_labels=[text_queries]
        )[0]

        detected = []
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy() if hasattr(results["labels"], "cpu") else results["labels"]

        for box, score, label_idx in zip(boxes, scores, labels):
            xmin, ymin, xmax, ymax = box.tolist()
            idx = int(label_idx)
            label_str = text_queries[idx] if 0 <= idx < len(text_queries) else "target object"
            detected.append(((xmin, ymin, xmax, ymax), label_str, float(score)))

        return detected


class GroundingEngine(BaseVLMEngine):
    """
    Spatial Grounding & Evidence Engine powered by Remote-Sensing Evidence Fusion Layer.
    Combines zero-shot OWLv2 semantic candidates with multispectral index masks (NDVI, NDWI, NDBI),
    spatial morphology connected components, shape linearity, and transparent candidate fusion scoring.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        use_mock: Optional[bool] = None,
        confidence_threshold: Optional[float] = None
    ):
        target_model = model_name or settings.GROUNDING_MODEL_NAME
        target_device = device or settings.GROUNDING_DEVICE
        target_use_mock = settings.USE_MOCK_GROUNDING if use_mock is None else use_mock
        super().__init__(model_name=target_model, device=target_device, use_mock=target_use_mock)
        self.confidence_threshold = confidence_threshold or settings.GROUNDING_CONF_THRESHOLD

    def _extract_target_queries(self, query: str) -> Dict[str, Any]:
        """Parses user query into target concept label and domain prompts for OWLv2."""
        q_lower = query.lower().strip()
        cleaned = re.sub(r"^(where\s+(is|are|can\s+i\s+find)|find|locate|show(\s+me)?|detect|identify|is\s+there(\s+any)?)\s+", "", q_lower)
        cleaned = cleaned.rstrip("?.").strip()

        # 1. Agriculture / Farmland / Crop Fields / Cropland / Cultivated Land
        if (
            re.search(r"\b(agricultur(e|al)|farmlands?|farms?|farming|croplands?|crops?|cultivated|cultivation|padd(y|ies)|orchards?|pastures?)\b", q_lower)
            or any(phrase in q_lower for phrase in ["agricultural land", "agricultural field", "agricultural fields", "crop field", "crop fields", "cultivated land", "crop land"])
        ):
            return {
                "category": "agriculture",
                "prompts": [
                    "agricultural field",
                    "crop field",
                    "farmland",
                    "cultivated land",
                    "cropland",
                    "agricultural land"
                ]
            }

        # 2. Buildings / Houses / Residential Structures
        elif (
            re.search(r"\b(buildings?|houses?|homes?|roofs?|residential|structures?|architecture|facilities|facility)\b", q_lower)
            or any(phrase in q_lower for phrase in ["residential structure", "residential structures", "residential building", "residential buildings"])
        ):
            return {
                "category": "building",
                "prompts": [
                    "building structure",
                    "houses",
                    "buildings",
                    "roofs",
                    "residential building"
                ]
            }

        # 3. Bare Land / Barren Land / Exposed Soil
        elif (
            re.search(r"\b(bare\s+(land|soil|ground)|barren(\s+land|\s+ground)?|exposed\s+soil|dirt\s+ground|unvegetated\s+ground)\b", q_lower)
            or any(phrase in q_lower for phrase in ["bare soil", "barren land", "exposed soil", "bare land", "bare ground"])
        ):
            return {
                "category": "bare_soil",
                "prompts": [
                    "bare land",
                    "barren land",
                    "exposed soil",
                    "bare ground"
                ]
            }

        # 4. Urban Area / City / Settlement / Built-up Area
        elif re.search(r"\b(urban(\s+areas?|\s+centers?|\s+zones?)?|cit(y|ies)|settlements?|built-?up(\s+areas?|\s+zones?)?|downtown|metropolitan)\b", q_lower):
            return {
                "category": "urban",
                "prompts": [
                    "urban area",
                    "city",
                    "settlement",
                    "built-up area",
                    "residential area"
                ]
            }

        # 5. Roads / Highways / Streets / Roadways
        elif re.search(r"\b(roads?|highways?|streets?|roadways?|freeways?|paved(\s+roads?)?|lanes?|intersections?|avenues?)\b", q_lower):
            return {
                "category": "road",
                "prompts": [
                    "road",
                    "highway",
                    "paved road",
                    "street"
                ]
            }

        # 6. Water Bodies / Rivers / Lakes / Ponds / Reservoirs / Oceans
        elif re.search(r"\b(waters?|rivers?|lakes?|ponds?|reservoirs?|seas?|oceans?|canals?|creeks?|streams?)\b", q_lower) or "water body" in q_lower or "water bodies" in q_lower:
            return {
                "category": "water",
                "prompts": [
                    "water body",
                    "river",
                    "lake",
                    "pond",
                    "reservoir"
                ]
            }

        # 7. Forest / Woodland / Trees / Dense Vegetation
        elif re.search(r"\b(forests?|woodlands?|trees?|dense\s+vegetation|vegetation|jungles?|canop(y|ies))\b", q_lower):
            return {
                "category": "vegetation",
                "prompts": [
                    "trees",
                    "dense vegetation",
                    "forest",
                    "woodland"
                ]
            }

        # 8. Domain Specific Objects
        elif re.search(r"\b(vehicles?|cars?|trucks?|automobiles?|buses?|parking)\b", q_lower):
            return {
                "category": "vehicle",
                "prompts": ["vehicle", "car", "truck", "automobile", "aerial view of cars"]
            }
        elif re.search(r"\b(ships?|boats?|vessels?|cargos?|marines?|ferr(y|ies))\b", q_lower):
            return {
                "category": "ship",
                "prompts": ["ship", "boat", "vessel", "cargo ship", "docked boat"]
            }
        elif re.search(r"\b(airports?|runways?|airfields?|hangars?|airplanes?|aerodromes?)\b", q_lower):
            return {
                "category": "airport",
                "prompts": ["airport", "runway", "airfield", "aircraft runway"]
            }
        elif re.search(r"\b(constructions?|excavations?|earthworks?)\b", q_lower) or "building site" in q_lower:
            return {
                "category": "construction area",
                "prompts": ["construction site", "construction area", "earthworks", "excavation site"]
            }
        else:
            target_str = cleaned if cleaned else "target object"
            return {
                "category": target_str,
                "prompts": [
                    target_str,
                    f"satellite view of {target_str}",
                    f"aerial {target_str}",
                    f"overhead {target_str}"
                ]
            }

    def detect_evidence(
        self,
        file_path: Path,
        query: str,
        confidence_threshold: Optional[float] = None,
        max_regions: Optional[int] = None,
        debug_mode: bool = False
    ) -> Tuple[List[EvidenceRegion], ImageDimensions, Dict[str, Any]]:
        """
        Detects spatial evidence regions for a query using Remote-Sensing Evidence Fusion.
        Returns Tuple of (evidence_regions, image_dimensions, model_info).
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Raster file not found at '{file_path}'")

        threshold = confidence_threshold if confidence_threshold is not None else self.confidence_threshold

        # 1. Load raster metadata and band array once (cached in memory for the entire query)
        meta = RasterReader.extract_metadata(file_path)
        all_bands = RasterReader.read_all_bands(file_path)
        orig_w = meta.width
        orig_h = meta.height
        img_dims = ImageDimensions(width=orig_w, height=orig_h)
        has_nir = BandManager.has_physical_nir(meta)

        # 2. Prepare normalized PIL image and capture scaling factor
        load_res = load_raster_as_pil_image(file_path, return_scale=True)
        pil_img, (p_w, p_h), (scale_x, scale_y) = load_res
        img_w, img_h = pil_img.size

        target_info = self._extract_target_queries(query)
        cat_label = target_info["category"]
        text_prompts = target_info["prompts"]

        self._ensure_model_loaded()

        if self.use_mock:
            regions = self._mock_detect_evidence(file_path, query, [cat_label], orig_w, orig_h, threshold)
            model_info = {"grounding_model": "MockGroundingEngine", "mode": "lightweight_mock", "status": "SUCCESS"}
            return regions, img_dims, model_info

        # Real Execution Mode using Evidence Fusion Engine
        raw_candidates = []
        try:
            # 3. Extract Spectral or Visual Color Masks (Matching original image dimensions)
            spectral_masks = RemoteSensingEvidenceFusionEngine.extract_spectral_masks(
                file_path=file_path,
                img_w=orig_w,
                img_h=orig_h,
                preloaded_bands=all_bands,
                preloaded_meta=meta
            )

            # 4. Extract Candidates from OWLv2 (Full Image + Tiled Crops)
            tile_size = settings.GROUNDING_TILE_SIZE
            overlap_pct = settings.GROUNDING_TILE_OVERLAP
            tiles = generate_tile_crops(pil_img, tile_size=tile_size, overlap_pct=overlap_pct)

            score_cutoff = max(0.035, min(0.055, threshold * 0.05 if threshold > 0.1 else threshold * 0.8))

            # Full image OWLv2 inference
            full_dets = OWLv2ModelManager.run_detection(pil_img, text_prompts, score_threshold=score_cutoff)
            for b, p_matched, conf in full_dets:
                # Map back to original image coordinates
                gx0 = b[0] * scale_x
                gy0 = b[1] * scale_y
                gx1 = b[2] * scale_x
                gy1 = b[3] * scale_y
                box = validate_and_clip_coords(gx0, gy0, gx1, gy1, orig_w, orig_h, min_size=8.0)
                if box:
                    coord_meta = {
                        "tile_origin": [0, 0],
                        "tile_size": [pil_img.size[0], pil_img.size[1]],
                        "raw_box": [round(v, 2) for v in b],
                        "processed_box": [round(v, 2) for v in b],
                        "tile_box": [round(v, 2) for v in b],
                        "global_box": [box.xmin, box.ymin, box.xmax, box.ymax],
                        "image_size": [orig_w, orig_h]
                    }
                    raw_candidates.append((box, float(conf), "full_image", p_matched, coord_meta))

            # Tiled OWLv2 inference
            for crop_img, (x0, y0, x1, y1) in tiles:
                tile_w = x1 - x0
                tile_h = y1 - y0
                tile_id = f"tile_x{x0}_y{y0}_w{tile_w}_h{tile_h}"
                t_dets = OWLv2ModelManager.run_detection(crop_img, text_prompts, score_threshold=score_cutoff)
                for b, p_matched, conf in t_dets:
                    lx0, ly0, lx1, ly1 = b
                    px0 = x0 + lx0
                    py0 = y0 + ly0
                    px1 = x0 + lx1
                    py1 = y0 + ly1
                    gx0 = px0 * scale_x
                    gy0 = py0 * scale_y
                    gx1 = px1 * scale_x
                    gy1 = py1 * scale_y
                    box = validate_and_clip_coords(gx0, gy0, gx1, gy1, orig_w, orig_h, min_size=8.0)
                    if box:
                        coord_meta = {
                            "tile_origin": [x0, y0],
                            "tile_size": [tile_w, tile_h],
                            "raw_box": [round(v, 2) for v in b],
                            "processed_box": [round(v, 2) for v in b],
                            "tile_box": [round(lx0, 2), round(ly0, 2), round(lx1, 2), round(ly1, 2)],
                            "global_box": [box.xmin, box.ymin, box.xmax, box.ymax],
                            "image_size": [orig_w, orig_h]
                        }
                        raw_candidates.append((box, float(conf), tile_id, p_matched, coord_meta))

            # 5. Add Candidates from Spectral Connected Components (only for authentic multispectral imagery)
            spec_candidates = RemoteSensingEvidenceFusionEngine.generate_spectral_segmentation_candidates(
                spectral_masks, cat_label, orig_w, orig_h, has_nir=has_nir
            )
            for s_box, s_conf, s_source in spec_candidates:
                coord_meta = {
                    "tile_origin": [0, 0],
                    "tile_size": [orig_w, orig_h],
                    "raw_box": [s_box.xmin, s_box.ymin, s_box.xmax, s_box.ymax],
                    "processed_box": [s_box.xmin, s_box.ymin, s_box.xmax, s_box.ymax],
                    "tile_box": [s_box.xmin, s_box.ymin, s_box.xmax, s_box.ymax],
                    "global_box": [s_box.xmin, s_box.ymin, s_box.xmax, s_box.ymax],
                    "image_size": [orig_w, orig_h]
                }
                raw_candidates.append((s_box, s_conf, s_source, cat_label, coord_meta))

            # 6. Evaluate Fusion Scores for All Candidates (In-memory, Zero Repeated Disk Reads)
            accepted_regions: List[EvidenceRegion] = []
            rejected_count = 0
            rejected_samples = []

            for box, raw_score, src_info, p_matched, coord_meta in raw_candidates:
                final_score, comp_scores, explanation = RemoteSensingEvidenceFusionEngine.evaluate_candidate_scores(
                    file_path=file_path,
                    box=box,
                    category=cat_label,
                    owlv2_raw_score=raw_score,
                    source_tile=src_info,
                    masks=spectral_masks,
                    img_w=orig_w,
                    img_h=orig_h,
                    preloaded_bands=all_bands,
                    preloaded_meta=meta
                )

                if final_score >= RemoteSensingEvidenceFusionEngine.FUSION_SCORE_THRESHOLD:
                    accepted_regions.append(EvidenceRegion(
                        box=box,
                        label=cat_label,
                        confidence=round(final_score, 4),
                        explanation=explanation,
                        owlv2_score=comp_scores["owlv2_score"],
                        spectral_score=comp_scores["spectral_score"],
                        spatial_score=comp_scores["spatial_score"],
                        shape_score=comp_scores["shape_score"],
                        final_score=comp_scores["final_score"],
                        source_tile=src_info,
                        coord_metadata=coord_meta
                    ))
                else:
                    rejected_count += 1
                    if len(rejected_samples) < 5:
                        rejected_samples.append((box, raw_score, src_info, explanation))

            # 7. Non-Maximum Suppression (NMS) Deduplication
            nms_threshold = settings.GROUNDING_NMS_IOU_THRESHOLD
            nms_regions = apply_nms(accepted_regions, iou_threshold=nms_threshold)

            # 8. Second-Stage Deduplication & Region Clustering
            max_limit = max_regions or settings.GROUNDING_MAX_EVIDENCE_REGIONS
            final_regions = RemoteSensingEvidenceFusionEngine.cluster_and_consolidate_regions(
                regions=nms_regions,
                category=cat_label,
                masks=spectral_masks,
                img_w=orig_w,
                img_h=orig_h,
                max_regions=max_limit
            )

            status_str = "SUCCESS" if final_regions else "NO_VALID_DETECTION"

            model_info = {
                "grounding_model": self.model_name,
                "engine": "RemoteSensingEvidenceFusionEngine",
                "mode": "multi_criteria_fusion_and_clustering",
                "category": cat_label,
                "prompts": text_prompts,
                "imagery_type": BandManager.get_imagery_type(meta),
                "has_nir": str(BandManager.has_physical_nir(meta)),
                "tile_size": str(tile_size),
                "tile_count": str(len(tiles)),
                "nms_threshold": str(nms_threshold),
                "max_evidence_regions": str(max_limit),
                "device": OWLv2ModelManager._device or "cpu",
                "status": status_str,
                "raw_candidates": str(len(raw_candidates)),
                "accepted_detections": str(len(final_regions)),
                "rejected_candidates": str(rejected_count)
            }

            if debug_mode:
                model_info["rejected_samples"] = [
                    {"box": [r[0].xmin, r[0].ymin, r[0].xmax, r[0].ymax], "score": r[1], "source": r[2], "reason": r[3]}
                    for r in rejected_samples
                ]

            return final_regions, img_dims, model_info

        except Exception as e:
            logger.error(f"Error during Evidence Fusion grounding: {e}")
            import traceback
            traceback.print_exc()
            model_info = {"grounding_model": self.model_name, "mode": "error", "status": "NO_VALID_DETECTION", "error": str(e)}
            return [], img_dims, model_info

    def _ensure_model_loaded(self) -> None:
        if self.use_mock:
            return
        success = OWLv2ModelManager.load_model(self.model_name, self.device_setting)
        if not success:
            logger.warning(f"Could not load OWLv2 model '{self.model_name}'.")
            self.use_mock = True

    def _mock_detect_evidence(
        self,
        file_path: Path,
        query: str,
        target_labels: List[str],
        img_w: int,
        img_h: int,
        threshold: float
    ) -> List[EvidenceRegion]:
        """Mock spatial evidence generator for unit test fixtures ONLY."""
        raw_regions = []
        for i, lbl in enumerate(target_labels):
            lbl_lower = lbl.lower()
            if "building" in lbl_lower or "structure" in lbl_lower or "urban" in lbl_lower:
                coords = (img_w * 0.10, img_h * 0.10, img_w * 0.39, img_h * 0.39)
            elif "water" in lbl_lower or "river" in lbl_lower or "lake" in lbl_lower:
                coords = (img_w * 0.10, img_h * 0.63, img_w * 0.88, img_h * 0.94)
            elif "vegetation" in lbl_lower or "forest" in lbl_lower or "tree" in lbl_lower:
                coords = (img_w * 0.58, img_h * 0.10, img_w * 0.94, img_h * 0.49)
            elif "road" in lbl_lower or "highway" in lbl_lower:
                coords = (img_w * 0.04, img_h * 0.48, img_w * 0.96, img_h * 0.53)
            else:
                coords = (img_w * 0.15, img_h * 0.15, img_w * 0.45, img_h * 0.45)

            box = validate_and_clip_coords(coords[0], coords[1], coords[2], coords[3], img_w, img_h)
            if box:
                raw_regions.append(EvidenceRegion(
                    box=box,
                    label=lbl,
                    confidence=0.85,
                    explanation=f"Mock visual evidence region for test fixture '{lbl}'",
                    owlv2_score=0.85,
                    spectral_score=0.85,
                    spatial_score=0.85,
                    shape_score=0.85,
                    final_score=0.85,
                    source_tile="mock_fixture"
                ))

        return apply_nms(raw_regions, iou_threshold=settings.GROUNDING_NMS_IOU_THRESHOLD)
