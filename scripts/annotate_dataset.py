"""
SatQuery AI — Dataset Annotation & Assisted Pseudo-Labeling Tool
Generates structured COCO-format annotations from reference images for human verification.
All machine-generated annotations are explicitly marked with `is_pseudo_label: true`.
"""

import sys
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import List, Dict, Any
from PIL import Image
import numpy as np

from backend.config import settings
from backend.models.grounding import OWLv2ModelManager, generate_tile_crops, validate_and_clip_box, apply_nms
from backend.models.base import load_raster_as_pil_image
from backend.api.schemas import BoundingBox, EvidenceRegion
from backend.utils.logger import get_logger

logger = get_logger("SatQueryAI.DatasetAnnotator")

CATEGORIES = [
    {"id": 1, "name": "building", "supercategory": "structure", "prompts": ["building", "house", "roof", "structure"]},
    {"id": 2, "name": "road", "supercategory": "infrastructure", "prompts": ["road", "highway", "paved road", "street"]},
    {"id": 3, "name": "water", "supercategory": "natural", "prompts": ["water body", "river", "lake", "pond"]},
    {"id": 4, "name": "vegetation", "supercategory": "natural", "prompts": ["trees", "forest", "dense vegetation"]},
    {"id": 5, "name": "vehicle", "supercategory": "transportation", "prompts": ["vehicle", "car", "truck", "automobile"]},
    {"id": 6, "name": "ship", "supercategory": "transportation", "prompts": ["ship", "boat", "vessel", "cargo ship"]},
    {"id": 7, "name": "airport", "supercategory": "infrastructure", "prompts": ["airport", "runway", "airfield"]},
    {"id": 8, "name": "agricultural field", "supercategory": "landuse", "prompts": ["agricultural field", "crop field", "farmland"]},
    {"id": 9, "name": "construction area", "supercategory": "landuse", "prompts": ["construction site", "construction area", "earthworks"]},
]


def annotate_dataset(
    image_dir: Path,
    output_json: Path,
    score_threshold: float = 0.04,
    device: str = "auto"
) -> Dict[str, Any]:
    """
    Scans image_dir, runs OWLv2 zero-shot detection for standard categories,
    and outputs a COCO-compliant JSON with explicit pseudo-label flags.
    """
    image_dir = Path(image_dir)
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    image_files = sorted([f for f in image_dir.iterdir() if f.suffix.lower() in image_extensions])

    if not image_files:
        logger.warning(f"No valid image files found in '{image_dir}'")
        return {}

    logger.info(f"Loading OWLv2 model on device '{device}'...")
    OWLv2ModelManager.load_model(device_setting=device)

    coco_data = {
        "info": {
            "description": "SatQuery AI Reference Dataset (Assisted Annotations)",
            "version": "1.0",
            "year": 2026,
            "date_created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "is_verified_ground_truth": False,
            "note": "All bounding boxes are candidate pseudo-labels. Must be verified/curated before supervised training."
        },
        "licenses": [{"id": 1, "name": "Proprietary / Internal Research", "url": ""}],
        "categories": [{"id": c["id"], "name": c["name"], "supercategory": c["supercategory"]} for c in CATEGORIES],
        "images": [],
        "annotations": []
    }

    annotation_id = 1

    for img_idx, img_path in enumerate(image_files, 1):
        logger.info(f"Processing image {img_idx}/{len(image_files)}: '{img_path.name}'...")
        pil_img = Image.open(img_path).convert("RGB")
        w, h = pil_img.size

        image_record = {
            "id": img_idx,
            "file_name": img_path.name,
            "width": w,
            "height": h,
            "source_path": str(img_path)
        }
        coco_data["images"].append(image_record)

        # Run detection category by category
        for cat in CATEGORIES:
            cat_id = cat["id"]
            cat_name = cat["name"]
            prompts = cat["prompts"]

            try:
                raw_dets = OWLv2ModelManager.run_detection(pil_img, prompts, score_threshold=score_threshold)
            except Exception as e:
                logger.error(f"Error detecting '{cat_name}' on {img_path.name}: {e}")
                continue

            candidates: List[EvidenceRegion] = []
            for b, p_matched, score in raw_dets:
                box = validate_and_clip_box(b, img_width=w, img_height=h)
                if box:
                    candidates.append(EvidenceRegion(box=box, label=cat_name, confidence=float(score)))

            # Deduplicate per category
            nms_dets = apply_nms(candidates, iou_threshold=0.40)

            for det in nms_dets:
                bx = det.box
                bw = round(bx.xmax - bx.xmin, 2)
                bh = round(bx.ymax - bx.ymin, 2)
                area = round(bw * bh, 2)

                # Skip tiny noise or full-canvas containers
                if area < 16.0 or (area / (w * h)) > 0.60:
                    continue

                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": img_idx,
                    "category_id": cat_id,
                    "category_name": cat_name,
                    "bbox": [bx.xmin, bx.ymin, bw, bh],
                    "area": area,
                    "confidence_score": round(det.confidence, 4),
                    "iscrowd": 0,
                    "is_pseudo_label": True,
                    "verification_status": "PENDING_HUMAN_REVIEW"
                })
                annotation_id += 1

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, indent=2)

    logger.info(f"Exported {len(coco_data['annotations'])} pseudo-labeled annotations across {len(coco_data['images'])} images to '{output_json}'.")
    return coco_data


if __name__ == "__main__":
    train_dir = Path(r"D:\Satquery Ai\Train model pictures")
    out_file = Path(r"D:\Satquery Ai\data\dataset\annotations\instances_pseudo_labeled.json")
    annotate_dataset(train_dir, out_file, score_threshold=0.04)
