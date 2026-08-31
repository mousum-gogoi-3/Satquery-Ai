import time
import uuid
import cv2
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from backend.config import settings
from backend.api.schemas import EvidenceRegion, BoundingBox
from backend.models.base import load_raster_as_pil_image
from backend.utils.logger import get_logger

logger = get_logger(__name__)


# Distinct color palette (RGB) per target category for GREEN final evidence
CATEGORY_COLORS = {
    "building": (46, 204, 113),       # Emerald Green
    "buildings": (46, 204, 113),
    "structure": (46, 204, 113),
    "urban": (46, 204, 113),
    "water": (46, 204, 113),
    "river": (46, 204, 113),
    "lake": (46, 204, 113),
    "vegetation": (46, 204, 113),
    "forest": (46, 204, 113),
    "road": (46, 204, 113),
    "highway": (46, 204, 113),
}

DEFAULT_GREEN = (46, 204, 113)        # Emerald Green for accepted final evidence
REJECTED_RED = (235, 64, 52)          # Red for rejected candidates in debug mode
MASK_BLUE = (30, 144, 255)            # Blue for spectral masks in debug mode


def _get_color_for_label(label: str) -> Tuple[int, int, int]:
    """Resolves RGB color for a given label string."""
    return DEFAULT_GREEN


def render_visual_evidence(
    source_image: Union[Path, Image.Image, np.ndarray],
    regions: List[EvidenceRegion],
    output_dir: Optional[Path] = None,
    filename_prefix: str = "evidence",
    show_debug_overlays: bool = False,
    rejected_candidates: Optional[List[Tuple[BoundingBox, float, str, str]]] = None,
    spectral_masks: Optional[Dict[str, np.ndarray]] = None
) -> Path:
    """
    Renders clean, minimal, non-overlapping visual evidence bounding boxes on a COPY of the source image.
    Normal mode: GREEN = final accepted evidence regions ONLY.
    Debug mode: GREEN = final accepted regions, RED = rejected candidates, BLUE = spectral mask contours.

    Returns:
        Path to the saved evidence image file.
    """
    output_dir = output_dir or settings.RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load source image as PIL Image in RGBA/RGB mode
    if isinstance(source_image, (str, Path)):
        source_path = Path(source_image)
        if not source_path.exists():
            raise FileNotFoundError(f"Source image not found for rendering: '{source_path}'")
        orig_pil = load_raster_as_pil_image(source_path)
    elif isinstance(source_image, np.ndarray):
        orig_pil = Image.fromarray(source_image).convert("RGB")
    elif isinstance(source_image, Image.Image):
        orig_pil = source_image.copy().convert("RGB")
    else:
        raise ValueError(f"Unsupported image type for rendering: {type(source_image)}")

    img_w, img_h = orig_pil.size

    overlay = Image.new("RGBA", (img_w, img_h), (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    canvas = orig_pil.convert("RGBA")
    canvas_draw = ImageDraw.Draw(canvas)

    line_thickness = max(2, int(min(img_w, img_h) * 0.005))
    font_size = max(12, int(min(img_w, img_h) * 0.022))

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # 2. Debug Overlays (RED rejected candidates & BLUE spectral mask contours)
    if show_debug_overlays:
        if spectral_masks:
            for cat_m, mask_arr in spectral_masks.items():
                if mask_arr is not None and np.any(mask_arr):
                    contours, _ = cv2.findContours(mask_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        pts = [(int(p[0][0]), int(p[0][1])) for p in cnt]
                        if len(pts) >= 3:
                            canvas_draw.polygon(pts, outline=MASK_BLUE + (255,), width=2)

        if rejected_candidates:
            for r_box, r_score, r_src, r_reason in rejected_candidates[:8]:
                rx0 = max(0.0, min(float(r_box.xmin), float(img_w)))
                ry0 = max(0.0, min(float(r_box.ymin), float(img_h)))
                rx1 = max(0.0, min(float(r_box.xmax), float(img_w)))
                ry1 = max(0.0, min(float(r_box.ymax), float(img_h)))
                if rx1 > rx0 and ry1 > ry0:
                    canvas_draw.rectangle([rx0, ry0, rx1, ry1], outline=REJECTED_RED + (255,), width=2)

    # 3. Draw clean, minimal GREEN final evidence regions
    for reg in regions:
        box = reg.box
        xmin = max(0.0, min(float(box.xmin), float(img_w)))
        ymin = max(0.0, min(float(box.ymin), float(img_h)))
        xmax = max(0.0, min(float(box.xmax), float(img_w)))
        ymax = max(0.0, min(float(box.ymax), float(img_h)))

        if xmax <= xmin or ymax <= ymin:
            continue

        rgb_color = DEFAULT_GREEN
        fill_color = rgb_color + (45,)  # 18% opacity fill

        overlay_draw.rectangle([xmin, ymin, xmax, ymax], fill=fill_color)
        canvas_draw.rectangle([xmin, ymin, xmax, ymax], outline=rgb_color + (255,), width=line_thickness)

        merged_str = f" [{reg.merged_detection_count} merged]" if reg.merged_detection_count and reg.merged_detection_count > 1 else ""
        label_text = f"{reg.label.upper()} ({reg.confidence * 100:.0f}%){merged_str}"

        text_bbox = canvas_draw.textbbox((xmin, ymin), label_text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        pill_xmin = xmin
        pill_ymin = max(0.0, ymin - text_h - 6)
        pill_xmax = min(img_w, xmin + text_w + 10)
        pill_ymax = min(img_h, pill_ymin + text_h + 6)

        canvas_draw.rectangle([pill_xmin, pill_ymin, pill_xmax, pill_ymax], fill=rgb_color + (255,))
        canvas_draw.text((pill_xmin + 5, pill_ymin + 2), label_text, fill=(255, 255, 255, 255), font=font)

    final_img = Image.alpha_composite(canvas, overlay).convert("RGB")

    unique_suffix = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"
    out_file = output_dir / f"{filename_prefix}_{unique_suffix}.png"
    final_img.save(out_file, format="PNG")

    logger.info(f"Rendered clean visual evidence image to '{out_file}' with {len(regions)} consolidated region(s).")
    return out_file
