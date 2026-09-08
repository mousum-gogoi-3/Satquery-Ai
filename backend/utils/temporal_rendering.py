import time
import uuid
import cv2
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from backend.config import settings
from backend.api.schemas import ChangeRegion, ChangeType, BoundingBox
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# Distinct RGB color palette per ChangeType
CHANGE_TYPE_COLORS: Dict[ChangeType, Tuple[int, int, int]] = {
    ChangeType.VEGETATION_LOSS: (231, 76, 60),           # Crimson Red
    ChangeType.VEGETATION_GAIN: (46, 204, 113),          # Emerald Green
    ChangeType.WATER_EXPANSION: (52, 152, 219),          # Marine Blue
    ChangeType.WATER_REDUCTION: (243, 156, 18),          # Sandy Amber
    ChangeType.NEW_BUILT_UP: (230, 126, 34),             # Vibrant Orange
    ChangeType.BUILT_UP_LOSS: (155, 89, 182),            # Purple / Slate
    ChangeType.ROAD_INFRASTRUCTURE_CHANGE: (26, 188, 156), # Cyan
    ChangeType.SURFACE_CHANGE: (241, 196, 15),           # Amber / Yellow
    ChangeType.GENERAL_LAND_COVER_CHANGE: (211, 84, 0),  # Rust
    ChangeType.BARE_SOIL_CHANGE: (160, 82, 45),          # Sienna / Brown
    ChangeType.UNKNOWN_CHANGE: (149, 165, 166),          # Slate Gray
}


def _get_color_for_change_type(ct: ChangeType) -> Tuple[int, int, int]:
    return CHANGE_TYPE_COLORS.get(ct, (241, 196, 15))


def render_temporal_evidence(
    bands_before: np.ndarray,
    bands_after: np.ndarray,
    magnitude_map: np.ndarray,
    regions: List[ChangeRegion],
    output_dir: Optional[Path] = None,
    filename_prefix: str = "temporal_evidence",
    debug_mode: bool = False
) -> Tuple[Path, Path]:
    """
    Renders a 4-panel visual evidence composite:
      [ BEFORE IMAGE ]       [ AFTER IMAGE ]
      [ CHANGE HEATMAP ]     [ GROUNDED CHANGE EVIDENCE ]
    Also saves a standalone change heatmap image.
    Guarantees non-destructive processing (source rasters are never modified).

    Returns:
        Tuple of (evidence_composite_path, heatmap_path).
    """
    output_dir = output_dir or settings.RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    h, w = magnitude_map.shape

    # 1. Generate 8-bit RGB representations for Before & After
    rgb_b = _to_rgb_uint8(bands_before, target_shape=(h, w))
    rgb_a = _to_rgb_uint8(bands_after, target_shape=(h, w))

    # 2. Render Heatmap using OpenCV Colormap JET / INFERNO
    mag_uint8 = (np.clip(magnitude_map, 0.0, 1.0) * 255.0).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(mag_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    heatmap_pil = Image.fromarray(heatmap_rgb)

    # Save standalone heatmap
    unique_suffix = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"
    heatmap_path = output_dir / f"heatmap_{filename_prefix}_{unique_suffix}.png"
    heatmap_pil.save(heatmap_path, format="PNG")

    # 3. Render Grounded Overlays onto After Image Canvas
    overlay = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    canvas = Image.fromarray(rgb_a).convert("RGBA")
    canvas_draw = ImageDraw.Draw(canvas)

    line_thickness = max(2, int(min(w, h) * 0.006))
    font_size = max(11, int(min(w, h) * 0.024))

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    for reg in regions:
        box = reg.bounding_box
        xmin = max(0.0, min(float(box.xmin), float(w)))
        ymin = max(0.0, min(float(box.ymin), float(h)))
        xmax = max(0.0, min(float(box.xmax), float(w)))
        ymax = max(0.0, min(float(box.ymax), float(h)))

        if xmax <= xmin or ymax <= ymin:
            continue

        rgb_color = _get_color_for_change_type(reg.change_type)
        fill_color = rgb_color + (50,)  # Semi-transparent fill

        overlay_draw.rectangle([xmin, ymin, xmax, ymax], fill=fill_color)
        canvas_draw.rectangle([xmin, ymin, xmax, ymax], outline=rgb_color + (255,), width=line_thickness)

        label_text = f"{reg.change_type.value.upper()} ({reg.confidence * 100:.0f}%)"
        text_bbox = canvas_draw.textbbox((xmin, ymin), label_text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        pill_xmin = xmin
        pill_ymin = max(0.0, ymin - text_h - 6)
        pill_xmax = min(w, xmin + text_w + 10)
        pill_ymax = min(h, pill_ymin + text_h + 6)

        canvas_draw.rectangle([pill_xmin, pill_ymin, pill_xmax, pill_ymax], fill=rgb_color + (255,))
        canvas_draw.text((pill_xmin + 5, pill_ymin + 2), label_text, fill=(255, 255, 255, 255), font=font)

    evidence_panel = Image.alpha_composite(canvas, overlay).convert("RGB")

    # 4. Build 4-Panel Grid
    # Add Header Banners to each panel
    panel_before = _add_panel_banner(Image.fromarray(rgb_b), "1. BEFORE IMAGE", (41, 128, 185))
    panel_after = _add_panel_banner(Image.fromarray(rgb_a), "2. AFTER IMAGE", (39, 174, 96))
    panel_heatmap = _add_panel_banner(heatmap_pil, "3. CHANGE MAGNITUDE HEATMAP", (142, 68, 173))
    panel_evidence = _add_panel_banner(evidence_panel, "4. GROUNDED CHANGE REGIONS", (230, 126, 34))

    p_w, p_h = panel_before.size
    grid_img = Image.new("RGB", (p_w * 2 + 16, p_h * 2 + 16), color=(240, 243, 244))

    # Paste panels
    grid_img.paste(panel_before, (4, 4))
    grid_img.paste(panel_after, (p_w + 12, 4))
    grid_img.paste(panel_heatmap, (4, p_h + 12))
    grid_img.paste(panel_evidence, (p_w + 12, p_h + 12))

    evidence_path = output_dir / f"{filename_prefix}_{unique_suffix}.png"
    grid_img.save(evidence_path, format="PNG")

    logger.info(f"Rendered 4-panel temporal visual evidence to '{evidence_path}' with {len(regions)} change region(s).")
    return evidence_path, heatmap_path


def _add_panel_banner(img: Image.Image, title: str, banner_color: Tuple[int, int, int]) -> Image.Image:
    """Adds a stylish header banner with title text above a panel image."""
    w, h = img.size
    banner_h = max(26, int(h * 0.08))

    composite = Image.new("RGB", (w, h + banner_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(composite)

    draw.rectangle([0, 0, w, banner_h], fill=banner_color)

    font_size = max(12, int(banner_h * 0.55))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    draw.text((10, (banner_h - font_size) // 2), title, fill=(255, 255, 255), font=font)
    composite.paste(img, (0, banner_h))
    return composite


def _to_rgb_uint8(bands: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Converts input multi-band array into an 8-bit RGB image."""
    h, w = target_shape
    b_count = bands.shape[0]

    if b_count >= 3:
        r = cv2.resize(bands[0].astype(np.float32), (w, h))
        g = cv2.resize(bands[1].astype(np.float32), (w, h))
        b = cv2.resize(bands[2].astype(np.float32), (w, h))
        rgb = np.stack([r, g, b], axis=-1)
    elif b_count == 2:
        gray = cv2.resize(bands[0].astype(np.float32), (w, h))
        rgb = np.stack([gray, gray, gray], axis=-1)
    else:
        gray = cv2.resize(bands[0].astype(np.float32), (w, h))
        rgb = np.stack([gray, gray, gray], axis=-1)

    p_min = float(np.percentile(rgb, 2.0))
    p_max = float(np.percentile(rgb, 98.0))

    if p_max > p_min:
        norm = np.clip((rgb - p_min) / (p_max - p_min) * 255.0, 0.0, 255.0).astype(np.uint8)
    else:
        norm = np.clip(rgb, 0.0, 255.0).astype(np.uint8)

    return norm
