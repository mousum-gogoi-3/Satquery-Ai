"""
SatQuery AI — Grounding Coordinate & Pipeline Debug Visualization Tool
Generates a comprehensive diagnostic composite image illustrating:
  1. Original image
  2. Tile boundaries with labeled offsets
  3. Raw OWLv2 detections per tile
  4. Converted global candidate boxes
  5. Post-NMS candidate boxes
  6. Final fused, consolidated evidence boxes
"""

import sys
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings
from backend.models.base import load_raster_as_pil_image
from backend.models.grounding import GroundingEngine, OWLv2ModelManager, generate_tile_crops, apply_nms
from backend.models.fusion import RemoteSensingEvidenceFusionEngine, validate_and_clip_coords
from backend.remote_sensing.raster import RasterReader
from backend.api.schemas import EvidenceRegion


def run_debug_visualization(
    image_path: Path,
    query: str = "Where is the building?",
    output_dir: Path = settings.RESULTS_DIR
) -> Path:
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("      SATQUERY AI — GROUNDING COORDINATE DEBUG VISUALIZATION")
    print("=" * 80)
    print(f"Target Image: '{image_path}'")
    print(f"Target Query: '{query}'")

    meta = RasterReader.extract_metadata(image_path)
    all_bands = RasterReader.read_all_bands(image_path)
    orig_w, orig_h = meta.width, meta.height

    pil_img, (p_w, p_h), (scale_x, scale_y) = load_raster_as_pil_image(image_path, return_scale=True)
    img_w, img_h = pil_img.size
    print(f"[Raster] Dimensions: {orig_w}x{orig_h} | PIL Size: {img_w}x{img_h} | Scale: ({scale_x:.2f}, {scale_y:.2f})")

    # 1. Base Canvas for Drawing
    canvas = pil_img.copy().convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(canvas)
    draw_overlay = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("arial.ttf", 12)
        title_font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()
        title_font = font

    # 2. Draw Tile Boundaries (Dashed / Thin Cyan)
    tile_size = settings.GROUNDING_TILE_SIZE
    overlap_pct = settings.GROUNDING_TILE_OVERLAP
    tiles = generate_tile_crops(pil_img, tile_size=tile_size, overlap_pct=overlap_pct)
    print(f"[Tiling] Generated {len(tiles)} tiles (size={tile_size}, overlap={overlap_pct*100:.0f}%)")

    for crop_img, (x0, y0, x1, y1) in tiles:
        draw.rectangle([x0, y0, x1, y1], outline=(0, 200, 255, 120), width=1)
        draw.text((x0 + 4, y0 + 4), f"({x0},{y0})", fill=(0, 220, 255, 200), font=font)

    # 3. OWLv2 Grounding Candidates
    OWLv2ModelManager.load_model()
    engine = GroundingEngine(use_mock=False)
    target_info = engine._extract_target_queries(query)
    cat_label = target_info["category"]
    text_prompts = target_info["prompts"]

    spectral_masks = RemoteSensingEvidenceFusionEngine.extract_spectral_masks(
        file_path=image_path,
        img_w=orig_w,
        img_h=orig_h,
        preloaded_bands=all_bands,
        preloaded_meta=meta
    )

    raw_candidates = []
    score_cutoff = 0.040

    # Full image inference
    full_dets = OWLv2ModelManager.run_detection(pil_img, text_prompts, score_threshold=score_cutoff)
    for b, p_matched, conf in full_dets:
        box = validate_and_clip_coords(b[0] * scale_x, b[1] * scale_y, b[2] * scale_x, b[3] * scale_y, orig_w, orig_h, min_size=8.0)
        if box:
            raw_candidates.append((box, float(conf), "full_image", p_matched))

    # Tiled inference
    for crop_img, (x0, y0, x1, y1) in tiles:
        tile_w, tile_h = x1 - x0, y1 - y0
        tile_id = f"tile_x{x0}_y{y0}_w{tile_w}_h{tile_h}"
        t_dets = OWLv2ModelManager.run_detection(crop_img, text_prompts, score_threshold=score_cutoff)
        for b, p_matched, conf in t_dets:
            gx0 = (x0 + b[0]) * scale_x
            gy0 = (y0 + b[1]) * scale_y
            gx1 = (x0 + b[2]) * scale_x
            gy1 = (y0 + b[3]) * scale_y
            box = validate_and_clip_coords(gx0, gy0, gx1, gy1, orig_w, orig_h, min_size=8.0)
            if box:
                raw_candidates.append((box, float(conf), tile_id, p_matched))

    print(f"[Inference] Raw OWLv2 Candidates: {len(raw_candidates)}")

    # 4. Evaluate fusion scores
    accepted_regions = []
    for box, raw_score, src_info, p_matched in raw_candidates:
        final_score, comp_scores, explanation = RemoteSensingEvidenceFusionEngine.evaluate_candidate_scores(
            file_path=image_path,
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
                source_tile=src_info
            ))

    # 5. Apply NMS
    nms_regions = apply_nms(accepted_regions, iou_threshold=settings.GROUNDING_NMS_IOU_THRESHOLD)
    print(f"[Filtering] Accepted Fusion Regions: {len(accepted_regions)} | Post-NMS Regions: {len(nms_regions)}")

    # 6. Cluster & Consolidate
    final_regions = RemoteSensingEvidenceFusionEngine.cluster_and_consolidate_regions(
        regions=nms_regions,
        category=cat_label,
        masks=spectral_masks,
        img_w=orig_w,
        img_h=orig_h,
        max_regions=10
    )
    print(f"[Consolidation] Final Consolidated Evidence Regions: {len(final_regions)}")

    # Draw Raw OWLv2 boxes in faint orange
    for box, conf, src, _ in raw_candidates[:40]:
        draw.rectangle([box.xmin, box.ymin, box.xmax, box.ymax], outline=(255, 165, 0, 70), width=1)

    # Draw Post-NMS boxes in purple/blue
    for r in nms_regions[:15]:
        draw.rectangle([r.box.xmin, r.box.ymin, r.box.xmax, r.box.ymax], outline=(180, 80, 255, 160), width=2)

    # Draw Final Evidence boxes in SOLID GREEN with detailed labels
    for idx, r in enumerate(final_regions):
        b = r.box
        draw_overlay.rectangle([b.xmin, b.ymin, b.xmax, b.ymax], fill=(46, 204, 113, 50))
        draw.rectangle([b.xmin, b.ymin, b.xmax, b.ymax], outline=(46, 204, 113, 255), width=3)
        label_text = f"#{idx+1} {r.label.upper()} | OWLv2:{r.owlv2_score:.3f} | Fused:{r.confidence:.3f} | {r.source_tile}"
        tb = draw.textbbox((b.xmin, b.ymin), label_text, font=title_font)
        draw.rectangle([b.xmin, max(0, b.ymin - 20), b.xmin + (tb[2] - tb[0]) + 10, b.ymin], fill=(46, 204, 113, 255))
        draw.text((b.xmin + 5, max(0, b.ymin - 18)), label_text, fill=(255, 255, 255, 255), font=title_font)

    composed = Image.alpha_composite(canvas, overlay).convert("RGB")
    out_file = output_dir / f"debug_grounding_coords_{image_path.stem}.png"
    composed.save(out_file)
    print(f"\n[Artifact] Saved detailed coordinate debug visualization to: '{out_file}'")
    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SatQuery AI Grounding Coordinate Debug Visualization")
    parser.add_argument("--image", type=str, default=r"data\uploads\1e4a3854-5536-4d91-973f-1bd8c052ea66_a.tif")
    parser.add_argument("--query", type=str, default="Where is the building?")
    args = parser.parse_args()

    run_debug_visualization(Path(args.image), args.query)
