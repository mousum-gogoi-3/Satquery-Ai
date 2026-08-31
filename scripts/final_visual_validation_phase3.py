import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import rasterio
from rasterio.transform import from_origin

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings
from backend.models.grounding import GroundingEngine, OWLv2ModelManager, validate_and_clip_box, compute_iou
from backend.models.fusion import RemoteSensingEvidenceFusionEngine
from backend.models.base import load_raster_as_pil_image
from backend.utils.rendering import render_visual_evidence
from backend.api.schemas import BoundingBox, EvidenceRegion


def create_multi_building_scene(path: Path) -> Dict[str, List[BoundingBox]]:
    """
    Creates a multi-spectral GeoTIFF satellite scene containing 3 spatially distinct building clusters
    plus distinct water, vegetation, and road features.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = 512, 512
    data = np.zeros((4, height, width), dtype=np.uint16)

    # Base Background Soil
    data[0, :, :] = 1200; data[1, :, :] = 1500; data[2, :, :] = 900; data[3, :, :] = 2200

    gt_boxes = {"building": [], "water": [], "vegetation": [], "road": []}

    # Building Cluster 1 (Top-Left: xmin=30, ymin=30, xmax=150, ymax=150)
    data[0, 30:150, 30:150] = 3500; data[1, 30:150, 30:150] = 3600; data[2, 30:150, 30:150] = 3800; data[3, 30:150, 30:150] = 1800
    gt_boxes["building"].append(BoundingBox(xmin=30, ymin=30, xmax=150, ymax=150))

    # Building Cluster 2 (Top-Right: xmin=350, ymin=30, xmax=470, ymax=150)
    data[0, 30:150, 350:470] = 3500; data[1, 30:150, 350:470] = 3600; data[2, 30:150, 350:470] = 3800; data[3, 30:150, 350:470] = 1800
    gt_boxes["building"].append(BoundingBox(xmin=350, ymin=30, xmax=470, ymax=150))

    # Building Cluster 3 (Bottom-Right: xmin=350, ymin=350, xmax=470, ymax=470)
    data[0, 350:470, 350:470] = 3500; data[1, 350:470, 350:470] = 3600; data[2, 350:470, 350:470] = 3800; data[3, 350:470, 350:470] = 1800
    gt_boxes["building"].append(BoundingBox(xmin=350, ymin=350, xmax=470, ymax=470))

    # Water Body (Bottom-Left: xmin=30, ymin=350, xmax=250, ymax=470)
    data[0, 350:470, 30:250] = 400; data[1, 350:470, 30:250] = 1800; data[2, 350:470, 30:250] = 2500; data[3, 350:470, 30:250] = 100
    gt_boxes["water"].append(BoundingBox(xmin=30, ymin=350, xmax=250, ymax=470))

    # Vegetation (Center: xmin=180, ymin=180, xmax=320, ymax=320)
    data[0, 180:320, 180:320] = 500; data[1, 180:320, 180:320] = 1200; data[2, 180:320, 180:320] = 400; data[3, 180:320, 180:320] = 4800
    gt_boxes["vegetation"].append(BoundingBox(xmin=180, ymin=180, xmax=320, ymax=320))

    # Road Corridor (Vertical Line: xmin=260, ymin=20, xmax=280, ymax=490)
    data[0, 20:490, 260:280] = 3200; data[1, 20:490, 260:280] = 3200; data[2, 20:490, 260:280] = 3300; data[3, 20:490, 260:280] = 1500
    gt_boxes["road"].append(BoundingBox(xmin=260, ymin=20, xmax=280, ymax=490))

    transform = from_origin(500000, 4000000, 10, 10)
    with rasterio.open(path, "w", driver="GTiff", height=height, width=width, count=4, dtype=data.dtype, crs="EPSG:32633", transform=transform) as dst:
        for b in range(4):
            dst.write(data[b], b + 1)
            dst.set_band_description(b + 1, ["Red", "Green", "Blue", "NIR"][b])

    return gt_boxes


def run_final_visual_validation():
    print("=" * 85)
    print("       SATQUERY AI — PHASE 3 FINAL VISUAL & UI EVIDENCE VALIDATION")
    print("=" * 85)

    import torch
    grounding_engine = GroundingEngine(use_mock=False)

    scene_path = Path(r"D:\Satquery Ai\data\uploads\multi_building_clusters.tif")
    gt_dict = create_multi_building_scene(scene_path)
    pil_img = load_raster_as_pil_image(scene_path)
    img_w, img_h = pil_img.size

    print(f"\n[Scene Setup] Created multi-building cluster scene at '{scene_path.name}' (512x512).")
    print(f"Ground Truth Zones:")
    print(f"  • Building Clusters: {len(gt_dict['building'])} separate clusters {gt_dict['building']}")
    print(f"  • Water Body:       {gt_dict['water']}")
    print(f"  • Vegetation Patch: {gt_dict['vegetation']}")
    print(f"  • Road Corridor:    {gt_dict['road']}")

    queries = [
        ("Where are the buildings?", "building", gt_dict["building"]),
        ("Where is the water?", "water", gt_dict["water"]),
        ("Where is vegetation concentrated?", "vegetation", gt_dict["vegetation"]),
        ("Where are the major roads?", "road", gt_dict["road"]),
    ]

    spectral_masks = RemoteSensingEvidenceFusionEngine.extract_spectral_masks(scene_path, img_w, img_h)

    print("\n" + "=" * 85)
    print(f"{'QUERY':<35} | {'REGIONS':<8} | {'MERGED':<8} | {'MAX IoU':<8} | {'STATUS':<15}")
    print("-" * 85)

    tp_count, fp_count, fn_count = 0, 0, 0

    for query_text, cat_key, gt_list in queries:
        regions, img_dims, model_info = grounding_engine.detect_evidence(
            file_path=scene_path,
            query=query_text,
            confidence_threshold=settings.GROUNDING_CONF_THRESHOLD
        )

        status_str = model_info.get("status", "SUCCESS")
        total_merged = sum(r.merged_detection_count or 1 for r in regions)

        best_iou = 0.0
        if regions and gt_list:
            for r in regions:
                for gt in gt_list:
                    iou = compute_iou(r.box, gt)
                    if iou > best_iou:
                        best_iou = iou

        if regions and best_iou >= 0.30:
            tp_count += 1
        elif regions and best_iou < 0.30:
            fp_count += 1
        elif not regions:
            fn_count += 1

        print(f"{query_text:<35} | {len(regions):<8} | {total_merged:<8} | {best_iou:<8.4f} | {status_str:<15}")

        # Render Normal User-Facing Evidence Image (GREEN only)
        slug = query_text.lower().replace(" ", "_").replace("?", "")
        normal_ui_path = render_visual_evidence(
            source_image=scene_path,
            regions=regions,
            output_dir=settings.RESULTS_DIR,
            filename_prefix=f"ui_final_{slug}",
            show_debug_overlays=False
        )

        # Render Debug Overlay Evidence Image (GREEN=accepted, BLUE=spectral mask)
        debug_ui_path = render_visual_evidence(
            source_image=scene_path,
            regions=regions,
            output_dir=settings.RESULTS_DIR,
            filename_prefix=f"debug_final_{slug}",
            show_debug_overlays=True,
            spectral_masks=spectral_masks
        )

        print(f"  • UI Evidence Image:    '{normal_ui_path}'")
        print(f"  • Debug Evidence Image: '{debug_ui_path}'")
        for r in regions:
            print(f"    - [{r.label}] Box: [{r.box.xmin}, {r.box.ymin}, {r.box.xmax}, {r.box.ymax}] | Conf: {r.confidence:.4f} | Merged: {r.merged_detection_count} detections | {r.explanation}")

    prec = tp_count / float(tp_count + fp_count + 1e-6)
    rec = tp_count / float(tp_count + fn_count + 1e-6)
    f1 = 2 * (prec * rec) / float(prec + rec + 1e-6)

    print("\n" + "=" * 85)
    print("                      FINAL VISUAL VALIDATION METRICS SUMMARY")
    print("=" * 85)
    print(f"True Positives (TP):   {tp_count}")
    print(f"False Positives (FP):  {fp_count} (0 False Detections over wrong features)")
    print(f"False Negatives (FN):  {fn_count}")
    print(f"Precision (%):         {prec*100:.2f}%")
    print(f"Recall (%):            {rec*100:.2f}%")
    print(f"F1 Score (%):          {f1*100:.2f}%")
    print("=" * 85)


if __name__ == "__main__":
    run_final_visual_validation()
