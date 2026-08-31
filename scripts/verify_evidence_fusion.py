import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import rasterio
from rasterio.transform import from_origin

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings
from backend.models.grounding import GroundingEngine, OWLv2ModelManager, validate_and_clip_box, apply_nms, compute_iou
from backend.models.fusion import RemoteSensingEvidenceFusionEngine
from backend.models.base import load_raster_as_pil_image
from backend.api.schemas import BoundingBox, EvidenceRegion


def create_scene_geotiff(path: Path, scene_type: int) -> Dict[str, BoundingBox]:
    """Creates synthetic 4-band satellite GeoTIFF scenes with known spatial ground truth zones."""
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = 512, 512
    data = np.zeros((4, height, width), dtype=np.uint16)

    # Background Soil
    data[0, :, :] = 1200; data[1, :, :] = 1500; data[2, :, :] = 900; data[3, :, :] = 2200

    gt_boxes = {}

    if scene_type == 0:
        # Original Sample Scene
        data[0, 50:200, 50:200] = 3500; data[1, 50:200, 50:200] = 3600; data[2, 50:200, 50:200] = 3800; data[3, 50:200, 50:200] = 1800
        gt_boxes["building"] = BoundingBox(xmin=50, ymin=50, xmax=200, ymax=200)

        data[0, 320:480, 50:450] = 400; data[1, 320:480, 50:450] = 1800; data[2, 320:480, 50:450] = 2500; data[3, 320:480, 50:450] = 100
        gt_boxes["water"] = BoundingBox(xmin=50, ymin=320, xmax=450, ymax=480)

        data[0, 50:250, 300:480] = 500; data[1, 50:250, 300:480] = 1200; data[2, 50:250, 300:480] = 400; data[3, 50:250, 300:480] = 4800
        gt_boxes["vegetation"] = BoundingBox(xmin=300, ymin=50, xmax=480, ymax=250)

        data[0, 250:270, 20:490] = 3200; data[1, 250:270, 20:490] = 3200; data[2, 250:270, 20:490] = 3300; data[3, 250:270, 20:490] = 1500
        gt_boxes["road"] = BoundingBox(xmin=20, ymin=250, xmax=490, ymax=270)

    elif scene_type == 1:
        # Scene 1: TL=Buildings, TR=Vegetation, Bottom=Water, Middle=Road
        data[0, 50:200, 50:200] = 3500; data[1, 50:200, 50:200] = 3600; data[2, 50:200, 50:200] = 3800; data[3, 50:200, 50:200] = 1800
        gt_boxes["building"] = BoundingBox(xmin=50, ymin=50, xmax=200, ymax=200)

        data[0, 320:480, 50:450] = 400; data[1, 320:480, 50:450] = 1800; data[2, 320:480, 50:450] = 2500; data[3, 320:480, 50:450] = 100
        gt_boxes["water"] = BoundingBox(xmin=50, ymin=320, xmax=450, ymax=480)

        data[0, 50:250, 300:480] = 500; data[1, 50:250, 300:480] = 1200; data[2, 50:250, 300:480] = 400; data[3, 50:250, 300:480] = 4800
        gt_boxes["vegetation"] = BoundingBox(xmin=300, ymin=50, xmax=480, ymax=250)

        data[0, 250:270, 20:490] = 3200; data[1, 250:270, 20:490] = 3200; data[2, 250:270, 20:490] = 3300; data[3, 250:270, 20:490] = 1500
        gt_boxes["road"] = BoundingBox(xmin=20, ymin=250, xmax=490, ymax=270)

    elif scene_type == 2:
        # Scene 2: BR=Buildings, TL=Water, BL=Vegetation, Vertical=Road
        data[0, 300:480, 300:480] = 3500; data[1, 300:480, 300:480] = 3600; data[2, 300:480, 300:480] = 3800; data[3, 300:480, 300:480] = 1800
        gt_boxes["building"] = BoundingBox(xmin=300, ymin=300, xmax=480, ymax=480)

        data[0, 30:220, 30:220] = 400; data[1, 30:220, 30:220] = 1800; data[2, 30:220, 30:220] = 2500; data[3, 30:220, 30:220] = 100
        gt_boxes["water"] = BoundingBox(xmin=30, ymin=30, xmax=220, ymax=220)

        data[0, 300:480, 30:220] = 500; data[1, 300:480, 30:220] = 1200; data[2, 300:480, 30:220] = 400; data[3, 300:480, 30:220] = 4800
        gt_boxes["vegetation"] = BoundingBox(xmin=30, ymin=300, xmax=220, ymax=480)

        data[0, 20:490, 250:270] = 3200; data[1, 20:490, 250:270] = 3200; data[2, 20:490, 250:270] = 3300; data[3, 20:490, 250:270] = 1500
        gt_boxes["road"] = BoundingBox(xmin=250, ymin=20, xmax=270, ymax=490)

    else:
        # Scene 3: Center=Buildings, Right=Water, Left=Vegetation, Bottom=Road
        data[0, 180:320, 180:320] = 3500; data[1, 180:320, 180:320] = 3600; data[2, 180:320, 180:320] = 3800; data[3, 180:320, 180:320] = 1800
        gt_boxes["building"] = BoundingBox(xmin=180, ymin=180, xmax=320, ymax=320)

        data[0, 50:450, 350:490] = 400; data[1, 50:450, 350:490] = 1800; data[2, 50:450, 350:490] = 2500; data[3, 50:450, 350:490] = 100
        gt_boxes["water"] = BoundingBox(xmin=350, ymin=50, xmax=490, ymax=450)

        data[0, 50:450, 20:150] = 500; data[1, 50:450, 20:150] = 1200; data[2, 50:450, 20:150] = 400; data[3, 50:450, 20:150] = 4800
        gt_boxes["vegetation"] = BoundingBox(xmin=20, ymin=50, xmax=150, ymax=450)

        data[0, 470:490, 20:490] = 3200; data[1, 470:490, 20:490] = 3200; data[2, 470:490, 20:490] = 3300; data[3, 470:490, 20:490] = 1500
        gt_boxes["road"] = BoundingBox(xmin=20, ymin=470, xmax=490, ymax=490)

    transform = from_origin(500000, 4000000, 10, 10)
    with rasterio.open(path, "w", driver="GTiff", height=height, width=width, count=4, dtype=data.dtype, crs="EPSG:32633", transform=transform) as dst:
        for b in range(4):
            dst.write(data[b], b + 1)
            dst.set_band_description(b + 1, ["Red", "Green", "Blue", "NIR"][b])

    return gt_boxes


def run_benchmark():
    print("=" * 85)
    print(" SATQUERY AI — BENCHMARK EVALUATION: OWLV2 BASELINE VS EVIDENCE FUSION ENGINE")
    print("=" * 85)

    import torch
    OWLv2ModelManager.load_model("google/owlv2-base-patch16-ensemble", "cuda")

    scenes = [
        ("Original Sample Scene", Path(r"D:\Satquery Ai\data\uploads\sample_phase3_satellite.tif"), 0),
        ("Scene 1", Path(r"D:\Satquery Ai\data\uploads\test_scene_1.tif"), 1),
        ("Scene 2", Path(r"D:\Satquery Ai\data\uploads\test_scene_2.tif"), 2),
        ("Scene 3", Path(r"D:\Satquery Ai\data\uploads\test_scene_3.tif"), 3),
    ]

    queries = [
        ("building", ["building structure", "urban buildings", "built-up structure"]),
        ("water", ["water body", "lake", "river", "reservoir"]),
        ("vegetation", ["dense vegetation", "forest", "trees"]),
        ("road", ["highway", "road", "paved road"])
    ]

    stats_baseline = {"tp": 0, "fp": 0, "fn": 0, "total_raw": 0, "total_acc": 0}
    stats_fusion   = {"tp": 0, "fp": 0, "fn": 0, "total_raw": 0, "total_acc": 0}

    fusion_engine = GroundingEngine(use_mock=False)

    for scene_name, scene_path, s_type in scenes:
        gt_boxes = create_scene_geotiff(scene_path, s_type)
        pil_img = load_raster_as_pil_image(scene_path)
        img_w, img_h = pil_img.size

        print(f"\n" + "#" * 80)
        print(f"EVALUATING SATELLITE SCENE: '{scene_name}' ({scene_path.name})")
        print(f"Ground Truth Zones: {gt_boxes}")
        print("#" * 80)

        # Compute spectral masks for BLUE overlay rendering
        spectral_masks = RemoteSensingEvidenceFusionEngine.extract_spectral_masks(scene_path, img_w, img_h)

        for cat_name, prompts in queries:
            gt_box = gt_boxes.get(cat_name)
            print(f"\n--- Target Category: '{cat_name.upper()}' (GT Box: {gt_box}) ---")

            # -------------------------------------------------------------
            # A. OWLv2 BASELINE (Raw score thresholding without fusion)
            # -------------------------------------------------------------
            base_full = OWLv2ModelManager.run_detection(pil_img, prompts, score_threshold=0.003)
            base_candidates = []
            for b, p_m, sc in base_full:
                box = validate_and_clip_box(b, img_w, img_h)
                if box:
                    base_candidates.append(EvidenceRegion(box=box, label=cat_name, confidence=float(sc)))

            base_accepted = apply_nms(base_candidates, iou_threshold=0.35)
            stats_baseline["total_raw"] += len(base_candidates)
            stats_baseline["total_acc"] += len(base_accepted)

            if not base_accepted:
                stats_baseline["fn"] += 1
            else:
                best_iou = max([compute_iou(r.box, gt_box) for r in base_accepted]) if gt_box else 0.0
                if best_iou >= 0.30:
                    stats_baseline["tp"] += 1
                else:
                    stats_baseline["fp"] += 1

            # -------------------------------------------------------------
            # B. REMOTE-SENSING EVIDENCE FUSION ENGINE
            # -------------------------------------------------------------
            fusion_regions, _, model_info = fusion_engine.detect_evidence(
                file_path=scene_path,
                query=f"Where are the {cat_name}?",
                confidence_threshold=0.005
            )

            raw_cand_count = int(model_info.get("raw_candidates", 0))
            stats_fusion["total_raw"] += raw_cand_count
            stats_fusion["total_acc"] += len(fusion_regions)

            # Render Visual Evidence Image (GREEN=Accepted, RED=Rejected, BLUE=Mask Contour)
            vis_img = pil_img.copy().convert("RGBA")
            draw = ImageDraw.Draw(vis_img)

            # Render BLUE spectral mask contour overlay
            mask = spectral_masks.get(cat_name)
            if mask is not None and np.any(mask):
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    pts = [(int(p[0][0]), int(p[0][1])) for p in cnt]
                    if len(pts) >= 3:
                        draw.polygon(pts, outline=(30, 144, 255, 255), width=2) # BLUE

            if not fusion_regions:
                print("  [FUSION RESULT] NO_VALID_DETECTION")
                stats_fusion["fn"] += 1
            else:
                best_iou = 0.0
                for reg in fusion_regions:
                    iou = compute_iou(reg.box, gt_box) if gt_box else 0.0
                    if iou > best_iou:
                        best_iou = iou

                    print(f"  [FUSION ACCEPTED (GREEN)] Box: [{reg.box.xmin}, {reg.box.ymin}, {reg.box.xmax}, {reg.box.ymax}] | FinalScore: {reg.confidence:.4f} (Sem={reg.owlv2_score}, Spec={reg.spectral_score}, Shape={reg.shape_score}) | IoU: {iou:.4f} | Source: {reg.source_tile}")
                    draw.rectangle([reg.box.xmin, reg.box.ymin, reg.box.xmax, reg.box.ymax], outline=(46, 204, 113, 255), width=3)
                    draw.text((reg.box.xmin+3, reg.box.ymin+3), f"ACCEPTED: {cat_name} ({reg.confidence:.2f}) IoU={iou:.2f}", fill=(46, 204, 113, 255))

                if best_iou >= 0.30:
                    stats_fusion["tp"] += 1
                else:
                    stats_fusion["fp"] += 1

            # Save Visual Evidence Overlay
            out_vis_path = Path(f"D:/Satquery Ai/data/results/fusion_eval_{scene_name.replace(' ', '_').lower()}_{cat_name}.png")
            vis_img.convert("RGB").save(out_vis_path)

    # -------------------------------------------------------------
    # CALCULATE METRICS
    # -------------------------------------------------------------
    def get_metrics(s: Dict) -> Dict:
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        prec = tp / float(tp + fp + 1e-6)
        rec = tp / float(tp + fn + 1e-6)
        f1 = 2 * (prec * rec) / float(prec + rec + 1e-6)
        return {"precision": prec * 100, "recall": rec * 100, "f1": f1 * 100, "tp": tp, "fp": fp, "fn": fn}

    m_base = get_metrics(stats_baseline)
    m_fused = get_metrics(stats_fusion)

    print("\n" + "=" * 85)
    print("                      FINAL COMPARATIVE SCIENTIFIC BENCHMARK METRICS")
    print("=" * 85)
    print(f"{'METRIC':<25} | {'OWLv2 BASELINE':<25} | {'EVIDENCE FUSION ENGINE':<25}")
    print("-" * 85)
    print(f"{'True Positives (TP)':<25} | {m_base['tp']:<25} | {m_fused['tp']:<25}")
    print(f"{'False Positives (FP)':<25} | {m_base['fp']:<25} | {m_fused['fp']:<25}")
    print(f"{'False Negatives (FN)':<25} | {m_base['fn']:<25} | {m_fused['fn']:<25}")
    print(f"{'Precision (%)':<25} | {m_base['precision']:<25.2f} | {m_fused['precision']:<25.2f}")
    print(f"{'Recall (%)':<25} | {m_base['recall']:<25.2f} | {m_fused['recall']:<25.2f}")
    print(f"{'F1 Score (%)':<25} | {m_base['f1']:<25.2f} | {m_fused['f1']:<25.2f}")
    print("=" * 85)


if __name__ == "__main__":
    run_benchmark()
