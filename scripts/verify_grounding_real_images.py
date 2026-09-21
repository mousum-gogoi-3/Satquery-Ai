"""
SatQuery AI — Real-Image Grounding Verification Script
Tests spatial grounding and visual evidence rendering on real reference images from:
D:\\Satquery Ai\\Train model pictures
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from backend.config import settings
from backend.models.grounding import GroundingEngine
from backend.utils.rendering import render_visual_evidence
from backend.utils.logger import get_logger

logger = get_logger("SatQueryAI.RealGroundingValidation")


def run_real_image_grounding_tests():
    print("=" * 85)
    print("      SATQUERY AI — REAL-IMAGE GROUNDING VERIFICATION BENCHMARK")
    print("=" * 85)

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[Hardware] PyTorch: {torch.__version__} | Device: {device_name} (CUDA: {torch.cuda.is_available()})")

    train_dir = Path(r"D:\Satquery Ai\Train model pictures")
    image_files = sorted(list(train_dir.glob("*.png")))

    if not image_files:
        print(f"[ERROR] No PNG reference images found in '{train_dir}'")
        return

    print(f"[Dataset] Found {len(image_files)} reference images in: '{train_dir}'")
    for f in image_files:
        print(f"  • {f.name} ({f.stat().st_size / (1024*1024):.2f} MB)")

    engine = GroundingEngine(use_mock=False)

    test_queries = [
        "Find buildings",
        "Find houses",
        "Find roads",
        "Find water"
    ]

    results_summary = []

    for img_idx, img_path in enumerate(image_files, 1):
        print("\n" + "#" * 85)
        print(f"[{img_idx}/{len(image_files)}] VALIDATING IMAGE: '{img_path.name}'")
        print("#" * 85)

        for q_idx, query in enumerate(test_queries, 1):
            t0 = time.perf_counter()
            regions, dims, model_info = engine.detect_evidence(
                file_path=img_path,
                query=query,
                confidence_threshold=0.010,
                max_regions=30
            )
            elapsed = time.perf_counter() - t0

            # Render visual evidence image
            slug = query.lower().replace(" ", "_").replace("/", "_")
            out_vis = render_visual_evidence(
                source_image=img_path,
                regions=regions,
                output_dir=settings.RESULTS_DIR,
                filename_prefix=f"real_eval_img{img_idx}_{slug}"
            )

            status = model_info.get("status", "UNKNOWN")
            raw_count = model_info.get("raw_candidates", 0)
            accepted_count = len(regions)
            top_conf = max([r.confidence for r in regions]) if regions else 0.0

            print(f"  Query: '{query:<25}' | Status: {status:<18} | Dets: {accepted_count:<3} | TopConf: {top_conf:.3f} | Time: {elapsed:.2f}s")
            if regions:
                for r in regions[:3]:
                    print(f"    - [{r.label}] [{r.box.xmin:.1f}, {r.box.ymin:.1f}, {r.box.xmax:.1f}, {r.box.ymax:.1f}] conf={r.confidence:.3f}")

            results_summary.append({
                "image": img_path.name,
                "query": query,
                "status": status,
                "raw_count": raw_count,
                "accepted_count": accepted_count,
                "top_conf": top_conf,
                "elapsed": elapsed,
                "output_image": str(out_vis.name)
            })

    print("\n" + "=" * 85)
    print("                       REAL-IMAGE GROUNDING BENCHMARK REPORT")
    print("=" * 85)
    print(f"{'IMAGE':<30} | {'QUERY':<25} | {'STATUS':<10} | {'DETS':<5} | {'TOP CONF':<8} | {'TIME(s)':<7}")
    print("-" * 85)
    for r in results_summary:
        print(f"{r['image'][:30]:<30} | {r['query']:<25} | {r['status']:<10} | {r['accepted_count']:<5} | {r['top_conf']:<8.3f} | {r['elapsed']:<7.2f}")
    print("=" * 85)


if __name__ == "__main__":
    run_real_image_grounding_tests()
