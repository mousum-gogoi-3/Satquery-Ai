"""
Comprehensive Validation Benchmark testing:
1. Original failing image verification (785x799 px)
2. Multiple different satellite images
3. Image with multiple buildings
4. Image with no obvious buildings
5. Different image sizes (not 785x799)
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings
from backend.models.grounding import GroundingEngine
from backend.utils.rendering import render_visual_evidence
from backend.remote_sensing.raster import RasterReader


def run_benchmark():
    print("=" * 85)
    print("        SATQUERY AI — COMPREHENSIVE GROUNDING VALIDATION BENCHMARK")
    print("=" * 85)

    engine = GroundingEngine(use_mock=False)

    test_cases = [
        {
            "name": "Case 1: Original Failing Image (785x799 px, Multiple Building Clusters)",
            "path": Path(r"data\uploads\1e4a3854-5536-4d91-973f-1bd8c052ea66_a.tif"),
            "query": "Where is the building?",
            "expected_has_building": True,
            "min_expected_regions": 1
        },
        {
            "name": "Case 2: High-Res Reference Scene (1647x905 px, Building Area & Construction)",
            "path": Path(r"Train model pictures\Screenshot 2026-09-21 095848.png"),
            "query": "Where is the building?",
            "expected_has_building": True,
            "min_expected_regions": 1
        },
        {
            "name": "Case 3: High Density Urban Apartments (1005x590 px, Multiple Buildings)",
            "path": Path(r"data\uploads\77838131-a220-4880-9a73-bf396d5bc6f8_c44de7a8-daff-4946-b771-5dd1f168b471.jpg"),
            "query": "Where is the building?",
            "expected_has_building": True,
            "min_expected_regions": 3
        },
        {
            "name": "Case 4: Large Coastal Urban Scene (1001x1001 px, not 785x799)",
            "path": Path(r"data\uploads\064a9e8d-3616-4934-bf84-9bafb505fe2e_sample.tif"),
            "query": "Where is the building?",
            "expected_has_building": True,
            "min_expected_regions": 1
        },
        {
            "name": "Case 5: Flooded River & Muddy Sediment (568x601 px, NO Obvious Buildings)",
            "path": Path(r"data\uploads\d1b9c13f-a43a-42b4-91ba-cd5d5ab9d851_2.jpg"),
            "query": "Where is the building?",
            "expected_has_building": False,
            "min_expected_regions": 0
        },
        {
            "name": "Case 6: Geographic State Map (1091x716 px, NO Obvious Buildings)",
            "path": Path(r"data\uploads\bface4a1-71cc-4de6-afdc-33b69ad9db4d_c87c2e68-ef96-462b-b98c-36bd6fb1acf9.jpg"),
            "query": "Where is the building?",
            "expected_has_building": False,
            "min_expected_regions": 0
        }
    ]

    all_passed = True
    summaries = []

    for c in test_cases:
        p = c["path"]
        print("\n" + "-" * 85)
        print(f"Executing: {c['name']}")
        print(f"Image Path: '{p}'")

        if not p.exists():
            print(f"[ERROR] Required test file does not exist: '{p}'")
            all_passed = False
            continue

        meta = RasterReader.extract_metadata(p)
        print(f"Dimensions: {meta.width}x{meta.height} | Bands: {meta.band_count} | Driver: {meta.driver}")

        t0 = time.perf_counter()
        regions, dims, info = engine.detect_evidence(
            file_path=p,
            query=c["query"],
            confidence_threshold=0.040,
            max_regions=8,
            debug_mode=True
        )
        elapsed = time.perf_counter() - t0

        slug = c["name"].split(":")[0].strip().replace(" ", "_").lower()
        out_vis = render_visual_evidence(
            source_image=p,
            regions=regions,
            output_dir=settings.RESULTS_DIR,
            filename_prefix=f"bench_{slug}"
        )

        status = info.get("status")
        reg_count = len(regions)
        top_conf = max([r.confidence for r in regions]) if regions else 0.0

        print(f"Status: {status} | Elapsed: {elapsed:.2f}s | Regions: {reg_count} | Top Conf: {top_conf:.3f}")
        for idx, r in enumerate(regions):
            b = r.box
            w = b.xmax - b.xmin
            h = b.ymax - b.ymin
            print(f"  [{idx+1}] box=[{b.xmin:.1f}, {b.ymin:.1f}, {b.xmax:.1f}, {b.ymax:.1f}] size={w:.1f}x{h:.1f} px | conf={r.confidence:.3f} | owlv2={r.owlv2_score} | merged={r.merged_detection_count}")

        # Verification rules
        passed = True
        err_msg = ""
        if c["expected_has_building"]:
            if reg_count < c["min_expected_regions"]:
                passed = False
                err_msg = f"Expected >= {c['min_expected_regions']} building regions, got {reg_count}"
            else:
                # Check that boxes are not tiny noise (<14px)
                for r in regions:
                    bw = r.box.xmax - r.box.xmin
                    bh = r.box.ymax - r.box.ymin
                    if bw < 14 or bh < 14:
                        passed = False
                        err_msg = f"Found degenerate noise box: {bw:.1f}x{bh:.1f} px"
                        break
        else:
            if reg_count != 0 or status != "NO_VALID_DETECTION":
                passed = False
                err_msg = f"Expected 0 regions & NO_VALID_DETECTION for image with no buildings, got {reg_count} regions (status: {status})"

        if passed:
            print(f"[PASS] {c['name']}")
        else:
            print(f"[FAIL] {c['name']} -> {err_msg}")
            all_passed = False

        summaries.append({
            "name": c["name"].split(":")[0],
            "desc": c["name"].split(":")[1].strip(),
            "dims": f"{meta.width}x{meta.height}",
            "status": status,
            "count": reg_count,
            "top_score": top_conf,
            "rendered": out_vis.name,
            "passed": passed,
            "err": err_msg
        })

    print("\n" + "=" * 95)
    print(f"{'CASE':<10} | {'DIMS':<10} | {'STATUS':<18} | {'COUNT':<6} | {'TOP CONF':<8} | {'RESULT':<8} | {'RENDERED IMAGE'}")
    print("-" * 95)
    for s in summaries:
        res = "PASS" if s["passed"] else "FAIL"
        print(f"{s['name']:<10} | {s['dims']:<10} | {s['status']:<18} | {s['count']:<6} | {s['top_score']:<8.3f} | {res:<8} | {s['rendered']}")
    print("=" * 95)

    if all_passed:
        print("\n>>> ALL BENCHMARK CASES PASSED PERFECTLY! <<<")
        return 0
    else:
        print("\n>>> ONE OR MORE BENCHMARK CASES FAILED! <<<")
        return 1


if __name__ == "__main__":
    sys.exit(run_benchmark())
