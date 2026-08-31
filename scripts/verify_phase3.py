import os
import sys
import time
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings
from backend.models.vqa import VQAEngine
from backend.models.grounding import GroundingEngine
from backend.utils.rendering import render_visual_evidence
from backend.utils.logger import get_logger

logger = get_logger("SatQueryAI.VerifyPhase3")


def create_synthetic_satellite_geotiff(output_path: Path) -> Path:
    """
    Creates a realistic 4-band (Red, Green, Blue, NIR) satellite GeoTIFF scene (512x512)
    containing distinct spectral land-cover zones (urban/buildings, water body, dense vegetation, roads).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = 512, 512

    # Initialize 4 bands: Red, Green, Blue, NIR
    data = np.zeros((4, height, width), dtype=np.uint16)

    # Base background: Soil/sparse vegetation
    data[0, :, :] = 1200  # Red
    data[1, :, :] = 1500  # Green
    data[2, :, :] = 900   # Blue
    data[3, :, :] = 2200  # NIR

    # Zone 1: Urban / Buildings (Upper left quadrant) - High Visible, Low/Moderate NIR
    data[0, 50:200, 50:200] = 3500  # High Red
    data[1, 50:200, 50:200] = 3600  # High Green
    data[2, 50:200, 50:200] = 3800  # High Blue (Concrete/Roof reflection)
    data[3, 50:200, 50:200] = 1800  # Moderate NIR

    # Zone 2: Water Body (Bottom quadrant) - Low Red, High Green/Blue, Very Low NIR (High NDWI)
    data[0, 320:480, 50:450] = 400   # Low Red
    data[1, 320:480, 50:450] = 1800  # Green reflection
    data[2, 320:480, 50:450] = 2500  # Blue reflection
    data[3, 320:480, 50:450] = 100   # Very Low NIR (Absorption)

    # Zone 3: Dense Vegetation (Upper right quadrant) - Low Red, High NIR (High NDVI)
    data[0, 50:250, 300:480] = 500   # Absorbed Red
    data[1, 50:250, 300:480] = 1200  # Moderate Green
    data[2, 50:250, 300:480] = 400   # Absorbed Blue
    data[3, 50:250, 300:480] = 4800  # Peak NIR Reflection

    # Zone 4: Major Road (Linear feature across middle)
    data[0, 250:270, 20:490] = 3200
    data[1, 250:270, 20:490] = 3200
    data[2, 250:270, 20:490] = 3300
    data[3, 250:270, 20:490] = 1500

    transform = from_origin(500000, 4000000, 10, 10)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=4,
        dtype=data.dtype,
        crs="EPSG:32633",
        transform=transform,
    ) as dst:
        for b in range(4):
            dst.write(data[b], b + 1)
            dst.set_band_description(b + 1, ["Red", "Green", "Blue", "NIR"][b])

    logger.info(f"Created real multi-spectral 4-band satellite GeoTIFF at '{output_path}'.")
    return output_path


def run_phase3_verification():
    print("=" * 80)
    print("      SATQUERY AI — PHASE 3 SCIENTIFIC GROUNDING VERIFICATION")
    print("=" * 80)

    import torch

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    cuda_avail = torch.cuda.is_available()

    print(f"[Hardware] PyTorch Version: {torch.__version__}")
    print(f"[Hardware] CUDA Available:  {cuda_avail}")
    print(f"[Hardware] Target Device:   {device_name}")

    if cuda_avail:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    test_geotiff = settings.DATA_DIR / "uploads" / "sample_phase3_satellite.tif"
    create_synthetic_satellite_geotiff(test_geotiff)

    print("\n[Initialization] Loading Qwen2-VL & OWLv2 Grounding Models onto GPU...")
    vqa_engine = VQAEngine(use_mock=False)
    grounding_engine = GroundingEngine(use_mock=False)

    test_queries = [
        "Where are the buildings?",
        "Where is the water?",
        "Where is vegetation concentrated?",
        "Where are the major roads?",
    ]

    results_report = []

    print("\n" + "-" * 80)
    print(f"{'QUERY':<35} | {'STATUS':<18} | {'REGIONS':<8} | {'TIME(s)':<7} | {'VRAM (MB)':<9}")
    print("-" * 80)

    total_start = time.perf_counter()

    for idx, query in enumerate(test_queries, 1):
        q_start = time.perf_counter()

        # Semantic VQA Answer
        answer, vqa_conf = vqa_engine.answer_question(test_geotiff, query)

        # Spatial Grounding Evidence Detection (Real OWLv2 Model Only - NO Mock Fallback)
        evidence_regions, img_dims, model_info = grounding_engine.detect_evidence(
            file_path=test_geotiff,
            query=query,
            confidence_threshold=settings.GROUNDING_CONF_THRESHOLD
        )

        status_str = model_info.get("status", "SUCCESS" if evidence_regions else "NO_VALID_DETECTION")

        # Render Visual Evidence Image
        slug = query.lower().replace(" ", "_").replace("?", "")
        output_image_path = render_visual_evidence(
            source_image=test_geotiff,
            regions=evidence_regions,
            filename_prefix=f"verify_phase3_q{idx}_{slug}"
        )

        q_elapsed = time.perf_counter() - q_start
        vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if cuda_avail else 0.0

        top_conf = max([r.confidence for r in evidence_regions]) if evidence_regions else 0.0

        print(f"{query:<35} | {status_str:<18} | {len(evidence_regions):<8} | {q_elapsed:<7.2f} | {vram_mb:<9.1f}")

        results_report.append({
            "query": query,
            "answer": answer,
            "status": status_str,
            "evidence_regions": evidence_regions,
            "output_path": output_image_path,
            "latency": q_elapsed,
            "model_info": model_info
        })

    total_elapsed = time.perf_counter() - total_start
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if cuda_avail else 0.0

    print("-" * 80)
    print(f"\n[SUMMARY] Completed 4 queries in {total_elapsed:.2f}s total.")
    print(f"[SUMMARY] Peak GPU VRAM Memory Allocated: {peak_vram_mb:.2f} MB (~{peak_vram_mb/1024:.2f} GB / 8.00 GB)")

    print("\n" + "=" * 80)
    print("DETAILED VERIFICATION RESULTS BY QUERY:")
    print("=" * 80)

    for item in results_report:
        print(f"\nQuery: '{item['query']}'")
        print(f"  • Answer:               {item['answer']}")
        print(f"  • Grounding Status:     {item['status']}")
        print(f"  • Model Info:           {item['model_info']}")
        print(f"  • Evidence Regions:     {len(item['evidence_regions'])} detected")
        if not item['evidence_regions']:
            print("    - Result: NO_VALID_DETECTION")
        else:
            for reg in item['evidence_regions']:
                print(f"    - [{reg.label}] Conf: {reg.confidence:.4f} | Box: [{reg.box.xmin}, {reg.box.ymin}, {reg.box.xmax}, {reg.box.ymax}] | {reg.explanation}")
        print(f"  • Rendered Evidence:    {item['output_path']}")
        print(f"  • Latency:              {item['latency']:.4f}s")

    print("\n[SUCCESS] Phase 3 Scientific Grounding Verification Complete!")


if __name__ == "__main__":
    run_phase3_verification()
