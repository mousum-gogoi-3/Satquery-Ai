"""
Verification of real API grounded queries on target image 9d0730f0-9297-41fa-8f35-49a406c41d26
"""

import sys
import asyncio
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from httpx import AsyncClient, ASGITransport
from backend.main import app
from backend.config import settings

TARGET_FILE_ID = "9d0730f0-9297-41fa-8f35-49a406c41d26"

QUERIES = [
    {
        "title": "Exact User Request (max_regions=1)",
        "payload": {
            "file_id": TARGET_FILE_ID,
            "query": "Where is the agricultural land?",
            "confidence_threshold": 1.0,
            "max_regions": 1,
            "debug_mode": False
        }
    },
    {
        "title": "Query 1: Building Query",
        "payload": {
            "file_id": TARGET_FILE_ID,
            "query": "Where is the building?",
            "confidence_threshold": 0.04,
            "max_regions": 5,
            "debug_mode": False
        }
    },
    {
        "title": "Query 2: Agricultural Land Query",
        "payload": {
            "file_id": TARGET_FILE_ID,
            "query": "Where is the agricultural land?",
            "confidence_threshold": 0.04,
            "max_regions": 5,
            "debug_mode": False
        }
    },
    {
        "title": "Query 3: Forest / Vegetation Query",
        "payload": {
            "file_id": TARGET_FILE_ID,
            "query": "Where is the forest?",
            "confidence_threshold": 0.04,
            "max_regions": 5,
            "debug_mode": False
        }
    },
    {
        "title": "Query 4: Road Query",
        "payload": {
            "file_id": TARGET_FILE_ID,
            "query": "Where is the road?",
            "confidence_threshold": 0.04,
            "max_regions": 5,
            "debug_mode": False
        }
    },
    {
        "title": "Query 5: Water Query",
        "payload": {
            "file_id": TARGET_FILE_ID,
            "query": "Where is the water?",
            "confidence_threshold": 0.04,
            "max_regions": 5,
            "debug_mode": False
        }
    }
]


async def run_queries():
    print("=" * 85)
    print("      SATQUERY AI — GROUNDED ANALYSIS API REAL-IMAGE VALIDATION")
    print(f"      Target File ID: {TARGET_FILE_ID}")
    print("=" * 85)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver", timeout=120.0) as client:
        for item in QUERIES:
            title = item["title"]
            payload = item["payload"]
            q = payload["query"]
            print("\n" + "-" * 85)
            print(f"RUNNING: {title}")
            print(f"Payload: {payload}")

            resp = await client.post("/api/v1/analysis/grounded", json=payload)
            if resp.status_code != 200:
                print(f"[ERROR] HTTP {resp.status_code}: {resp.text}")
                continue

            data = resp.json()
            model_info = data.get("model_info", {})
            regions = data.get("evidence_regions", [])

            cat = model_info.get("category", "N/A")
            prompts = model_info.get("prompts", [])
            raw_c = model_info.get("raw_candidates", "0")
            acc_d = model_info.get("accepted_detections", "0")
            status = model_info.get("status", "N/A")
            overall_conf = data.get("overall_confidence", 0.0)
            img_path = data.get("evidence_image_path", "N/A")
            answer = data.get("answer", "")

            print(f"Query:               {q}")
            print(f"Category:            {cat}")
            print(f"Prompts:             {prompts}")
            print(f"Raw Candidates:      {raw_c}")
            print(f"Accepted Detections: {acc_d}")
            print(f"Status:              {status}")
            print(f"Overall Confidence:  {overall_conf:.4f}")
            print(f"VQA Answer:          {answer}")
            print(f"Evidence Image:      {img_path}")
            print(f"Final Evidence Regions ({len(regions)}):")

            for idx, r in enumerate(regions):
                b = r["box"]
                w = b["xmax"] - b["xmin"]
                h = b["ymax"] - b["ymin"]
                conf = r.get("confidence", 0.0)
                owl = r.get("owlv2_score", 0.0)
                merged = r.get("merged_detection_count", 1)
                src = r.get("source_tile", "N/A")
                print(f"  [{idx+1}] [{b['xmin']:.1f}, {b['ymin']:.1f}, {b['xmax']:.1f}, {b['ymax']:.1f}] size={w:.1f}x{h:.1f} px | conf={conf:.4f} | owlv2={owl:.4f} | merged={merged} | src={src}")

    print("\n" + "=" * 85)
    print("VALIDATION COMPLETE")
    print("=" * 85)


if __name__ == "__main__":
    asyncio.run(run_queries())
