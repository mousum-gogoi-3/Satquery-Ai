import pytest
from pathlib import Path
import numpy as np
from PIL import Image
import tifffile
from httpx import AsyncClient, ASGITransport

from backend.main import app
from backend.config import settings


def create_water_vegetation_geotiff(path: Path) -> Path:
    """Creates a 256x256 GeoTIFF with distinct blue water body and green vegetated terrain."""
    arr = np.zeros((3, 256, 256), dtype=np.uint8)
    # Left half: water (deep blue)
    arr[0, :, :128] = 20
    arr[1, :, :128] = 60
    arr[2, :, :128] = 200
    # Right half: vegetation (lush green)
    arr[0, :, 128:] = 35
    arr[1, :, 128:] = 160
    arr[2, :, 128:] = 40

    extratags = [
        (33550, 'd', 3, (0.0001, 0.0001, 0.0)),
        (33922, 'd', 6, (0.0, 0.0, 0.0, 10.0, 50.0, 0.0)),
        (34735, 'h', 8, (1, 1, 0, 1, 2048, 0, 1, 4326)),
    ]
    tifffile.imwrite(path, arr, extratags=extratags, planarconfig='SEPARATE')
    return path


def create_urban_buildings_geotiff(path: Path) -> Path:
    """Creates a 256x256 GeoTIFF with high-contrast bright rectangular structures (buildings) on dark ground."""
    arr = np.full((3, 256, 256), 50, dtype=np.uint8)  # dark background
    # Add bright building blocks
    for r in range(30, 220, 60):
        for c in range(30, 220, 60):
            arr[:, r:r+35, c:c+35] = 240  # bright roofs/structures

    # Horizontal and vertical road lines
    arr[:, 110:125, :] = 100
    arr[:, :, 110:125] = 100

    extratags = [
        (33550, 'd', 3, (0.0001, 0.0001, 0.0)),
        (33922, 'd', 6, (0.0, 0.0, 0.0, 10.0, 50.0, 0.0)),
        (34735, 'h', 8, (1, 1, 0, 1, 2048, 0, 1, 4326)),
    ]
    tifffile.imwrite(path, arr, extratags=extratags, planarconfig='SEPARATE')
    return path


def create_single_band_panchromatic_geotiff(path: Path) -> Path:
    """Creates a 263x236 single-band grayscale raster matching user dimensions."""
    arr = np.linspace(40, 220, 263 * 236, dtype=np.uint8).reshape((263, 236))
    extratags = [
        (33550, 'd', 3, (0.0001, 0.0001, 0.0)),
        (33922, 'd', 6, (0.0, 0.0, 0.0, 10.0, 50.0, 0.0)),
        (34735, 'h', 8, (1, 1, 0, 1, 2048, 0, 1, 4326)),
    ]
    tifffile.imwrite(path, arr, extratags=extratags)
    return path


GENERIC_CANNED_SUBSTRINGS = [
    "A single-band unknown panchromatic raster scene",
    "A single-band uint8 panchromatic raster scene",
    "A single-band float32 panchromatic raster scene",
    "showing surface reflectance variations and structural features",
    "High intensity values correspond to developed structures and bare surfaces",
    "depicts an active terrestrial landscape with visible clusters of developed structures",
]


@pytest.mark.asyncio
async def test_different_images_produce_different_answers(tmp_data_dir):
    """
    Step 11 & Acceptance Test A/B:
    Upload two visually distinct images and ask 'Explain this image'.
    Outputs must NOT be identical and must NOT match the canned generic template.
    """
    img_a_path = create_water_vegetation_geotiff(tmp_data_dir / "image_A.tif")
    img_b_path = create_urban_buildings_geotiff(tmp_data_dir / "image_B.tif")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        # Upload image A
        with open(img_a_path, "rb") as f:
            res_a = await client.post("/api/v1/raster/upload", files={"file": ("image_A.tif", f, "image/tiff")})
        assert res_a.status_code == 201
        file_id_a = res_a.json()["file_id"]

        # Upload image B
        with open(img_b_path, "rb") as f:
            res_b = await client.post("/api/v1/raster/upload", files={"file": ("image_B.tif", f, "image/tiff")})
        assert res_b.status_code == 201
        file_id_b = res_b.json()["file_id"]

        assert file_id_a != file_id_b

        # VQA Query on Image A
        vqa_a = await client.post(
            "/api/v1/analysis/vqa",
            json={"file_id": file_id_a, "question": "Explain this image"}
        )
        assert vqa_a.status_code == 200
        data_a = vqa_a.json()

        # VQA Query on Image B
        vqa_b = await client.post(
            "/api/v1/analysis/vqa",
            json={"file_id": file_id_b, "question": "Explain this image"}
        )
        assert vqa_b.status_code == 200
        data_b = vqa_b.json()

        answer_a = data_a["answer"]
        answer_b = data_b["answer"]

        # 1. Responses must succeed and include image_used
        assert data_a["image_used"] is True
        assert data_b["image_used"] is True

        # 2. Responses must NOT be identical
        assert answer_a != answer_b, f"Expected distinct answers for different images, but got identical: {answer_a}"

        # 3. Neither answer may be the canned generic string
        for canned in GENERIC_CANNED_SUBSTRINGS:
            assert canned not in answer_a, f"Answer A contained generic canned text: {canned}"
            assert canned not in answer_b, f"Answer B contained generic canned text: {canned}"

        print(f"\n[Test A Answer]: {answer_a}")
        print(f"\n[Test B Answer]: {answer_b}")


@pytest.mark.asyncio
async def test_single_band_panchromatic_vqa(tmp_data_dir):
    """
    Step 4 & 8: Single-band 263x236 raster must not return the canned template
    and must provide valid visual inference.
    """
    img_path = create_single_band_panchromatic_geotiff(tmp_data_dir / "single_band_263x236.tif")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        with open(img_path, "rb") as f:
            res = await client.post("/api/v1/raster/upload", files={"file": ("single_band_263x236.tif", f, "image/tiff")})
        assert res.status_code == 201
        file_id = res.json()["file_id"]

        vqa_res = await client.post(
            "/api/v1/analysis/vqa",
            json={"file_id": file_id, "question": "Explain this image"}
        )
        assert vqa_res.status_code == 200
        data = vqa_res.json()
        answer = data["answer"]

        # Verify not canned
        for canned in GENERIC_CANNED_SUBSTRINGS:
            assert canned not in answer, f"Found canned template in single-band answer: {canned}"

        assert data["image_used"] is True
        print(f"\n[Single-Band Answer]: {answer}")


@pytest.mark.asyncio
async def test_different_questions_same_image(tmp_data_dir):
    """
    Step 15 Acceptance Test D:
    Ask different questions about the same image.
    The model must return different, question-specific answers.
    """
    img_path = create_urban_buildings_geotiff(tmp_data_dir / "image_C.tif")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        with open(img_path, "rb") as f:
            res = await client.post("/api/v1/raster/upload", files={"file": ("image_C.tif", f, "image/tiff")})
        assert res.status_code == 201
        file_id = res.json()["file_id"]

        # Question 1: Explain this image
        r1 = await client.post("/api/v1/analysis/vqa", json={"file_id": file_id, "question": "Explain this image"})
        assert r1.status_code == 200
        ans1 = r1.json()["answer"]

        # Question 2: Where are the buildings?
        r2 = await client.post("/api/v1/analysis/vqa", json={"file_id": file_id, "question": "Where are the buildings?"})
        assert r2.status_code == 200
        ans2 = r2.json()["answer"]

        # Question 3: Is there vegetation visible?
        r3 = await client.post("/api/v1/analysis/vqa", json={"file_id": file_id, "question": "Is there vegetation visible?"})
        assert r3.status_code == 200
        ans3 = r3.json()["answer"]

        assert ans1 != ans2, "Answers to Q1 and Q2 should not be identical"
        assert ans2 != ans3, "Answers to Q2 and Q3 should not be identical"
        assert ans1 != ans3, "Answers to Q1 and Q3 should not be identical"

        print(f"\n[Q1 - Explain]: {ans1}")
        print(f"\n[Q2 - Buildings]: {ans2}")
        print(f"\n[Q3 - Vegetation]: {ans3}")


@pytest.mark.asyncio
async def test_grounded_router_uses_vqa_with_user_question(tmp_data_dir):
    """
    Verify that POST /api/v1/analysis/grounded routes IMAGE_UNDERSTANDING queries
    to real VQA inference and returns image_used in model_info.
    """
    img_path = create_water_vegetation_geotiff(tmp_data_dir / "grounded_test.tif")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        with open(img_path, "rb") as f:
            res = await client.post("/api/v1/raster/upload", files={"file": ("grounded_test.tif", f, "image/tiff")})
        assert res.status_code == 201
        file_id = res.json()["file_id"]

        grounded_res = await client.post(
            "/api/v1/analysis/grounded",
            json={"file_id": file_id, "query": "Explain this image"}
        )
        assert grounded_res.status_code == 200
        data = grounded_res.json()
        assert data["analysis_type"] == "image_understanding"
        assert data["model_info"]["image_used"] is True
        assert len(data["answer"]) > 0

        for canned in GENERIC_CANNED_SUBSTRINGS:
            assert canned not in data["answer"]


@pytest.mark.asyncio
async def test_vqa_answers_not_truncated(tmp_data_dir):
    """
    Step 9: Test that answers end naturally and are not abruptly truncated mid-sentence.
    """
    img_path = create_urban_buildings_geotiff(tmp_data_dir / "image_trunc_test.tif")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        with open(img_path, "rb") as f:
            res = await client.post("/api/v1/raster/upload", files={"file": ("image_trunc_test.tif", f, "image/tiff")})
        assert res.status_code == 201
        file_id = res.json()["file_id"]

        for q in [
            "Explain this image",
            "Where are the buildings?",
            "Is there vegetation visible?",
            "What are the main features visible in this scene?"
        ]:
            vqa_res = await client.post("/api/v1/analysis/vqa", json={"file_id": file_id, "question": q})
            assert vqa_res.status_code == 200
            ans = vqa_res.json()["answer"].strip()

            assert len(ans) > 20
            last_word = ans.split()[-1].lower().rstrip(".,;:!?")
            assert last_word not in ["or", "and", "the", "a", "an", "with", "of", "to", "in", "for", "on", "at", "by"], (
                f"Answer ended mid-sentence with '{last_word}': {ans}"
            )
            print(f"\n[Completed Answer for '{q}'] (len={len(ans)}):\n{ans}")
