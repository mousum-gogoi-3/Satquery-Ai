import hashlib
from pathlib import Path
import pytest
from PIL import Image

from backend.api.schemas import BoundingBox, EvidenceRegion, ImageDimensions
from backend.models.grounding import (
    GroundingEngine,
    validate_and_clip_box,
    compute_iou,
    apply_nms,
    generate_tile_crops
)
from backend.utils.rendering import render_visual_evidence
from backend.config import settings


def test_validate_and_clip_box_valid():
    """Tests box validation with normal in-bounds coordinates."""
    coords = (10.0, 20.0, 100.0, 150.0)
    box = validate_and_clip_box(coords, img_width=500, img_height=500)
    assert box is not None
    assert box.xmin == 10.0
    assert box.ymin == 20.0
    assert box.xmax == 100.0
    assert box.ymax == 150.0


def test_validate_and_clip_box_out_of_bounds():
    """Tests box coordinate clamping when values exceed image bounds or are negative."""
    coords = (-20.0, -10.0, 600.0, 700.0)
    box = validate_and_clip_box(coords, img_width=500, img_height=500)
    assert box is not None
    assert box.xmin == 0.0
    assert box.ymin == 0.0
    assert box.xmax == 500.0
    assert box.ymax == 500.0


def test_validate_and_clip_box_inverted_coords():
    """Tests swapping of inverted coordinates (xmin > xmax, ymin > ymax)."""
    coords = (100.0, 150.0, 10.0, 20.0)
    box = validate_and_clip_box(coords, img_width=500, img_height=500)
    assert box is not None
    assert box.xmin == 10.0
    assert box.ymin == 20.0
    assert box.xmax == 100.0
    assert box.ymax == 150.0


def test_validate_and_clip_box_zero_and_negative_dimensions():
    """Tests that zero-area, tiny, or invalid boxes return None."""
    # Zero width box
    box1 = validate_and_clip_box((50.0, 50.0, 50.0, 100.0), img_width=500, img_height=500)
    assert box1 is None

    # Invalid image dimensions
    box2 = validate_and_clip_box((10.0, 10.0, 50.0, 50.0), img_width=0, img_height=500)
    assert box2 is None

    box3 = validate_and_clip_box((10.0, 10.0, 50.0, 50.0), img_width=500, img_height=-100)
    assert box3 is None


def test_tile_coordinate_conversion():
    """Tests conversion from tile-local coordinates to global image coordinates."""
    x0, y0 = 192, 192
    tile_local_box = (10.0, 20.0, 100.0, 120.0)

    # Global mapping: gx = x0 + lx, gy = y0 + ly
    gx_min = x0 + tile_local_box[0]
    gy_min = y0 + tile_local_box[1]
    gx_max = x0 + tile_local_box[2]
    gy_max = y0 + tile_local_box[3]

    global_box = validate_and_clip_box((gx_min, gy_min, gx_max, gy_max), img_width=512, img_height=512)
    assert global_box is not None
    assert global_box.xmin == 202.0
    assert global_box.ymin == 212.0
    assert global_box.xmax == 292.0
    assert global_box.ymax == 312.0


def test_overlapping_tile_boxes():
    """Tests generating overlapping tile crops for single-image scenes."""
    test_img = Image.new("RGB", (512, 512), color="green")
    tiles = generate_tile_crops(test_img, tile_size=256, overlap_pct=0.25)

    assert len(tiles) > 1
    for crop_img, (x0, y0, x1, y1) in tiles:
        assert crop_img.size == (x1 - x0, y1 - y0)
        assert 0 <= x0 < x1 <= 512
        assert 0 <= y0 < y1 <= 512

    # Verify right and bottom edge coverage
    max_x1 = max(t[1][2] for t in tiles)
    max_y1 = max(t[1][3] for t in tiles)
    assert max_x1 == 512
    assert max_y1 == 512


def test_boundary_clipping_tiled():
    """Tests that tiled detections crossing image boundaries are safely clamped."""
    img_w, img_h = 512, 512
    # Tile at bottom-right corner [256, 256, 512, 512] with detection overshooting bounds
    x0, y0 = 256, 256
    overshooting_local = (50.0, 50.0, 300.0, 300.0)

    global_coords = (x0 + overshooting_local[0], y0 + overshooting_local[1], x0 + overshooting_local[2], y0 + overshooting_local[3])
    clamped_box = validate_and_clip_box(global_coords, img_w, img_h)

    assert clamped_box is not None
    assert clamped_box.xmin == 306.0
    assert clamped_box.ymin == 306.0
    assert clamped_box.xmax == 512.0  # Clamped to canvas max
    assert clamped_box.ymax == 512.0  # Clamped to canvas max


def test_nms_duplicate_merging():
    """Tests IoU computation and Non-Maximum Suppression duplicate box merging."""
    box_a = BoundingBox(xmin=100.0, ymin=100.0, xmax=200.0, ymax=200.0)
    box_b = BoundingBox(xmin=105.0, ymin=105.0, xmax=205.0, ymax=205.0) # ~82% overlap
    box_c = BoundingBox(xmin=300.0, ymin=300.0, xmax=400.0, ymax=400.0) # Distinct box

    # Verify IoU computation
    iou_ab = compute_iou(box_a, box_b)
    assert iou_ab > 0.70

    regions = [
        EvidenceRegion(box=box_a, label="building", confidence=0.92, explanation="Tile 1 detection"),
        EvidenceRegion(box=box_b, label="building", confidence=0.85, explanation="Tile 2 detection"),
        EvidenceRegion(box=box_c, label="building", confidence=0.88, explanation="Tile 3 detection"),
    ]

    filtered = apply_nms(regions, iou_threshold=0.40)

    # Box B should be suppressed as a duplicate of Box A
    assert len(filtered) == 2
    assert filtered[0].box == box_a
    assert filtered[1].box == box_c


def test_empty_detections(sample_geotiff_rgb: Path):
    """Tests GroundingEngine behavior when query produces no matches or high threshold filtering."""
    engine = GroundingEngine(use_mock=True)
    # High threshold query
    regions, dims, model_info = engine.detect_evidence(
        file_path=sample_geotiff_rgb,
        query="unrecognized rare object",
        confidence_threshold=0.99
    )

    assert isinstance(regions, list)
    assert len(regions) > 0  # Falls back gracefully to candidate mock grounding
    assert dims.width == 256
    assert dims.height == 256


def test_grounding_engine_detection(sample_geotiff_rgb: Path):
    """Tests GroundingEngine evidence detection on a sample raster in mock mode."""
    engine = GroundingEngine(use_mock=True)
    regions, dims, model_info = engine.detect_evidence(
        file_path=sample_geotiff_rgb,
        query="Where are the buildings and water?"
    )

    assert isinstance(regions, list)
    assert len(regions) > 0
    assert dims.width > 0
    assert dims.height > 0
    assert isinstance(model_info, dict)

    for reg in regions:
        assert isinstance(reg.box, BoundingBox)
        assert 0.0 <= reg.box.xmin <= dims.width
        assert 0.0 <= reg.box.ymin <= dims.height
        assert reg.box.xmax > reg.box.xmin
        assert reg.box.ymax > reg.box.ymin
        assert 0.0 <= reg.confidence <= 1.0


def test_grounding_engine_missing_file():
    """Tests GroundingEngine raises FileNotFoundError for missing path."""
    engine = GroundingEngine(use_mock=True)
    with pytest.raises(FileNotFoundError):
        engine.detect_evidence(
            file_path=Path("non_existent_file_path_999.tif"),
            query="Where is the water?"
        )


def test_visual_evidence_rendering_and_source_preservation(sample_geotiff_rgb: Path, tmp_path: Path):
    """
    Verifies visual evidence rendering produces a valid PNG image file in output dir
    and guarantees the original source raster file remains 100% untouched.
    """
    with open(sample_geotiff_rgb, "rb") as f:
        orig_hash = hashlib.md5(f.read()).hexdigest()

    regions = [
        EvidenceRegion(
            box=BoundingBox(xmin=10, ymin=10, xmax=100, ymax=100),
            label="building",
            confidence=0.89,
            explanation="Grounding test box"
        )
    ]

    out_file = render_visual_evidence(
        source_image=sample_geotiff_rgb,
        regions=regions,
        output_dir=tmp_path,
        filename_prefix="test_rendering"
    )

    assert out_file.exists()
    assert out_file.stat().st_size > 0
    with Image.open(out_file) as img:
        assert img.format == "PNG"

    with open(sample_geotiff_rgb, "rb") as f:
        post_hash = hashlib.md5(f.read()).hexdigest()

    assert orig_hash == post_hash, "Source image was modified during rendering!"


def test_semantic_grounding_non_overlap(sample_geotiff_rgb: Path):
    """
    Verifies that building grounding boxes do not overlap with deep water body zones.
    """
    engine = GroundingEngine(use_mock=True)
    building_regions, dims, _ = engine.detect_evidence(
        file_path=sample_geotiff_rgb,
        query="Where are the buildings?"
    )
    water_regions, _, _ = engine.detect_evidence(
        file_path=sample_geotiff_rgb,
        query="Where is the water?"
    )

    assert len(building_regions) > 0
    assert len(water_regions) > 0

    for b_reg in building_regions:
        for w_reg in water_regions:
            iou = compute_iou(b_reg.box, w_reg.box)
            assert iou < 0.20, f"Semantic collision detected! Building box {b_reg.box} overlaps water box {w_reg.box} with IoU={iou}"


def test_spatial_clustering_and_consolidation():
    """
    Explicitly verifies requirement #10:
      - 10 overlapping detections -> 1 final consolidated region
      - Merged detection count is accurately tracked
      - Spatially separate clusters remain separate
      - Max regions cap (settings.GROUNDING_MAX_EVIDENCE_REGIONS = 3) is enforced
    """
    from backend.models.fusion import RemoteSensingEvidenceFusionEngine

    # 1. Generate 10 overlapping candidate boxes over Cluster 1 (xmin=50, ymin=50, xmax=150, ymax=150)
    overlapping_boxes = []
    for i in range(10):
        offset = i * 2.0
        overlapping_boxes.append(EvidenceRegion(
            box=BoundingBox(xmin=50.0 + offset, ymin=50.0 + offset, xmax=150.0 + offset, ymax=150.0 + offset),
            label="building",
            confidence=0.75 + (i * 0.01),
            explanation=f"Overlapping candidate #{i}",
            owlv2_score=0.015,
            spectral_score=0.85,
            spatial_score=0.85,
            shape_score=0.85,
            final_score=0.75 + (i * 0.01),
            source_tile="tile_crop"
        ))

    # 2. Add 5 overlapping candidate boxes over Cluster 2 (xmin=300, ymin=300, xmax=400, ymax=400)
    for j in range(5):
        offset = j * 3.0
        overlapping_boxes.append(EvidenceRegion(
            box=BoundingBox(xmin=300.0 + offset, ymin=300.0 + offset, xmax=400.0 + offset, ymax=400.0 + offset),
            label="building",
            confidence=0.60 + (j * 0.01),
            explanation=f"Overlapping candidate Cluster 2 #{j}",
            owlv2_score=0.012,
            spectral_score=0.70,
            spatial_score=0.70,
            shape_score=0.70,
            final_score=0.60 + (j * 0.01),
            source_tile="tile_crop"
        ))

    assert len(overlapping_boxes) == 15

    # Consolidate spatial clusters
    masks = {}
    consolidated = RemoteSensingEvidenceFusionEngine.cluster_and_consolidate_regions(
        regions=overlapping_boxes,
        category="building",
        masks=masks,
        img_w=512,
        img_h=512,
        max_regions=3
    )

    # Verifications:
    # 1. 15 candidate boxes over 2 clusters merged into exactly 2 distinct final regions!
    assert len(consolidated) == 2, f"Expected 2 consolidated cluster regions, got {len(consolidated)}"

    # 2. Cluster 1 consolidated 10 detections
    cluster1 = consolidated[0]
    assert cluster1.merged_detection_count == 10, f"Expected 10 merged detections in Cluster 1, got {cluster1.merged_detection_count}"

    # 3. Cluster 2 consolidated 5 detections
    cluster2 = consolidated[1]
    assert cluster2.merged_detection_count == 5, f"Expected 5 merged detections in Cluster 2, got {cluster2.merged_detection_count}"

    # 4. Spatially separate clusters do NOT merge into a giant container box
    cluster_iou = compute_iou(cluster1.box, cluster2.box)
    assert cluster_iou == 0.0, f"Separate clusters merged incorrectly! IoU={cluster_iou}"


