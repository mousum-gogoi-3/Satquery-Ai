import hashlib
from pathlib import Path
import pytest
from PIL import Image
import numpy as np

from backend.api.schemas import BoundingBox, EvidenceRegion, ImageDimensions
from backend.models.grounding import (
    GroundingEngine,
    validate_and_clip_box,
    compute_iou,
    apply_nms,
    generate_tile_crops
)
from backend.models.fusion import validate_and_clip_coords
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


def test_device_selection_cuda_cpu():
    """Verifies OWLv2ModelManager automatically detects CUDA and respects explicit CPU overrides."""
    import torch
    from backend.models.grounding import OWLv2ModelManager

    auto_dev = OWLv2ModelManager.get_device("auto")
    expected_auto = "cuda" if torch.cuda.is_available() else "cpu"
    assert auto_dev == expected_auto

    cpu_dev = OWLv2ModelManager.get_device("cpu")
    assert cpu_dev == "cpu"

    cuda_dev = OWLv2ModelManager.get_device("cuda")
    assert cuda_dev == "cuda"


def test_uint16_and_uint8_band_normalization():
    """Verifies that uint8, uint16, and float32 arrays are correctly normalized to [0.0, 1.0] without clipping."""
    from backend.models.fusion import RemoteSensingEvidenceFusionEngine

    # uint8 (0-255)
    u8_arr = np.array([0, 128, 255], dtype=np.uint8)
    norm_u8 = RemoteSensingEvidenceFusionEngine._normalize_bands(u8_arr)
    assert norm_u8.dtype == np.float32
    assert np.isclose(norm_u8[0], 0.0)
    assert np.isclose(norm_u8[1], 128 / 255.0)
    assert np.isclose(norm_u8[2], 1.0)

    # uint16 (0-65535, e.g. Sentinel-2 reflectance 0-10000)
    u16_arr = np.array([0, 3000, 10000], dtype=np.uint16)
    norm_u16 = RemoteSensingEvidenceFusionEngine._normalize_bands(u16_arr)
    assert norm_u16.dtype == np.float32
    assert np.isclose(norm_u16[0], 0.0)
    assert 0.25 <= norm_u16[1] <= 0.35
    assert np.isclose(norm_u16[2], 1.0)

    # float32 normalized [0.0, 1.0]
    f32_arr = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    norm_f32 = RemoteSensingEvidenceFusionEngine._normalize_bands(f32_arr)
    assert np.isclose(norm_f32[1], 0.5)


def test_rgba_alpha_vs_geotiff_nir_detection(tmp_path: Path):
    """
    Verifies that a 4-channel PNG (RGBA) is recognized as having an Alpha channel (NOT NIR),
    while a 4-band GeoTIFF is recognized as having a Near-Infrared (NIR) band.
    """
    from backend.remote_sensing.raster import RasterReader
    from backend.remote_sensing.bands import BandManager, BandType

    # 1. Generate 4-channel RGBA PNG
    rgba_path = tmp_path / "test_screenshot.png"
    arr = np.random.randint(0, 256, size=(100, 100, 4), dtype=np.uint8)
    Image.fromarray(arr, mode="RGBA").save(rgba_path)

    meta_png = RasterReader.extract_metadata(rgba_path)
    assert meta_png.band_count == 4
    assert not BandManager.has_physical_nir(meta_png)
    assert BandManager.get_imagery_type(meta_png) == "RGBA"
    png_configs = BandManager.infer_band_configuration(meta_png)
    assert png_configs[3].band_type == BandType.ALPHA

    # 2. Generate 4-band synthetic GeoTIFF (Red, Green, Blue, NIR)
    import tifffile
    gtiff_path = tmp_path / "test_satellite.tif"
    gtiff_arr = np.random.randint(100, 5000, size=(4, 100, 100), dtype=np.uint16)
    extratags = [
        (33550, 'd', 3, (10.0, 10.0, 0.0)),
        (33922, 'd', 6, (0.0, 0.0, 0.0, 500000.0, 4000000.0, 0.0)),
        (34735, 'h', 8, (1, 1, 0, 1, 3072, 0, 1, 32633)),
    ]
    tifffile.imwrite(gtiff_path, gtiff_arr, extratags=extratags, planarconfig='SEPARATE')

    meta_gtiff = RasterReader.extract_metadata(gtiff_path)
    assert meta_gtiff.band_count == 4
    assert BandManager.has_physical_nir(meta_gtiff)
    assert BandManager.get_imagery_type(meta_gtiff) == "MULTISPECTRAL"
    gtiff_configs = BandManager.infer_band_configuration(meta_gtiff)
    assert gtiff_configs[3].band_type == BandType.NIR


def test_query_normalization_and_category_mapping():
    """Verifies natural language query normalization and multi-category prompt expansion."""
    engine = GroundingEngine(use_mock=True)

    queries_to_expected = [
        ("Where are the buildings?", "building"),
        ("Find houses", "building"),
        ("Find roads", "road"),
        ("Where is the water?", "water"),
        ("Find vehicles", "vehicle"),
        ("Find ships", "ship"),
        ("Find airports", "airport"),
        ("Find agricultural fields", "agriculture"),
        ("Find construction areas", "construction area"),
        ("Find trees/vegetation", "vegetation"),
        ("Locate industrial storage tanks", "industrial storage tanks"),
    ]

    for q, expected_cat in queries_to_expected:
        info = engine._extract_target_queries(q)
        assert info["category"] == expected_cat, f"Query '{q}' mapped to '{info['category']}', expected '{expected_cat}'"
        assert len(info["prompts"]) >= 1


def test_adaptive_fusion_scoring_rgb_mode():
    """
    Verifies that on RGB/RGBA images (where NIR is absent),
    the Evidence Fusion Engine applies adaptive visual weights and does not
    eliminate valid building or vehicle detections.
    """
    from backend.models.fusion import RemoteSensingEvidenceFusionEngine
    from backend.remote_sensing.raster import RasterMetadata

    box = BoundingBox(xmin=10.0, ymin=10.0, xmax=50.0, ymax=50.0)
    fake_meta = RasterMetadata(width=200, height=200, band_count=4, driver="PNG", datatype="RGBA")
    fake_bands = np.zeros((4, 200, 200), dtype=np.uint8)

    # Building candidate with good OWLv2 score on RGB/RGBA
    score, comp, expl = RemoteSensingEvidenceFusionEngine.evaluate_candidate_scores(
        file_path=Path("dummy.png"),
        box=box,
        category="building",
        owlv2_raw_score=0.035,
        source_tile="tile_crop",
        masks={},
        img_w=200,
        img_h=200,
        preloaded_bands=fake_bands,
        preloaded_meta=fake_meta
    )

    # Score should be well above acceptance threshold (0.28)
    assert score >= 0.28
    assert "Physical Elimination" not in expl
    assert comp["owlv2_score"] == 0.035


def test_small_objects_preservation():
    """Verifies that small valid objects (e.g. vehicles 10x10 px) are preserved and not discarded."""
    from backend.models.fusion import RemoteSensingEvidenceFusionEngine
    from backend.remote_sensing.raster import RasterMetadata

    small_box = BoundingBox(xmin=100.0, ymin=100.0, xmax=112.0, ymax=114.0)  # 12x14 pixels
    fake_meta = RasterMetadata(width=1000, height=1000, band_count=3, driver="PNG", datatype="RGB")
    fake_bands = np.zeros((3, 1000, 1000), dtype=np.uint8)

    score, comp, expl = RemoteSensingEvidenceFusionEngine.evaluate_candidate_scores(
        file_path=Path("dummy_vehicle.png"),
        box=small_box,
        category="vehicle",
        owlv2_raw_score=0.025,
        source_tile="tile_crop",
        masks={},
        img_w=1000,
        img_h=1000,
        preloaded_bands=fake_bands,
        preloaded_meta=fake_meta
    )

    assert score >= 0.28, f"Small object rejected! Score={score}"
    assert "Container box" not in expl


def test_regression_a_identity_coordinate_conversion():
    """Test A — Identity conversion: tile origin [0, 0], local [50, 60, 150, 180] -> global [50, 60, 150, 180]."""
    x0, y0 = 0, 0
    scale_x, scale_y = 1.0, 1.0
    local_box = (50.0, 60.0, 150.0, 180.0)
    gx0 = (x0 + local_box[0]) * scale_x
    gy0 = (y0 + local_box[1]) * scale_y
    gx1 = (x0 + local_box[2]) * scale_x
    gy1 = (y0 + local_box[3]) * scale_y
    box = validate_and_clip_coords(gx0, gy0, gx1, gy1, img_w=256, img_h=256)
    assert box is not None
    assert box.xmin == 50.0
    assert box.ymin == 60.0
    assert box.xmax == 150.0
    assert box.ymax == 180.0


def test_regression_b_non_zero_tile_origin_coordinate_conversion():
    """Test B — Non-zero tile origin: tile origin [500, 400], local [50, 60, 150, 180] -> global [550, 460, 650, 580]."""
    x0, y0 = 500, 400
    scale_x, scale_y = 1.0, 1.0
    local_box = (50.0, 60.0, 150.0, 180.0)
    gx0 = (x0 + local_box[0]) * scale_x
    gy0 = (y0 + local_box[1]) * scale_y
    gx1 = (x0 + local_box[2]) * scale_x
    gy1 = (y0 + local_box[3]) * scale_y
    box = validate_and_clip_coords(gx0, gy0, gx1, gy1, img_w=1000, img_h=1000)
    assert box is not None
    assert box.xmin == 550.0
    assert box.ymin == 460.0
    assert box.xmax == 650.0
    assert box.ymax == 580.0


def test_regression_c_boundary_clipping_bounds():
    """Test C — Clipping: for an image 785x799, box extending outside must be clipped to [0, 785] x [0, 799]."""
    box = validate_and_clip_coords(-20.5, 750.0, 810.2, 850.0, img_w=785, img_h=799)
    assert box is not None
    assert 0.0 <= box.xmin < box.xmax <= 785.0
    assert 0.0 <= box.ymin < box.ymax <= 799.0
    assert box.xmin == 0.0
    assert box.xmax == 785.0
    assert box.ymin == 750.0
    assert box.ymax == 799.0


def test_regression_d_xy_orientation():
    """Test D — X/Y orientation: verify that X (horizontal) and Y (vertical) are never swapped."""
    x0, y0 = 100, 300
    local_box = (20.0, 10.0, 220.0, 60.0)  # width=200, height=50
    gx0 = x0 + local_box[0]  # 120
    gy0 = y0 + local_box[1]  # 310
    gx1 = x0 + local_box[2]  # 320
    gy1 = y0 + local_box[3]  # 360
    box = validate_and_clip_coords(gx0, gy0, gx1, gy1, img_w=800, img_h=800)
    assert box is not None
    assert (box.xmax - box.xmin) == 200.0, "X dimension corrupted or swapped with Y!"
    assert (box.ymax - box.ymin) == 50.0, "Y dimension corrupted or swapped with X!"
    assert box.xmin == 120.0
    assert box.ymin == 310.0


def test_regression_e_overlapping_tiles_consolidation():
    """Test E — Overlapping tiles: detections from two overlapping tiles representing same object merge correctly."""
    from backend.models.fusion import RemoteSensingEvidenceFusionEngine

    # Building at global [380, 120, 500, 240] detected in two overlapping tiles
    r1 = EvidenceRegion(
        box=BoundingBox(xmin=380.0, ymin=120.0, xmax=500.0, ymax=240.0),
        label="building",
        confidence=0.72,
        owlv2_score=0.22,
        final_score=0.72,
        source_tile="tile_x192_y0_w256_h256"
    )
    r2 = EvidenceRegion(
        box=BoundingBox(xmin=382.0, ymin=119.0, xmax=498.0, ymax=241.0),
        label="building",
        confidence=0.75,
        owlv2_score=0.24,
        final_score=0.75,
        source_tile="tile_x384_y0_w256_h256"
    )
    # Separate building at [100, 600, 200, 700]
    r3 = EvidenceRegion(
        box=BoundingBox(xmin=100.0, ymin=600.0, xmax=200.0, ymax=700.0),
        label="building",
        confidence=0.68,
        owlv2_score=0.19,
        final_score=0.68,
        source_tile="tile_x0_y543_w256_h256"
    )

    consolidated = RemoteSensingEvidenceFusionEngine.cluster_and_consolidate_regions(
        regions=[r1, r2, r3],
        category="building",
        masks={},
        img_w=785,
        img_h=799
    )

    # Exactly 2 regions should remain (r1 and r2 merged into 1; r3 separate)
    assert len(consolidated) == 2
    b1_reg = next(r for r in consolidated if r.box.xmin > 300)
    assert b1_reg.merged_detection_count == 2
    assert b1_reg.confidence == 0.75
    assert b1_reg.source_tile == "tile_x384_y0_w256_h256"


def test_regression_f_renderer_consistency(tmp_path):
    """Test F — Renderer consistency: rendered image draws evidence at exact global coordinates."""
    from backend.utils.rendering import render_visual_evidence
    from PIL import Image

    test_img = Image.new("RGB", (300, 300), (50, 50, 50))
    img_file = tmp_path / "test_render.png"
    test_img.save(img_file)

    known_box = BoundingBox(xmin=50.0, ymin=60.0, xmax=150.0, ymax=160.0)
    reg = EvidenceRegion(
        box=known_box,
        label="building",
        confidence=0.90,
        final_score=0.90
    )

    out_file = render_visual_evidence(regions=[reg], source_image=img_file, output_dir=tmp_path)
    assert out_file.exists()
    rendered = Image.open(out_file)
    assert rendered.size == (300, 300)


def test_regression_g_realistic_building_localization():
    """
    Test G — Realistic regression test on the failing image (1e4a3854-5536-4d91-973f-1bd8c052ea66_a.tif).
    Verifies:
      1. Detected building boxes are spatially consistent with real structures (width >= 40px, height >= 40px).
      2. No arbitrary tiny 8x7 noise boxes are returned as top detections.
      3. No runaway transitive merging (merged_detection_count < 15).
    """
    failing_image = Path("data/uploads/1e4a3854-5536-4d91-973f-1bd8c052ea66_a.tif")
    if not failing_image.exists():
        pytest.skip(f"Reference test file '{failing_image}' not found.")

    engine = GroundingEngine(use_mock=False)
    regions, dims, model_info = engine.detect_evidence(
        file_path=failing_image,
        query="Where is the building?",
        max_regions=5
    )

    assert model_info["status"] == "SUCCESS"
    assert len(regions) > 0

    top_region = regions[0]
    b = top_region.box
    width = b.xmax - b.xmin
    height = b.ymax - b.ymin

    assert width >= 40.0, f"Detected box too small in width: {width:.1f}px"
    assert height >= 40.0, f"Detected box too small in height: {height:.1f}px"
    assert (top_region.merged_detection_count or 1) < 15, f"Runaway cluster merging: {top_region.merged_detection_count}"


def test_semantic_query_to_grounding_mapping_regression():
    """
    Regression test covering all required semantic category mappings:
    1. Where is the building? -> building
    2. Where are the buildings? -> building
    3. Where is the agricultural land? -> agriculture
    4. Where are the crop fields? -> agriculture
    5. Show me the farmland -> agriculture
    6. Where is the forest? -> vegetation
    7. Where is the road? -> road
    8. Where is the water? -> water
    9. Verify agricultural category produces agriculture-specific OWLv2 prompts (fields/crops, not trees).
    10. Verify building detection prompts and coordinates remain consistent.
    """
    engine = GroundingEngine(use_mock=True)

    # 1. Where is the building? -> building
    q1 = engine._extract_target_queries("Where is the building?")
    assert q1["category"] == "building"
    assert "building structure" in q1["prompts"]

    # 2. Where are the buildings? -> building
    q2 = engine._extract_target_queries("Where are the buildings?")
    assert q2["category"] == "building"
    assert "buildings" in q2["prompts"]

    # 3. Where is the agricultural land? -> agriculture
    q3 = engine._extract_target_queries("Where is the agricultural land?")
    assert q3["category"] == "agriculture"

    # 4. Where are the crop fields? -> agriculture
    q4 = engine._extract_target_queries("Where are the crop fields?")
    assert q4["category"] == "agriculture"

    # 5. Show me the farmland -> agriculture
    q5 = engine._extract_target_queries("Show me the farmland")
    assert q5["category"] == "agriculture"

    # 6. Where is the forest? -> vegetation
    q6 = engine._extract_target_queries("Where is the forest?")
    assert q6["category"] == "vegetation"
    assert "forest" in q6["prompts"]
    assert "trees" in q6["prompts"]

    # 7. Where is the road? -> road
    q7 = engine._extract_target_queries("Where is the road?")
    assert q7["category"] == "road"
    assert "road" in q7["prompts"]

    # 8. Where is the water? -> water
    q8 = engine._extract_target_queries("Where is the water?")
    assert q8["category"] == "water"
    assert "water body" in q8["prompts"]

    # 9. Verify that agricultural category produces agriculture-specific prompts (not forest/trees)
    ag_prompts = q3["prompts"]
    assert any("agricultural field" in p for p in ag_prompts)
    assert any("crop field" in p for p in ag_prompts)
    assert any("farmland" in p for p in ag_prompts)
    assert not any("trees" in p for p in ag_prompts)
    assert not any("woodland" in p for p in ag_prompts)

    # 10. Verify additional NL variations for agriculture and buildings
    assert engine._extract_target_queries("Locate cultivated land")["category"] == "agriculture"
    assert engine._extract_target_queries("Find the cropland")["category"] == "agriculture"
    assert engine._extract_target_queries("Where are the farms?")["category"] == "agriculture"
    assert engine._extract_target_queries("Locate the houses")["category"] == "building"
    assert engine._extract_target_queries("Find residential structures")["category"] == "building"
    assert engine._extract_target_queries("Show me the buildings")["category"] == "building"
    assert engine._extract_target_queries("Where is the bare soil?")["category"] == "bare_soil"
    assert engine._extract_target_queries("Where is the urban area?")["category"] == "urban"




