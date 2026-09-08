import pytest
from pathlib import Path
from typing import List, Dict, Tuple, Any
import numpy as np
import tifffile

from backend.config import settings
from backend.models.temporal_engine import BiTemporalAnalysisEngine
from backend.models.fusion import compute_box_iou
from backend.api.schemas import BoundingBox, ChangeType, ChangeRegion
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def compute_evaluation_metrics(
    predicted_regions: List[ChangeRegion],
    ground_truth_boxes: List[BoundingBox],
    expected_type: ChangeType,
    iou_threshold: float = 0.50
) -> Dict[str, float]:
    """
    Calculates TP, FP, FN, Precision, Recall, F1 Score, and Mean IoU
    between detected change regions and ground-truth bounding boxes.
    """
    # Filter predictions to the expected change type if matching
    matching_preds = [r for r in predicted_regions if r.change_type == expected_type]
    
    tp = 0
    fp = 0
    fn = 0
    ious: List[float] = []

    matched_gt = set()

    for pred in matching_preds:
        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, gt in enumerate(ground_truth_boxes):
            iou = compute_box_iou(pred.bounding_box, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
            tp += 1
            matched_gt.add(best_gt_idx)
            ious.append(best_iou)
        else:
            fp += 1

    fn = len(ground_truth_boxes) - len(matched_gt)

    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    f1 = 2 * (precision * recall) / max(1e-6, (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_iou = float(np.mean(ious)) if ious else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_iou": round(mean_iou, 4),
    }


@pytest.fixture
def eval_vegetation_pair(tmp_path: Path) -> Tuple[Path, Path, BoundingBox]:
    """Known Synthetic Ground-Truth: Vegetation Loss at [50, 50, 150, 150]."""
    pb = tmp_path / "eval_veg_b.tif"
    pa = tmp_path / "eval_veg_a.tif"

    h, w = 256, 256
    before = np.full((4, h, w), 500, dtype=np.uint16)
    before[0, 50:150, 50:150] = 300   # Red
    before[1, 50:150, 50:150] = 600   # Green
    before[2, 50:150, 50:150] = 300   # Blue
    before[3, 50:150, 50:150] = 3500  # NIR -> NDVI=0.84

    after = np.full((4, h, w), 500, dtype=np.uint16)
    after[0, 50:150, 50:150] = 2500  # Red
    after[1, 50:150, 50:150] = 2000  # Green
    after[2, 50:150, 50:150] = 1800  # Blue
    after[3, 50:150, 50:150] = 900   # NIR -> NDVI=-0.47

    extratags = [
        (33550, 'd', 3, (10.0, 10.0, 0.0)),
        (33922, 'd', 6, (0.0, 0.0, 0.0, 500000.0, 6000000.0, 0.0)),
        (34735, 'h', 8, (1, 1, 0, 1, 3072, 0, 1, 32633)),
    ]

    tifffile.imwrite(pb, before, extratags=extratags, planarconfig='SEPARATE')
    tifffile.imwrite(pa, after, extratags=extratags, planarconfig='SEPARATE')

    gt_box = BoundingBox(xmin=50.0, ymin=50.0, xmax=150.0, ymax=150.0)
    return pb, pa, gt_box


@pytest.fixture
def eval_water_pair(tmp_path: Path) -> Tuple[Path, Path, BoundingBox]:
    """Known Synthetic Ground-Truth: Water Expansion / Flooding at [100, 100, 200, 200]."""
    pb = tmp_path / "eval_water_b.tif"
    pa = tmp_path / "eval_water_a.tif"

    h, w = 256, 256
    before = np.full((4, h, w), 1000, dtype=np.uint16)
    before[0, 100:200, 100:200] = 1500  # Red
    before[1, 100:200, 100:200] = 1200  # Green
    before[2, 100:200, 100:200] = 1000  # Blue
    before[3, 100:200, 100:200] = 2500  # NIR -> NDWI=-0.35

    after = np.full((4, h, w), 1000, dtype=np.uint16)
    after[0, 100:200, 100:200] = 300   # Red
    after[1, 100:200, 100:200] = 2500  # Green
    after[2, 100:200, 100:200] = 3000  # Blue
    after[3, 100:200, 100:200] = 200   # NIR -> NDWI=+0.85

    extratags = [
        (33550, 'd', 3, (10.0, 10.0, 0.0)),
        (33922, 'd', 6, (0.0, 0.0, 0.0, 500000.0, 6000000.0, 0.0)),
        (34735, 'h', 8, (1, 1, 0, 1, 3072, 0, 1, 32633)),
    ]

    tifffile.imwrite(pb, before, extratags=extratags, planarconfig='SEPARATE')
    tifffile.imwrite(pa, after, extratags=extratags, planarconfig='SEPARATE')

    gt_box = BoundingBox(xmin=100.0, ymin=100.0, xmax=200.0, ymax=200.0)
    return pb, pa, gt_box


@pytest.fixture
def eval_built_up_pair(tmp_path: Path) -> Tuple[Path, Path, BoundingBox]:
    """Known Synthetic Ground-Truth: New Construction / Built-up at [70, 70, 170, 170]."""
    pb = tmp_path / "eval_built_b.tif"
    pa = tmp_path / "eval_built_a.tif"

    h, w = 256, 256
    before = np.full((4, h, w), 800, dtype=np.uint16)
    # Bare soil in Before: Moderate Red (1200), Moderate NIR (1500) -> NDBI=-0.11
    before[0, 70:170, 70:170] = 1200
    before[1, 70:170, 70:170] = 1100
    before[2, 70:170, 70:170] = 1000
    before[3, 70:170, 70:170] = 1500

    after = np.full((4, h, w), 800, dtype=np.uint16)
    # Built-up concrete structure in After: Very High Red/Blue reflection (3500), Lower NIR (1200) -> NDBI=+0.48
    after[0, 70:170, 70:170] = 3500
    after[1, 70:170, 70:170] = 3000
    after[2, 70:170, 70:170] = 3600
    after[3, 70:170, 70:170] = 1200

    extratags = [
        (33550, 'd', 3, (10.0, 10.0, 0.0)),
        (33922, 'd', 6, (0.0, 0.0, 0.0, 500000.0, 6000000.0, 0.0)),
        (34735, 'h', 8, (1, 1, 0, 1, 3072, 0, 1, 32633)),
    ]

    tifffile.imwrite(pb, before, extratags=extratags, planarconfig='SEPARATE')
    tifffile.imwrite(pa, after, extratags=extratags, planarconfig='SEPARATE')

    gt_box = BoundingBox(xmin=70.0, ymin=70.0, xmax=170.0, ymax=170.0)
    return pb, pa, gt_box


def test_scientific_evaluation_vegetation_change(eval_vegetation_pair, tmp_data_dir):
    """Evaluates Precision, Recall, F1, and IoU for Vegetation Loss."""
    pb, pa, gt_box = eval_vegetation_pair
    engine = BiTemporalAnalysisEngine()

    res = engine.analyze_change(
        before_path=pb,
        after_path=pa,
        file_id_before="eval_veg_b",
        file_id_after="eval_veg_a",
        query="Where did vegetation decrease?"
    )

    assert res.status == "SUCCESS"
    assert res.dominant_change_type == ChangeType.VEGETATION_LOSS

    metrics = compute_evaluation_metrics(
        predicted_regions=res.change_regions,
        ground_truth_boxes=[gt_box],
        expected_type=ChangeType.VEGETATION_LOSS,
        iou_threshold=0.50
    )

    assert metrics["tp"] == 1
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["mean_iou"] >= 0.90


def test_scientific_evaluation_water_change(eval_water_pair, tmp_data_dir):
    """Evaluates Precision, Recall, F1, and IoU for Water Expansion."""
    pb, pa, gt_box = eval_water_pair
    engine = BiTemporalAnalysisEngine()

    res = engine.analyze_change(
        before_path=pb,
        after_path=pa,
        file_id_before="eval_water_b",
        file_id_after="eval_water_a",
        query="Did the water body expand?"
    )

    assert res.status == "SUCCESS"
    assert res.dominant_change_type == ChangeType.WATER_EXPANSION

    metrics = compute_evaluation_metrics(
        predicted_regions=res.change_regions,
        ground_truth_boxes=[gt_box],
        expected_type=ChangeType.WATER_EXPANSION,
        iou_threshold=0.50
    )

    assert metrics["tp"] == 1
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["mean_iou"] >= 0.90


def test_scientific_evaluation_built_up_change(eval_built_up_pair, tmp_data_dir):
    """Evaluates Precision, Recall, F1, and IoU for New Built-up Construction."""
    pb, pa, gt_box = eval_built_up_pair
    engine = BiTemporalAnalysisEngine()

    res = engine.analyze_change(
        before_path=pb,
        after_path=pa,
        file_id_before="eval_built_b",
        file_id_after="eval_built_a",
        query="Where did new construction happen?"
    )

    assert res.status == "SUCCESS"
    assert res.dominant_change_type == ChangeType.NEW_BUILT_UP

    metrics = compute_evaluation_metrics(
        predicted_regions=res.change_regions,
        ground_truth_boxes=[gt_box],
        expected_type=ChangeType.NEW_BUILT_UP,
        iou_threshold=0.50
    )

    assert metrics["tp"] == 1
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["mean_iou"] >= 0.90


def test_scientific_evaluation_overall_benchmark_report(
    eval_vegetation_pair,
    eval_water_pair,
    eval_built_up_pair,
    tmp_data_dir
):
    """
    Generates and asserts the quantitative scientific evaluation benchmark across all test pairs.
    """
    engine = BiTemporalAnalysisEngine()
    test_cases = [
        ("Vegetation Loss", eval_vegetation_pair, ChangeType.VEGETATION_LOSS, "Where did vegetation decrease?"),
        ("Water Expansion", eval_water_pair, ChangeType.WATER_EXPANSION, "Did the water body expand?"),
        ("New Built-Up", eval_built_up_pair, ChangeType.NEW_BUILT_UP, "Where are the new buildings?"),
    ]

    all_metrics = []

    for name, (pb, pa, gt_box), ctype, query in test_cases:
        res = engine.analyze_change(
            before_path=pb,
            after_path=pa,
            file_id_before=f"eval_{name}_b",
            file_id_after=f"eval_{name}_a",
            query=query
        )
        metrics = compute_evaluation_metrics(
            predicted_regions=res.change_regions,
            ground_truth_boxes=[gt_box],
            expected_type=ctype,
            iou_threshold=0.50
        )
        metrics["name"] = name
        all_metrics.append(metrics)

    avg_prec = float(np.mean([m["precision"] for m in all_metrics]))
    avg_rec  = float(np.mean([m["recall"] for m in all_metrics]))
    avg_f1   = float(np.mean([m["f1"] for m in all_metrics]))
    avg_iou  = float(np.mean([m["mean_iou"] for m in all_metrics]))

    logger.info("=== Phase 4 Scientific Evaluation Results ===")
    for m in all_metrics:
        logger.info(f"{m['name']}: TP={m['tp']}, FP={m['fp']}, FN={m['fn']}, Precision={m['precision']:.2f}, Recall={m['recall']:.2f}, F1={m['f1']:.2f}, IoU={m['mean_iou']:.2f}")
    logger.info(f"Overall Average: Precision={avg_prec:.2f}, Recall={avg_rec:.2f}, F1={avg_f1:.2f}, Mean IoU={avg_iou:.2f}")

    assert avg_prec >= 0.90
    assert avg_rec >= 0.90
    assert avg_f1 >= 0.90
    assert avg_iou >= 0.85
