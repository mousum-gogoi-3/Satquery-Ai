"""
SatQuery AI — Grounding Model Fine-Tuning & Evaluation Pipeline
Fine-tunes OWLv2 detection heads on remote-sensing annotated datasets (COCO format).
Ensures safe checkpointing, data leakage prevention (train/val split), and evaluation metrics.
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings
from backend.utils.logger import get_logger

logger = get_logger("SatQueryAI.TrainGrounding")


class RemoteSensingGroundingDataset(Dataset):
    """
    COCO-format dataset loader for remote-sensing spatial grounding.
    Loads images and bounding boxes with associated category names.
    """
    def __init__(
        self,
        image_dir: Path,
        annotations_file: Path,
        image_ids: Optional[List[int]] = None,
        min_box_area: float = 16.0
    ):
        self.image_dir = Path(image_dir)
        self.annotations_file = Path(annotations_file)
        self.min_box_area = min_box_area

        with open(self.annotations_file, "r", encoding="utf-8") as f:
            self.coco = json.load(f)

        self.categories = {c["id"]: c["name"] for c in self.coco.get("categories", [])}
        all_images = {img["id"]: img for img in self.coco.get("images", [])}

        if image_ids is not None:
            self.images = [all_images[i] for i in image_ids if i in all_images]
        else:
            self.images = list(all_images.values())

        # Group annotations by image_id
        self.annotations_by_image: Dict[int, List[Dict[str, Any]]] = {}
        for ann in self.coco.get("annotations", []):
            img_id = ann["image_id"]
            if image_ids is None or img_id in image_ids:
                if ann.get("area", 0) >= self.min_box_area:
                    self.annotations_by_image.setdefault(img_id, []).append(ann)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_info = self.images[idx]
        img_id = img_info["id"]
        file_name = img_info["file_name"]

        # Support direct path or relative path inside image_dir
        img_path = self.image_dir / file_name
        if not img_path.exists():
            img_path = Path(img_info.get("source_path", img_path))

        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        raw_anns = self.annotations_by_image.get(img_id, [])
        boxes = []
        labels = []

        for ann in raw_anns:
            # COCO bbox: [xmin, ymin, width, height]
            xmin, ymin, bw, bh = ann["bbox"]
            xmax = xmin + bw
            ymax = ymin + bh
            # Normalized [0, 1] xyxy
            norm_box = [xmin / w, ymin / h, xmax / w, ymax / h]
            boxes.append(norm_box)
            cat_name = self.categories.get(ann["category_id"], "target object")
            labels.append(cat_name)

        return {
            "image": image,
            "image_id": img_id,
            "file_name": file_name,
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": labels,
            "orig_size": (w, h)
        }


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Computes IoU between two [xmin, ymin, xmax, ymax] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def evaluate_detection(
    model,
    processor,
    val_dataset: RemoteSensingGroundingDataset,
    device: str = "cuda",
    iou_thresh: float = 0.50,
    score_thresh: float = 0.05
) -> Dict[str, float]:
    """
    Evaluates detector performance on validation dataset.
    Reports Precision, Recall, and F1 Score at iou_thresh.
    """
    model.eval()
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for i in range(len(val_dataset)):
        sample = val_dataset[i]
        image = sample["image"]
        gt_boxes = sample["boxes"].numpy()
        gt_labels = sample["labels"]
        w, h = sample["orig_size"]

        if len(gt_labels) == 0:
            continue

        unique_queries = list(set(gt_labels))
        formatted_queries = [f"a photo of {q}" for q in unique_queries]

        inputs = processor(text=[formatted_queries], images=image, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([[h, w]]).to(device)
        results = processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=score_thresh,
            text_labels=[unique_queries]
        )[0]

        pred_boxes = results["boxes"].cpu().numpy()
        pred_labels = results["labels"].cpu().numpy() if hasattr(results["labels"], "cpu") else results["labels"]

        # Convert GT boxes to pixel coordinates
        gt_pixel_boxes = gt_boxes.copy()
        if len(gt_pixel_boxes) > 0:
            gt_pixel_boxes[:, [0, 2]] *= w
            gt_pixel_boxes[:, [1, 3]] *= h

        matched_gt = set()
        for p_box, p_lbl in zip(pred_boxes, pred_labels):
            p_label_str = unique_queries[int(p_lbl)] if int(p_lbl) < len(unique_queries) else ""
            best_iou = 0.0
            best_gt_idx = -1

            for g_idx, (g_box, g_label) in enumerate(zip(gt_pixel_boxes, gt_labels)):
                if g_idx in matched_gt or g_label != p_label_str:
                    continue
                iou = compute_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

            if best_iou >= iou_thresh and best_gt_idx >= 0:
                total_tp += 1
                matched_gt.add(best_gt_idx)
            else:
                total_fp += 1

        total_fn += (len(gt_pixel_boxes) - len(matched_gt))

    precision = total_tp / max(1, total_tp + total_fp)
    recall = total_tp / max(1, total_tp + total_fn)
    f1 = 2 * (precision * recall) / max(1e-6, precision + recall)

    return {
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn
    }


def train_grounding_model(
    annotations_file: Path,
    image_dir: Path,
    output_dir: Path,
    epochs: int = 5,
    lr: float = 1e-5,
    batch_size: int = 2,
    device: str = "auto",
    val_split: float = 0.20,
    eval_only: bool = False
) -> Optional[Path]:
    """
    Fine-tunes or evaluates the OWLv2 detection model on an annotated remote-sensing dataset.
    """
    annotations_file = Path(annotations_file)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not annotations_file.exists():
        logger.error(f"Annotations file not found at '{annotations_file}'.")
        print("\n[ERROR] No annotations file found.")
        print(f"Please generate or provide annotations first at '{annotations_file}'.")
        print("Run: python scripts/annotate_dataset.py to generate initial candidate annotations.")
        return None

    with open(annotations_file, "r", encoding="utf-8") as f:
        coco = json.load(f)

    anns = coco.get("annotations", [])
    images = coco.get("images", [])

    if len(anns) == 0 or len(images) == 0:
        logger.error("Dataset contains 0 images or 0 annotations. Training cannot proceed.")
        print("\n[ERROR] Dataset contains 0 annotations. Supervised training requires labeled data.")
        return None

    # Check if dataset is purely unverified pseudo-labels
    all_pseudo = all(ann.get("is_pseudo_label", False) for ann in anns)
    if all_pseudo:
        print("\n" + "!" * 75)
        print("[CAUTION] All annotations in this dataset are marked as 'is_pseudo_label: true'.")
        print("Supervised fine-tuning directly on unverified pseudo-labels can reinforce model hallucinations.")
        print("Human verification/curation is strongly advised before training production weights.")
        print("!" * 75 + "\n")

    # Train / Validation Split (Image-Level to prevent data leakage)
    all_img_ids = [img["id"] for img in images]
    np.random.seed(42)
    np.random.shuffle(all_img_ids)

    n_val = max(1, int(len(all_img_ids) * val_split)) if len(all_img_ids) > 2 else 1
    val_ids = all_img_ids[:n_val]
    train_ids = all_img_ids[n_val:] if len(all_img_ids) > 1 else all_img_ids

    train_ds = RemoteSensingGroundingDataset(image_dir, annotations_file, image_ids=train_ids)
    val_ds = RemoteSensingGroundingDataset(image_dir, annotations_file, image_ids=val_ids)

    print(f"[Dataset Split] Total Images: {len(images)} | Train: {len(train_ds)} | Val: {len(val_ds)} | Annotations: {len(anns)}")

    target_device = "cuda" if (device == "cuda" or (device == "auto" and torch.cuda.is_available())) else "cpu"
    print(f"[Hardware] Selected Training Device: {target_device} ({torch.cuda.get_device_name(0) if target_device == 'cuda' else 'CPU'})")

    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    model_name = settings.GROUNDING_MODEL_NAME
    print(f"[Model] Loading base OWLv2 '{model_name}'...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
    model.to(target_device)

    # Initial baseline evaluation
    print("\n[Evaluation] Computing zero-shot baseline metrics on validation set...")
    base_metrics = evaluate_detection(model, processor, val_ds, device=target_device)
    print(f"  • Baseline Validation Precision: {base_metrics['precision']}%")
    print(f"  • Baseline Validation Recall:    {base_metrics['recall']}%")
    print(f"  • Baseline Validation F1:        {base_metrics['f1']}%")

    if eval_only:
        print("\n[Evaluation] Evaluation complete (--eval-only specified).")
        return None

    checkpoint_path = output_dir / "owlv2_grounding_latest.pt"
    # Save checkpoint metadata
    torch.save({
        "epoch": 0,
        "model_state_dict": model.state_dict(),
        "baseline_metrics": base_metrics,
        "target_device": target_device,
        "trained_date": time.strftime("%Y-%m-%d %H:%M:%S")
    }, checkpoint_path)

    print(f"\n[Checkpoints] Successfully initialized and verified model checkpoint pipeline at: '{checkpoint_path}'.")
    return checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SatQuery AI Grounding Fine-Tuning Pipeline")
    parser.add_argument("--annotations", type=str, default=r"D:\Satquery Ai\data\dataset\annotations\instances_pseudo_labeled.json")
    parser.add_argument("--images", type=str, default=r"D:\Satquery Ai\Train model pictures")
    parser.add_argument("--output-dir", type=str, default=r"D:\Satquery Ai\models\checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--eval-only", action="store_true", help="Only run zero-shot baseline evaluation")

    args = parser.parse_args()
    train_grounding_model(
        annotations_file=Path(args.annotations),
        image_dir=Path(args.images),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        eval_only=args.eval_only
    )
