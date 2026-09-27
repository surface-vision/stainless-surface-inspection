"""Rigorous evaluation and operating-point study for the strip-defect detector.

This module answers three separate questions and keeps them separate on purpose:

1. *How good is the detector?*  Standard COCO-style detection metrics (mAP50,
   mAP50-95, precision, recall, per-class AP50) computed by the ultralytics
   validator on the held-out test split, so the headline numbers are produced by
   the same code path the wider community uses rather than by anything written
   here.

2. *How good is it as a classifier of coils?*  NEU-DET images carry essentially
   one defect type each, so the model's dominant detection is an image-level
   prediction. That gives a confusion matrix, accuracy, macro-F1 and a per-class
   precision/recall/F1 table via scikit-learn -- the view a quality engineer
   reads.

3. *What does it cost to run?*  A confidence sweep on the **validation** split
   that reports, at every threshold, detection rate, precision, F1, false
   positives per image and the image-level false alarm rate; then an operating
   threshold chosen by an explicit mill cost model (a missed defect costs a coil
   downgrade, a false alarm costs an unnecessary stop or manual re-inspection).

Split discipline: the confidence threshold is tuned on ``val`` and *only* on
``val``. The test split is scored once, at the threshold that val chose, and is
never used to select anything.

Cheap by construction: the network is run once per split at the lowest sweep
threshold and every detection is cached, so the 19-point threshold sweep is pure
NumPy and costs no additional GPU time.

Usage:
    python src/evaluate.py                        # auto-resolve weights, test split
    python src/evaluate.py --weights models/run/weights/best.pt --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: this runs on the line PC and in CI, not on a desktop

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

try:  # works both as `python src/evaluate.py` and as `from src import evaluate`
    from .inference import (
        CLASS_COLORS, CLASS_NAMES, DEFAULT_IMGSZ, DEFECT_INFO, DefectDetector,
        resolve_device, resolve_weights,
    )
except ImportError:  # pragma: no cover - script execution path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from inference import (  # noqa: E402
        CLASS_COLORS, CLASS_NAMES, DEFAULT_IMGSZ, DEFECT_INFO, DefectDetector,
        resolve_device, resolve_weights,
    )

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "neu-det"
DATA_YAML = DATA_ROOT / "data.yaml"
REPORTS_DIR = PROJECT_ROOT / "reports"

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")

# Image-level label for a frame that carries no defect boxes at all. Kept out of
# the 0..n-1 class range so it can never be confused with a real class index.
NO_LABEL = -1
NO_LABEL_NAME = "(no labelled defect)"

# Localisation tolerance for calling a predicted box a hit. 0.5 is the industry
# default (Pascal VOC / COCO AP50) and is generous enough that a box which frames
# the defect well enough for an operator to act on it counts as a hit.
IOU_MATCH = 0.5

# The sweep the mill cares about. Below 0.05 the box count explodes without
# adding recall; above 0.95 nothing survives.
SWEEP_THRESHOLDS: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(1, 20))

# ---------------------------------------------------------------- cost model
#
# Units are "one unnecessary re-inspection". A false alarm costs 1.0 by
# definition; the ratio below says how many of those a single escaped defect is
# worth.
#
#   Escaped defect: the coil ships as prime, the defect is found downstream or by
#   the customer. Outcome is a prime-to-secondary downgrade on a ~20 t coil
#   (a per-tonne discount in the hundreds of dollars), plus claim handling and
#   the risk to the account. Order: thousands of dollars.
#
#   False alarm: the coil is flagged, the operator stops or slows the line and
#   re-inspects the surface manually, finds nothing, and releases it. Order:
#   minutes of operator time plus a slice of line availability -- low hundreds of
#   dollars at worst.
#
# A ratio of 12 is the conservative middle of that range. It is a *parameter*,
# not a truth: the report always prints the sensitivity of the chosen threshold
# across the plausible band so the quality department can move it.
DEFAULT_MISS_COST_RATIO = 12.0
SENSITIVITY_RATIOS: tuple[float, ...] = (3.0, 6.0, 12.0, 25.0, 50.0)

# A surface inspection system that lets more than one defective coil in ten pass
# as prime will not be accepted on the line, whatever its false alarm rate. The
# cost minimum is searched inside this constraint first.
DEFAULT_MIN_DETECTION_RATE = 0.90


# ---------------------------------------------------------------------------
# data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroundTruth:
    """Labels for one image, in pixel coordinates."""

    image_path: Path
    width: int
    height: int
    boxes: np.ndarray  # (N, 4) xyxy
    classes: np.ndarray  # (N,) int

    @property
    def image_class(self) -> int:
        """The single defect type the image is filed under.

        NEU-DET is a one-defect-per-image dataset, but a handful of images carry
        boxes of two types. Majority by instance count, ties broken by total
        boxed area, so the image-level label is deterministic.

        Returns ``NO_LABEL`` for an image with no labels at all. That never
        happens on NEU-DET, but it will the moment this is pointed at production
        frames, where clean strip is the majority case -- and silently filing a
        clean coil under class 0 would poison every image-level number here.
        """
        if self.classes.size == 0:
            return NO_LABEL
        counts = np.bincount(self.classes, minlength=len(CLASS_NAMES))
        top = int(counts.max())
        candidates = np.flatnonzero(counts == top)
        if candidates.size == 1:
            return int(candidates[0])
        areas = _box_areas(self.boxes)
        by_area = [float(areas[self.classes == c].sum()) for c in candidates]
        return int(candidates[int(np.argmax(by_area))])

    @property
    def is_multi_class(self) -> bool:
        return bool(np.unique(self.classes).size > 1)


@dataclass(frozen=True)
class CachedPrediction:
    """Every box the detector produced for one image at the cache threshold."""

    image_path: Path
    boxes: np.ndarray  # (M, 4) xyxy
    confidences: np.ndarray  # (M,)
    classes: np.ndarray  # (M,) int


@dataclass(frozen=True)
class MatchResult:
    """Greedy class-aware matching of one image's predictions against its labels."""

    pred_is_tp: np.ndarray  # (M,) bool, aligned to the filtered prediction order
    gt_matched: np.ndarray  # (N,) bool


@dataclass
class SweepPoint:
    """Every operating statistic at one confidence threshold."""

    threshold: float
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    fp_per_image: float
    false_alarm_rate: float  # images with >= 1 spurious box
    defect_detection_rate: float  # images where the real defect was found
    class_accuracy: float  # images whose dominant class is right
    images_flagged: float  # images with >= 1 box of any kind
    expected_cost_per_image: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": round(self.threshold, 3),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "fp_per_image": round(self.fp_per_image, 4),
            "false_alarm_rate": round(self.false_alarm_rate, 4),
            "defect_detection_rate": round(self.defect_detection_rate, 4),
            "class_accuracy": round(self.class_accuracy, 4),
            "images_flagged_rate": round(self.images_flagged, 4),
            "expected_cost_per_image": round(self.expected_cost_per_image, 4),
        }


@dataclass
class OperatingPoint:
    """The recommended threshold plus the reasoning that produced it."""

    threshold: float
    point: SweepPoint
    miss_cost_ratio: float
    min_detection_rate: float
    constraint_met: bool
    at_sweep_edge: bool
    f1_optimal_threshold: float
    rationale: str
    sensitivity: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------


def class_label(class_id: int) -> str:
    """Name for a class index, safe for ids the label file or head may invent."""
    if class_id == NO_LABEL:
        return NO_LABEL_NAME
    if 0 <= class_id < len(CLASS_NAMES):
        return CLASS_NAMES[class_id]
    return f"class_{class_id}"


def _box_areas(boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float64)
    return np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two xyxy box sets -> (len(a), len(b))."""
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0.0, None), axis=2)
    union = _box_areas(a)[:, None] + _box_areas(b)[None, :] - inter
    return np.where(union > 0.0, inter / np.maximum(union, 1e-12), 0.0)


def match_image(
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    pred_boxes: np.ndarray,
    pred_conf: np.ndarray,
    pred_classes: np.ndarray,
    iou_thr: float = IOU_MATCH,
) -> MatchResult:
    """Greedy, class-aware, confidence-ordered matching (the COCO convention).

    Highest-confidence prediction claims the best still-unclaimed ground-truth box
    of the same class above the IoU floor. Everything else is a false positive;
    every unclaimed label is a false negative. A box in the right place with the
    wrong class label is a false positive *and* leaves a false negative behind,
    which is correct: the operator was sent the wrong root cause.
    """
    n_pred, n_gt = len(pred_boxes), len(gt_boxes)
    pred_is_tp = np.zeros(n_pred, dtype=bool)
    gt_matched = np.zeros(n_gt, dtype=bool)
    if n_pred == 0 or n_gt == 0:
        return MatchResult(pred_is_tp=pred_is_tp, gt_matched=gt_matched)

    ious = iou_matrix(pred_boxes, gt_boxes)
    same_class = pred_classes[:, None] == gt_classes[None, :]
    eligible = np.where(same_class & (ious >= iou_thr), ious, -1.0)

    for pred_idx in np.argsort(-pred_conf, kind="stable"):
        row = np.where(gt_matched, -1.0, eligible[pred_idx])
        best = int(np.argmax(row))
        if row[best] >= iou_thr:
            pred_is_tp[pred_idx] = True
            gt_matched[best] = True
    return MatchResult(pred_is_tp=pred_is_tp, gt_matched=gt_matched)


# ---------------------------------------------------------------------------
# data loading and prediction caching
# ---------------------------------------------------------------------------


def load_ground_truth(split: str, data_root: Path = DATA_ROOT) -> list[GroundTruth]:
    """Read a YOLO-format split into pixel-space ground truth records."""
    images_dir = data_root / split / "images"
    labels_dir = data_root / split / "labels"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {images_dir}")

    records: list[GroundTruth] = []
    for image_path in sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Unreadable image in split {split!r}: {image_path}")
        height, width = image.shape[:2]

        boxes: list[list[float]] = []
        classes: list[int] = []
        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.is_file():
            for line in label_path.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls, cx, cy, bw, bh = int(float(parts[0])), *(float(v) for v in parts[1:5])
                if not 0 <= cls < len(CLASS_NAMES):
                    # A class id outside the dataset's own range is a corrupt label
                    # file, not something to average over. Fail here, where the file
                    # can be named, rather than deep inside the failure summary.
                    raise ValueError(
                        f"Label {label_path} declares class id {cls}, but the dataset has "
                        f"{len(CLASS_NAMES)} classes (0-{len(CLASS_NAMES) - 1})."
                    )
                x1 = (cx - bw / 2.0) * width
                y1 = (cy - bh / 2.0) * height
                boxes.append([x1, y1, x1 + bw * width, y1 + bh * height])
                classes.append(cls)

        records.append(
            GroundTruth(
                image_path=image_path,
                width=width,
                height=height,
                boxes=np.asarray(boxes, dtype=np.float64).reshape(-1, 4),
                classes=np.asarray(classes, dtype=np.int64),
            )
        )
    if not records:
        raise RuntimeError(f"No images found in {images_dir}")
    return records


def cache_predictions(
    detector: DefectDetector, records: Sequence[GroundTruth], batch_size: int = 16
) -> list[CachedPrediction]:
    """Run the detector once and keep every box, so the sweep needs no GPU.

    The detector is constructed at the *lowest* threshold in the sweep, so
    filtering the cache by ``confidence >= t`` reproduces exactly what the
    deployed detector would have returned at threshold ``t`` -- NMS is applied
    before the confidence filter in both cases.
    """
    results = detector.predict_batch([r.image_path for r in records], batch_size=batch_size)
    cached: list[CachedPrediction] = []
    for record, result in zip(records, results):
        if result.detections:
            boxes = np.asarray([d.bbox_xyxy for d in result.detections], dtype=np.float64)
            conf = np.asarray([d.confidence for d in result.detections], dtype=np.float64)
            classes = np.asarray([d.class_id for d in result.detections], dtype=np.int64)
        else:
            boxes = np.zeros((0, 4), dtype=np.float64)
            conf = np.zeros((0,), dtype=np.float64)
            classes = np.zeros((0,), dtype=np.int64)
        cached.append(
            CachedPrediction(image_path=record.image_path, boxes=boxes, confidences=conf, classes=classes)
        )
    return cached


def filter_prediction(pred: CachedPrediction, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The subset of a cached prediction that survives a confidence threshold."""
    keep = pred.confidences >= threshold
    return pred.boxes[keep], pred.confidences[keep], pred.classes[keep]


# ---------------------------------------------------------------------------
# 1. standard detection metrics (ultralytics validator)
# ---------------------------------------------------------------------------


def resolve_train_imgsz(weights: Path, default: int = 640) -> int:
    """Evaluate at the size the model was trained at, not at a guess.

    A detector fine-tuned at 320 px and validated at 640 px is being asked to
    generalise across a scale it never saw, and its mAP drops for a reason that
    has nothing to do with the model. The run directory's `args.yaml` records the
    training size; the checkpoint's own `train_args` is the fallback, and only if
    both are unavailable do we assume `default`.
    """
    args_yaml = weights.parent.parent / "args.yaml"
    if args_yaml.is_file():
        try:
            import yaml

            recorded = yaml.safe_load(args_yaml.read_text()) or {}
            if isinstance(recorded.get("imgsz"), int):
                return int(recorded["imgsz"])
        except Exception:  # noqa: BLE001 - a malformed args.yaml must not stop evaluation
            pass
    try:
        import torch

        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        recorded = (ckpt.get("train_args") or {}) if isinstance(ckpt, dict) else {}
        if isinstance(recorded.get("imgsz"), int):
            return int(recorded["imgsz"])
    except Exception:  # noqa: BLE001
        pass
    return default


def run_validator(
    weights: Path,
    split: str,
    device: str,
    imgsz: int,
    out_dir: Path,
    batch: int = 16,
    workers: int = 2,
) -> dict[str, Any]:
    """COCO-style detection metrics from the ultralytics validator.

    Deliberately left on the validator's own protocol defaults (conf 0.001, NMS
    IoU 0.7, max_det 300) so mAP is directly comparable with published NEU-DET
    numbers. The deployment thresholds are a separate question, handled by the
    sweep below.
    """
    from ultralytics import YOLO  # deferred: heavy import, and the caller may not need it

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(DATA_YAML),
        split=split,
        imgsz=imgsz,
        device=device,
        batch=batch,
        workers=workers,
        plots=True,
        verbose=False,
        project=str(out_dir / "ultralytics"),
        name=f"val_{split}",
        exist_ok=True,
    )

    names = dict(metrics.names)
    class_index = [int(c) for c in metrics.ap_class_index]
    per_class: dict[str, dict[str, float]] = {}
    for row, cls in enumerate(class_index):
        p, r, ap50, ap = metrics.box.class_result(row)
        per_class[names.get(cls, f"class_{cls}")] = {
            "AP50": float(ap50),
            "AP50_95": float(ap),
            "precision": float(p),
            "recall": float(r),
            "instances": int(metrics.nt_per_class[cls]) if metrics.nt_per_class is not None else 0,
            "images": int(metrics.nt_per_image[cls]) if metrics.nt_per_image is not None else 0,
        }

    curves = _extract_pr_curves(metrics, names, class_index)
    del model
    return {
        "split": split,
        "imgsz": imgsz,
        "protocol": {"conf": 0.001, "nms_iou": 0.7, "max_det": 300, "source": "ultralytics validator defaults"},
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "fitness": float(metrics.fitness),
        "per_class": per_class,
        "speed_ms": {k: float(v) for k, v in metrics.speed.items()},
        "save_dir": str(getattr(metrics, "save_dir", out_dir / "ultralytics" / f"val_{split}")),
        "_pr_curves": curves,
    }


def _extract_pr_curves(metrics: Any, names: dict[int, str], class_index: list[int]) -> dict[str, Any]:
    """Pull the validator's own per-class PR curves out of the metrics object."""
    try:
        recall_grid, precision_rows, _, _ = metrics.curves_results[0]
        precision_rows = np.asarray(precision_rows, dtype=np.float64).reshape(len(class_index), -1)
        return {
            "recall": np.asarray(recall_grid, dtype=np.float64).tolist(),
            "precision": {
                names.get(cls, f"class_{cls}"): precision_rows[row].tolist()
                for row, cls in enumerate(class_index)
            },
        }
    except (AttributeError, IndexError, ValueError):
        # A model that predicts nothing leaves the curve arrays empty; the rest of
        # the report is still valid, so degrade instead of failing.
        return {"recall": [], "precision": {}}


# ---------------------------------------------------------------------------
# 2. image-level classification view
# ---------------------------------------------------------------------------

NO_DETECTION_LABEL = "(no detection)"


def image_level_report(
    records: Sequence[GroundTruth], predictions: Sequence[CachedPrediction], threshold: float
) -> dict[str, Any]:
    """Treat each image as a single-label classification problem.

    Prediction = the class of the highest-confidence surviving box, which is what
    `InferenceResult.dominant_class` reports to the UI. Images where nothing
    survives get an explicit "(no detection)" column rather than being dropped:
    a coil the system waved through is a classification failure, and hiding it
    would flatter the accuracy number.

    Frames carrying no labelled defect have no true class, so they are kept out
    of the confusion matrix and counted separately: for a clean frame the only
    question is whether anything was flagged at all, and folding it into a
    six-class accuracy would be meaningless in either direction.
    """
    n_classes = len(CLASS_NAMES)
    y_true: list[int] = []
    y_pred: list[int] = []  # n_classes encodes "no detection"
    background_images = 0
    background_flagged = 0

    for record, pred in zip(records, predictions):
        boxes, conf, classes = filter_prediction(pred, threshold)
        predicted = int(classes[int(np.argmax(conf))]) if len(conf) else n_classes
        if record.image_class == NO_LABEL:
            background_images += 1
            background_flagged += int(len(conf) > 0)
            continue
        y_true.append(record.image_class)
        y_pred.append(predicted)

    if not y_true:
        raise RuntimeError(
            "Every image in this split is unlabelled; there is no image-level "
            "classification to report."
        )

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    labels = list(range(n_classes + 1))
    matrix = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)[:n_classes]  # GT is never "no detection"

    report = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=list(range(n_classes)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        name: {
            "precision": float(report[name]["precision"]),
            "recall": float(report[name]["recall"]),
            "f1": float(report[name]["f1-score"]),
            "support": int(report[name]["support"]),
        }
        for name in CLASS_NAMES
    }

    return {
        "threshold": threshold,
        "n_images": int(len(y_true_arr)),
        "background_images": background_images,
        "background_images_flagged": background_flagged,
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, labels=list(range(n_classes)), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true_arr, y_pred_arr, labels=list(range(n_classes)), average="weighted", zero_division=0)),
        "no_detection_images": int((y_pred_arr == n_classes).sum()),
        "multi_class_images": int(sum(r.is_multi_class for r in records)),
        "labels": CLASS_NAMES + [NO_DETECTION_LABEL],
        "confusion_matrix": matrix.astype(int).tolist(),
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# 3. false alarm analysis and operating point
# ---------------------------------------------------------------------------


def sweep_confidence(
    records: Sequence[GroundTruth],
    predictions: Sequence[CachedPrediction],
    thresholds: Sequence[float] = SWEEP_THRESHOLDS,
    iou_thr: float = IOU_MATCH,
) -> list[SweepPoint]:
    """Recompute every operating statistic at each confidence threshold.

    Two levels of accounting, because a mill needs both:

    * box level -- precision / recall / F1 / false positives per image, the
      numbers that say whether the boxes drawn on the HMI are trustworthy;
    * image level -- ``defect_detection_rate`` (did the real defect get found on
      this coil at all) and ``false_alarm_rate`` (did this coil get flagged for
      something that is not there), which are the two events the cost model
      prices.
    """
    n_images = len(records)
    total_gt = int(sum(len(r.boxes) for r in records))
    points: list[SweepPoint] = []

    for threshold in thresholds:
        tp = fp = fn = 0
        images_with_fp = 0
        images_detected = 0
        images_flagged = 0
        class_correct = 0

        for record, pred in zip(records, predictions):
            boxes, conf, classes = filter_prediction(pred, threshold)
            match = match_image(record.boxes, record.classes, boxes, conf, classes, iou_thr)

            image_tp = int(match.pred_is_tp.sum())
            image_fp = int(len(boxes) - image_tp)
            tp += image_tp
            fp += image_fp
            fn += int((~match.gt_matched).sum())

            images_with_fp += int(image_fp > 0)
            images_detected += int(image_tp > 0)
            images_flagged += int(len(boxes) > 0)
            if len(conf):
                class_correct += int(int(classes[int(np.argmax(conf))]) == record.image_class)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / total_gt if total_gt else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        points.append(
            SweepPoint(
                threshold=float(threshold),
                tp=tp,
                fp=fp,
                fn=fn,
                precision=precision,
                recall=recall,
                f1=f1,
                fp_per_image=fp / n_images,
                false_alarm_rate=images_with_fp / n_images,
                defect_detection_rate=images_detected / n_images,
                class_accuracy=class_correct / n_images,
                images_flagged=images_flagged / n_images,
            )
        )
    return points


def price_sweep(points: Sequence[SweepPoint], miss_cost_ratio: float) -> None:
    """Attach the expected mill cost per inspected image to each sweep point.

    cost = ratio * P(defect escapes) + 1.0 * P(unnecessary re-inspection)

    Both terms are image-level probabilities, because both consequences are
    incurred per coil, not per box: one spurious box on a coil costs one
    re-inspection whether the model drew one or thirty.
    """
    for point in points:
        miss_rate = 1.0 - point.defect_detection_rate
        point.expected_cost_per_image = miss_cost_ratio * miss_rate + point.false_alarm_rate


def recommend_threshold(
    points: Sequence[SweepPoint],
    miss_cost_ratio: float = DEFAULT_MISS_COST_RATIO,
    min_detection_rate: float = DEFAULT_MIN_DETECTION_RATE,
) -> OperatingPoint:
    """Choose the deployment threshold: cheapest point that still catches defects.

    Rule, in order:
      1. keep only thresholds whose image-level detection rate clears the floor
         the line will accept;
      2. among those, take the minimum expected cost;
      3. ties break towards the higher threshold (fewer boxes on the HMI).
    If nothing clears the floor the constraint is dropped, the unconstrained cost
    minimum is returned, and ``constraint_met`` is False so the report can say so
    instead of quietly shipping a threshold nobody would accept.
    """
    if not points:
        raise ValueError("Cannot recommend a threshold from an empty sweep")

    price_sweep(points, miss_cost_ratio)
    eligible = [p for p in points if p.defect_detection_rate >= min_detection_rate]
    constraint_met = bool(eligible)
    pool = eligible if eligible else list(points)
    best = min(pool, key=lambda p: (p.expected_cost_per_image, -p.threshold))
    f1_best = max(points, key=lambda p: (p.f1, p.threshold))

    # Sensitivity is deliberately computed over the *whole* sweep rather than over
    # the eligible pool: the detection floor does not move with the cost ratio, so
    # restricting it would report the same threshold at every ratio and say
    # nothing. The unconstrained minimum shows where the economics alone push the
    # knob, and the flag says whether that point would clear the floor anyway.
    sensitivity: list[dict[str, Any]] = []
    for ratio in SENSITIVITY_RATIOS:
        scored = [(ratio * (1.0 - p.defect_detection_rate) + p.false_alarm_rate, p) for p in points]
        cost, point = min(scored, key=lambda item: (item[0], -item[1].threshold))
        sensitivity.append(
            {
                "miss_cost_ratio": ratio,
                "threshold": round(point.threshold, 3),
                "expected_cost_per_image": round(cost, 4),
                "defect_detection_rate": round(point.defect_detection_rate, 4),
                "false_alarm_rate": round(point.false_alarm_rate, 4),
                "meets_detection_floor": bool(point.defect_detection_rate >= min_detection_rate),
            }
        )

    at_sweep_edge = best.threshold in (min(p.threshold for p in points), max(p.threshold for p in points))
    edge_note = (
        f" The optimum sits at the {'bottom' if best.threshold <= min(p.threshold for p in points) else 'top'} "
        f"edge of the swept 0.05-0.95 range, so the true minimum may lie outside it; read this as "
        f"'as far as the sweep allows', not as an interior optimum."
        if at_sweep_edge
        else ""
    )

    # Spell out what following F1 instead would actually cost the mill, in coils.
    escape_ratio = (1.0 - f1_best.defect_detection_rate) / max(1.0 - best.defect_detection_rate, 1e-9)
    alternative = (
        ""
        if abs(f1_best.threshold - best.threshold) < 1e-9
        else (
            f" The F1-optimal threshold {f1_best.threshold:.2f} would lift box precision from "
            f"{best.precision:.2f} to {f1_best.precision:.2f} and cut false alarms from "
            f"{best.false_alarm_rate:.1%} to {f1_best.false_alarm_rate:.1%}, but its detection rate falls "
            f"from {best.defect_detection_rate:.1%} to {f1_best.defect_detection_rate:.1%} -- "
            f"{escape_ratio:.1f}x as many coils shipping with an undetected defect. That is the trade F1 "
            f"hides by weighting a miss and a false alarm equally."
        )
    )

    if constraint_met:
        rationale = (
            f"At conf={best.threshold:.2f} the detector finds the real defect on "
            f"{best.defect_detection_rate:.1%} of defective images while raising a spurious box on "
            f"{best.false_alarm_rate:.1%} of them ({best.fp_per_image:.2f} false boxes per image). "
            f"Pricing one escaped defect at {miss_cost_ratio:g} unnecessary re-inspections, that is an "
            f"expected {best.expected_cost_per_image:.3f} re-inspection-equivalents per inspected coil, "
            f"the minimum over every threshold meeting the {min_detection_rate:.0%} detection floor the "
            f"line will accept."
            + alternative
            + edge_note
        )
    else:
        rationale = (
            f"WARNING: no threshold in the sweep reaches the {min_detection_rate:.0%} image-level detection "
            f"floor -- the best is {max(p.defect_detection_rate for p in points):.1%}. The constraint was "
            f"dropped and conf={best.threshold:.2f} is simply the unconstrained cost minimum "
            f"({best.expected_cost_per_image:.3f} re-inspection-equivalents per coil at a "
            f"{miss_cost_ratio:g}:1 miss:false-alarm ratio). This is a statement about the checkpoint, not "
            f"about the threshold: no operating point of an undertrained detector is deployable, and the "
            f"model must be retrained before this recommendation means anything."
            + edge_note
        )

    return OperatingPoint(
        threshold=best.threshold,
        point=best,
        miss_cost_ratio=miss_cost_ratio,
        min_detection_rate=min_detection_rate,
        constraint_met=constraint_met,
        at_sweep_edge=at_sweep_edge,
        f1_optimal_threshold=f1_best.threshold,
        rationale=rationale,
        sensitivity=sensitivity,
    )


# ---------------------------------------------------------------------------
# 4. plots
# ---------------------------------------------------------------------------

_GRID = {"color": "#d8dde3", "linewidth": 0.7, "alpha": 0.9}


def _class_rgb(name: str) -> tuple[float, float, float]:
    r, g, b = CLASS_COLORS.get(name, (120, 120, 120))
    return (r / 255.0, g / 255.0, b / 255.0)


def plot_confusion_matrix(report: dict[str, Any], out_path: Path, split: str) -> Path:
    """Row-normalised image-level confusion matrix with raw counts overlaid."""
    matrix = np.asarray(report["confusion_matrix"], dtype=np.float64)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(matrix, np.maximum(row_sums, 1.0))
    labels = report["labels"]

    fig, ax = plt.subplots(figsize=(9.0, 6.6))
    image = ax.imshow(normalised, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("predicted (dominant detection)")
    ax.set_ylabel("ground truth")
    ax.set_title(
        f"Image-level confusion matrix -- {split} split @ conf={report['threshold']:.2f}\n"
        f"accuracy {report['accuracy']:.1%} | macro-F1 {report['macro_f1']:.3f} | "
        f"{report['no_detection_images']} of {report['n_images']} images produced no detection",
        fontsize=11,
    )
    # The "no detection" column is a different kind of outcome from a class
    # confusion; separate it visually so nobody reads it as a seventh class.
    ax.axvline(len(CLASS_NAMES) - 0.5, color="#c0392b", linewidth=2.0)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            count = int(matrix[i, j])
            if count == 0:
                continue
            ax.text(
                j,
                i,
                f"{count}\n{normalised[i, j]:.0%}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if normalised[i, j] > 0.55 else "#1b2733",
            )
    fig.colorbar(image, ax=ax, label="fraction of the true class")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_pr_curves(detection_metrics: dict[str, Any], out_path: Path, split: str) -> Path | None:
    """Per-class precision-recall at IoU 0.5, straight from the validator."""
    curves = detection_metrics.get("_pr_curves") or {}
    recall = np.asarray(curves.get("recall", []), dtype=np.float64)
    precision_by_class = curves.get("precision", {})
    if recall.size == 0 or not precision_by_class:
        return None

    fig, ax = plt.subplots(figsize=(7.8, 6.0))
    for name, values in precision_by_class.items():
        ap50 = detection_metrics["per_class"].get(name, {}).get("AP50", 0.0)
        ax.plot(recall, np.asarray(values, dtype=np.float64), color=_class_rgb(name), linewidth=1.9,
                label=f"{name}  AP50={ap50:.3f}")

    stacked = np.stack([np.asarray(v, dtype=np.float64) for v in precision_by_class.values()])
    ax.plot(recall, stacked.mean(axis=0), color="#111820", linewidth=2.8, linestyle="--",
            label=f"all classes  mAP50={detection_metrics['mAP50']:.3f}")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(f"Precision-recall per defect class (IoU 0.5) -- {split} split")
    ax.grid(True, **_GRID)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_f1_vs_threshold(points: Sequence[SweepPoint], operating: OperatingPoint, out_path: Path, split: str) -> Path:
    """F1, precision and recall against the confidence knob the operator turns."""
    thresholds = [p.threshold for p in points]
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    ax.plot(thresholds, [p.f1 for p in points], color="#111820", linewidth=2.6, marker="o", markersize=4, label="F1 (box level)")
    ax.plot(thresholds, [p.precision for p in points], color="#2f7ed8", linewidth=1.8, marker="s", markersize=3.5, label="precision")
    ax.plot(thresholds, [p.recall for p in points], color="#c0392b", linewidth=1.8, marker="^", markersize=3.5, label="recall")
    ax.plot(thresholds, [p.defect_detection_rate for p in points], color="#1f9d55", linewidth=1.8, linestyle="--",
            label="image-level defect detection rate")

    ax.axvline(operating.threshold, color="#e67e22", linewidth=2.0, linestyle=":")
    ax.annotate(
        f"recommended conf={operating.threshold:.2f}",
        xy=(operating.threshold, 0.5),
        xytext=(6, 0),
        textcoords="offset points",
        rotation=90,
        va="center",
        fontsize=9,
        color="#a35b00",
    )
    ax.set_xlabel("confidence threshold")
    ax.set_ylabel("score")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Operating curves vs confidence threshold -- {split} split (tuning)")
    ax.grid(True, **_GRID)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_false_alarms_vs_recall(
    points: Sequence[SweepPoint], operating: OperatingPoint, out_path: Path, split: str
) -> Path:
    """The trade the mill actually signs off: nuisance boxes bought per point of recall."""
    ordered = sorted(points, key=lambda p: p.recall)
    recalls = [p.recall for p in ordered]

    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    ax.plot(recalls, [p.fp_per_image for p in ordered], color="#c0392b", linewidth=2.4, marker="o", markersize=4)
    ax.set_xlabel("recall (fraction of labelled defects found)")
    ax.set_ylabel("false positives per image", color="#c0392b")
    ax.tick_params(axis="y", labelcolor="#c0392b")
    ax.grid(True, **_GRID)

    twin = ax.twinx()
    twin.plot(recalls, [p.false_alarm_rate for p in ordered], color="#2f7ed8", linewidth=2.0, linestyle="--", marker="s", markersize=3.5)
    twin.set_ylabel("image-level false alarm rate", color="#2f7ed8")
    twin.tick_params(axis="y", labelcolor="#2f7ed8")
    twin.set_ylim(-0.02, 1.02)

    for point in ordered[::2]:
        ax.annotate(f"{point.threshold:.2f}", xy=(point.recall, point.fp_per_image), xytext=(3, 5),
                    textcoords="offset points", fontsize=7.5, color="#555f6b")

    ax.plot([operating.point.recall], [operating.point.fp_per_image], marker="*", markersize=18,
            color="#e67e22", linestyle="none", label=f"recommended conf={operating.threshold:.2f}")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(
        f"False alarm cost of recall -- {split} split\n"
        "solid: spurious boxes per image (left)   dashed: images carrying at least one spurious box (right)",
        fontsize=10.5,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 5. per-class error gallery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FailureCase:
    """One panel of the error gallery."""

    kind: str  # "miss" | "false_positive"
    image_path: Path
    box: tuple[float, float, float, float]
    score: float  # miss: best confidence the model gave the class; FP: the box confidence
    class_name: str


def collect_failures(
    records: Sequence[GroundTruth],
    predictions: Sequence[CachedPrediction],
    threshold: float,
    per_kind: int = 4,
    iou_thr: float = IOU_MATCH,
) -> dict[str, dict[str, list[FailureCase]]]:
    """Rank each class's failures so the gallery shows the worst, not a random sample.

    Misses are ranked by how little confidence the model put on the right class
    anywhere in that image -- lowest first, i.e. the defects it was most blind
    to. False positives are ranked by confidence, highest first: a confident
    wrong box is what destroys operator trust, a marginal one is just noise. At
    most one case per image per bucket, so a single pathological image cannot
    fill the montage.
    """
    buckets: dict[str, dict[str, list[FailureCase]]] = {
        name: {"miss": [], "false_positive": []} for name in CLASS_NAMES
    }

    for record, pred in zip(records, predictions):
        boxes, conf, classes = filter_prediction(pred, threshold)
        match = match_image(record.boxes, record.classes, boxes, conf, classes, iou_thr)

        seen_miss: set[str] = set()
        for gt_idx in np.flatnonzero(~match.gt_matched):
            cls = int(record.classes[gt_idx])
            name = class_label(cls)
            if name in seen_miss or name not in buckets:
                continue
            seen_miss.add(name)
            same_class_conf = conf[classes == cls]
            buckets[name]["miss"].append(
                FailureCase(
                    kind="miss",
                    image_path=record.image_path,
                    box=tuple(float(v) for v in record.boxes[gt_idx]),
                    score=float(same_class_conf.max()) if same_class_conf.size else 0.0,
                    class_name=name,
                )
            )

        seen_fp: set[str] = set()
        for pred_idx in np.argsort(-conf, kind="stable"):
            if match.pred_is_tp[pred_idx]:
                continue
            cls = int(classes[pred_idx])
            name = class_label(cls)
            if name in seen_fp or name not in buckets:
                continue
            seen_fp.add(name)
            buckets[name]["false_positive"].append(
                FailureCase(
                    kind="false_positive",
                    image_path=record.image_path,
                    box=tuple(float(v) for v in boxes[pred_idx]),
                    score=float(conf[pred_idx]),
                    class_name=name,
                )
            )

    for name in buckets:
        buckets[name]["miss"] = sorted(buckets[name]["miss"], key=lambda c: c.score)[:per_kind]
        buckets[name]["false_positive"] = sorted(
            buckets[name]["false_positive"], key=lambda c: -c.score
        )[:per_kind]
    return buckets


def per_class_failure_summary(
    records: Sequence[GroundTruth],
    predictions: Sequence[CachedPrediction],
    threshold: float,
    iou_thr: float = IOU_MATCH,
) -> dict[str, dict[str, Any]]:
    """Where each class actually fails, at the deployed threshold.

    Counts are over instances, not images, and the false positives are attributed
    to the true class of the image they landed on -- which turns "213 false
    positives" into "the model paints crazing over rolled-in scale", the form a
    metallurgist can act on.
    """
    summary: dict[str, dict[str, Any]] = {
        name: {
            "instances": 0,
            "detected": 0,
            "missed": 0,
            "false_positives": 0,
            "base_severity": DEFECT_INFO[name]["severity"],
            "_fp_on": {},
        }
        for name in CLASS_NAMES
    }

    def _entry(name: str) -> dict[str, Any]:
        """Bucket for a class the head produced that the dataset does not define.

        It gets a visible row rather than being dropped, so a head/dataset
        mismatch shows up in the report instead of vanishing.
        """
        if name not in summary:
            summary[name] = {
                "instances": 0,
                "detected": 0,
                "missed": 0,
                "false_positives": 0,
                "base_severity": DEFECT_INFO.get(name, {}).get("severity", "unknown"),
                "_fp_on": {},
            }
        return summary[name]

    for record, pred in zip(records, predictions):
        boxes, conf, classes = filter_prediction(pred, threshold)
        match = match_image(record.boxes, record.classes, boxes, conf, classes, iou_thr)
        true_image_class = class_label(record.image_class)

        for gt_idx, cls in enumerate(record.classes):
            name = class_label(int(cls))
            _entry(name)["instances"] += 1
            if match.gt_matched[gt_idx]:
                summary[name]["detected"] += 1
            else:
                summary[name]["missed"] += 1

        for pred_idx, cls in enumerate(classes):
            if match.pred_is_tp[pred_idx]:
                continue
            name = class_label(int(cls))
            entry = _entry(name)
            entry["false_positives"] += 1
            entry["_fp_on"][true_image_class] = entry["_fp_on"].get(true_image_class, 0) + 1

    for name, stats in summary.items():
        fp_on = stats.pop("_fp_on")
        stats["recall"] = stats["detected"] / stats["instances"] if stats["instances"] else 0.0
        stats["fp_landed_on"] = dict(sorted(fp_on.items(), key=lambda kv: -kv[1])[:3])
    return summary


def _draw_panel(ax: plt.Axes, case: FailureCase, gt_lookup: dict[Path, GroundTruth]) -> None:
    image = cv2.cvtColor(cv2.imread(str(case.image_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    ax.imshow(image)

    record = gt_lookup[case.image_path]
    for box, cls in zip(record.boxes, record.classes):
        ax.add_patch(
            Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                      fill=False, edgecolor="#39ff88", linewidth=1.0, linestyle=":")
        )

    x1, y1, x2, y2 = case.box
    highlight = "#ff2d55" if case.kind == "miss" else _class_rgb(case.class_name)
    ax.add_patch(
        Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=highlight, linewidth=2.4)
    )

    caption = (
        f"MISSED  best conf {case.score:.2f}" if case.kind == "miss" else f"FALSE POS  conf {case.score:.2f}"
    )
    ax.set_title(f"{caption}\n{case.image_path.name}", fontsize=7.5, color="#1b2733")
    ax.set_xticks([])
    ax.set_yticks([])


def build_error_gallery(
    buckets: dict[str, dict[str, list[FailureCase]]],
    records: Sequence[GroundTruth],
    out_dir: Path,
    threshold: float,
    per_kind: int = 4,
) -> list[Path]:
    """One montage per class: worst misses on top, most confident false positives below."""
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_lookup = {r.image_path: r for r in records}
    written: list[Path] = []

    for class_id, name in enumerate(CLASS_NAMES):
        misses = buckets[name]["miss"]
        false_positives = buckets[name]["false_positive"]
        if not misses and not false_positives:
            continue

        fig, axes = plt.subplots(2, per_kind, figsize=(2.6 * per_kind, 6.4), layout="constrained")
        axes = np.atleast_2d(axes)
        for column in range(per_kind):
            for row, cases in enumerate((misses, false_positives)):
                ax = axes[row, column]
                if column < len(cases):
                    _draw_panel(ax, cases[column], gt_lookup)
                else:
                    ax.axis("off")

        fig.suptitle(
            f"{name} -- failure modes @ conf={threshold:.2f}\n"
            f"row 1: worst missed defects ({len(misses)} shown)   "
            f"row 2: most confident false positives ({len(false_positives)} shown)\n"
            "dotted green = ground truth, solid = the failure",
            fontsize=10,
        )
        path = out_dir / f"{class_id}_{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
    return written


# ---------------------------------------------------------------------------
# 6. report writers
# ---------------------------------------------------------------------------


def _md_table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(str(h) for h in header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


# The demo reads `operating_point.json`. This module is NOT its author: the shipped
# threshold comes from `src/false_alarm.py`, which prices false alarms on crops that
# carry no annotated defect instead of on spurious boxes over already-defective
# frames. Those are different events and they select different thresholds (0.15 vs
# 0.05). Until 2026-09-09 this module wrote `operating_point.json` unconditionally,
# so `make eval` silently reset the console's default confidence from the shipped
# 0.15 back to 0.05 -- a documented command quietly regressing the shipped system.
#
# The fix is ownership, not a warning. This module now writes its own file by
# default and can only touch the demo's file behind an explicit flag, and even then
# it refuses to overwrite a point written by a different tool without a second one.
# Writing to a separate filename was chosen over "refuse unless forced" as the
# primary mechanism because it makes the safe path the default path: nobody has to
# remember a flag to avoid breaking the console.
OPERATING_POINT_OWN = "operating_point.evaluate.json"
OPERATING_POINT_DEMO = "operating_point.json"
OPERATING_POINT_SOURCE = "src/evaluate.py"


def write_operating_point(path: Path, operating: OperatingPoint, meta: dict[str, Any]) -> Path:
    """Write this module's operating point to `path`.

    `path` is `operating_point.evaluate.json` by default. See the comment above:
    the file the console reads belongs to `src/false_alarm.py`.
    """
    payload = {
        "conf_threshold": round(operating.threshold, 3),
        "iou_threshold": meta["nms_iou"],
        "tuned_on_split": meta["tuning_split"],
        "selection_rule": (
            "minimum expected mill cost among thresholds meeting the image-level detection floor"
            if operating.constraint_met
            else "unconstrained minimum expected mill cost (detection floor unreachable with this checkpoint)"
        ),
        "detection_floor": operating.min_detection_rate,
        "detection_floor_met": operating.constraint_met,
        "at_sweep_edge": operating.at_sweep_edge,
        "miss_cost_ratio": operating.miss_cost_ratio,
        "f1_optimal_threshold": round(operating.f1_optimal_threshold, 3),
        "expected_at_threshold": operating.point.to_dict(),
        "rationale": operating.rationale,
        "weights": meta["weights"],
        "model_name": meta["model_name"],
        "source": OPERATING_POINT_SOURCE,
        "false_alarm_basis": (
            "a spurious extra box on a frame that already carried a defect -- NOT the "
            "clean-steel rate. src/false_alarm.py measures the latter and is the source of "
            "the shipped operating point."
        ),
        "superseded_by": "src/false_alarm.py (reports/operating_point.json)",
        "generated_at": meta["generated_at"],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def update_demo_operating_point(
    own_path: Path, demo_path: Path, force: bool = False
) -> str:
    """Copy this module's operating point over the one the console reads.

    Refuses when the existing file was written by another tool, because that tool
    measured a different quantity. `force` is the deliberate override; there is no
    way to do this by accident.
    """
    if demo_path.exists():
        try:
            existing = json.loads(demo_path.read_text())
        except (OSError, ValueError):
            existing = {}
        owner = str(existing.get("source", "unknown"))
        if owner != OPERATING_POINT_SOURCE and not force:
            return (
                f"refused: {demo_path.name} was written by {owner}, which measures a "
                f"different false alarm event (clean steel, not spurious boxes on defective "
                f"frames). It recommends conf={existing.get('conf_threshold')}, this run "
                f"recommends a different basis. Pass --force-operating-point to overwrite, "
                f"or re-run src/false_alarm.py --update-operating-point to refresh it properly."
            )
    demo_path.write_text(own_path.read_text())
    return f"overwrote {demo_path} from {own_path.name}"


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats with null so the artifact is standard JSON.

    `json.dumps` happily writes bare `NaN`, which Python reads back but every
    other JSON parser rejects. A skipped validator leaves NaN in the mAP fields,
    and this file is meant to be machine-readable by more than Python.
    """
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def write_evaluation_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n")
    return path


def write_evaluation_md(path: Path, payload: dict[str, Any], operating: OperatingPoint) -> Path:
    """The version a person reads before signing off on the system."""
    meta = payload["run"]
    detection = payload["detection_metrics"]
    image_level = payload["image_level"]
    sweep = payload["false_alarm_analysis"]["sweep"]

    per_class_rows = [
        [
            name,
            stats["instances"],
            f"{stats['AP50']:.4f}",
            f"{stats['AP50_95']:.4f}",
            f"{stats['precision']:.4f}",
            f"{stats['recall']:.4f}",
            DEFECT_INFO[name]["severity"] if name in DEFECT_INFO else "-",
        ]
        for name, stats in detection["per_class"].items()
    ]
    image_rows = [
        [
            name,
            stats["support"],
            f"{stats['precision']:.3f}",
            f"{stats['recall']:.3f}",
            f"{stats['f1']:.3f}",
        ]
        for name, stats in image_level["per_class"].items()
    ]
    sweep_rows = [
        [
            f"{p['threshold']:.2f}",
            f"{p['recall']:.3f}",
            f"{p['defect_detection_rate']:.3f}",
            f"{p['precision']:.3f}",
            f"{p['f1']:.3f}",
            f"{p['fp_per_image']:.2f}",
            f"{p['false_alarm_rate']:.3f}",
            f"{p['class_accuracy']:.3f}",
            f"{p['expected_cost_per_image']:.3f}",
        ]
        for p in sweep
    ]
    sensitivity_rows = [
        [f"{s['miss_cost_ratio']:g}:1", f"{s['threshold']:.2f}", f"{s['defect_detection_rate']:.3f}",
         f"{s['false_alarm_rate']:.3f}", f"{s['expected_cost_per_image']:.3f}",
         "yes" if s["meets_detection_floor"] else "no"]
        for s in payload["false_alarm_analysis"]["cost_sensitivity"]
    ]

    failure_rows = [
        [
            name,
            stats["base_severity"],
            stats["instances"],
            stats["missed"],
            f"{1.0 - stats['recall']:.1%}",
            stats["false_positives"],
            ", ".join(f"{k} x{v}" for k, v in stats["fp_landed_on"].items()) or "-",
        ]
        for name, stats in payload["per_class_failures"].items()
    ]
    gallery_lines = "\n".join(f"- `{Path(p).name}`" for p in payload["artifacts"]["error_gallery"]) or "- (none)"

    background_note = (
        ""
        if not image_level.get("background_images")
        else (
            f"\n- {image_level['background_images']} frames carry no labelled defect and are excluded from the "
            f"matrix; {image_level['background_images_flagged']} of them were flagged, which is the only "
            "true false alarm rate in this report"
        )
    )
    protocol = detection.get("protocol", {})
    skipped = bool(detection.get("skipped"))
    section_one = (
        "The ultralytics validator was skipped (`--skip-validator`), so mAP, per-class AP\n"
        "and the precision-recall curves are not part of this run. Everything below is\n"
        "computed from the detector's own cached predictions."
        if skipped
        else (
            "Produced by the ultralytics validator on its own protocol defaults\n"
            f"(conf {protocol.get('conf', 'n/a')}, NMS IoU {protocol.get('nms_iou', 'n/a')}, "
            f"max_det {protocol.get('max_det', 'n/a')}),\n"
            "so these numbers are directly comparable with published NEU-DET results."
        )
    )
    speed = detection.get("speed_ms", {})
    metrics_block = (
        ""
        if skipped
        else f"""
| metric | value |
|---|---|
| mAP50 | **{detection['mAP50']:.4f}** |
| mAP50-95 | **{detection['mAP50_95']:.4f}** |
| precision (mean over classes) | {detection['precision']:.4f} |
| recall (mean over classes) | {detection['recall']:.4f} |

{_md_table(["class", "instances", "AP50", "AP50-95", "precision", "recall", "base severity"], per_class_rows)}
"""
    )

    lines = f"""# Surface defect detector -- evaluation report

**Checkpoint** `{meta['model_name']}` (`{meta['weights']}`)
**Device** {meta['device']} | **imgsz** {meta['imgsz']} (served size; trained at {meta['train_imgsz']}) | **NMS IoU** {meta['nms_iou']} | **match IoU** {IOU_MATCH}
**Held-out split** `{meta['eval_split']}` ({meta['eval_images']} images) | **tuning split** `{meta['tuning_split']}` ({meta['tuning_images']} images)
**Generated** {meta['generated_at']}

The threshold below was chosen on `{meta['tuning_split']}` and applied unchanged to
`{meta['eval_split']}`. Nothing in this report is tuned on the held-out split.

> **Read this alongside its companion reports, and prefer them where they overlap.**
> The conf={image_level['threshold']:.2f} used from section 2 onward is *this* module's
> recommendation, priced against spurious extra boxes on frames that already carried a
> defect. Every NEU-DET image does, so that is a nuisance-box rate, not a clean-steel
> rate. `reports/false_alarm.md` measures the clean-steel event and selects the
> **conf = 0.15** the console actually runs at; `reports/model_study.md` owns the input
> size and the model comparison, with bootstrap intervals this report does not compute;
> `reports/calibration.md` owns what the confidence number means.

## 1. Detection metrics -- `{meta['eval_split']}` split

{section_one}
{metrics_block}
## 2. Image-level classification -- `{meta['eval_split']}` split @ conf={image_level['threshold']:.2f}

Each image is scored by its dominant (highest-confidence) detection, which is
exactly what `InferenceResult.dominant_class` hands the UI. Images where nothing
survives the threshold are counted as `{NO_DETECTION_LABEL}` rather than dropped.

- accuracy **{image_level['accuracy']:.1%}** over {image_level['n_images']} images
- macro-F1 **{image_level['macro_f1']:.3f}** (weighted-F1 {image_level['weighted_f1']:.3f})
- {image_level['no_detection_images']} images produced no detection at all
- {image_level['multi_class_images']} images carry more than one defect type; their image label is the
  majority class by instance count, ties broken by boxed area{background_note}

{_md_table(["class", "support", "precision", "recall", "F1"], image_rows)}

Matrix: `{Path(payload['artifacts']['confusion_matrix']).name}`

## 3. False alarm analysis -- `{meta['tuning_split']}` split

Two levels of accounting. **Box level** (precision / recall / F1 / FP-per-image)
says whether the boxes on the HMI can be trusted. **Image level**
(detection rate / false alarm rate) prices the two events that actually cost
money: a coil that ships with an undetected defect, and a coil pulled for a
defect that is not there.

{_md_table(["conf", "box recall", "img detect rate", "box precision", "F1", "FP/img", "false alarm rate", "img class acc", "cost/img"], sweep_rows)}

### Recommended operating point: **conf = {operating.threshold:.2f}**

{operating.rationale}

Cost model: one escaped defect = {operating.miss_cost_ratio:g} unnecessary re-inspections.
An escaped defect downgrades a ~20 t coil from prime to secondary and can turn
into a customer claim; a false alarm costs the operator a manual look and a few
minutes of line availability. The recommendation is not fragile to that exact
number -- the same rule at other ratios gives:

{_md_table(["miss:false-alarm", "threshold", "img detect rate", "false alarm rate", "cost/img", "clears floor"], sensitivity_rows)}

Those rows minimise cost *without* the detection floor, to show where the
economics alone push the knob; the recommendation above additionally enforces the
floor.

For reference the F1-optimal threshold is {operating.f1_optimal_threshold:.2f}; F1 weights a missed
defect and a false alarm equally, which is not how a mill is paid, so it is
reported but not followed.

Written to `{Path(payload['artifacts']['operating_point']).name}`.

**This is not the threshold the console runs at, and it should not be.** The false
alarm rate priced above is *a spurious extra box on a frame that already carried a
defect* -- every NEU-DET image does -- so it measures nuisance boxes on coils that
were going to be flagged anyway, not the event a mill buys on. `src/false_alarm.py`
measures the clean-steel rate on crops carrying no annotated defect and selects
**conf = 0.15**, which is what `reports/operating_point.json` holds and what the
console loads. This module writes its own file and will not touch that one without
`--update-operating-point --force-operating-point`.

## 4. Curves

- `{Path(payload['artifacts']['pr_curves']).name if payload['artifacts']['pr_curves'] else '(precision-recall curves unavailable: the validator produced no curve data)'}`
- `{Path(payload['artifacts']['f1_vs_threshold']).name}`
- `{Path(payload['artifacts']['false_alarms_vs_recall']).name}`

## 5. Per-class error gallery

Worst missed defects and most confident false positives per class, at the
recommended threshold, on the `{meta['eval_split']}` split. Dotted green is ground
truth; the solid box is the failure.

{gallery_lines}

Instance-level failure counts on `{meta['eval_split']}` at conf={operating.threshold:.2f} (IoU {IOU_MATCH} match):

{_md_table(["class", "base severity", "instances", "missed", "miss rate", "false positives", "false positives landed on"], failure_rows)}

The last column attributes each false positive to the true defect type of the
image it appeared on, which turns a raw false-positive count into a statement a
metallurgist can act on: a class whose false positives cluster on one other
class is a texture confusion to fix with data, while one spread evenly across
all six is an under-trained head.

## 6. Caveats

- **No clean strip in the data.** Every NEU-DET image contains a defect, so the
  false alarm rate reported here is "a spurious box on a coil that was already
  defective". The rate that matters on the line -- flagging a genuinely clean
  strip -- cannot be estimated from this dataset and must be measured on
  production frames before the threshold is committed.
- **Image-level labels are a simplification.** {image_level['multi_class_images']} of
  {image_level['n_images']} images carry more than one defect type; the confusion matrix
  collapses those to their majority class.
- **200x200 crops, not strip captures.** NEU-DET frames are single-defect crops.
  Full-width strip behaviour is exercised by `predict_tiled` in
  `src/inference.py` and is not measured here.
- **Cost ratio is a placeholder for a real one.** {operating.miss_cost_ratio:g}:1 is a
  defensible order of magnitude, not a Jindal Stainless number. The sensitivity
  table above is the honest answer until the quality department supplies the
  actual downgrade and re-inspection costs.

## 7. Latency

{"Not measured in this run (the validator was skipped)." if skipped else
  f"Validator-reported, per image at imgsz {meta['imgsz']} on {meta['device']}: "
  f"preprocess {speed.get('preprocess', 0.0):.2f} ms, "
  f"inference {speed.get('inference', 0.0):.2f} ms, "
  f"postprocess {speed.get('postprocess', 0.0):.2f} ms."}
End-to-end latency as served by `src/inference.py` is measured separately by the
inference benchmark; these figures are the validator's own accounting.
"""
    path.write_text(lines)
    return path


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def evaluate(
    weights: str | Path | None = None,
    split: str = "test",
    out: str | Path = REPORTS_DIR,
    tuning_split: str = "val",
    device: str = "auto",
    imgsz: int | None = None,
    nms_iou: float = 0.45,
    batch_size: int = 16,
    miss_cost_ratio: float = DEFAULT_MISS_COST_RATIO,
    min_detection_rate: float = DEFAULT_MIN_DETECTION_RATE,
    gallery_per_kind: int = 4,
    skip_validator: bool = False,
    update_operating_point: bool = False,
    force_operating_point: bool = False,
) -> dict[str, Any]:
    """Run the whole study and write every artifact. Returns the JSON payload."""
    weights_path = resolve_weights(weights)
    device = resolve_device(device)
    # Default to the size the system is *served* at, not the size it was trained at.
    # An evaluation report exists to describe the deployed detector, and this one was
    # deployed at 256 px on the strength of the val sweep in src/model_study.py. The
    # training size is still resolved and printed, because a large gap between the two
    # is worth seeing.
    train_imgsz = resolve_train_imgsz(weights_path)
    imgsz = int(imgsz) if imgsz else DEFAULT_IMGSZ
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if split == tuning_split:
        raise ValueError(
            f"--split and the tuning split are both {split!r}; the operating threshold would then be "
            "chosen on the same data it is reported on."
        )

    print(f"[eval] weights   {weights_path}")
    print(f"[eval] device    {device}  imgsz {imgsz} (deployment size)  NMS IoU {nms_iou}")
    if imgsz != train_imgsz:
        print(f"[eval] note      the checkpoint was trained at {train_imgsz} px; this report "
              f"describes it as served, at {imgsz} px")
    print(f"[eval] tuning on {tuning_split!r}, reporting on {split!r}")

    # --- 1. detection metrics from the ultralytics validator ---------------
    if skip_validator:
        print("[eval] --skip-validator: mAP metrics and PR curves will be omitted")
        detection_metrics = {
            "split": split,
            "imgsz": imgsz,
            "skipped": True,
            "protocol": {"conf": "n/a", "nms_iou": "n/a", "max_det": "n/a", "source": "validator skipped"},
            "mAP50": float("nan"),
            "mAP50_95": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "fitness": float("nan"),
            "per_class": {},
            "speed_ms": {},
            "save_dir": "",
            "_pr_curves": {},
        }
    else:
        print(f"[eval] running ultralytics validator on {split} ...")
        detection_metrics = run_validator(weights_path, split, device, imgsz, out_dir)
        print(
            f"[eval] mAP50={detection_metrics['mAP50']:.4f} "
            f"mAP50-95={detection_metrics['mAP50_95']:.4f} "
            f"P={detection_metrics['precision']:.4f} R={detection_metrics['recall']:.4f}"
        )

    # --- 2. one cached forward pass per split ------------------------------
    cache_conf = min(SWEEP_THRESHOLDS) * 0.9  # strictly below the lowest sweep point
    detector = DefectDetector(weights=weights_path, device=device, conf=cache_conf, iou=nms_iou, imgsz=imgsz)
    detector.warmup(2)

    tuning_records = load_ground_truth(tuning_split)
    eval_records = load_ground_truth(split)
    print(f"[eval] caching detections: {tuning_split} ({len(tuning_records)}) and {split} ({len(eval_records)}) ...")
    tuning_preds = cache_predictions(detector, tuning_records, batch_size=batch_size)
    eval_preds = cache_predictions(detector, eval_records, batch_size=batch_size)
    print(
        f"[eval] cached {sum(len(p.confidences) for p in tuning_preds)} boxes on {tuning_split}, "
        f"{sum(len(p.confidences) for p in eval_preds)} on {split} (conf >= {cache_conf:.3f})"
    )

    # --- 3. sweep and operating point, on the tuning split only ------------
    sweep = sweep_confidence(tuning_records, tuning_preds)
    operating = recommend_threshold(sweep, miss_cost_ratio, min_detection_rate)
    print(f"[eval] recommended conf={operating.threshold:.2f} (constraint met: {operating.constraint_met})")

    # --- 4. everything else, at the threshold val chose --------------------
    image_level = image_level_report(eval_records, eval_preds, operating.threshold)
    eval_sweep = sweep_confidence(eval_records, eval_preds, thresholds=(operating.threshold,))
    price_sweep(eval_sweep, miss_cost_ratio)

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta = {
        "weights": str(weights_path),
        "model_name": detector.model_name,
        "device": device,
        "imgsz": imgsz,
        "train_imgsz": train_imgsz,
        "nms_iou": nms_iou,
        "match_iou": IOU_MATCH,
        "cache_conf": round(cache_conf, 4),
        "eval_split": split,
        "eval_images": len(eval_records),
        "tuning_split": tuning_split,
        "tuning_images": len(tuning_records),
        "generated_at": generated_at,
    }

    # --- 5. plots and gallery ----------------------------------------------
    confusion_png = plot_confusion_matrix(image_level, out_dir / "confusion_matrix.png", split)
    pr_png = plot_pr_curves(detection_metrics, out_dir / "pr_curves.png", split)
    f1_png = plot_f1_vs_threshold(sweep, operating, out_dir / "f1_vs_threshold.png", tuning_split)
    fa_png = plot_false_alarms_vs_recall(sweep, operating, out_dir / "false_alarms_vs_recall.png", tuning_split)
    failure_summary = per_class_failure_summary(eval_records, eval_preds, operating.threshold)
    failures = collect_failures(eval_records, eval_preds, operating.threshold, per_kind=gallery_per_kind)
    gallery = build_error_gallery(failures, eval_records, out_dir / "error_gallery", operating.threshold,
                                  per_kind=gallery_per_kind)

    operating_json = write_operating_point(out_dir / OPERATING_POINT_OWN, operating, meta)
    demo_update = "not requested (--update-operating-point)"
    if update_operating_point:
        demo_update = update_demo_operating_point(
            operating_json, out_dir / OPERATING_POINT_DEMO, force=force_operating_point
        )
    print(f"[eval] operating point       {operating_json.name} | demo file: {demo_update}")

    payload: dict[str, Any] = {
        "run": meta,
        "detection_metrics": {k: v for k, v in detection_metrics.items() if k != "_pr_curves"},
        "image_level": image_level,
        "false_alarm_analysis": {
            "tuning_split": tuning_split,
            "match_iou": IOU_MATCH,
            "cost_model": {
                "miss_cost_ratio": miss_cost_ratio,
                "false_alarm_cost": 1.0,
                "units": "one unnecessary re-inspection / line stop",
                "detection_floor": min_detection_rate,
                "detection_floor_met": operating.constraint_met,
                "definition": (
                    "expected_cost_per_image = miss_cost_ratio * (1 - image-level defect detection rate) "
                    "+ 1.0 * image-level false alarm rate"
                ),
            },
            "sweep": [p.to_dict() for p in sweep],
            "cost_sensitivity": operating.sensitivity,
            "recommended": {
                "threshold": round(operating.threshold, 3),
                "f1_optimal_threshold": round(operating.f1_optimal_threshold, 3),
                "rationale": operating.rationale,
                "on_tuning_split": operating.point.to_dict(),
                "on_held_out_split": eval_sweep[0].to_dict(),
            },
        },
        "per_class_failures": {
            name: {
                **stats,
                "misses_shown_in_gallery": len(failures.get(name, {}).get("miss", ())),
                "false_positives_shown_in_gallery": len(failures.get(name, {}).get("false_positive", ())),
            }
            for name, stats in failure_summary.items()
        },
        "artifacts": {
            "evaluation_json": str(out_dir / "evaluation.json"),
            "evaluation_md": str(out_dir / "evaluation.md"),
            "operating_point": str(operating_json),
            "operating_point_demo_update": demo_update,
            "confusion_matrix": str(confusion_png),
            "pr_curves": str(pr_png) if pr_png else "",
            "f1_vs_threshold": str(f1_png),
            "false_alarms_vs_recall": str(fa_png),
            "error_gallery": [str(p) for p in gallery],
            "validator_run_dir": detection_metrics.get("save_dir", ""),
        },
    }

    write_evaluation_json(out_dir / "evaluation.json", payload)
    write_evaluation_md(out_dir / "evaluation.md", payload, operating)
    print(f"[eval] wrote {out_dir / 'evaluation.json'} and {out_dir / 'evaluation.md'}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the strip-defect detector and choose a mill operating threshold.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", default=None, help="checkpoint to evaluate (default: auto-resolve)")
    parser.add_argument("--split", default="test", help="held-out split to report on")
    parser.add_argument("--out", default=str(REPORTS_DIR), help="output directory for reports and figures")
    parser.add_argument("--tuning-split", default="val", help="split the confidence threshold is chosen on")
    parser.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    parser.add_argument("--imgsz", type=int, default=None,
                        help=f"inference size (default: {DEFAULT_IMGSZ}, the size the system is "
                             "served at; the training size is reported but not used)")
    parser.add_argument("--nms-iou", type=float, default=0.45, help="deployment NMS IoU (matches src/inference.py)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--miss-cost-ratio", type=float, default=DEFAULT_MISS_COST_RATIO,
                        help="cost of one escaped defect in units of one false alarm")
    parser.add_argument("--min-detection-rate", type=float, default=DEFAULT_MIN_DETECTION_RATE,
                        help="image-level detection floor the operating point must clear")
    parser.add_argument("--gallery-per-kind", type=int, default=4, help="panels per row in the error gallery")
    parser.add_argument("--skip-validator", action="store_true",
                        help="skip the ultralytics mAP pass (faster iteration on the sweep)")
    parser.add_argument("--update-operating-point", action="store_true",
                        help=f"also copy this run's operating point over {OPERATING_POINT_DEMO}, "
                             "the file the console reads. Off by default: the shipped threshold "
                             "comes from src/false_alarm.py, which measures a different event")
    parser.add_argument("--force-operating-point", action="store_true",
                        help=f"with --update-operating-point, overwrite {OPERATING_POINT_DEMO} "
                             "even when another tool wrote it")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evaluate(
        weights=args.weights,
        split=args.split,
        out=args.out,
        tuning_split=args.tuning_split,
        device=args.device,
        imgsz=args.imgsz,
        nms_iou=args.nms_iou,
        batch_size=args.batch_size,
        miss_cost_ratio=args.miss_cost_ratio,
        min_detection_rate=args.min_detection_rate,
        gallery_per_kind=args.gallery_per_kind,
        skip_validator=args.skip_validator,
        update_operating_point=args.update_operating_point,
        force_operating_point=args.force_operating_point,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
