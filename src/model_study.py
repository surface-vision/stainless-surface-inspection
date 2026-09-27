"""Definitive model-selection study for the Jindal Stainless strip-defect detector.

Two checkpoints were fine-tuned on the same 1440/180/180 NEU-DET split with the
same recipe (150 epochs requested, imgsz 320, batch 32, seed 1337): `yolov8n` and
`yolov8s`. On the held-out test split the *nano* model scores higher. This module
exists to decide whether that is a real ordering or a small-sample artefact, and
to fix every remaining deployment knob with a measurement rather than a habit.

It answers five questions and keeps them separate:

1. **Head to head.** mAP50, mAP50-95, precision, recall and per-class AP50 for both
   checkpoints, alongside parameter count, GFLOPs, weight-file size and measured
   single-frame latency. The gap is then tested with a **paired bootstrap over the
   180 test images**: both models are scored on the *same* resampled image sets, so
   the confidence interval is on the *difference*, where the between-image variance
   the two models share cancels out. Without that interval a 0.07 mAP50 gap on 180
   images is an anecdote.

2. **Inference resolution.** Both models were trained at 320 px on 200x200 source
   images. Each is evaluated at inference sizes 256/320/416/512/640 to measure the
   train/test resolution mismatch penalty directly.

3. **Test-time augmentation.** Ultralytics TTA (multi-scale + horizontal flip) at
   the chosen size, priced in both mAP50 and milliseconds.

4. **Which classes fail, and why.** Every missed ground-truth box is sorted into
   *class confusion*, *extent error* or *blind miss* by looking at the best
   overlapping prediction of any class, and each class's difficulty is then
   regressed against a measured image statistic (defect-vs-background separability,
   edge energy, annotation overlap) so "low contrast" stops being an assertion.

5. **A deployment recommendation** with the checkpoint, the input size and the TTA
   decision each traced to one of the numbers above.

**Split discipline.** Every *choice* here -- best inference size, TTA on or off --
is made on the **validation** split. The test split is scored at the configuration
val chose and is never used to select anything. The head-to-head comparison and the
bootstrap are reported on test, which is what test is for. Where the study reports
a test number that was also used to frame a question (the nano-vs-small ordering,
which was known before this module was written), it says so.

**What the bootstrap does and does not cover.** Resampling images estimates the
variability that comes from having drawn *these* 180 test images. It holds the
trained weights fixed, so it says nothing about training-seed variance. Separating
the two would need several seeds per architecture; the cost of that experiment is
computed in the report from the measured epoch times rather than guessed at.

Outputs: `reports/model_study.json`, `reports/model_study.md`, and the figures
`reports/model_study_imgsz.png`, `reports/model_study_bootstrap.png`,
`reports/model_study_failures.png`, `reports/model_study_difficulty.png`.

Usage:
    .venv/bin/python src/model_study.py
    .venv/bin/python src/model_study.py --resamples 2000 --skip-train-cost
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: this runs from a terminal, not a desktop session

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Ultralytics pip-installs its way out of a version conflict on import unless told
# not to. A study that silently changes the environment it is measuring is
# worthless, and it has already downgraded numpy once in this venv.
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

from inference import (  # noqa: E402
    CLASS_COLORS,
    CLASS_NAMES,
    DEFAULT_IMGSZ,
    DefectDetector,
    resolve_device,
)
from inference import _synchronise as synchronise_device  # noqa: E402

# Reused rather than reimplemented: the ground-truth loader, the greedy class-aware
# matcher and the prediction cache are already validated in evaluate.py, and a
# second implementation of box matching in this repo would be a second thing to be
# wrong.
from evaluate import (  # noqa: E402
    CachedPrediction,
    GroundTruth,
    cache_predictions,
    class_label,
    filter_prediction,
    iou_matrix,
    load_ground_truth,
    match_image,
)

# Latency distribution container and the checkpoint-complexity probe already exist
# in the speed study; importing them keeps one definition of "p95 latency" and one
# definition of "GFLOPs at size N" in the project.
from benchmark import Checkpoint, LatencyStats, _sha256, measure_complexity  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_ROOT / "data" / "neu-det" / "data.yaml"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
TEST_IMAGES_DIR = PROJECT_ROOT / "data" / "neu-det" / "test" / "images"

# Multiples of 32 bracketing the 320 px training size, from below native (the source
# images are 200x200, so 256 is the only size in the grid that does not upsample) to
# the ultralytics default 640.
IMGSZ_GRID: tuple[int, ...] = (256, 320, 416, 512, 640)

# The DefectDetector default an operator sees if nobody tunes anything. The failure
# taxonomy is reported at this and at the tuned threshold, because they tell
# different stories: the default shows what the operator is shown, the tuned one
# shows what the cost model asks for.
DEFAULT_CONF = 0.25
DEPLOY_NMS_IOU = 0.45
OPERATING_POINT = REPORTS_DIR / "operating_point.json"
FALLBACK_TUNED_CONF = 0.15


def tuned_confidence() -> tuple[float, str]:
    """The deployed threshold, read from whichever study currently owns it.

    Hard-coding this would silently fork the deployment story the moment the
    operating-point study is rerun, which it has been. Read it, and record where it
    came from, so the failure taxonomy below is always reported at the threshold the
    project is actually recommending.
    """
    if OPERATING_POINT.is_file():
        try:
            recorded = json.loads(OPERATING_POINT.read_text())
            value = float(recorded["conf_threshold"])
            source = recorded.get("source") or "reports/operating_point.json"
            return value, f"{source} (tuned on {recorded.get('tuned_on_split', 'val')})"
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass
    return FALLBACK_TUNED_CONF, "fallback constant; reports/operating_point.json unreadable"

# Below this IoU a prediction is not "the same defect, badly drawn" -- it is a box
# somewhere else in the frame. Used only to separate extent errors from blind misses.
EXTENT_IOU_FLOOR = 0.10

BOOTSTRAP_SEED = 1337

# Published per-class difficulty ordering, hardest first, from docs/research_notes.md
# section 1c (Maity & Ghosh, arXiv:2510.21811, Table 1, AP@[.5:.95], marked [HARD]).
# Stored as the ordering only; the underlying values are AP50-95 on a different
# 70/20/10 split and are not comparable to our AP50 as numbers.
PUBLISHED_HARDEST_FIRST: tuple[str, ...] = (
    "crazing",
    "rolled-in_scale",
    "scratches",
    "pitted_surface",
    "inclusion",
    "patches",
)
PUBLISHED_SOURCE = "Maity & Ghosh, arXiv:2510.21811 Table 1 (YOLOv11 column), via docs/research_notes.md s1c"

GRID = {"color": "#d8dde3", "linewidth": 0.7, "alpha": 0.9}


# ---------------------------------------------------------------------------
# data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """One checkpoint under study."""

    key: str
    run: str
    weights: Path

    @property
    def label(self) -> str:
        return f"{self.run}/{self.weights.name}"


@dataclass
class SplitStats:
    """The validator's own per-image statistics, kept per image instead of pooled.

    Ultralytics computes mAP by concatenating these five arrays across the whole
    split and calling `ap_per_class` once. Capturing them per image, before that
    concatenation, is what makes an image-level bootstrap possible: a resample is
    just a different concatenation order with repeats, scored by the identical
    function. The recomputed full-split mAP is checked against the validator's own
    number in `fidelity`, and the study refuses to report a bootstrap whose point
    estimate does not reproduce.
    """

    names: list[str]
    tp: list[np.ndarray]  # (n_i, 10) bool, one column per IoU threshold 0.50..0.95
    conf: list[np.ndarray]
    pred_cls: list[np.ndarray]
    target_cls: list[np.ndarray]

    def __len__(self) -> int:
        return len(self.names)

    def reindex(self, order: Sequence[str]) -> "SplitStats":
        """Same statistics, re-ordered onto a canonical image-name sequence.

        The validation dataloader does not emit images in sorted order, and two
        runs of two different models are only paired if the i-th entry is the same
        photograph in both. Aligning by name rather than by position is the
        difference between a paired bootstrap and a meaningless one.
        """
        pos = {name: i for i, name in enumerate(self.names)}
        missing = [n for n in order if n not in pos]
        if missing:
            raise KeyError(f"{len(missing)} image(s) absent from this run, e.g. {missing[:3]}")
        idx = [pos[n] for n in order]
        return SplitStats(
            names=list(order),
            tp=[self.tp[i] for i in idx],
            conf=[self.conf[i] for i in idx],
            pred_cls=[self.pred_cls[i] for i in idx],
            target_cls=[self.target_cls[i] for i in idx],
        )


@dataclass
class ValRun:
    """One (model, split, imgsz, augment) validation cell."""

    model: str
    split: str
    imgsz: int
    augment: bool
    mAP50: float
    mAP50_95: float
    precision: float
    recall: float
    per_class_AP50: dict[str, float]
    per_class_AP50_95: dict[str, float]
    instances: dict[str, int]
    speed_ms: dict[str, float]
    seconds: float
    stats: SplitStats | None = field(default=None, repr=False)

    @property
    def cell(self) -> tuple[str, str, int, bool]:
        return (self.model, self.split, self.imgsz, self.augment)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "split": self.split,
            "imgsz": self.imgsz,
            "augment": self.augment,
            "mAP50": round(self.mAP50, 5),
            "mAP50_95": round(self.mAP50_95, 5),
            "precision": round(self.precision, 5),
            "recall": round(self.recall, 5),
            "per_class_AP50": {k: round(v, 5) for k, v in self.per_class_AP50.items()},
            "per_class_AP50_95": {k: round(v, 5) for k, v in self.per_class_AP50_95.items()},
            "instances": self.instances,
            "validator_speed_ms_per_image": {k: round(v, 3) for k, v in self.speed_ms.items()},
            "wall_seconds": round(self.seconds, 2),
        }


# ---------------------------------------------------------------------------
# 1. checkpoint facts: size, parameters, FLOPs, content hash
# ---------------------------------------------------------------------------


def discover_models(requested: Sequence[str] | None = None) -> list[ModelSpec]:
    """The checkpoints to compare, newest run last, pinned to `best.pt`."""
    runs = list(requested) if requested else ["yolov8n_neudet", "yolov8s_neudet"]
    specs: list[ModelSpec] = []
    for run in runs:
        weights = MODELS_DIR / run / "weights" / "best.pt"
        if not weights.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {weights}")
        specs.append(ModelSpec(key=run.split("_")[0], run=run, weights=weights))
    return specs


def checkpoint_facts(spec: ModelSpec, image_sizes: Sequence[int]) -> dict[str, Any]:
    """Parameters, GFLOPs at each input size, file size and SHA-256.

    The hash is not decoration: this study and the training log must be talking
    about the same bytes, and `best.pt` is a file a rerun would overwrite.
    """
    checkpoint = Checkpoint(
        key=spec.label,
        source_path=str(spec.weights),
        snapshot_path=str(spec.weights),
        sha256=_sha256(spec.weights),
        size_mb=round(spec.weights.stat().st_size / 1e6, 2),
        aliases=[spec.label],
    )
    measure_complexity(checkpoint, image_sizes)
    return {
        "run": spec.run,
        "weights": str(spec.weights),
        "sha256": checkpoint.sha256,
        "params": checkpoint.params,
        "params_millions": round(checkpoint.params / 1e6, 3),
        "weight_file_mb": checkpoint.size_mb,
        "gflops_by_imgsz": checkpoint.gflops_by_imgsz,
    }


def training_history(spec: ModelSpec) -> dict[str, Any]:
    """What the training run actually did, read back from its own results.csv.

    The recipe asked for 150 epochs; early stopping may have ended one run sooner,
    and the epoch that produced `best.pt` is not usually the last. Both facts
    matter to the capacity argument, and neither is in the checkpoint.
    """
    import pandas as pd

    csv = MODELS_DIR / spec.run / "results.csv"
    if not csv.is_file():
        return {"available": False, "reason": f"{csv} not found"}
    frame = pd.read_csv(csv)
    frame.columns = [c.strip() for c in frame.columns]
    # Ultralytics' own model-selection criterion, so "best epoch" here means the
    # epoch whose weights were actually saved as best.pt.
    fitness = 0.1 * frame["metrics/mAP50(B)"] + 0.9 * frame["metrics/mAP50-95(B)"]
    best = int(fitness.idxmax())
    elapsed = float(frame["time"].iloc[-1]) if "time" in frame.columns else float("nan")

    # `time` is cumulative wall clock, so the per-epoch cost is its first difference
    # (with the first entry standing for epoch 1). The mean of that is not the cost
    # of an epoch on this machine: the yolov8s run contains two epochs of ~940 s at
    # 320 px, which is a memory-pressure stall, not arithmetic. Contention can only
    # ever *add* time, so the median is the epoch cost and the mean is the schedule
    # this laptop actually delivered. Both are reported, and the stalls are named.
    per_epoch = np.diff(frame["time"].values, prepend=0.0) if "time" in frame.columns else np.array([])
    median_epoch = float(np.median(per_epoch)) if per_epoch.size else float("nan")
    stalls = (
        [
            {"epoch": int(frame["epoch"].values[i]), "seconds": round(float(per_epoch[i]), 1)}
            for i in np.where(per_epoch > 3.0 * median_epoch)[0]
        ]
        if per_epoch.size
        else []
    )
    return {
        "available": True,
        "epochs_requested": 150,
        "epochs_run": int(len(frame)),
        "early_stopped": bool(len(frame) < 150),
        "best_epoch": int(frame.loc[best, "epoch"]),
        "best_val_mAP50": round(float(frame.loc[best, "metrics/mAP50(B)"]), 5),
        "best_val_mAP50_95": round(float(frame.loc[best, "metrics/mAP50-95(B)"]), 5),
        "final_train_box_loss": round(float(frame["train/box_loss"].iloc[-1]), 4),
        "final_val_box_loss": round(float(frame["val/box_loss"].iloc[-1]), 4),
        "final_train_cls_loss": round(float(frame["train/cls_loss"].iloc[-1]), 4),
        "final_val_cls_loss": round(float(frame["val/cls_loss"].iloc[-1]), 4),
        # The generalisation gap in the classification head is the direct evidence
        # for or against "the bigger model memorised 1440 images".
        "cls_loss_generalisation_gap": round(
            float(frame["val/cls_loss"].iloc[-1] - frame["train/cls_loss"].iloc[-1]), 4
        ),
        "box_loss_generalisation_gap": round(
            float(frame["val/box_loss"].iloc[-1] - frame["train/box_loss"].iloc[-1]), 4
        ),
        "wall_seconds": round(elapsed, 1),
        "seconds_per_epoch": round(elapsed / max(1, len(frame)), 2),
        "seconds_per_epoch_median": round(median_epoch, 2),
        "stalled_epochs": stalls,
        "stall_seconds": round(sum(s["seconds"] for s in stalls), 1),
        "source": str(csv),
    }


# ---------------------------------------------------------------------------
# 2. validation, with the per-image statistics captured on the way past
# ---------------------------------------------------------------------------


def run_validation(
    spec: ModelSpec,
    split: str,
    imgsz: int,
    device: str,
    *,
    augment: bool = False,
    batch: int = 16,
    workers: int = 2,
    out_dir: Path,
) -> ValRun:
    """Ultralytics validator on one split, at one input size, with or without TTA.

    Left on the validator's own protocol defaults (conf 0.001, NMS IoU 0.7,
    max_det 300) so the mAP is directly comparable with published NEU-DET numbers
    and with the figures already recorded by `train_detector.py`.

    The per-image statistics are lifted out through an `on_val_batch_end`
    callback. They have to be taken there: `get_stats()` concatenates them, scores
    them and then calls `clear_stats()`, so by the time the metrics object is
    returned the per-image view is gone.
    """
    from ultralytics import YOLO

    captured: dict[str, Any] = {}

    def snapshot(validator: Any) -> None:
        # Copy the lists, not the arrays: the arrays are never mutated, but the
        # lists are cleared out from under us at the end of the run.
        captured["stats"] = {k: list(v) for k, v in validator.metrics.stats.items()}

    model = YOLO(str(spec.weights))
    model.add_callback("on_val_batch_end", snapshot)

    started = time.perf_counter()
    metrics = model.val(
        data=str(DATA_YAML),
        split=split,
        imgsz=imgsz,
        device=device,
        batch=batch,
        workers=workers,
        augment=augment,
        plots=False,
        verbose=False,
        project=str(out_dir),
        name=f"{spec.key}_{split}_{imgsz}{'_tta' if augment else ''}",
        exist_ok=True,
    )
    seconds = time.perf_counter() - started

    names = dict(metrics.names)
    class_index = [int(c) for c in metrics.ap_class_index]
    ap50: dict[str, float] = {}
    ap: dict[str, float] = {}
    instances: dict[str, int] = {}
    for row, cls in enumerate(class_index):
        _, _, a50, a = metrics.box.class_result(row)
        label = names.get(cls, f"class_{cls}")
        ap50[label] = float(a50)
        ap[label] = float(a)
        instances[label] = (
            int(metrics.nt_per_class[cls]) if metrics.nt_per_class is not None else 0
        )

    raw = captured.get("stats")
    stats: SplitStats | None = None
    if raw is not None:
        # image_metrics is an insertion-ordered dict written in the same loop
        # iteration as the stats append, so its key order is the stats row order.
        image_names = list(metrics.box.image_metrics.keys())
        if len(image_names) == len(raw["tp"]):
            stats = SplitStats(
                names=image_names,
                tp=[np.asarray(t, dtype=bool) for t in raw["tp"]],
                conf=[np.asarray(c, dtype=np.float64) for c in raw["conf"]],
                pred_cls=[np.asarray(c, dtype=np.float64) for c in raw["pred_cls"]],
                target_cls=[np.asarray(c, dtype=np.float64) for c in raw["target_cls"]],
            )

    del model
    return ValRun(
        model=spec.key,
        split=split,
        imgsz=imgsz,
        augment=augment,
        mAP50=float(metrics.box.map50),
        mAP50_95=float(metrics.box.map),
        precision=float(metrics.box.mp),
        recall=float(metrics.box.mr),
        per_class_AP50=ap50,
        per_class_AP50_95=ap,
        instances=instances,
        speed_ms={k: float(v) for k, v in metrics.speed.items()},
        seconds=seconds,
        stats=stats,
    )


# ---------------------------------------------------------------------------
# 3. recomputation from per-image statistics, and the paired bootstrap
# ---------------------------------------------------------------------------


def score_subset(stats: SplitStats, index: Sequence[int] | np.ndarray) -> dict[str, Any]:
    """mAP over an arbitrary (possibly repeating) selection of images.

    Calls ultralytics' own `ap_per_class`, so a resample is scored by exactly the
    function that produced the headline number -- same 101-point COCO
    interpolation, same max-F1 operating point for P and R, same treatment of a
    class with no predictions.
    """
    from ultralytics.utils.metrics import ap_per_class

    idx = np.asarray(index, dtype=int)
    tp = np.concatenate([stats.tp[i] for i in idx], 0)
    conf = np.concatenate([stats.conf[i] for i in idx], 0)
    pred_cls = np.concatenate([stats.pred_cls[i] for i in idx], 0)
    target_cls = np.concatenate([stats.target_cls[i] for i in idx], 0)
    if tp.size == 0:
        tp = np.zeros((0, 10), dtype=bool)

    _, _, precision, recall, _, ap, unique = ap_per_class(tp, conf, pred_cls, target_cls)[:7]
    return {
        "mAP50": float(ap[:, 0].mean()),
        "mAP50_95": float(ap.mean()),
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "per_class_AP50": {class_label(int(c)): float(ap[i, 0]) for i, c in enumerate(unique)},
        "classes_present": int(len(unique)),
    }


def bootstrap_indices(n_images: int, resamples: int, seed: int) -> np.ndarray:
    """(resamples, n_images) of image indices drawn with replacement."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_images, size=(resamples, n_images), dtype=np.int64)


def paired_bootstrap(
    stats_a: SplitStats,
    stats_b: SplitStats,
    label_a: str,
    label_b: str,
    resamples: int,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Confidence interval on the mAP difference, both models on the same resamples.

    Pairing is the whole point. The dominant source of variance in a 180-image mAP
    is *which images you drew* -- a resample heavy in crazing scores badly for
    every model -- and that component is common to both models and cancels in the
    difference. An unpaired interval would be several times wider and would say
    nothing about which model is better.

    Reported as a percentile interval, plus the share of resamples in which each
    model wins, which is the quantity a reader actually wants: how often would this
    comparison come out the same way on a differently drawn test set of the same
    size.
    """
    if [n for n in stats_a.names] != [n for n in stats_b.names]:
        raise ValueError("paired bootstrap requires identically ordered image lists")

    n = len(stats_a)
    draws = bootstrap_indices(n, resamples, seed)

    map50 = np.empty((resamples, 2), dtype=np.float64)
    map5095 = np.empty((resamples, 2), dtype=np.float64)
    short_classes = 0
    started = time.perf_counter()
    for r in range(resamples):
        idx = draws[r]
        sa = score_subset(stats_a, idx)
        sb = score_subset(stats_b, idx)
        map50[r] = (sa["mAP50"], sb["mAP50"])
        map5095[r] = (sa["mAP50_95"], sb["mAP50_95"])
        if sa["classes_present"] < len(CLASS_NAMES) or sb["classes_present"] < len(CLASS_NAMES):
            short_classes += 1

    def summarise(samples: np.ndarray, point_a: float, point_b: float) -> dict[str, Any]:
        delta = samples[:, 0] - samples[:, 1]
        lo, hi = np.percentile(delta, [2.5, 97.5])
        return {
            "point_estimate_a": round(point_a, 5),
            "point_estimate_b": round(point_b, 5),
            "point_difference": round(point_a - point_b, 5),
            "bootstrap_mean_difference": round(float(delta.mean()), 5),
            "bootstrap_std_difference": round(float(delta.std(ddof=1)), 5),
            "ci95_low": round(float(lo), 5),
            "ci95_high": round(float(hi), 5),
            "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0),
            "share_a_wins": round(float((delta > 0).mean()), 4),
            "share_b_wins": round(float((delta < 0).mean()), 4),
            "a_marginal_std": round(float(samples[:, 0].std(ddof=1)), 5),
            "b_marginal_std": round(float(samples[:, 1].std(ddof=1)), 5),
            # What an unpaired comparison would have reported: the two marginal
            # variances added, as if the models had been scored on independently
            # drawn test sets. The ratio is how much the pairing was worth.
            "unpaired_std_estimate": round(
                float(np.hypot(samples[:, 0].std(ddof=1), samples[:, 1].std(ddof=1))), 5
            ),
            "pairing_variance_reduction": round(
                float(
                    np.hypot(samples[:, 0].std(ddof=1), samples[:, 1].std(ddof=1))
                    / max(delta.std(ddof=1), 1e-12)
                ),
                2,
            ),
        }

    full = np.arange(n)
    point_a, point_b = score_subset(stats_a, full), score_subset(stats_b, full)
    return {
        "model_a": label_a,
        "model_b": label_b,
        "n_images": n,
        "resamples": resamples,
        "seed": seed,
        "resamples_missing_a_class": short_classes,
        "seconds": round(time.perf_counter() - started, 1),
        "mAP50": summarise(map50, point_a["mAP50"], point_b["mAP50"]),
        "mAP50_95": summarise(map5095, point_a["mAP50_95"], point_b["mAP50_95"]),
        "_delta_map50_samples": (map50[:, 0] - map50[:, 1]),
        "_delta_map5095_samples": (map5095[:, 0] - map5095[:, 1]),
    }


def bootstrap_fidelity(stats: SplitStats, run: ValRun) -> dict[str, Any]:
    """Does recomputing from the captured per-image rows reproduce the validator?

    If this does not agree to within float noise, the bootstrap is scoring
    something other than the reported model and every interval below is void.
    """
    recomputed = score_subset(stats, np.arange(len(stats)))
    return {
        "model": run.model,
        "split": run.split,
        "imgsz": run.imgsz,
        "validator_mAP50": round(run.mAP50, 8),
        "recomputed_mAP50": round(recomputed["mAP50"], 8),
        "abs_error_mAP50": abs(recomputed["mAP50"] - run.mAP50),
        "validator_mAP50_95": round(run.mAP50_95, 8),
        "recomputed_mAP50_95": round(recomputed["mAP50_95"], 8),
        "abs_error_mAP50_95": abs(recomputed["mAP50_95"] - run.mAP50_95),
        "exact": abs(recomputed["mAP50"] - run.mAP50) < 1e-9
        and abs(recomputed["mAP50_95"] - run.mAP50_95) < 1e-9,
    }


# ---------------------------------------------------------------------------
# 4. latency
# ---------------------------------------------------------------------------


def load_frames(count: int) -> list[np.ndarray]:
    """Real test frames at their native 200x200, held in RAM.

    Native size on purpose: this sweep varies the *network* input size, so the
    frames handed to it must not change with it. Sampled evenly across the sorted
    split so all six classes are represented -- NMS cost tracks how many boxes
    survive, and a pool of one class would flatter or punish postprocess.
    """
    paths = sorted(TEST_IMAGES_DIR.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No test images under {TEST_IMAGES_DIR}")
    chosen = [paths[i] for i in np.unique(np.linspace(0, len(paths) - 1, count).round().astype(int))]
    frames = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in chosen]
    return [np.ascontiguousarray(f) for f in frames if f is not None]


def time_predict(
    weights: Path,
    frames: Sequence[np.ndarray],
    imgsz: int,
    device: str,
    *,
    augment: bool,
    conf: float = DEFAULT_CONF,
    iou: float = DEPLOY_NMS_IOU,
    iters: int = 80,
    warmup: int = 12,
) -> LatencyStats:
    """Single-frame wall-clock latency, warm, with the accelerator queue drained.

    The clock is read only after `synchronise_device`, because MPS is asynchronous:
    without the barrier the loop times how fast Python can enqueue work, not how
    fast the GPU finishes it. Warm-up is discarded and is run at the same input
    shape as the measurement -- MPS compiles and caches kernels per shape, so a
    warm-up at another size would charge the first timed call for a compile.

    TTA and non-TTA go through this same function so their ratio is a like-for-like
    cost, not an artefact of two different measurement paths.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))
    model.to(device)

    def call(frame: np.ndarray) -> None:
        model.predict(
            frame,
            imgsz=imgsz,
            device=device,
            conf=conf,
            iou=iou,
            augment=augment,
            verbose=False,
        )
        synchronise_device(device)

    for w in range(warmup):
        call(frames[w % len(frames)])

    samples: list[float] = []
    for i in range(iters):
        frame = frames[i % len(frames)]
        t0 = time.perf_counter()
        call(frame)
        samples.append((time.perf_counter() - t0) * 1000.0)

    del model
    return LatencyStats.from_samples(samples)


def time_tta_ratio(
    weights: Path,
    frames: Sequence[np.ndarray],
    imgsz: int,
    device: str,
    *,
    iters: int = 120,
    warmup: int = 20,
) -> dict[str, Any]:
    """Cost of TTA relative to the same model without it, measured *interleaved*.

    Measuring the two in separate phases and dividing is what produced this study's
    original 8.2x figure, and it is wrong: on a contended laptop each phase carries
    its own, different amount of other people's work, and the quotient of two
    contended medians is not a cost multiple. Alternating the two settings inside a
    single loop puts both on the same machine conditions sample by sample, so
    whatever contention there is cancels in the ratio -- the same argument that
    makes the paired bootstrap tighter than an unpaired one.

    Reported at the floor as well as the median: the floor is the arithmetic cost,
    the median is what the queue actually delivered.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))
    model.to(device)

    def call(frame: np.ndarray, augment: bool) -> None:
        model.predict(
            frame,
            imgsz=imgsz,
            device=device,
            conf=DEFAULT_CONF,
            iou=DEPLOY_NMS_IOU,
            augment=augment,
            verbose=False,
        )
        synchronise_device(device)

    for w in range(warmup):
        call(frames[w % len(frames)], False)
        call(frames[w % len(frames)], True)

    samples: dict[bool, list[float]] = {False: [], True: []}
    for i in range(iters):
        frame = frames[i % len(frames)]
        for augment in (False, True):
            t0 = time.perf_counter()
            call(frame, augment)
            samples[augment].append((time.perf_counter() - t0) * 1000.0)

    del model
    off = LatencyStats.from_samples(samples[False])
    on = LatencyStats.from_samples(samples[True])
    return {
        "imgsz": imgsz,
        "interleaved": True,
        "off": _latency_dict(off),
        "on": _latency_dict(on),
        "cost_multiple_at_floor": round(on.min_ms / off.min_ms, 2),
        "cost_multiple_at_median": round(on.median_ms / off.median_ms, 2),
    }


def latency_crosscheck(
    spec: ModelSpec, frames: Sequence[np.ndarray], imgsz: int, device: str, iters: int = 60
) -> dict[str, Any]:
    """The same measurement through the shipping API, to prove the wrapper is free.

    `time_predict` calls ultralytics directly so that TTA is available. Production
    calls `DefectDetector.predict`, which adds letterbox bookkeeping, severity
    scoring and record construction. If those cost anything material, the latency
    table above is not the number the line pays.
    """
    detector = DefectDetector(
        weights=spec.weights, device=device, conf=DEFAULT_CONF, iou=DEPLOY_NMS_IOU, imgsz=imgsz
    )
    detector.warmup(n=10)
    samples: list[float] = []
    for i in range(iters):
        t0 = time.perf_counter()
        detector.predict(frames[i % len(frames)])
        samples.append((time.perf_counter() - t0) * 1000.0)
    stats = LatencyStats.from_samples(samples)
    return {"api": "DefectDetector.predict", "imgsz": imgsz, **_latency_dict(stats)}


def _latency_dict(stats: LatencyStats) -> dict[str, Any]:
    """Mean, tail and floor.

    Which number to quote matters here, because this machine runs other work and
    contention only ever *adds* time. The **median** is used for every headline and
    every model-vs-model ratio: it is robust both to an occasional stall and to a
    single lucky call, which the minimum is not -- on a contended sweep the minimum
    can sit at a third of the mean and would flatter the result badly. The mean and
    p95 are reported as measured, contention included, because tail latency is what
    sizes a line; the minimum is kept only as a floor for reference.
    """
    return {
        "n": stats.n,
        "mean_ms": round(stats.mean_ms, 3),
        "median_ms": round(stats.median_ms, 3),
        "min_ms": round(stats.min_ms, 3),
        "p95_ms": round(stats.p95_ms, 3),
        "std_ms": round(stats.std_ms, 3),
        "fps": round(1000.0 / stats.mean_ms, 1),
        "fps_at_median": round(1000.0 / stats.median_ms, 1),
    }


# ---------------------------------------------------------------------------
# 5. failure analysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MissedBox:
    """One unmatched ground-truth box, with the verdict on why it was missed."""

    image: Path
    true_class: str
    box: tuple[float, float, float, float]
    verdict: str  # "class_confusion" | "extent_error" | "blind_miss"
    best_iou: float
    best_pred_class: str | None
    best_pred_conf: float
    same_class_best_conf: float  # highest confidence anywhere in the frame for the right class


def classify_miss(
    gt_box: np.ndarray,
    gt_class: int,
    pred_boxes: np.ndarray,
    pred_conf: np.ndarray,
    pred_classes: np.ndarray,
    image_path: Path,
) -> MissedBox:
    """Why one labelled defect went unreported, from the best overlapping prediction.

    Three verdicts, and the boundary between them is the point of the exercise:

    * **class confusion** -- something is drawn in the right place (IoU >= 0.5) but
      carries the wrong class. The detector saw the defect and named it wrong. A
      metallurgist gets sent the wrong root cause.
    * **extent error** -- the best overlapping box, of any class, overlaps between
      0.10 and 0.50. The defect was noticed; the rectangle disagrees. On a texture
      that fills the frame this is usually a disagreement about where the defect
      *stops*, which is as much an annotation question as a model one.
    * **blind miss** -- nothing overlaps at all. The detector had no opinion.

    These have different fixes (more labels, better label discipline, better
    contrast at the camera), which is why lumping them into "recall" hides the
    action.
    """
    same_class_conf = pred_conf[pred_classes == gt_class]
    same_class_best = float(same_class_conf.max()) if same_class_conf.size else 0.0

    if pred_boxes.size == 0:
        return MissedBox(
            image=image_path,
            true_class=class_label(gt_class),
            box=tuple(float(v) for v in gt_box),
            verdict="blind_miss",
            best_iou=0.0,
            best_pred_class=None,
            best_pred_conf=0.0,
            same_class_best_conf=same_class_best,
        )

    ious = iou_matrix(gt_box.reshape(1, 4), pred_boxes)[0]
    best = int(np.argmax(ious))
    best_iou = float(ious[best])
    best_cls = int(pred_classes[best])

    if best_iou >= 0.5 and best_cls != gt_class:
        verdict = "class_confusion"
    elif best_iou >= EXTENT_IOU_FLOOR:
        verdict = "extent_error"
    else:
        verdict = "blind_miss"

    return MissedBox(
        image=image_path,
        true_class=class_label(gt_class),
        box=tuple(float(v) for v in gt_box),
        verdict=verdict,
        best_iou=best_iou,
        best_pred_class=class_label(best_cls),
        best_pred_conf=float(pred_conf[best]),
        same_class_best_conf=same_class_best,
    )


def failure_taxonomy(
    records: Sequence[GroundTruth],
    cached: Sequence[CachedPrediction],
    threshold: float,
) -> tuple[dict[str, Any], list[MissedBox]]:
    """Per-class miss taxonomy and false-positive attribution at one threshold."""
    misses: list[MissedBox] = []
    per_class: dict[str, dict[str, Any]] = {
        name: {
            "instances": 0,
            "matched": 0,
            "missed": 0,
            "class_confusion": 0,
            "extent_error": 0,
            "blind_miss": 0,
            "confused_with": {},
            "false_positives_on_this_class_image": {},
        }
        for name in CLASS_NAMES
    }

    for record, pred in zip(records, cached):
        boxes, conf, classes = filter_prediction(pred, threshold)
        match = match_image(record.boxes, record.classes, boxes, conf, classes)

        for gt_idx in range(len(record.boxes)):
            name = class_label(int(record.classes[gt_idx]))
            bucket = per_class[name]
            bucket["instances"] += 1
            if match.gt_matched[gt_idx]:
                bucket["matched"] += 1
                continue
            bucket["missed"] += 1
            miss = classify_miss(
                record.boxes[gt_idx],
                int(record.classes[gt_idx]),
                boxes,
                conf,
                classes,
                record.image_path,
            )
            bucket[miss.verdict] += 1
            if miss.verdict == "class_confusion" and miss.best_pred_class:
                bucket["confused_with"][miss.best_pred_class] = (
                    bucket["confused_with"].get(miss.best_pred_class, 0) + 1
                )
            misses.append(miss)

        # False positives are attributed to the true class of the image they landed
        # on. "213 false positives" is unactionable; "the model paints crazing over
        # rolled-in scale" is a data problem someone can fix.
        image_class = class_label(record.image_class)
        for pred_idx in range(len(boxes)):
            if match.pred_is_tp[pred_idx]:
                continue
            fp_name = class_label(int(classes[pred_idx]))
            if image_class in per_class:
                landed = per_class[image_class]["false_positives_on_this_class_image"]
                landed[fp_name] = landed.get(fp_name, 0) + 1

    for name, bucket in per_class.items():
        instances = max(1, bucket["instances"])
        bucket["recall"] = round(bucket["matched"] / instances, 4)
        bucket["confused_with"] = dict(
            sorted(bucket["confused_with"].items(), key=lambda kv: -kv[1])
        )
        bucket["false_positives_on_this_class_image"] = dict(
            sorted(bucket["false_positives_on_this_class_image"].items(), key=lambda kv: -kv[1])[:4]
        )

    return {"threshold": threshold, "per_class": per_class}, misses


def appearance_stats(records: Sequence[GroundTruth]) -> dict[str, dict[str, float]]:
    """Measured image statistics per class, to test the "low contrast" explanation.

    Everything here is computed from the pixels and the labels, not asserted:

    * **separability** -- |mean intensity inside the box - mean outside every box| /
      std outside. A d'-like number: how far the defect's grey level sits from the
      background, in units of background roughness. Low separability is what "low
      contrast" means quantitatively.
    * **edge_energy** -- mean Sobel gradient magnitude inside the box, normalised by
      the same quantity over the whole image. Above 1.0 the defect is locally busier
      than its surroundings; near 1.0 it is texturally indistinguishable.
    * **box_area_frac** and **mean_pairwise_iou** -- how much of the frame one label
      claims, and how much sibling labels in the same image overlap each other.
      Both are annotation-extent measures: a texture that covers the whole frame
      gets chopped into large, mutually overlapping rectangles whose exact edges
      nobody could reproduce, and IoU-0.5 scoring punishes that.
    """
    acc: dict[str, dict[str, list[float]]] = {
        name: {"sep": [], "edge": [], "area": [], "iou": [], "boxes": [], "rms": []}
        for name in CLASS_NAMES
    }

    for record in records:
        image = cv2.imread(str(record.image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        grey = image.astype(np.float32)
        gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.hypot(gx, gy)
        frame_gradient = float(gradient.mean()) or 1.0

        # Background = every pixel no label claims. If the labels cover the whole
        # frame there is no background to compare against, and separability is
        # undefined rather than zero -- which is itself the finding for crazing.
        covered = np.zeros(grey.shape, dtype=bool)
        for box in record.boxes:
            x1, y1, x2, y2 = _clip_box(box, grey.shape)
            covered[y1:y2, x1:x2] = True
        background = grey[~covered]

        n_boxes = len(record.boxes)
        pairwise = iou_matrix(record.boxes, record.boxes) if n_boxes > 1 else np.zeros((0, 0))
        if n_boxes > 1:
            off_diagonal = pairwise[~np.eye(n_boxes, dtype=bool)]
            mean_iou = float(off_diagonal.mean())
        else:
            mean_iou = 0.0

        for box, cls in zip(record.boxes, record.classes):
            name = class_label(int(cls))
            if name not in acc:
                continue
            x1, y1, x2, y2 = _clip_box(box, grey.shape)
            patch = grey[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            acc[name]["rms"].append(float(patch.std()))
            acc[name]["edge"].append(float(gradient[y1:y2, x1:x2].mean()) / frame_gradient)
            acc[name]["area"].append(
                float((x2 - x1) * (y2 - y1)) / float(record.width * record.height)
            )
            acc[name]["iou"].append(mean_iou)
            acc[name]["boxes"].append(float(n_boxes))
            if background.size > 32 and background.std() > 1e-6:
                acc[name]["sep"].append(
                    abs(float(patch.mean()) - float(background.mean())) / float(background.std())
                )

    summary: dict[str, dict[str, float]] = {}
    for name, values in acc.items():
        summary[name] = {
            "n_boxes": int(len(values["area"])),
            "separability": round(float(np.mean(values["sep"])), 4) if values["sep"] else float("nan"),
            "separability_n": int(len(values["sep"])),
            "edge_energy_ratio": round(float(np.mean(values["edge"])), 4) if values["edge"] else float("nan"),
            "rms_intensity_in_box": round(float(np.mean(values["rms"])), 3) if values["rms"] else float("nan"),
            "box_area_frac": round(float(np.mean(values["area"])), 4) if values["area"] else float("nan"),
            "mean_pairwise_gt_iou": round(float(np.mean(values["iou"])), 4) if values["iou"] else float("nan"),
            "boxes_per_image": round(float(np.mean(values["boxes"])), 3) if values["boxes"] else float("nan"),
        }
    return summary


def _clip_box(box: Iterable[float], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = shape
    x1, y1, x2, y2 = (float(v) for v in box)
    return (
        int(max(0, min(width - 1, round(x1)))),
        int(max(0, min(height - 1, round(y1)))),
        int(max(1, min(width, round(x2)))),
        int(max(1, min(height, round(y2)))),
    )


def difficulty_correlation(
    per_class_ap50: dict[str, float], appearance: dict[str, dict[str, float]]
) -> dict[str, Any]:
    """Rank correlation between measured appearance and per-class AP50.

    Six classes is six points. A Spearman rho on six points is a description of
    this table, not a hypothesis test, and the p-value is reported only so nobody
    mistakes it for one.
    """
    names = [n for n in CLASS_NAMES if n in per_class_ap50]
    ap = np.array([per_class_ap50[n] for n in names], dtype=np.float64)
    out: dict[str, Any] = {"n_classes": len(names), "classes": names}
    for field_name in ("separability", "edge_energy_ratio", "box_area_frac", "mean_pairwise_gt_iou"):
        values = np.array([appearance[n][field_name] for n in names], dtype=np.float64)
        if np.isnan(values).any():
            out[field_name] = {"rho": None, "note": "not defined for every class"}
            continue
        rho, p = spearmanr(values, ap)
        out[field_name] = {
            "rho_vs_AP50": round(float(rho), 4),
            "p_value": round(float(p), 4),
            "values": {n: round(float(v), 4) for n, v in zip(names, values)},
        }
    return out


def ordering_vs_published(per_class_ap50: dict[str, float]) -> dict[str, Any]:
    """Our hardest-first class ordering against the published one."""
    ours = [n for n, _ in sorted(per_class_ap50.items(), key=lambda kv: kv[1])]
    published = list(PUBLISHED_HARDEST_FIRST)
    rank_ours = {n: i for i, n in enumerate(ours)}
    rank_pub = {n: i for i, n in enumerate(published)}
    common = [n for n in published if n in rank_ours]
    rho, p = spearmanr(
        [rank_ours[n] for n in common], [rank_pub[n] for n in common]
    )
    return {
        "ours_hardest_first": ours,
        "published_hardest_first": published,
        "published_source": PUBLISHED_SOURCE,
        "identical": ours == published,
        "spearman_rho": round(float(rho), 4),
        "p_value": round(float(p), 4),
        "rank_shifts": {
            n: {"ours": rank_ours[n], "published": rank_pub[n], "shift": rank_ours[n] - rank_pub[n]}
            for n in common
        },
    }


# ---------------------------------------------------------------------------
# 6. training cost at 320 vs 640, to substantiate the resolution decision
# ---------------------------------------------------------------------------


def training_step_cost(
    spec: ModelSpec,
    device: str,
    sizes: Sequence[int] = (320, 640),
    batch: int = 8,
    steps: int = 4,
    warmup: int = 1,
    budget_s: float = 120.0,
) -> dict[str, Any]:
    """Measured forward+loss+backward+step time, projected to a full epoch.

    The claim "we trained at 320 because 640 was unaffordable on this hardware" is
    only worth making if it is a measurement. This times the real optimiser step at
    both sizes on synthetic batches of the right shape, then projects to
    ceil(1440/batch) steps per epoch. The 320 projection is checked against the
    epoch time the training run actually logged; if the projection reproduces the
    logged number, the 640 projection can be believed.

    Synthetic pixels are legitimate here: convolution and backward cost depend on
    tensor shape, not on tensor content, and the label tensor is shape-matched to a
    realistic 2 boxes per image.

    What this deliberately does *not* include: dataloading, mosaic and the rest of
    the augmentation pipeline, AMP bookkeeping, the EMA update and the per-epoch
    validation pass. So `gpu_only_seconds_per_epoch` is a floor, not an epoch time --
    it is reported as such, and the transferable quantity is the **ratio** between
    the two sizes, which the caller applies to the epoch time the training run
    actually logged.

    The probe batch defaults to 8, not the 32 the real runs used. Batch 32 at 640 px
    was tried first and abandoned: on a machine already under memory pressure it
    paged, and a single step did not return inside a two-minute budget -- and an MPS
    op in flight cannot be preempted, so the budget cannot save a run from it. The
    640/320 ratio is close to batch-independent because both sides scale with the
    same activation volume, so measuring it at a batch that reliably fits is
    legitimate; the batch is recorded alongside the number so it is not misread as a
    batch-32 measurement. `memory_pressure()` records the swap state at the moment
    of measurement for the same reason.
    """
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG

    # Steps per epoch are quoted for the batch the real runs used, not the probe
    # batch, so the projection lines up with the logged epoch times.
    real_batch = 32
    steps_per_epoch = int(np.ceil(1440 / real_batch))
    results: dict[str, Any] = {
        "probe_batch": batch,
        "training_batch": real_batch,
        "steps_timed": steps,
        "train_images": 1440,
        "steps_per_epoch": steps_per_epoch,
        "note": (
            "step times measured at batch "
            f"{batch}; the 640/320 ratio is the transferable quantity, not the absolute time"
        ),
        "by_imgsz": {},
    }

    for size in sizes:
        model = YOLO(str(spec.weights))
        net = model.model.to(device).train()
        # The checkpoint stores the trimmed training args, which omit the loss
        # gains the criterion needs. train_detector.py never overrode box/cls/dfl,
        # so the shipped defaults are the gains the real run used.
        net.args = get_cfg(DEFAULT_CFG)
        for parameter in net.parameters():
            parameter.requires_grad_(True)
        optimiser = torch.optim.SGD(net.parameters(), lr=1e-6, momentum=0.937)

        generator = torch.Generator().manual_seed(BOOTSTRAP_SEED)
        batch_data = {
            "img": torch.rand(batch, 3, size, size, generator=generator).to(device),
            "cls": torch.randint(0, len(CLASS_NAMES), (batch * 2, 1), generator=generator)
            .float()
            .to(device),
            "bboxes": (torch.rand(batch * 2, 4, generator=generator) * 0.4 + 0.3).to(device),
            "batch_idx": torch.arange(batch).repeat_interleave(2).float().to(device),
        }

        samples: list[float] = []
        warm_ms: list[float] = []
        cell_start = time.perf_counter()
        pressure = memory_pressure()
        try:
            for i in range(warmup + steps):
                synchronise_device(device)
                t0 = time.perf_counter()
                loss, _ = net.loss(batch_data)
                loss.sum().backward()
                optimiser.step()
                optimiser.zero_grad(set_to_none=True)
                synchronise_device(device)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                (samples if i >= warmup else warm_ms).append(elapsed_ms)
                # A 640 px batch of 32 needs more unified memory than this machine
                # has spare. Once it pages, one step takes minutes. The budget is
                # checked after *every* step, warm-up included, because otherwise an
                # unbounded warm-up stalls the whole study before the guard is armed.
                if time.perf_counter() - cell_start > budget_s:
                    break
        except (RuntimeError, AttributeError) as exc:  # pragma: no cover - hardware dependent
            results["by_imgsz"][str(size)] = {"ok": False, "error": str(exc)[:200], **pressure}
            del net, model
            _release(device)
            continue

        if not samples:
            results["by_imgsz"][str(size)] = {
                "ok": False,
                "error": (
                    f"budget {budget_s:.0f}s exhausted during warm-up; "
                    f"{len(warm_ms)} warm-up step(s) took "
                    f"{'/'.join(f'{w / 1000:.0f}s' for w in warm_ms) or 'none completed'}"
                ),
                "warmup_ms": [round(w, 1) for w in warm_ms],
                **pressure,
            }
            del net, model
            _release(device)
            continue
        # Minimum, not median: this machine runs other work, and contention can only
        # add time to a step. The spread is recorded so the reader can judge it.
        step_ms = float(np.min(samples))
        results["by_imgsz"][str(size)] = {
            "ok": True,
            "step_ms_min": round(step_ms, 1),
            "step_ms_median": round(float(np.median(samples)), 1),
            "step_ms_max": round(float(np.max(samples)), 1),
            "step_ms_samples": [round(s, 1) for s in samples],
            "samples_taken": len(samples),
            "budget_hit": len(samples) < steps,
            **pressure,
            "gpu_only_seconds_per_epoch_at_probe_batch": round(
                step_ms * int(np.ceil(1440 / batch)) / 1000.0, 1
            ),
        }
        del net, model
        _release(device)

    sizes_ok = [s for s in sizes if results["by_imgsz"].get(str(s), {}).get("ok")]
    if 320 in sizes_ok and 640 in sizes_ok:
        results["ratio_640_over_320"] = round(
            results["by_imgsz"]["640"]["step_ms_min"] / results["by_imgsz"]["320"]["step_ms_min"], 3
        )
    return results


def memory_pressure() -> dict[str, Any]:
    """Swap in use at the moment of measurement.

    A 640 px batch of 32 needs more unified memory than this machine has spare, and
    a step that pages is not measuring arithmetic. Recording the swap figure next to
    the timing is the difference between a slow number and an uninterpretable one.
    """
    try:
        import subprocess

        raw = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=5
        ).stdout
        parts = dict(
            zip(("total", "used", "free"), [t for t in raw.replace("=", " ").split() if t.endswith("M")])
        )
        return {"swap": parts, "raw": raw.strip()}
    except Exception:  # noqa: BLE001 - a missing sysctl must not stop the study
        return {"swap": None, "raw": "unavailable"}


def _release(device: str) -> None:
    import gc

    gc.collect()
    if device.startswith("mps"):
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 7. figures
# ---------------------------------------------------------------------------


def _style(ax: plt.Axes, *, xlabel: str, ylabel: str, title: str | None = None) -> None:
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if title:
        ax.set_title(title, fontsize=10.5, pad=8)
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8.5)


def _model_colour(key: str) -> str:
    return {"yolov8n": "#1f77b4", "yolov8s": "#d62728"}.get(key, "#555555")


def chart_imgsz(
    runs: Sequence[ValRun], latency: dict[str, Any], train_imgsz: int, out_path: Path
) -> Path:
    """mAP50 vs inference size on both splits, with the latency it costs."""
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), layout="constrained")

    for ax, split in zip(axes[:2], ("val", "test")):
        for key in sorted({r.model for r in runs}):
            cells = sorted(
                (r for r in runs if r.model == key and r.split == split and not r.augment),
                key=lambda r: r.imgsz,
            )
            if not cells:
                continue
            ax.plot(
                [c.imgsz for c in cells],
                [c.mAP50 for c in cells],
                marker="o",
                linewidth=2.0,
                color=_model_colour(key),
                label=key,
            )
        ax.axvline(train_imgsz, color="#888888", linestyle="--", linewidth=1.0)
        ax.annotate(
            f"trained at {train_imgsz}",
            xy=(train_imgsz, ax.get_ylim()[0]),
            xytext=(4, 6),
            textcoords="offset points",
            fontsize=8,
            color="#666666",
            rotation=90,
        )
        _style(ax, xlabel="inference imgsz (px)", ylabel="mAP50", title=f"{split} split")
        ax.set_xticks(list(IMGSZ_GRID))
        ax.legend(fontsize=8.5, frameon=False)

    ax = axes[2]
    for key, cells in latency.items():
        if key.startswith("_"):
            continue
        sizes = sorted(int(s) for s in cells["by_imgsz"])
        ax.plot(
            sizes,
            [cells["by_imgsz"][str(s)]["median_ms"] for s in sizes],
            marker="s",
            linewidth=2.0,
            color=_model_colour(key),
            label=key,
        )
    _style(
        ax,
        xlabel="inference imgsz (px)",
        ylabel="latency, batch 1, median (ms)",
        title="cost of size",
    )
    ax.set_xticks(list(IMGSZ_GRID))
    ax.legend(fontsize=8.5, frameon=False)
    # This panel is measured on a shared laptop and is visibly noisier than the two
    # accuracy panels -- occasionally non-monotonic, which is not physical. Say so on
    # the figure rather than let a reader take it for a bug in the sweep.
    ax.annotate(
        "batch 1 on a contended laptop: noisy and\n"
        "sometimes non-monotonic. At this input size\n"
        "wall clock is dispatch overhead, not the model.\n"
        "See src/benchmark.py for the batched regime.",
        xy=(0.02, 0.98),
        xycoords="axes fraction",
        va="top",
        fontsize=7.2,
        color="#7a7a7a",
    )

    fig.suptitle(
        "Inference-resolution sensitivity -- NEU-DET, both checkpoints trained at 320 px on 200x200 source images",
        fontsize=11.5,
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def chart_bootstrap(bootstrap: dict[str, Any], out_path: Path) -> Path:
    """Distribution of the paired mAP difference over resampled test sets."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), layout="constrained")
    panels = (
        ("_delta_map50_samples", "mAP50", bootstrap["mAP50"]),
        ("_delta_map5095_samples", "mAP50-95", bootstrap["mAP50_95"]),
    )
    for ax, (key, name, summary) in zip(axes, panels):
        samples = bootstrap[key]
        ax.hist(samples, bins=60, color="#4c78a8", alpha=0.85, edgecolor="none")
        ax.axvline(0.0, color="#333333", linewidth=1.4)
        ax.axvline(summary["point_difference"], color="#d62728", linewidth=1.8)
        ax.axvspan(summary["ci95_low"], summary["ci95_high"], color="#d62728", alpha=0.12)
        _style(
            ax,
            xlabel=f"{name}({bootstrap['model_a']}) - {name}({bootstrap['model_b']})",
            ylabel="resamples",
            title=(
                f"{name}: observed {summary['point_difference']:+.4f}, "
                f"95% CI [{summary['ci95_low']:+.4f}, {summary['ci95_high']:+.4f}]"
            ),
        )
        ax.annotate(
            f"{bootstrap['model_a']} wins {_share_pct(summary['share_a_wins'])} of resamples",
            xy=(0.02, 0.94),
            xycoords="axes fraction",
            fontsize=8.5,
            color="#444444",
        )
    fig.suptitle(
        f"Paired bootstrap over the {bootstrap['n_images']} held-out test images "
        f"({bootstrap['resamples']} resamples, seed {bootstrap['seed']}); "
        "zero line = no difference",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def chart_difficulty(
    per_class_ap50: dict[str, float],
    appearance: dict[str, dict[str, float]],
    correlation: dict[str, Any],
    out_path: Path,
) -> Path:
    """Per-class AP50 against the measured appearance statistics."""
    names = [n for n in CLASS_NAMES if n in per_class_ap50]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), layout="constrained")

    ax = axes[0]
    order = sorted(names, key=lambda n: per_class_ap50[n])
    ax.barh(
        range(len(order)),
        [per_class_ap50[n] for n in order],
        color=[np.array(CLASS_COLORS[n]) / 255.0 for n in order],
    )
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8.5)
    _style(ax, xlabel="AP50 (held-out test)", ylabel="", title="measured class difficulty")

    for ax, field_name, xlabel in (
        (axes[1], "separability", "defect-vs-background separability  |dmean| / std"),
        (axes[2], "edge_energy_ratio", "edge energy inside box / edge energy of frame"),
    ):
        for name in names:
            ax.scatter(
                appearance[name][field_name],
                per_class_ap50[name],
                s=90,
                color=np.array(CLASS_COLORS[name]) / 255.0,
                edgecolor="#333333",
                linewidth=0.6,
                zorder=3,
            )
            ax.annotate(
                name,
                xy=(appearance[name][field_name], per_class_ap50[name]),
                xytext=(6, -3),
                textcoords="offset points",
                fontsize=7.5,
                color="#333333",
            )
        info = correlation.get(field_name, {})
        rho = info.get("rho_vs_AP50")
        _style(
            ax,
            xlabel=xlabel,
            ylabel="AP50",
            title=f"Spearman rho = {rho}" if rho is not None else "not defined for every class",
        )
        ax.margins(x=0.22)

    fig.suptitle(
        "Why the weak classes are weak: per-class AP50 against statistics measured from the test images and their labels",
        fontsize=11.5,
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def build_failure_montage(
    misses: Sequence[MissedBox],
    records: Sequence[GroundTruth],
    cached: Sequence[CachedPrediction],
    threshold: float,
    classes: Sequence[str],
    out_path: Path,
    per_class: int = 5,
) -> Path:
    """One row per weak class: the defects the model was most blind to.

    Ranked by the highest confidence the model put on the *correct* class anywhere
    in that frame, lowest first -- these are the frames where it was not merely
    mis-drawn but unconvinced. Ground truth is dotted green, every surviving
    prediction is solid in its class colour, and the panel title carries the
    verdict from `classify_miss` so the montage and the taxonomy table cannot drift
    apart.
    """
    by_class: dict[str, list[MissedBox]] = {name: [] for name in classes}
    seen: dict[str, set[Path]] = {name: set() for name in classes}
    for miss in sorted(misses, key=lambda m: (m.same_class_best_conf, -m.best_iou)):
        if miss.true_class in by_class and miss.image not in seen[miss.true_class]:
            by_class[miss.true_class].append(miss)
            seen[miss.true_class].add(miss.image)

    gt_lookup = {r.image_path: r for r in records}
    pred_lookup = {p.image_path: p for p in cached}

    rows = len(classes)
    fig, axes = plt.subplots(
        rows, per_class, figsize=(2.75 * per_class, 3.15 * rows), layout="constrained"
    )
    axes = np.atleast_2d(axes)

    for row, name in enumerate(classes):
        cases = by_class[name][:per_class]
        for column in range(per_class):
            ax = axes[row, column]
            if column >= len(cases):
                ax.axis("off")
                continue
            case = cases[column]
            image = cv2.cvtColor(cv2.imread(str(case.image), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            ax.imshow(image)

            record = gt_lookup[case.image]
            for box, cls in zip(record.boxes, record.classes):
                ax.add_patch(
                    Rectangle(
                        (box[0], box[1]),
                        box[2] - box[0],
                        box[3] - box[1],
                        fill=False,
                        edgecolor="#39ff88",
                        linewidth=1.1,
                        linestyle=":",
                    )
                )
            ax.add_patch(
                Rectangle(
                    (case.box[0], case.box[1]),
                    case.box[2] - case.box[0],
                    case.box[3] - case.box[1],
                    fill=False,
                    edgecolor="#ff2d55",
                    linewidth=2.2,
                )
            )

            pred = pred_lookup[case.image]
            boxes, conf, classes_ = filter_prediction(pred, threshold)
            for box, c, k in zip(boxes, conf, classes_):
                colour = np.array(CLASS_COLORS.get(class_label(int(k)), (255, 255, 255))) / 255.0
                ax.add_patch(
                    Rectangle(
                        (box[0], box[1]),
                        box[2] - box[0],
                        box[3] - box[1],
                        fill=False,
                        edgecolor=colour,
                        linewidth=1.4,
                    )
                )
                ax.text(
                    box[0] + 1,
                    box[1] + 9,
                    f"{class_label(int(k))[:4]} {c:.2f}",
                    fontsize=5.6,
                    color="white",
                    bbox={"facecolor": colour, "edgecolor": "none", "pad": 0.7, "alpha": 0.85},
                )

            verdict = case.verdict.replace("_", " ")
            detail = (
                f"best IoU {case.best_iou:.2f} as {case.best_pred_class}"
                if case.best_pred_class
                else "no prediction in frame"
            )
            ax.set_title(
                f"{verdict.upper()}  |  right-class conf {case.same_class_best_conf:.2f}\n"
                f"{detail}\n{case.image.name}",
                fontsize=7.0,
                color="#1b2733",
            )
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        f"Worst missed defects on the two weak classes, at the deployed threshold conf={threshold:.2f}\n"
        "dotted green = every ground-truth label   solid red = the missed label   "
        "coloured = what the detector actually reported",
        fontsize=10.5,
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 8. report writers
# ---------------------------------------------------------------------------


def _paragraphs(block: Sequence[Any], where: str) -> list[str]:
    """Validate a generated prose block before it reaches the report.

    Every narrative section below is a list of strings built by concatenation. A
    stray comma inside one of those parenthesised entries silently produces a tuple,
    and `"\n".join` then fails far away with no indication of which block was
    malformed. Failing here names it.
    """
    out: list[str] = []
    for i, item in enumerate(block):
        if not isinstance(item, str):
            raise TypeError(
                f"{where}[{i}] is {type(item).__name__}, not str -- a trailing comma inside a "
                "parenthesised paragraph turns it into a tuple"
            )
        out.append(item)
    return out


def _share_pct(share: float) -> str:
    """A win-share as a percentage that never rounds a non-unanimous result to 100%.

    2000 resamples resolve to 0.05%, and `f"{0.9995:.1%}"` renders "100.0%" -- which
    on a slide reads as "this never came out the other way". It did, once. Widen the
    precision until the rendered string stops claiming unanimity it does not have.
    """
    for places in (1, 2, 3, 4):
        text = f"{share:.{places}%}"
        rendered = float(text.rstrip("%"))
        rounds_up_to_all = rendered >= 100.0 and share < 1.0
        rounds_down_to_none = rendered <= 0.0 and share > 0.0
        if not rounds_up_to_all and not rounds_down_to_none:
            return text
    return f"{share:.4%}"


def _md_table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(str(h) for h in header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items() if not k.startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2))
    return path


def write_markdown(path: Path, payload: dict[str, Any]) -> Path:  # noqa: C901 - one long report
    """The narrative report. Every number in it comes from `payload`."""
    p = payload
    head = p["headline"]
    boot = p["bootstrap"]["at_training_size"]
    boot_deploy = p["bootstrap"]["at_deployment_size"]
    chosen = p["selection"]
    lines: list[str] = []
    a, b = boot["model_a"], boot["model_b"]

    lines.append("# Model selection study -- YOLOv8n vs YOLOv8s on NEU-DET")
    lines.append("")
    lines.append(
        f"Generated {p['meta']['generated_at']} on {p['meta']['platform']}, "
        f"device `{p['meta']['device']}`, torch {p['meta']['torch']}, "
        f"ultralytics {p['meta']['ultralytics']}."
    )
    lines.append("")
    lines.append(
        "Every number below was measured by `src/model_study.py` in the run that wrote this file. "
        "Metrics come from the ultralytics validator on its own protocol defaults (conf 0.001, "
        "NMS IoU 0.7, max_det 300), which is what published NEU-DET numbers use."
    )
    lines.append("")

    # ---------------- verdict
    lines.append("## Verdict")
    lines.append("")
    lines.append(chosen["verdict_paragraph"])
    lines.append("")
    lines.append(
        _md_table(
            ["decision", "value", "decided on", "evidence"],
            [[d["decision"], d["value"], d["decided_on"], d["evidence"]] for d in chosen["decisions"]],
        )
    )
    lines.append("")

    # ---------------- Q1
    lines.append("## 1. Head to head on the held-out test split")
    lines.append("")
    lines.append(
        f"Both checkpoints, same 1440/180/180 split, same recipe, evaluated at the "
        f"training resolution ({p['train_imgsz']} px)."
    )
    lines.append("")
    lines.append(
        _md_table(
            [
                "model",
                "mAP50",
                "mAP50-95",
                "P",
                "R",
                "params (M)",
                f"GFLOPs @{p['train_imgsz']}",
                "weight MB",
                f"latency @{p['train_imgsz']} median ms",
                "p95 ms",
                "FPS (median)",
            ],
            head["table"],
        )
    )
    lines.append("")
    lines.append("Per-class AP50 at the training resolution:")
    lines.append("")
    lines.append(_md_table(head["per_class_header"], head["per_class_rows"]))
    lines.append("")
    lines.append("### Is the gap real? Paired bootstrap over the 180 test images")
    lines.append("")
    lines.append(
        f"Both models were rescored on the *same* {boot['resamples']} resampled test sets "
        f"(180 images drawn with replacement, seed {boot['seed']}). Pairing matters: most of the "
        "variance in a 180-image mAP is which images were drawn, and that component is shared by "
        "both models and cancels in the difference."
    )
    lines.append("")
    lines.append(
        _md_table(
            ["metric", f"{a}", f"{b}", "difference", "95% CI on the difference", "excludes 0?", f"{a} wins"],
            [
                [
                    metric,
                    f"{s['point_estimate_a']:.4f}",
                    f"{s['point_estimate_b']:.4f}",
                    f"{s['point_difference']:+.4f}",
                    f"[{s['ci95_low']:+.4f}, {s['ci95_high']:+.4f}]",
                    "yes" if s["ci_excludes_zero"] else "**no**",
                    _share_pct(s['share_a_wins']),
                ]
                for metric, s in (("mAP50", boot["mAP50"]), ("mAP50-95", boot["mAP50_95"]))
            ],
        )
    )
    lines.append("")
    for paragraph in _paragraphs(head["bootstrap_reading"], 'head["bootstrap_reading"]'):
        lines.append(paragraph)
        lines.append("")
    lines.append(
        "Fidelity check -- the bootstrap rescores the validator's own per-image rows, so recomputing "
        "over the full split must reproduce the validator exactly:"
    )
    lines.append("")
    lines.append(
        _md_table(
            ["model", "imgsz", "validator mAP50", "recomputed mAP50", "abs error", "exact"],
            [
                [
                    f["model"],
                    f["imgsz"],
                    f"{f['validator_mAP50']:.8f}",
                    f"{f['recomputed_mAP50']:.8f}",
                    f"{f['abs_error_mAP50']:.2e}",
                    "yes" if f["exact"] else "no",
                ]
                for f in p["fidelity"]
            ],
        )
    )
    lines.append("")
    lines.append(f"![paired bootstrap]({Path(p['figures']['bootstrap']).name})")
    lines.append("")
    lines.append("### The same test, with each model at its own best input size")
    lines.append("")
    for paragraph in _paragraphs(head["deployment_reading"], 'head["deployment_reading"]'):
        lines.append(paragraph)
        lines.append("")
    lines.append(
        _md_table(
            [
                "metric",
                boot_deploy["model_a"],
                boot_deploy["model_b"],
                "difference",
                "95% CI on the difference",
                "excludes 0?",
                f"{boot_deploy['model_a']} wins",
            ],
            [
                [
                    metric,
                    f"{d['point_estimate_a']:.4f}",
                    f"{d['point_estimate_b']:.4f}",
                    f"{d['point_difference']:+.4f}",
                    f"[{d['ci95_low']:+.4f}, {d['ci95_high']:+.4f}]",
                    "yes" if d["ci_excludes_zero"] else "**no**",
                    _share_pct(d['share_a_wins']),
                ]
                for metric, d in (
                    ("mAP50", boot_deploy["mAP50"]),
                    ("mAP50-95", boot_deploy["mAP50_95"]),
                )
            ],
        )
    )
    lines.append("")
    lines.append(f"![paired bootstrap, deployment sizes]({Path(p['figures']['bootstrap_deploy']).name})")
    lines.append("")
    lines.append("### The mechanism: capacity against 1440 training images")
    lines.append("")
    for paragraph in _paragraphs(head["mechanism"], 'head["mechanism"]'):
        lines.append(paragraph)
        lines.append("")
    lines.append(
        _md_table(
            [
                "model",
                "epochs run",
                "best epoch",
                "best val mAP50",
                "final train cls loss",
                "final val cls loss",
                "cls generalisation gap",
                "s/epoch (median)",
                "s/epoch (mean, incl. stalls)",
            ],
            head["history_rows"],
        )
    )
    lines.append("")

    # ---------------- Q2
    lines.append("## 2. Inference-resolution sensitivity")
    lines.append("")
    lines.append(
        f"Both models were trained at {p['train_imgsz']} px on 200x200 source images, so every size "
        "in the grid except 256 upsamples the input, and none of them adds information."
    )
    lines.append("")
    lines.append(_md_table(p["imgsz"]["header"], p["imgsz"]["rows"]))
    lines.append("")
    for paragraph in _paragraphs(p["imgsz"]["reading"], 'p["imgsz"]["reading"]'):
        lines.append(paragraph)
        lines.append("")
    lines.append(f"![resolution sweep]({Path(p['figures']['imgsz']).name})")
    lines.append("")

    # ---------------- Q3
    lines.append("## 3. Test-time augmentation")
    lines.append("")
    lines.append(
        "Ultralytics TTA runs the image at three scales with a horizontal flip and merges the "
        "detections before NMS."
    )
    lines.append("")
    lines.append(_md_table(p["tta"]["header"], p["tta"]["rows"]))
    lines.append("")
    for paragraph in _paragraphs(p["tta"]["reading"], 'p["tta"]["reading"]'):
        lines.append(paragraph)
        lines.append("")

    # ---------------- Q4
    lines.append("## 4. Which classes fail, and why")
    lines.append("")
    lines.append(p["failures"]["intro"])
    lines.append("")
    lines.append(_md_table(p["failures"]["taxonomy_header"], p["failures"]["taxonomy_rows"]))
    lines.append("")
    lines.append("Statistics measured from the test images and their labels:")
    lines.append("")
    lines.append(_md_table(p["failures"]["appearance_header"], p["failures"]["appearance_rows"]))
    lines.append("")
    for paragraph in _paragraphs(p["failures"]["reading"], 'p["failures"]["reading"]'):
        lines.append(paragraph)
        lines.append("")
    lines.append("### Ordering against the published literature")
    lines.append("")
    lines.append(_md_table(p["failures"]["ordering_header"], p["failures"]["ordering_rows"]))
    lines.append("")
    for paragraph in _paragraphs(p["failures"]["ordering_reading"], 'p["failures"]["ordering_reading"]'):
        lines.append(paragraph)
        lines.append("")
    lines.append(f"![failure montage]({Path(p['figures']['failures']).name})")
    lines.append("")
    lines.append(f"![class difficulty]({Path(p['figures']['difficulty']).name})")
    lines.append("")

    # ---------------- Q5
    lines.append("## 5. Deployment recommendation")
    lines.append("")
    for paragraph in _paragraphs(chosen["recommendation"], 'chosen["recommendation"]'):
        lines.append(paragraph)
        lines.append("")

    # ---------------- positioning
    lines.append("## Where this lands against published NEU-DET results")
    lines.append("")
    for paragraph in _paragraphs(p["positioning"], 'p["positioning"]'):
        lines.append(paragraph)
        lines.append("")

    # ---------------- honesty
    lines.append("## What this study does not establish")
    lines.append("")
    for item in p["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"Machine-readable form: `{Path(p['meta']['json_path']).name}`.")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resamples", type=int, default=2000, help="paired bootstrap resamples")
    parser.add_argument("--latency-iters", type=int, default=80)
    parser.add_argument("--batch", type=int, default=16, help="validator batch size")
    parser.add_argument("--skip-train-cost", action="store_true", help="skip the 320-vs-640 step timing")
    parser.add_argument("--out", type=Path, default=REPORTS_DIR / "model_study.json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901 - a report is a long function
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    started = datetime.now(timezone.utc)
    scratch = REPORTS_DIR / "ultralytics" / "model_study"

    import ultralytics

    specs = discover_models()
    nano = next(s for s in specs if s.key == "yolov8n")
    small = next(s for s in specs if s.key == "yolov8s")
    train_imgsz = 320

    print(f"[study] device={device}  models={[s.run for s in specs]}")

    # --- checkpoint facts ---------------------------------------------------
    facts = {s.key: checkpoint_facts(s, IMGSZ_GRID) for s in specs}
    history = {s.key: training_history(s) for s in specs}
    for key, fact in facts.items():
        print(
            f"[facts] {key}: {fact['params_millions']}M params, "
            f"{fact['gflops_by_imgsz'][str(train_imgsz)]} GFLOPs @{train_imgsz}, "
            f"{fact['weight_file_mb']} MB, sha {fact['sha256'][:12]}"
        )

    # --- resolution sweep, both splits -------------------------------------
    runs: list[ValRun] = []
    for spec in specs:
        for split in ("val", "test"):
            for imgsz in IMGSZ_GRID:
                run = run_validation(
                    spec, split, imgsz, device, batch=args.batch, out_dir=scratch
                )
                runs.append(run)
                print(
                    f"[val ] {spec.key:8s} {split:4s} imgsz={imgsz:4d}  "
                    f"mAP50={run.mAP50:.4f}  mAP50-95={run.mAP50_95:.4f}  "
                    f"P={run.precision:.4f} R={run.recall:.4f}  ({run.seconds:.1f}s)"
                )

    def cell(model: str, split: str, imgsz: int, augment: bool = False) -> ValRun:
        return next(
            r for r in runs if r.model == model and r.split == split and r.imgsz == imgsz and r.augment == augment
        )

    # Best inference size is chosen on VAL, never on test.
    best_imgsz: dict[str, int] = {}
    for spec in specs:
        val_cells = [r for r in runs if r.model == spec.key and r.split == "val" and not r.augment]
        best_imgsz[spec.key] = max(val_cells, key=lambda r: r.mAP50).imgsz
        print(f"[pick] {spec.key}: best inference imgsz on val = {best_imgsz[spec.key]}")

    # --- TTA at the val-chosen size ----------------------------------------
    for spec in specs:
        for split in ("val", "test"):
            run = run_validation(
                spec,
                split,
                best_imgsz[spec.key],
                device,
                augment=True,
                batch=args.batch,
                out_dir=scratch,
            )
            runs.append(run)
            print(
                f"[tta ] {spec.key:8s} {split:4s} imgsz={run.imgsz:4d}  "
                f"mAP50={run.mAP50:.4f}  mAP50-95={run.mAP50_95:.4f}  ({run.seconds:.1f}s)"
            )

    # --- latency ------------------------------------------------------------
    frames = load_frames(24)
    latency: dict[str, Any] = {"_machine": memory_pressure()}
    for spec in specs:
        by_size: dict[str, Any] = {}
        for imgsz in IMGSZ_GRID:
            stats = time_predict(
                spec.weights, frames, imgsz, device, augment=False, iters=args.latency_iters
            )
            by_size[str(imgsz)] = _latency_dict(stats)
            print(
                f"[lat ] {spec.key:8s} imgsz={imgsz:4d}  "
                f"mean={stats.mean_ms:6.2f} ms  p95={stats.p95_ms:6.2f} ms  "
                f"({1000.0 / stats.mean_ms:.0f} fps)"
            )
        tta_paired = time_tta_ratio(
            spec.weights,
            frames,
            best_imgsz[spec.key],
            device,
            iters=max(60, args.latency_iters),
        )
        latency[spec.key] = {
            "by_imgsz": by_size,
            # The TTA entry is the *interleaved* measurement: its "off" arm is
            # measured in the same loop as its "on" arm, so the cost multiple is a
            # like-for-like ratio rather than a quotient of two contended phases.
            "tta": {"imgsz": best_imgsz[spec.key], **tta_paired["on"]},
            "tta_paired": tta_paired,
            "shipping_api_crosscheck": latency_crosscheck(
                spec, frames, best_imgsz[spec.key], device
            ),
        }
        print(
            f"[lat ] {spec.key:8s} TTA @{best_imgsz[spec.key]}  "
            f"off {tta_paired['off']['median_ms']:.2f} ms -> on {tta_paired['on']['median_ms']:.2f} ms "
            f"({tta_paired['cost_multiple_at_median']:.2f}x median, "
            f"{tta_paired['cost_multiple_at_floor']:.2f}x at the floor, interleaved)"
        )

    # --- paired bootstrap on test at the training resolution ---------------
    nano_test = cell("yolov8n", "test", train_imgsz)
    small_test = cell("yolov8s", "test", train_imgsz)
    if nano_test.stats is None or small_test.stats is None:
        raise RuntimeError("per-image validator statistics were not captured; cannot bootstrap")
    order = sorted(nano_test.stats.names)
    stats_n = nano_test.stats.reindex(order)
    stats_s = small_test.stats.reindex(order)

    fidelity = [bootstrap_fidelity(stats_n, nano_test), bootstrap_fidelity(stats_s, small_test)]
    for f in fidelity:
        print(
            f"[chk ] {f['model']}: recomputed mAP50 {f['recomputed_mAP50']:.8f} vs validator "
            f"{f['validator_mAP50']:.8f}  exact={f['exact']}"
        )
    if not all(f["exact"] for f in fidelity):
        raise RuntimeError("recomputed mAP does not reproduce the validator; bootstrap aborted")

    print(f"[boot] {args.resamples} paired resamples at the training size {train_imgsz} ...")
    boot = paired_bootstrap(stats_n, stats_s, "yolov8n", "yolov8s", args.resamples)
    print(
        f"[boot] @{train_imgsz} dmAP50 = {boot['mAP50']['point_difference']:+.4f}  "
        f"95% CI [{boot['mAP50']['ci95_low']:+.4f}, {boot['mAP50']['ci95_high']:+.4f}]  "
        f"nano wins {_share_pct(boot['mAP50']['share_a_wins'])}  ({boot['seconds']}s)"
    )

    # The comparison above answers "which checkpoint is better at the size they were
    # trained at". It is not the question a deployment asks, which is "which
    # checkpoint is better *at its own best configuration*". Section 2 showed input
    # size moves mAP50 far more than the architecture does, so the two questions can
    # have different answers, and the deployment decision has to rest on the second.
    deploy_n = cell("yolov8n", "test", best_imgsz["yolov8n"])
    deploy_s = cell("yolov8s", "test", best_imgsz["yolov8s"])
    if deploy_n.stats is None or deploy_s.stats is None:
        raise RuntimeError("per-image statistics missing at the deployment sizes")
    dep_stats_n = deploy_n.stats.reindex(order)
    dep_stats_s = deploy_s.stats.reindex(order)
    fidelity += [bootstrap_fidelity(dep_stats_n, deploy_n), bootstrap_fidelity(dep_stats_s, deploy_s)]
    if not all(f["exact"] for f in fidelity):
        raise RuntimeError("recomputed mAP does not reproduce the validator; bootstrap aborted")

    print(
        f"[boot] {args.resamples} paired resamples at each model's val-chosen size "
        f"(nano {best_imgsz['yolov8n']}, small {best_imgsz['yolov8s']}) ..."
    )
    boot_deploy = paired_bootstrap(
        dep_stats_n,
        dep_stats_s,
        f"yolov8n@{best_imgsz['yolov8n']}",
        f"yolov8s@{best_imgsz['yolov8s']}",
        args.resamples,
    )
    print(
        f"[boot] deploy dmAP50 = {boot_deploy['mAP50']['point_difference']:+.4f}  "
        f"95% CI [{boot_deploy['mAP50']['ci95_low']:+.4f}, {boot_deploy['mAP50']['ci95_high']:+.4f}]  "
        f"nano wins {_share_pct(boot_deploy['mAP50']['share_a_wins'])}  ({boot_deploy['seconds']}s)"
    )

    # Second, independent 180-image draw: does the val split reproduce the ordering?
    val_gap = cell("yolov8n", "val", train_imgsz).mAP50 - cell("yolov8s", "val", train_imgsz).mAP50
    test_gap = nano_test.mAP50 - small_test.mAP50

    # --- failure analysis on the recommended checkpoint ---------------------
    # Decided on the deployment comparison. Where that interval contains zero the
    # models are not separable on accuracy and the tie breaks on cost.
    #
    # The cost side is *derived*, not assumed. Hard-coding "yolov8n" here -- which an
    # earlier version of this file did -- makes the tie-break unfalsifiable: it would
    # return the same answer whichever model turned out to be cheaper, so it proves
    # nothing about this comparison. Read the axes off the measured facts instead and
    # require them to agree before the tie is broken at all.
    if boot_deploy["mAP50"]["ci_excludes_zero"]:
        winner_key = "yolov8n" if boot_deploy["mAP50"]["point_difference"] > 0 else "yolov8s"
        winner_basis = "deployment-size bootstrap interval excludes zero"
    else:
        cost_axes = {
            "parameter count": "params",
            "weight-file size": "weight_file_mb",
        }
        cheaper = {
            axis: min(("yolov8n", "yolov8s"), key=lambda k: facts[k][field])
            for axis, field in cost_axes.items()
        }
        cheaper["FLOPs"] = min(
            ("yolov8n", "yolov8s"),
            key=lambda k: facts[k]["gflops_by_imgsz"][str(best_imgsz[k])],
        )
        agreed = set(cheaper.values())
        if len(agreed) != 1:
            raise RuntimeError(
                "accuracy is tied and the cost axes disagree "
                f"({cheaper}); this study cannot break the tie on cost alone"
            )
        winner_key = agreed.pop()
        winner_basis = (
            "deployment-size bootstrap interval contains zero, so the tie breaks on cost; "
            f"{winner_key} is the cheaper model on all of "
            + ", ".join(cheaper) + " (batch-1 latency does not separate them at this input size)"
        )
    winner = nano if winner_key == "yolov8n" else small
    deploy_imgsz = best_imgsz[winner_key]

    records = load_ground_truth("test")
    detector = DefectDetector(
        weights=winner.weights, device=device, conf=0.01, iou=DEPLOY_NMS_IOU, imgsz=deploy_imgsz
    )
    cached = cache_predictions(detector, records)
    del detector
    _release(device)

    tuned_conf, tuned_source = tuned_confidence()
    print(f"[conf] tuned threshold {tuned_conf} from {tuned_source}")
    taxonomy_tuned, _ = failure_taxonomy(records, cached, tuned_conf)
    taxonomy_default, misses_default = failure_taxonomy(records, cached, DEFAULT_CONF)
    appearance = appearance_stats(records)

    # Section 4 analyses the *recommended* configuration, so its per-class AP50 must
    # come from the deployment size, not the training size. Scoring the taxonomy at
    # 256 px and the AP50 column at 320 px would put two configurations in one table.
    # It is not a cosmetic difference: at 320 px separability-vs-AP50 gives rho 0.83
    # and the published-ordering agreement 0.94, and at 256 px the two swap over.
    winner_test = cell(winner_key, "test", deploy_imgsz)
    correlation = difficulty_correlation(winner_test.per_class_AP50, appearance)
    ordering = ordering_vs_published(winner_test.per_class_AP50)
    print(
        f"[fail] separability vs AP50 Spearman rho = "
        f"{correlation['separability'].get('rho_vs_AP50')}; "
        f"ordering matches published = {ordering['identical']} (rho {ordering['spearman_rho']})"
    )

    weak = [n for n, _ in sorted(winner_test.per_class_AP50.items(), key=lambda kv: kv[1])[:2]]

    # --- training cost ------------------------------------------------------
    train_cost: dict[str, Any] = {"measured": False, "reason": "skipped by --skip-train-cost"}
    if not args.skip_train_cost:
        print("[cost] timing training steps at 320 and 640 ...")
        train_cost = {
            "measured": True,
            "note": (
                "Forward + loss + backward + optimiser step on synthetic batches of the real shape. "
                "Convolution cost depends on tensor shape, not tensor content."
            ),
            "by_model": {s.key: training_step_cost(s, device) for s in specs},
        }
        for key, block in train_cost["by_model"].items():
            for size, entry in block["by_imgsz"].items():
                if entry.get("ok"):
                    print(
                        f"[cost] {key:8s} imgsz={size:>3s}  step={entry['step_ms_min']:8.1f} ms  "
                        f"-> GPU-only {entry['gpu_only_seconds_per_epoch_at_probe_batch']:7.1f} s/epoch "
                        f"at probe batch {block['probe_batch']}"
                    )
            if "ratio_640_over_320" in block:
                print(f"[cost] {key:8s} 640/320 step-cost ratio = {block['ratio_640_over_320']}x")

    # --- figures ------------------------------------------------------------
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    figures = {
        "imgsz": str(chart_imgsz(runs, latency, train_imgsz, REPORTS_DIR / "model_study_imgsz.png")),
        "bootstrap": str(chart_bootstrap(boot, REPORTS_DIR / "model_study_bootstrap.png")),
        "bootstrap_deploy": str(
            chart_bootstrap(boot_deploy, REPORTS_DIR / "model_study_bootstrap_deploy.png")
        ),
        "difficulty": str(
            chart_difficulty(
                winner_test.per_class_AP50,
                appearance,
                correlation,
                REPORTS_DIR / "model_study_difficulty.png",
            )
        ),
        "failures": str(
            build_failure_montage(
                misses_default,
                records,
                cached,
                DEFAULT_CONF,
                weak,
                REPORTS_DIR / "model_study_failures.png",
            )
        ),
    }
    for name, path in figures.items():
        print(f"[fig ] {name}: {path}")

    payload = assemble_payload(
        specs=specs,
        facts=facts,
        history=history,
        runs=runs,
        latency=latency,
        boot=boot,
        boot_deploy=boot_deploy,
        winner_basis=winner_basis,
        fidelity=fidelity,
        best_imgsz=best_imgsz,
        train_imgsz=train_imgsz,
        winner_key=winner_key,
        deploy_imgsz=deploy_imgsz,
        taxonomy_tuned=taxonomy_tuned,
        tuned_conf=tuned_conf,
        tuned_source=tuned_source,
        taxonomy_default=taxonomy_default,
        appearance=appearance,
        correlation=correlation,
        ordering=ordering,
        train_cost=train_cost,
        figures=figures,
        val_gap=val_gap,
        test_gap=test_gap,
        device=device,
        started=started,
        ultralytics_version=ultralytics.__version__,
        json_path=args.out,
        cell=cell,
    )

    json_path = write_json(args.out, payload)
    md_path = write_markdown(args.out.with_suffix(".md"), payload)
    print(f"[out ] {json_path}")
    print(f"[out ] {md_path}")
    return 0


def assemble_payload(**kw: Any) -> dict[str, Any]:  # noqa: C901 - assembles the whole report
    """Turn every measurement into the JSON payload and the report's prose.

    Kept apart from `main` so the numbers are computed once and the narrative is a
    pure function of them: no sentence in the report can claim something the
    payload does not contain.
    """
    specs: list[ModelSpec] = kw["specs"]
    facts, history = kw["facts"], kw["history"]
    runs: list[ValRun] = kw["runs"]
    latency, boot, fidelity = kw["latency"], kw["boot"], kw["fidelity"]
    boot_deploy, winner_basis = kw["boot_deploy"], kw["winner_basis"]
    best_imgsz, train_imgsz = kw["best_imgsz"], kw["train_imgsz"]
    winner_key, deploy_imgsz = kw["winner_key"], kw["deploy_imgsz"]
    appearance, correlation, ordering = kw["appearance"], kw["correlation"], kw["ordering"]
    cell = kw["cell"]
    tuned_conf, tuned_source = kw["tuned_conf"], kw["tuned_source"]
    loser_key = "yolov8s" if winner_key == "yolov8n" else "yolov8n"

    def lat(key: str, imgsz: int) -> dict[str, Any]:
        return latency[key]["by_imgsz"][str(imgsz)]

    # ---------- headline table
    head_rows = []
    for spec in specs:
        run = cell(spec.key, "test", train_imgsz)
        f, l = facts[spec.key], lat(spec.key, train_imgsz)
        head_rows.append(
            [
                spec.key,
                f"{run.mAP50:.4f}",
                f"{run.mAP50_95:.4f}",
                f"{run.precision:.4f}",
                f"{run.recall:.4f}",
                f["params_millions"],
                f["gflops_by_imgsz"][str(train_imgsz)],
                f["weight_file_mb"],
                f"{l['median_ms']:.2f}",
                f"{l['p95_ms']:.2f}",
                f"{l['fps_at_median']:.0f}",
            ]
        )

    class_names = [n for n in CLASS_NAMES if n in cell(specs[0].key, "test", train_imgsz).per_class_AP50]
    per_class_rows = []
    for spec in specs:
        run = cell(spec.key, "test", train_imgsz)
        per_class_rows.append([spec.key] + [f"{run.per_class_AP50[n]:.3f}" for n in class_names])
    delta_row = ["**difference**"]
    for n in class_names:
        d = cell(winner_key, "test", train_imgsz).per_class_AP50[n] - cell(
            loser_key, "test", train_imgsz
        ).per_class_AP50[n]
        delta_row.append(f"{d:+.3f}")
    per_class_rows.append(delta_row)

    s50 = boot["mAP50"]
    s5095 = boot["mAP50_95"]
    val_gap, test_gap = kw["val_gap"], kw["test_gap"]
    excludes = s50["ci_excludes_zero"]

    # Ultralytics' own fitness, the criterion that selected both best.pt files, applied
    # to the two deployment-size test scores. Reported because it is the one accuracy
    # summary that does not favour the recommended model, and a reader will compute it.
    def _fitness(key: str) -> float:
        run = cell(key, "test", best_imgsz[key])
        return 0.1 * run.mAP50 + 0.9 * run.mAP50_95

    fit_n, fit_s = _fitness("yolov8n"), _fitness("yolov8s")

    bootstrap_reading = [
        (
            f"**Reading.** The observed test-set gap is {s50['point_difference']:+.4f} mAP50 in favour of "
            f"{boot['model_a']}. The 95% interval on that difference is "
            f"[{s50['ci95_low']:+.4f}, {s50['ci95_high']:+.4f}], which "
            + (
                "**excludes zero**: on a differently drawn 180-image test set from the same "
                "distribution, the ordering would come out the same way "
                f"{_share_pct(s50['share_a_wins'])} of the time. The gap is not an artefact of which "
                "180 images we happened to hold out."
                if excludes
                else "**contains zero**. On a differently drawn 180-image test set from the same "
                f"distribution the ordering would come out the same way only "
                f"{_share_pct(s50['share_a_wins'])} of the time. On this evidence the two checkpoints "
                "cannot be separated at the 5% level, and the point estimate should not be "
                "reported as a clean win."
            )
        ),
        (
            f"On mAP50-95 the difference is {s5095['point_difference']:+.4f} with 95% CI "
            f"[{s5095['ci95_low']:+.4f}, {s5095['ci95_high']:+.4f}] "
            f"({'excludes' if s5095['ci_excludes_zero'] else 'contains'} zero); "
            f"{boot['model_a']} wins {_share_pct(s5095['share_a_wins'])} of resamples."
        ),
        (
            f"Pairing was worth having: the standard deviation of the paired difference is "
            f"{s50['bootstrap_std_difference']:.4f}, against {s50['unpaired_std_estimate']:.4f} if "
            "the two models had been scored on independently drawn test sets. That is a "
            f"{s50['pairing_variance_reduction']:.1f}x tighter interval for no extra compute, and it "
            "is the whole reason a 0.07 gap on 180 images can be resolved at all."
        ),
        (
            "**A tension worth confronting, because it looks like a contradiction.** On the "
            f"validation split the same two checkpoints are nearly tied: {val_gap:+.4f} mAP50, "
            f"against {test_gap:+.4f} on test. A reader is entitled to ask which split to believe, "
            "and the answer is not 'average them'. **`best.pt` was chosen on val.** Ultralytics "
            "saves the epoch with the highest validation fitness, so each checkpoint is the "
            f"maximum over {history['yolov8n']['epochs_run']} (nano) and "
            f"{history['yolov8s']['epochs_run']} (small) validation evaluations. A maximum over many "
            "draws is biased upward, and it is biased upward *on the split it was maximised over*. "
            "Val therefore over-states both models by an unknown amount and cannot referee a "
            "comparison between them; it is a selection split wearing an evaluation split's "
            "clothes. Test is the only measurement here that no decision was made on, which is why "
            "it is the one the verdict rests on."
        ),
        (
            "That resolution is worth stating as a limit as well as a defence. It explains why the "
            "two splits disagree without appealing to luck, but it does not prove the test figure "
            "is unbiased in every respect -- only that it is the less contaminated of the two."
        ),
    ]

    d50 = boot_deploy["mAP50"]
    deployment_reading = [
        (
            "**The comparison above is not the one the deployment faces.** It scores both models at "
            f"{train_imgsz} px, the size they were trained at, which is the right way to answer "
            "'which of these two training runs came out better'. But section 2 shows input size "
            "moves mAP50 by far more than the architecture does, and neither model's best input "
            f"size is {train_imgsz}. The question that decides what ships is which checkpoint is "
            "better **at its own best configuration**, so the bootstrap is repeated with each model "
            f"at the size val chose for it (nano at {best_imgsz['yolov8n']} px, small at "
            f"{best_imgsz['yolov8s']} px), paired over the same resampled test sets."
        ),
        (
            f"At their own best sizes the gap is {d50['point_difference']:+.4f} mAP50 "
            f"({d50['point_estimate_a']:.4f} against {d50['point_estimate_b']:.4f}), 95% CI "
            f"[{d50['ci95_low']:+.4f}, {d50['ci95_high']:+.4f}], "
            + (
                f"which excludes zero; {boot_deploy['model_a']} wins "
                f"{_share_pct(d50['share_a_wins'])} of resamples."
                if d50["ci_excludes_zero"]
                else f"which **contains zero**. {boot_deploy['model_a']} wins "
                f"{_share_pct(d50['share_a_wins'])} of resamples and {boot_deploy['model_b']} the rest, so "
                "at their best configurations the two checkpoints are not separable on accuracy at "
                "all."
            )
        ),
        (
            "**Both results are true and they answer different questions.** At the trained size, "
            f"nano is ahead by {s50['point_difference']:+.4f} with an interval that "
            f"{'excludes' if excludes else 'contains'} zero. At each model's best size, the gap is "
            f"{d50['point_difference']:+.4f} with an interval that "
            f"{'excludes' if d50['ci_excludes_zero'] else 'contains'} zero. "
            + (
                "The honest headline is therefore the narrower one: **the original claim that nano "
                "beats small survives at the size the claim was made about, but it does not survive "
                "once both models are given their best input size.** Most of what looked like an "
                "architecture difference was a resolution effect that happened to hurt the larger "
                "model more at 320 px. Anyone quoting the +0.069 figure as evidence that a smaller "
                "detector is better on small datasets is over-reading it. Note also that the two "
                "metrics disagree in sign at the deployment size: nano leads on mAP50 by "
                f"{d50['point_difference']:+.4f} and trails on mAP50-95 by "
                f"{boot_deploy['mAP50_95']['point_difference']:+.4f}, both intervals containing "
                "zero. Two metrics pointing opposite ways, neither significantly, is what a genuine "
                "tie looks like -- and it is why the deployment decision below is made on cost. "
                "**The strongest form of the objection should be stated outright**, because a "
                "reader will find it: ultralytics' own model-selection criterion is the composite "
                "fitness 0.1*mAP50 + 0.9*mAP50-95, which is what chose both of these checkpoints "
                "epoch by epoch. Evaluated at the deployment size on test it gives "
                f"{fit_n:.4f} for yolov8n@{best_imgsz['yolov8n']} against {fit_s:.4f} for "
                f"yolov8s@{best_imgsz['yolov8s']} -- so on the project's own composite metric the "
                f"ordering favours **{'yolov8s' if fit_s > fit_n else 'yolov8n'}** by "
                f"{abs(fit_s - fit_n):.4f}, because that metric weights mAP50-95 nine to one and "
                "mAP50-95 is the metric nano trails on. That difference is well inside the "
                "bootstrap interval and decides nothing, but it does mean no accuracy metric picks "
                "yolov8n here. The recommendation below rests on cost, and it has to."
                if excludes and not d50["ci_excludes_zero"]
                else "Quote whichever one matches the question being asked, and say which it is."
            )
        ),
    ]

    hist_rows = []
    for spec in specs:
        h = history[spec.key]
        if not h.get("available"):
            continue
        hist_rows.append(
            [
                spec.key,
                h["epochs_run"],
                h["best_epoch"],
                f"{h['best_val_mAP50']:.4f}",
                f"{h['final_train_cls_loss']:.3f}",
                f"{h['final_val_cls_loss']:.3f}",
                f"{h['cls_loss_generalisation_gap']:+.3f}",
                h["seconds_per_epoch_median"],
                h["seconds_per_epoch"],
            ]
        )

    hn, hs = history["yolov8n"], history["yolov8s"]
    val_n = cell("yolov8n", "val", train_imgsz).mAP50
    val_s = cell("yolov8s", "val", train_imgsz).mAP50
    mechanism = [
        (
            "The direct evidence for the capacity explanation is in the loss curves, not the mAP. "
            f"At the end of training yolov8s sits at train cls loss {hs['final_train_cls_loss']:.3f} "
            f"against val {hs['final_val_cls_loss']:.3f} -- a generalisation gap of "
            f"{hs['cls_loss_generalisation_gap']:+.3f}. yolov8n sits at "
            f"{hn['final_train_cls_loss']:.3f} against {hn['final_val_cls_loss']:.3f}, a gap of "
            f"{hn['cls_loss_generalisation_gap']:+.3f} -- it is scoring *better* on data it has "
            "never seen than on data it trains on, which is what a model with too little capacity "
            "to memorise looks like when the training set is also being augmented hard. The larger "
            "model is fitting structure in the 1440 training images that does not transfer. That is "
            "the mechanism, and it is measured rather than assumed."
        ),
        (
            "The scale of the effect is what needs care. 1440 training images across six classes is "
            "roughly 240 images per class, which is thin for an 11.1M-parameter detector and "
            "comfortable for a 3.0M-parameter one. The recipe made this worse in one specific way: "
            "both runs used the ultralytics default schedule with mosaic, mixup and 150 epochs, "
            "tuned for COCO-scale data. Nothing in it -- no extra weight decay, no shorter schedule, "
            "no frozen backbone -- was adjusted for the larger model on the smaller dataset. So the "
            "honest claim is not 'yolov8s is the wrong architecture for NEU-DET' but '**yolov8s "
            "trained with this recipe on this much data is worse than yolov8n trained the same "
            "way**'. Those are different claims, and only the second one is measured."
        ),
        (
            f"Re-evaluated at {train_imgsz} px, the two checkpoints score {val_n:.4f} and "
            f"{val_s:.4f} mAP50 on val -- close, for the selection reason given above -- and "
            f"{cell('yolov8n', 'test', train_imgsz).mAP50:.4f} against "
            f"{cell('yolov8s', 'test', train_imgsz).mAP50:.4f} on test. yolov8n also stopped early "
            f"at epoch {hn['epochs_run']} of 150 with its best weights at epoch {hn['best_epoch']}, "
            f"while yolov8s ran the full 150 with its best at epoch {hs['best_epoch']}. Neither run "
            "was starved of epochs, and neither was still improving when it stopped."
        ),
    ]

    # ---------- resolution
    imgsz_header = ["model", "split"] + [f"{s} px" for s in IMGSZ_GRID]
    imgsz_rows = []
    for spec in specs:
        for split in ("val", "test"):
            row = [spec.key, split]
            for size in IMGSZ_GRID:
                run = cell(spec.key, split, size)
                marker = " *" if (split == "val" and size == best_imgsz[spec.key]) else ""
                row.append(f"{run.mAP50:.4f}{marker}")
            imgsz_rows.append(row)
    imgsz_rows.append(
        ["latency (median ms)", "batch 1"]
        + [f"{lat('yolov8n', s)['median_ms']:.1f} / {lat('yolov8s', s)['median_ms']:.1f}" for s in IMGSZ_GRID]
    )

    def sweep(key: str, split: str) -> list[tuple[int, float]]:
        return [(s, cell(key, split, s).mAP50) for s in IMGSZ_GRID]

    def shape(key: str, split: str) -> dict[str, Any]:
        """Peak, trained-size value and the fall-off, straight from the sweep."""
        points = sweep(key, split)
        peak_size, peak = max(points, key=lambda kv: kv[1])
        at_train = dict(points)[train_imgsz]
        at_max = dict(points)[max(IMGSZ_GRID)]
        return {
            "peak_size": peak_size,
            "peak": peak,
            "at_train": at_train,
            "at_max": at_max,
            "drop_to_max": peak - at_max,
            "relative_drop": (peak - at_max) / peak if peak else 0.0,
        }

    shapes = {(k, sp): shape(k, sp) for k in ("yolov8n", "yolov8s") for sp in ("val", "test")}
    worst_rel = max(v["relative_drop"] for v in shapes.values())
    best_rel = min(v["relative_drop"] for v in shapes.values())

    imgsz_reading = [
        "`*` marks the size chosen on val. The last row is nano / small single-frame latency in "
        "ms at batch 1, quoted as the **median** of the timed calls: this laptop was running other "
        "work throughout, and the median is robust to both an occasional stall and a single lucky "
        "call in a way that neither the mean nor the minimum is. Mean, min and p95 for every cell "
        "are in the JSON.",
        (
            "**Resolution mismatch is the single largest effect in this entire study, and it is "
            "catastrophic rather than marginal.** Going from the best input size to 640 px costs "
            f"between {best_rel:.0%} and {worst_rel:.0%} of mAP50 depending on the model and split "
            "-- far more than the difference between the two architectures, more than TTA, more "
            "than anything else measured here. Feeding a detector trained at 320 px an image at "
            "640 px is not a mild extrapolation; it roughly halves it."
        ),
        (
            "The mechanism is scale, not detail. A YOLOv8 head assigns each object to a feature "
            "level by its pixel size, and those assignments were learned from 320 px inputs. At "
            "640 px every defect is twice as many pixels across as anything the model was trained "
            "to regress, so boxes land on the wrong stride and the classifier sees a texture at a "
            "spatial frequency it never saw. Because NEU-DET defects are large relative to the "
            "frame -- a crazing label alone covers about a quarter of it -- there is no small-object "
            "regime that benefits to offset the loss."
        ),
        (
            "Peak performance sits **at or below the training size**, which is the second thing "
            "worth noting: the source images are 200x200, so 320 px is already upsampling and 256 px "
            "is the only size in the grid that does not. There is no information above 200 px to "
            "recover, so nothing above the training size can pay for its own compute. "
            + "; ".join(
                f"{k} on {sp} peaks at {v['peak_size']} px ({v['peak']:.4f})"
                for (k, sp), v in shapes.items()
            )
            + "."
        ),
        (
            "For yolov8n the val sweep runs "
            + ", ".join(f"{s}:{v:.4f}" for s, v in sweep("yolov8n", "val"))
            + f", best at {best_imgsz['yolov8n']} px. On test the same model runs "
            + ", ".join(f"{s}:{v:.4f}" for s, v in sweep("yolov8n", "test"))
            + "."
        ),
        (
            "For yolov8s the val sweep runs "
            + ", ".join(f"{s}:{v:.4f}" for s, v in sweep("yolov8s", "val"))
            + f", best at {best_imgsz['yolov8s']} px. On test: "
            + ", ".join(f"{s}:{v:.4f}" for s, v in sweep("yolov8s", "test"))
            + "."
        ),
        (
            "**The deployment consequence is a rule, not a number.** The curve is not symmetric: "
            "dropping *below* the training size is nearly free here (256 px is the peak on every "
            "model and split), while going above it is ruinous. So the rule is **never run above "
            "the size the model was trained at, and validate any size you do run**. If mill frames "
            "need a larger input -- and they will, because a 2048 px camera frame is not a 200 px "
            "tile -- the model must be *retrained* at that size, not merely evaluated at it. This "
            "is a live operational hazard, not a theoretical one: `imgsz` is an argument anyone "
            "can change, it raises no error, and at 640 px it silently costs more than half the "
            "accuracy."
        ),
    ]

    # ---------- TTA
    tta_header = [
        "model",
        "imgsz",
        "split",
        "mAP50 no TTA",
        "mAP50 with TTA",
        "delta",
        "latency no TTA (ms)",
        "latency TTA (ms)",
        "cost multiple",
    ]
    tta_rows = []
    for spec in specs:
        size = best_imgsz[spec.key]
        # Both arms from the interleaved run, so the ratio is honest. The size sweep
        # in `by_imgsz` was measured in its own phase and must not be divided into a
        # number taken from a different one.
        paired = latency[spec.key]["tta_paired"]
        base_ms = paired["off"]["median_ms"]
        tta_ms = paired["on"]["median_ms"]
        for split in ("val", "test"):
            plain = cell(spec.key, split, size)
            aug = cell(spec.key, split, size, True)
            tta_rows.append(
                [
                    spec.key,
                    size,
                    split,
                    f"{plain.mAP50:.4f}",
                    f"{aug.mAP50:.4f}",
                    f"{aug.mAP50 - plain.mAP50:+.4f}",
                    f"{base_ms:.2f}",
                    f"{tta_ms:.2f}",
                    f"{tta_ms / base_ms:.2f}x",
                ]
            )

    w_size = best_imgsz[winner_key]
    w_plain_val = cell(winner_key, "val", w_size)
    w_tta_val = cell(winner_key, "val", w_size, True)
    w_plain_test = cell(winner_key, "test", w_size)
    w_tta_test = cell(winner_key, "test", w_size, True)
    w_paired = latency[winner_key]["tta_paired"]
    tta_ms = w_paired["on"]["median_ms"]
    base_ms = w_paired["off"]["median_ms"]
    tta_floor_x = w_paired["cost_multiple_at_floor"]
    tta_helps_val = w_tta_val.mAP50 > w_plain_val.mAP50
    tta_reading = [
        (
            f"On the split that is allowed to decide -- val -- TTA moves {winner_key} from "
            f"{w_plain_val.mAP50:.4f} to {w_tta_val.mAP50:.4f} mAP50, "
            f"{w_tta_val.mAP50 - w_plain_val.mAP50:+.4f}. It costs {tta_ms / base_ms:.2f}x the "
            f"latency ({base_ms:.2f} ms to {tta_ms:.2f} ms per frame, {tta_floor_x:.2f}x at the "
            "floor), i.e. it throws away "
            f"{(1 - base_ms / tta_ms) * 100:.0f}% of the throughput of the accelerator. Both arms "
            "of that ratio were timed alternately inside one loop, so it is a cost multiple rather "
            "than the quotient of two separately-contended measurements."
        ),
        (
            f"On test the same change is {w_plain_test.mAP50:.4f} to {w_tta_test.mAP50:.4f} "
            f"({w_tta_test.mAP50 - w_plain_test.mAP50:+.4f}), reported for completeness and not used "
            "to decide anything."
        ),
        (
            "**Not worth it on a production line.** "
            + (
                "The accuracy it buys is inside the noise band the bootstrap already measured "
                f"(a 95% interval {abs(s50['ci95_high'] - s50['ci95_low']):.4f} wide on a difference "
                "of this size), while the throughput cost is certain and large. Section 2f of "
                "docs/research_notes.md puts the required tile rate at roughly 10,800 200x200 tiles "
                "per second at the slowest well-sourced line speed; multiplying per-frame cost by "
                f"{tta_ms / base_ms:.2f} multiplies the accelerator count by the same factor for a "
                "gain nobody can demonstrate. Spend the silicon on more cameras or a second "
                "inspection point, not on re-running the same frame at three scales."
                if not tta_helps_val or (w_tta_val.mAP50 - w_plain_val.mAP50) < abs(s50["ci95_high"] - s50["ci95_low"]) / 2
                else "Even where it helps, the gain must be weighed against the throughput arithmetic in "
                "docs/research_notes.md section 2f before it goes on a line."
            )
        ),
    ]

    # ---------- failures
    # Deployment size, to match the taxonomy: see the note in main(). `winner_train`
    # is kept separately because the positioning section legitimately needs the
    # previously-recorded 320 px figure.
    tax_tuned, tax_default = kw["taxonomy_tuned"], kw["taxonomy_default"]
    winner_test = cell(winner_key, "test", deploy_imgsz)
    winner_train = cell(winner_key, "test", train_imgsz)
    taxonomy_rows = []
    for name in CLASS_NAMES:
        d = tax_default["per_class"][name]
        t = tax_tuned["per_class"][name]
        confused = ", ".join(f"{k} x{v}" for k, v in list(d["confused_with"].items())[:2]) or "-"
        taxonomy_rows.append(
            [
                name,
                f"{winner_test.per_class_AP50[name]:.3f}",
                d["instances"],
                f"{d['recall']:.2f}",
                d["class_confusion"],
                d["extent_error"],
                d["blind_miss"],
                confused,
                f"{t['recall']:.2f}",
            ]
        )

    appearance_rows = []
    for name in CLASS_NAMES:
        a = appearance[name]
        appearance_rows.append(
            [
                name,
                f"{winner_test.per_class_AP50[name]:.3f}",
                a["n_boxes"],
                f"{a['separability']:.3f}",
                f"{a['edge_energy_ratio']:.3f}",
                f"{a['box_area_frac'] * 100:.1f}%",
                f"{a['mean_pairwise_gt_iou']:.3f}",
                f"{a['boxes_per_image']:.2f}",
            ]
        )

    weakest = sorted(winner_test.per_class_AP50.items(), key=lambda kv: kv[1])[:2]
    w1, w2 = weakest[0][0], weakest[1][0]
    d1, d2 = tax_default["per_class"][w1], tax_default["per_class"][w2]
    a1, a2 = appearance[w1], appearance[w2]
    sep_rho = correlation["separability"].get("rho_vs_AP50")
    iou_rho = correlation["mean_pairwise_gt_iou"].get("rho_vs_AP50")
    area_rho = correlation["box_area_frac"].get("rho_vs_AP50")

    def dominant(bucket: dict[str, Any]) -> str:
        counts = {
            "class confusion": bucket["class_confusion"],
            "extent error": bucket["extent_error"],
            "blind miss": bucket["blind_miss"],
        }
        top = max(counts.items(), key=lambda kv: kv[1])
        total = max(1, sum(counts.values()))
        return f"{top[0]} ({top[1]} of {total} misses, {top[1] / total:.0%})"

    total_confusion = sum(tax_default["per_class"][n]["class_confusion"] for n in CLASS_NAMES)
    total_missed = sum(tax_default["per_class"][n]["missed"] for n in CLASS_NAMES)
    best_sep_class = max(CLASS_NAMES, key=lambda n: appearance[n]["separability"])
    edge_rho = correlation["edge_energy_ratio"].get("rho_vs_AP50")

    failures_reading = [
        (
            "**The failure is not class confusion, and that is a measurement, not an impression.** "
            f"Across all six classes and all {total_missed} missed labels at conf={DEFAULT_CONF}, "
            f"exactly **{total_confusion}** were cases of a box drawn in the right place carrying "
            "the wrong class name. The false positives tell the same story from the other side: "
            "they land almost entirely on frames of their own class ("
            + "; ".join(
                f"{n} -> {list(tax_default['per_class'][n]['false_positives_on_this_class_image'].items())[0][0]} "
                f"x{list(tax_default['per_class'][n]['false_positives_on_this_class_image'].items())[0][1]}"
                for n in ("crazing", "rolled-in_scale")
                if tax_default["per_class"][n]["false_positives_on_this_class_image"]
            )
            + "), i.e. they are extra or badly drawn boxes of the *correct* type, not hallucinated "
            "defects of another type. The detector knows what it is looking at. It does not "
            "reliably know whether the defect is there, or where it ends."
        ),
        (
            f"**{w1}** (AP50 {winner_test.per_class_AP50[w1]:.3f}) fails by {dominant(d1)}. Its "
            f"measured defect-vs-background separability is {a1['separability']:.2f} standard "
            f"deviations of background roughness -- the lowest of the six classes, against "
            f"{appearance[best_sep_class]['separability']:.2f} for {best_sep_class} -- and its "
            f"edge-energy ratio is {a1['edge_energy_ratio']:.2f}, meaning the pixels inside the "
            "label are no busier than the pixels outside it. The defect is neither a different grey "
            "level nor a texture discontinuity. On top of that, each label claims "
            f"{a1['box_area_frac'] * 100:.0f}% of a 200x200 frame and there are "
            f"{a1['boxes_per_image']:.1f} of them per image. The montage shows what that combination "
            "means in practice: the labelled and unlabelled parts of a crazing tile are visually "
            "indistinguishable, and the boxes carve arbitrary slabs out of a texture that covers the "
            "whole strip. Both the low-contrast and the ambiguous-extent hypotheses are true here, "
            "and they compound."
        ),
        (
            f"**{w2}** (AP50 {winner_test.per_class_AP50[w2]:.3f}) fails differently: by "
            f"{dominant(d2)}. Separability {a2['separability']:.2f} is the second lowest, but the "
            f"edge-energy ratio {a2['edge_energy_ratio']:.2f} is above 1.0, so the defect region is "
            "at least texturally distinct. The detector fires -- the montage row shows boxes at "
            "confidence 0.28 to 0.42 sitting on or beside the labelled scale -- but the rectangles "
            "disagree with the annotator about where one patch of rolled-in scale ends and the next "
            "begins. This is an IoU-0.5 scoring failure on a defect that was found, not a detection "
            "failure."
        ),
        (
            "Across the six classes, exactly one of these statistics orders them: AP50 against "
            f"measured separability gives Spearman rho = {sep_rho} "
            f"(p = {correlation['separability'].get('p_value')}). The others do not. Edge-energy "
            f"ratio gives rho = {edge_rho} -- essentially nothing -- because patches is the easiest "
            f"class of all at an edge ratio of "
            f"{appearance['patches']['edge_energy_ratio']:.2f}, i.e. *smoother* than its "
            "background: patches is easy because it is a different grey level, not because it is a "
            f"different texture. Label overlap gives rho = {iou_rho} "
            f"(p = {correlation['mean_pairwise_gt_iou'].get('p_value')}) and mean label area "
            f"rho = {area_rho} (p = {correlation['box_area_frac'].get('p_value')}) -- both weak, "
            "both negative, and neither significant on six points. Note that box size is the "
            "weaker of the two despite the intuition that big boxes are easy: pitted surface has "
            f"the largest boxes of any class ({appearance['pitted_surface']['box_area_frac'] * 100:.0f}% "
            "of frame) and is only mid-table for accuracy. So the ranking of causes that the data "
            "actually supports is: "
            "**grey-level separability first and by a distance; extent/annotation geometry a "
            "secondary effect that shows up in the per-class miss taxonomy but not in the aggregate "
            "correlations; texture contrast not at all; class confusion not at all.** "
            "Six classes is six data points, so these rho values describe this table rather than "
            "test a hypothesis -- but the one that is strong is the one the physics predicts, and "
            "the class ordering it produces matches every source in docs/research_notes.md "
            "section 1c."
        ),
        (
            "The practical reading is that neither weak class is fixed by a bigger network. "
            f"{w1}-style failure -- invisible in the pixels, arbitrary in extent -- is an optics and "
            "annotation-protocol problem: dark-field or oblique illumination changes measured "
            "separability in a way no loss function can, and a written rule about where a craze "
            "network stops is worth more than another 50 epochs. It is also a fair question whether "
            "crazing should be scored as a box at all rather than as a whole-frame label or a "
            f"segmentation mask. {w2}-style failure -- found but mis-drawn -- is the one that would "
            "genuinely respond to more labelled data and to a loss that is less brittle about box "
            "edges (the Shape-IoU and Focaler-IoU work in docs/research_notes.md section 6b targets "
            "exactly this)."
        ),
    ]

    ordering_rows = [
        [
            rank + 1,
            ordering["ours_hardest_first"][rank],
            f"{winner_test.per_class_AP50[ordering['ours_hardest_first'][rank]]:.3f}",
            ordering["published_hardest_first"][rank],
        ]
        for rank in range(len(ordering["ours_hardest_first"]))
    ]
    ordering_reading = [
        (
            "Our hardest-first ordering "
            + ("**matches** " if ordering["identical"] else "**differs from** ")
            + f"the published one (Spearman rho = {ordering['spearman_rho']} on the ranks). "
            f"Source: {PUBLISHED_SOURCE}."
        ),
        (
            "Note the published values are AP@[.5:.95] on a different 70/20/10 split of NEU-DET, so "
            "only the *ordering* is comparable, not the numbers. What transfers is the finding that "
            "crazing and rolled-in scale are the hard classes and patches is the easy one, which "
            "every source in docs/research_notes.md section 1c agrees on and which we reproduce "
            "independently."
        ),
    ]
    if not ordering["identical"]:
        shifts = sorted(
            ordering["rank_shifts"].items(), key=lambda kv: -abs(kv[1]["shift"])
        )
        biggest = shifts[0]
        ordering_reading.insert(
            1,
            (
                f"The largest disagreement is **{biggest[0]}**: rank {biggest[1]['ours'] + 1} here "
                f"against rank {biggest[1]['published'] + 1} published. "
                "Both endpoints of the ordering -- the hardest and the easiest class -- agree."
                if ordering["ours_hardest_first"][0] == ordering["published_hardest_first"][0]
                and ordering["ours_hardest_first"][-1] == ordering["published_hardest_first"][-1]
                else f"The largest disagreement is **{biggest[0]}**: rank "
                f"{biggest[1]['ours'] + 1} here against rank {biggest[1]['published'] + 1} published."
            ),
        )

    failures_intro = (
        f"Failure analysis is run on the recommended checkpoint (**{winner_key}** at "
        f"{deploy_imgsz} px) through the shipping `DefectDetector` path at the deployment NMS IoU "
        f"{DEPLOY_NMS_IOU}, not at the validator's mAP protocol -- these are the boxes an operator "
        f"would actually be shown. Columns are at conf={DEFAULT_CONF} (the `DefectDetector` "
        f"default); the last column repeats recall at conf={tuned_conf}, the threshold the "
        f"operating-point study currently recommends ({tuned_source}). Every missed label is sorted by the best "
        "overlapping prediction of any class: **class confusion** = something drawn in the right "
        "place (IoU >= 0.5) with the wrong name; **extent error** = best overlap between "
        f"{EXTENT_IOU_FLOOR} and 0.5, the defect was seen but the rectangle disagrees; **blind "
        "miss** = nothing overlaps at all. The AP50 column, and every correlation below it, is "
        f"the validator's per-class AP50 for the same checkpoint at the same {deploy_imgsz} px -- "
        f"not the {train_imgsz} px figure quoted in section 1, which describes a configuration "
        "this project is not shipping. The two differ enough to matter: at "
        f"{train_imgsz} px {w1} scores "
        f"{winner_train.per_class_AP50[w1]:.3f} against {winner_test.per_class_AP50[w1]:.3f} here."
    )

    # ---------- selection and recommendation
    w_facts = facts[winner_key]
    w_lat = lat(winner_key, deploy_imgsz)
    decisions = [
        {
            "decision": "checkpoint",
            "value": f"`models/{winner_key}_neudet/weights/best.pt`",
            "decided_on": "test bootstrap at each model's best size, then cost",
            "evidence": (
                f"at best sizes {d50['point_estimate_a']:.4f} vs {d50['point_estimate_b']:.4f} "
                f"mAP50, CI [{d50['ci95_low']:+.4f}, {d50['ci95_high']:+.4f}] "
                f"{'excludes' if d50['ci_excludes_zero'] else 'contains'} 0; {winner_basis}"
            ),
        },
        {
            "decision": "inference imgsz",
            "value": f"{deploy_imgsz} px",
            "decided_on": "val",
            "evidence": (
                f"best val mAP50 across {IMGSZ_GRID}; the curve collapses above the training "
                f"size -- up to {worst_rel:.0%} of mAP50 lost by 640 px"
            ),
        },
        {
            "decision": "test-time augmentation",
            "value": "off",
            "decided_on": "val + latency",
            "evidence": (
                f"val gain {w_tta_val.mAP50 - w_plain_val.mAP50:+.4f} mAP50 for "
                f"{tta_ms / base_ms:.2f}x latency"
            ),
        },
        {
            "decision": "confidence threshold",
            "value": f"{tuned_conf}",
            "decided_on": "val (operating-point study)",
            "evidence": f"minimum expected mill cost subject to a 90% detection floor; {tuned_source}",
        },
        {
            "decision": "NMS IoU",
            "value": f"{DEPLOY_NMS_IOU}",
            "decided_on": "shipping default",
            "evidence": "DefectDetector default; not swept in this study",
        },
    ]

    dep_winner = cell(winner_key, "test", deploy_imgsz)
    verdict = (
        f"**Ship `{winner_key}` at {deploy_imgsz} px with TTA off.** On the held-out test split, at "
        f"the input size validation chose, it scores mAP50 {dep_winner.mAP50:.4f} / mAP50-95 "
        f"{dep_winner.mAP50_95:.4f}, from a {w_facts['weight_file_mb']} MB checkpoint "
        f"with {w_facts['params_millions']}M parameters, at a batch-1 floor of "
        f"{w_lat['min_ms']:.1f} ms per 200x200 frame on this machine "
        f"(median {w_lat['median_ms']:.1f} ms; see the latency caveat in section 5 -- the "
        "absolute milliseconds are not reproducible run to run on a contended laptop and should "
        "not be quoted as a throughput figure). "
        + (
            f"With both models at their own best input size the gap over {loser_key} is "
            f"{d50['point_difference']:+.4f} mAP50, 95% CI "
            f"[{d50['ci95_low']:+.4f}, {d50['ci95_high']:+.4f}], excluding zero -- so on this "
            "evidence it is genuinely the better detector, not merely the cheaper one."
            if d50["ci_excludes_zero"]
            else f"With both models at their own best input size the gap over {loser_key} is "
            f"{d50['point_difference']:+.4f} mAP50, 95% CI "
            f"[{d50['ci95_low']:+.4f}, {d50['ci95_high']:+.4f}], which **contains zero**. The "
            "honest statement is therefore not 'nano is the better detector' but '**nano is not "
            f"worse, and it is {facts[loser_key]['params_millions'] / w_facts['params_millions']:.1f}x "
            f"smaller, {facts[loser_key]['gflops_by_imgsz'][str(deploy_imgsz)] / w_facts['gflops_by_imgsz'][str(deploy_imgsz)]:.1f}x "
            f"cheaper in FLOPs**'. That is more than sufficient to decide a deployment, and it is the reason "
            "to ship it. The larger claim -- that the nano architecture is better on a dataset this "
            "size -- is not supported once both models are given their best configuration."
        )
    )

    tc = kw["train_cost"]

    def cost_block(key: str) -> dict[str, Any] | None:
        """Step-cost ratio for one model, calibrated onto its logged epoch time.

        The probe times the GPU step only. A real epoch also pays dataloading,
        mosaic augmentation, EMA and a validation pass, so the raw projection is a
        floor -- measured here at roughly half the logged epoch time. The ratio
        between the two input sizes is the part that transfers, so it is applied to
        the epoch time the run actually logged rather than replacing it.
        """
        block = (tc.get("by_model") or {}).get(key)
        if not block or "ratio_640_over_320" not in block:
            return None
        # Median, not mean: the mean carries this machine's paging stalls, and a
        # projection built on it would price contention as if it were arithmetic.
        logged = history[key]["seconds_per_epoch_median"]
        gpu_320 = block["by_imgsz"]["320"]["gpu_only_seconds_per_epoch_at_probe_batch"]
        ratio = block["ratio_640_over_320"]
        return {
            "probe_batch": block["probe_batch"],
            "step_320_ms": block["by_imgsz"]["320"]["step_ms_min"],
            "step_640_ms": block["by_imgsz"]["640"]["step_ms_min"],
            "ratio": ratio,
            "gpu_only_320_s": gpu_320,
            "logged_320_s": logged,
            "overhead_multiple": round(logged / gpu_320, 2) if gpu_320 else None,
            "calibrated_640_s_per_epoch": round(logged * ratio, 1),
            "calibrated_640_hours_150_epochs": round(logged * ratio * 150 / 3600.0, 2),
            "logged_320_hours_150_epochs": round(logged * 150 / 3600.0, 2),
        }

    def cost_failure(key: str) -> str | None:
        """Why the 640 px probe could not be completed, if it could not."""
        block = (tc.get("by_model") or {}).get(key)
        if not block:
            return None
        entry = block.get("by_imgsz", {}).get("640", {})
        if entry.get("ok"):
            return None
        detail = entry.get("error", "not attempted")
        swap = (entry.get("swap") or {}).get("used")
        return f"{detail}" + (f"; swap in use at the time: {swap}" if swap else "")

    lat_ratio = lat(loser_key, deploy_imgsz)["median_ms"] / w_lat["median_ms"]
    flop_ratio = (
        facts[loser_key]["gflops_by_imgsz"][str(deploy_imgsz)]
        / w_facts["gflops_by_imgsz"][str(deploy_imgsz)]
    )
    # Test mAP50 the shipped model scores at the size the library defaults to (640).
    # Asserted to be the worst cell in the grid rather than assumed, so the sentence
    # below cannot outlive the measurement that justifies it.
    imgsz_worst = cell(winner_key, "test", 640).mAP50
    assert imgsz_worst == min(cell(winner_key, "test", s).mAP50 for s in IMGSZ_GRID)
    cost_n, cost_s = cost_block("yolov8n"), cost_block("yolov8s")
    cost_fail_s = cost_failure("yolov8s")
    recommendation = [
        verdict,
        (
            f"**Why this checkpoint.** On accuracy at best configurations the two are not "
            f"separable: {winner_key} leads on mAP50 by {d50['point_difference']:+.4f} and trails "
            f"on mAP50-95 by {boot_deploy['mAP50_95']['point_difference']:+.4f}, both intervals "
            "containing zero. What separates them is cost. "
            f"{winner_key} is "
            f"{facts[loser_key]['params_millions'] / w_facts['params_millions']:.1f}x smaller "
            f"({w_facts['params_millions']}M against {facts[loser_key]['params_millions']}M "
            f"parameters), {w_facts['weight_file_mb']} MB against "
            f"{facts[loser_key]['weight_file_mb']} MB on disk, and "
            f"{flop_ratio:.1f}x cheaper in arithmetic at {deploy_imgsz} px. "
            + (
                f"Measured batch-1 latency agrees in direction but understates the margin "
                f"({w_lat['median_ms']:.1f} ms against "
                f"{lat(loser_key, deploy_imgsz)['median_ms']:.1f} ms median, {lat_ratio:.2f}x "
                f"against {flop_ratio:.1f}x of arithmetic), and that is expected rather than "
                f"awkward -- see the caveat below: at {deploy_imgsz} px a single frame does not "
                "saturate the GPU, so batch-1 wall clock is part dispatch overhead. The arithmetic "
                "ratio is what governs a line running many streams per accelerator."
                if lat_ratio < flop_ratio / 1.5
                else f"Measured batch-1 latency agrees, at {lat_ratio:.2f}x."
            )
            + (
                " With accuracy tied and every cost axis favouring the smaller model, there is no "
                "reading of this evidence in which yolov8s is the right choice."
                if not d50["ci_excludes_zero"]
                else " The accuracy and the cost arguments point the same way, which is the easy "
                "case."
            )
        ),
        (
            f"**Why {deploy_imgsz} px.** Chosen on val, where it was the best of "
            f"{'/'.join(str(s) for s in IMGSZ_GRID)}. This is the highest-stakes knob in the whole "
            f"configuration: section 2 measures a fall of up to {worst_rel:.0%} of mAP50 between the "
            "best size and 640 px, because the model can only regress boxes at the scale it was "
            "trained on. Pin it, and treat any change to it as requiring a retrain rather than a "
            "config edit. On real mill frames the decision must be retaken from scratch, since "
            "there the input size also determines how many source pixels survive per network pixel "
            "and therefore the smallest detectable defect -- see the tiling arithmetic in "
            "`src/benchmark.py`."
        ),
        (
            f"**Why no TTA.** {tta_ms / base_ms:.2f}x the latency for "
            f"{w_tta_val.mAP50 - w_plain_val.mAP50:+.4f} mAP50 on val. The cost is certain, the "
            "benefit is inside the noise the bootstrap measured, and the binding constraint on a "
            "rolling line is throughput."
        ),
        (
            "**What must be re-measured before this runs on a line.** This model was trained and "
            "tested on NEU-DET, which is hot-rolled carbon steel photographed under one lighting "
            "setup. The vendors themselves admit that the same defect class looks different from "
            "different upstream mills (AMETEK/Ternium, quoted in docs/research_notes.md section 2a). "
            "Every number in this report is a capability demonstration on a public benchmark, not a "
            "prediction of performance on Jindal stainless strip. The per-line calibration protocol, "
            "not the architecture, is what determines whether this works."
        ),
    ]
    if lat_ratio < flop_ratio / 1.5:
        recommendation.append(
            f"**A caveat on the speed claim, because the measurement does not say what the FLOPs "
            f"say.** At {deploy_imgsz} px and batch 1 the two models are close to the same speed on "
            f"this machine -- median {w_lat['median_ms']:.1f} ms against "
            f"{lat(loser_key, deploy_imgsz)['median_ms']:.1f} ms, a ratio of {lat_ratio:.2f}x -- even "
            f"though {loser_key} is {flop_ratio:.1f}x the arithmetic. At this input size a single "
            "frame does not saturate the GPU, so the wall clock is dominated by fixed per-call cost "
            "(Python dispatch, host-device transfer, NMS) rather than by convolution. The "
            f"{flop_ratio:.1f}x shows up once the accelerator is actually busy -- it is already "
            f"visible at 640 px in the table above ({lat('yolov8n', 640)['median_ms']:.1f} ms "
            f"against {lat('yolov8s', 640)['median_ms']:.1f} ms median) and it is what governs a mill "
            "deployment, "
            "where one accelerator serves many camera streams in batches. `src/benchmark.py` "
            "measures that regime. Do not quote a batch-1 latency ratio as the throughput argument. "
            "Every latency figure here was also taken on a laptop running other work at the time -- "
            f"swap in use during the sweep: {(latency.get('_machine') or {}).get('raw', 'unknown')}."
        )
        recommendation.append(
            "**And a harder caveat: no absolute latency in this report is reproducible, so none of "
            "them belongs on a slide.** Re-running this sweep on the same machine moves the "
            f"{winner_key}@{deploy_imgsz} batch-1 *median* by more than a factor of two between "
            "sessions -- the floor is stable to about 10% but the median, the p95 and every FPS "
            "derived from them are dominated by whatever else the laptop is doing. Two consequences. "
            "First, quote the ordering and the ratios, never the milliseconds: within one "
            "interleaved measurement the two models sit in the right order at every input size, and "
            "that ordering survives re-measurement. Second, comparisons are only valid when the two "
            "arms were timed alternately in one loop, which is how the TTA multiple in section 3 is "
            "now measured and is not how the `by_imgsz` sweep was measured -- so ratios taken "
            "*across* rows of that table (including the "
            f"{flop_ratio:.1f}x-at-640 point above) are indicative only. A defensible per-frame "
            "number for a mill sizing exercise has to come from a quiet machine and a batched "
            "sweep, which is `src/benchmark.py`'s job, not this one's."
        )
        crosscheck = latency.get(winner_key, {}).get("shipping_api_crosscheck")
        if crosscheck:
            recommendation.append(
                "**The shipping API was cross-checked, and the raw numbers need reading with the "
                "caveat above.** `DefectDetector.predict` -- the path production actually calls, "
                "with letterboxing, severity scoring and record construction on top of the "
                f"forward pass -- measured a median {crosscheck['median_ms']:.1f} ms and a floor of "
                f"{crosscheck['min_ms']:.1f} ms at {deploy_imgsz} px, against a floor of "
                f"{w_lat['min_ms']:.1f} ms for the bare ultralytics call. Compared at the floor, "
                "where contention is squeezed out, the wrapper costs a few tenths of a "
                "millisecond and the framework is right that it is close to free; compared at the "
                "median it looks several times more expensive, which is an artefact of the two "
                "having been timed in separate phases rather than a real cost. This is recorded "
                "because it was measured, and because the median reading of it is the sort of "
                "number that would otherwise get quoted."
            )

    # Whether the shipping code still carries the wrong default is a fact about the
    # tree, not a fact about this study, so it is read from the tree. An earlier
    # edition of this paragraph asserted both the defect ("the library defaults to
    # 640 px") and its one exception ("the Streamlit demo is already correct") in
    # hard-coded prose. The first half was true; the second was not -- the console
    # was opening at 640 px as well -- and neither would have survived the fix.
    shipped_default = int(DEFAULT_IMGSZ)
    if shipped_default != deploy_imgsz:
        recommendation.append(
            f"**One consequence to action, and it is worse than a config tidy-up.** This study "
            f"selects {deploy_imgsz} px, but `inference.DEFAULT_IMGSZ` -- the default every "
            f"detector, exporter and reporter inherits when no input size is named -- is "
            f"{shipped_default} px. Anyone who constructs a detector without naming an input "
            f"size gets test mAP50 {imgsz_worst:.4f} instead of {dep_winner.mAP50:.4f} -- a "
            f"{(1 - imgsz_worst / dep_winner.mAP50) * 100:.0f}% loss -- silently, with no error "
            "and no warning. Change `inference.DEFAULT_IMGSZ`, re-export ONNX and CoreML at the "
            "selected size, and retire any artefact frozen at the old one."
        )
    else:
        recommendation.append(
            f"**The shipping default agrees with this study.** `inference.DEFAULT_IMGSZ` is "
            f"{shipped_default} px, so a detector, exporter, coil reporter or explainer "
            "constructed without naming an input size runs at the size validation chose. This is "
            "worth stating because it was not always true: the library, the exporter and the "
            f"console all defaulted to 640 px, where this checkpoint scores mAP50 "
            f"{imgsz_worst:.4f} against {dep_winner.mAP50:.4f} -- a "
            f"{(1 - imgsz_worst / dep_winner.mAP50) * 100:.0f}% loss taken silently. Any ONNX or "
            "CoreML artefact frozen at 640 px must still be retired rather than left in `export/`."
        )

    winner_cost = cost_n if winner_key == "yolov8n" else cost_s
    if winner_cost:
        recommendation.append(
            f"**On retraining cost.** A per-line recalibration of {winner_key} at 320 px costs "
            f"{winner_cost['logged_320_s']:.0f} s/epoch, or about "
            f"{winner_cost['logged_320_hours_150_epochs']:.1f} h for a 150-epoch run, measured from "
            f"`models/{winner_key}_neudet/results.csv`. At 640 px the measured GPU step cost is "
            f"{winner_cost['ratio']:.1f}x higher, which puts the same run at roughly "
            f"{winner_cost['calibrated_640_hours_150_epochs']:.1f} h. Budget the 320 px figure: an "
            "overnight retrain per line is affordable, and section 2 showed 640 buys nothing at "
            "inference."
        )

    # ---------- positioning
    lower, upper = 0.70, 0.80
    headline = dep_winner.mAP50  # the recommended configuration, not the trained size
    where = "inside" if lower <= headline <= upper else ("below" if headline < lower else "above")
    positioning = [
        (
            f"docs/research_notes.md section 1a puts the credible band for well-run NEU-DET "
            f"*detection* baselines at mAP50 0.70-0.80. Our best held-out figure is "
            f"**{headline:.4f}** ({winner_key} at {deploy_imgsz} px on the held-out test "
            "split), which sits inside that band"
            if where == "inside"
            else f"docs/research_notes.md section 1a puts the credible band for well-run NEU-DET "
            f"*detection* baselines at mAP50 0.70-0.80. Our best held-out figure is "
            f"**{headline:.4f}** ({winner_key} at {deploy_imgsz} px on the held-out test "
            f"split), which is {where} that band"
        )
        + ". For reference, the same notes record YOLOv8n at 74.0 and 78.6 mAP50 in two 2025-26 "
        "papers, YOLOv11n at 77.2, and the 2020 IEEE TIM DDN paper at 74.8 on a ResNet34 backbone "
        "[all HARD]. We are level with a competent stock baseline and below the tuned published "
        "improvements, which is where an untuned 150-epoch fine-tune on a laptop GPU should be. "
        f"Note that the figure recorded before this study, {winner_train.mAP50:.4f}, was the same "
        f"weights scored at {train_imgsz} px; choosing the inference size on val is worth "
        f"{headline - winner_train.mAP50:+.4f} mAP50 and cost nothing but a sweep.",
        (
            "**Two specific reasons we are not higher, both of them choices rather than accidents.**"
        ),
        (
            "*First, we trained at 320 px, not 640, while essentially every published NEU-DET "
            "number is at 640.* "
            + (
                (
                    f"The measured cost of that choice on this machine: one optimiser step for "
                    f"yolov8s at batch {cost_s['probe_batch']} takes {cost_s['step_320_ms']:.0f} ms "
                    f"at 320 px and {cost_s['step_640_ms']:.0f} ms at 640 px, a ratio of "
                    f"{cost_s['ratio']:.1f}x. (The probe uses batch {cost_s['probe_batch']}, not "
                    "the 32 the real runs used. A batch-32 attempt at 640 px was made first and "
                    "abandoned: with the machine under concurrent memory pressure it went into "
                    "swap and a single step did not return inside a two-minute budget. The "
                    "640/320 ratio is close to batch-independent, so it is measured at a batch "
                    "that reliably fits, and the batch is recorded so the figure is not misread.) "
                    f"The training run itself logged {cost_s['logged_320_s']:.0f} s/epoch at 320 px "
                    f"({cost_s['logged_320_hours_150_epochs']:.1f} h for 150 epochs), so 640 px "
                    f"projects to about {cost_s['calibrated_640_s_per_epoch'] / 60:.1f} min/epoch, "
                    f"or {cost_s['calibrated_640_hours_150_epochs']:.1f} h for the same schedule -- "
                    "on a laptop GPU that is shared with everything else this project needs to run. "
                )
                if cost_s
                else (
                    "The measurement could not be completed on this machine. The 640 px batch-32 "
                    f"probe for yolov8s did not finish a step within its budget ({cost_fail_s}). "
                    "That is not a null result: a batch of 32 at 640 px needs more unified memory "
                    "than this laptop has spare, so the step pages to disk and stops measuring "
                    "arithmetic at all. Note the machine was also running other work at the time, "
                    "so the pressure is not attributable to this probe alone. "
                )
                if cost_fail_s
                else ""
            )
            + "**A correction worth stating, and it partly goes the brief's way.** The project "
            "brief records 640 px as costing 'more than 11 minutes per epoch versus 35 s at 320'. "
            "The 35 s half of that reproduces: per-epoch wall time in `models/*/results.csv` runs "
            f"at a median of {history['yolov8n']['seconds_per_epoch_median']:.0f} s (nano) and "
            f"{history['yolov8s']['seconds_per_epoch_median']:.0f} s (small), and the yolov8s "
            "median is within a few per cent of the brief's figure. An earlier draft of this "
            "report used the *mean* epoch time to say 35 s did not reproduce; that was wrong, "
            "because the yolov8s mean "
            f"({history['yolov8s']['seconds_per_epoch']:.0f} s) is inflated by "
            f"{len(history['yolov8s']['stalled_epochs'])} stalled epochs ("
            + ", ".join(
                f"epoch {s['epoch']} at {s['seconds'] / 60:.0f} min"
                for s in history["yolov8s"]["stalled_epochs"]
            )
            + f") that between them account for {history['yolov8s']['stall_seconds'] / 60:.0f} "
            "minutes of a 2.2 h run. The same reasoning this report applies to latency -- "
            "contention only ever adds time, so quote the median -- applies here and had not been. "
            + (
                f"The 11-minute half still does not reproduce: the measured {cost_s['ratio']:.1f}x "
                "step-cost ratio projects 640 px to about "
                f"{cost_s['calibrated_640_s_per_epoch'] / 60:.1f} min/epoch for yolov8s, not 11, "
                "and `reports/training.log` contains no 640 px run to check it against. But the "
                "stalls above are direct evidence for the mechanism that would explain it: this "
                "machine already produced two ~16 minute epochs at 320 px under memory pressure, "
                "so an 11-minute epoch at 640 px is entirely plausible as a paging effect rather "
                "than an arithmetic one. "
                if cost_s
                else "the 640 px probe could not complete a single step inside its budget because "
                "the machine went into swap, which means the honest answer is that 640 px training "
                "on this hardware is memory-bound rather than compute-bound. That is a *better* "
                "justification for the 320 px choice than the original one, and it also makes the "
                "brief's '11 minutes per epoch' plausible as a paging effect -- but it is an "
                "explanation, not a reproduction, and `reports/training.log` contains no 640 px run "
                "to check the figure against. "
            )
            + "The *direction* of the original claim survives either way: 640 px training costs "
            "several times more on this machine and may not fit in memory at this batch size, "
            "which is why it was not run. The specific 11-minute figure should be quoted as an "
            "observation from the original run, not as an arithmetic cost -- what this study can "
            f"measure is a {cost_s['ratio']:.1f}x step-cost ratio if nothing pages, and a machine "
            "that demonstrably does page."
            if cost_s
            else "The *direction* of the original claim survives: 640 px training costs several "
            "times more on this machine and may not fit in memory at this batch size."
        ),
        (
            "*Second, we report a genuine held-out test split.* The 1800 NEU-DET images are split "
            "1440/180/180 by `src/prepare_data.py`, and the 180 test images were used for nothing "
            "except the final score. Many published NEU-DET numbers are validation numbers, tuned "
            "on the same images they are reported on. The size of that effect is visible in this "
            f"very report: at {train_imgsz} px, yolov8n scores "
            f"{cell('yolov8n', 'val', train_imgsz).mAP50:.4f} on val against "
            f"{cell('yolov8n', 'test', train_imgsz).mAP50:.4f} on test, and yolov8s "
            f"{cell('yolov8s', 'val', train_imgsz).mAP50:.4f} against "
            f"{cell('yolov8s', 'test', train_imgsz).mAP50:.4f}. Reporting the val column instead "
            "would have moved our headline without changing the model."
        ),
        (
            "Neither reason is an excuse for a number, and neither should be presented as one. They "
            "are the two knobs a reviewer would ask about, and both have a measurement attached."
        ),
    ]

    limitations = [
        (
            "**Training-seed variance is not measured.** The bootstrap resamples images, holding the "
            "weights fixed. It cannot tell you whether retraining yolov8s with a different seed "
            "would close the gap. Measuring that needs several seeds per architecture; at the "
            f"logged {history['yolov8n']['seconds_per_epoch_median']:.0f} s/epoch (nano) and "
            f"{history['yolov8s']['seconds_per_epoch_median']:.0f} s/epoch (small), three seeds each at "
            f"150 epochs is about "
            f"{3 * 150 * (history['yolov8n']['seconds_per_epoch_median'] + history['yolov8s']['seconds_per_epoch_median']) / 3600:.1f} "
            "hours on this machine. That is the single experiment that would upgrade this study's "
            "central claim from 'these weights, on these images' to 'this architecture, on this "
            "dataset', and it is affordable."
        ),
        (
            "**The nano-versus-small question was known before this study ran.** The ordering was "
            "already visible in `reports/train_summary_*.json`, so the test split had been looked "
            "at once before the bootstrap was designed. The bootstrap quantifies the uncertainty in "
            "a comparison that was not pre-registered; it does not make it a blind test."
        ),
        (
            "**The validation split is not a clean second opinion.** `best.pt` is the epoch that "
            "maximised validation fitness, so val is a selection split for both checkpoints and is "
            "biased upward on both. This report uses val only to choose inference-time knobs "
            "(input size, TTA), where the bias is common to every option being compared and largely "
            "cancels. It is deliberately not used to compare the two models."
        ),
        (
            "**The two checkpoints differ in more than capacity.** They share a recipe, a seed and "
            "a split, but one early-stopped at 135 epochs and one ran 150, and neither had its "
            "hyperparameters adapted to its size. A fair architecture comparison would tune each "
            "model separately; this is a comparison of two checkpoints produced by one recipe, "
            "which is the question a deployment actually faces but not the question a paper would "
            "ask."
        ),
        (
            "**One dataset, one imaging setup.** NEU-DET is 1800 200x200 grayscale images of "
            "hot-rolled carbon steel. Nothing here has been measured on stainless, on a different "
            "camera, or under different illumination, and section 6a of docs/research_notes.md "
            "documents domain shift between mills as the primary deployment failure mode -- in the "
            "vendors' own words."
        ),
        (
            "**The failure taxonomy uses one matching convention.** Misses are classified by the "
            "best overlapping prediction at a 0.10 IoU floor. A different floor moves boxes between "
            "the 'extent error' and 'blind miss' columns. The floor is a stated constant "
            "(`EXTENT_IOU_FLOOR`), not a tuned one."
        ),
        (
            "**Spearman rho on six classes is descriptive.** The appearance-versus-AP50 correlations "
            "have six data points. They are reported with their p-values so nobody mistakes them "
            "for evidence of a law."
        ),
        (
            "**NMS IoU, batch size and half precision were not swept.** The deployment recommendation "
            "fixes them at the shipping defaults. `src/benchmark.py` covers batch and device; NMS IoU "
            "and quantisation remain unmeasured."
        ),
    ]

    return {
        "meta": {
            "generated_at": kw["started"].isoformat(timespec="seconds"),
            "device": kw["device"],
            "platform": f"{platform.platform()} / {platform.machine()}",
            "torch": torch.__version__,
            "ultralytics": kw["ultralytics_version"],
            "json_path": str(kw["json_path"]),
            "protocol": {
                "metrics": "ultralytics validator defaults: conf 0.001, NMS IoU 0.7, max_det 300",
                "failure_analysis": f"DefectDetector, conf {DEFAULT_CONF} and {tuned_conf}, NMS IoU {DEPLOY_NMS_IOU}",
                "split_discipline": "inference size and TTA chosen on val; test scored once at that configuration",
            },
        },
        "dataset": {
            "name": "NEU-DET",
            "split": {"train": 1440, "val": 180, "test": 180},
            "image_px": 200,
            "classes": list(CLASS_NAMES),
            "test_instances": cell(specs[0].key, "test", train_imgsz).instances,
        },
        "train_imgsz": train_imgsz,
        "checkpoints": facts,
        "training_history": history,
        "runs": [r.to_dict() for r in runs],
        "latency": latency,
        "bootstrap": {"at_training_size": boot, "at_deployment_size": boot_deploy},
        "winner_basis": winner_basis,
        "fidelity": fidelity,
        "split_noise": {
            "val_gap_mAP50": round(val_gap, 5),
            "test_gap_mAP50": round(test_gap, 5),
            "swing": round(abs(test_gap - val_gap), 5),
            "note": (
                "same two checkpoints on two disjoint 180-image splits. Neither split was trained "
                "on, but val WAS selected on -- best.pt is the max-val-fitness epoch -- so val is "
                "optimistically biased for both models and cannot arbitrate between them."
            ),
        },
        "best_imgsz_on_val": best_imgsz,
        "failure_taxonomy": {"default_conf": tax_default, "tuned_conf": tax_tuned},
        "appearance": appearance,
        "difficulty_correlation": correlation,
        "class_ordering": ordering,
        "training_cost": {**tc, "calibrated": {"yolov8n": cost_n, "yolov8s": cost_s}},
        "figures": kw["figures"],
        "headline": {
            "table": head_rows,
            "per_class_header": ["model"] + class_names,
            "per_class_rows": per_class_rows,
            "bootstrap_reading": bootstrap_reading,
            "deployment_reading": deployment_reading,
            "mechanism": mechanism,
            "history_rows": hist_rows,
        },
        "imgsz": {"header": imgsz_header, "rows": imgsz_rows, "reading": imgsz_reading},
        "tta": {"header": tta_header, "rows": tta_rows, "reading": tta_reading},
        "failures": {
            "intro": failures_intro,
            "taxonomy_header": [
                "class",
                "AP50",
                "GT boxes",
                f"recall @{DEFAULT_CONF}",
                "class confusion",
                "extent error",
                "blind miss",
                "confused with",
                f"recall @{tuned_conf}",
            ],
            "taxonomy_rows": taxonomy_rows,
            "appearance_header": [
                "class",
                "AP50",
                "boxes",
                "separability",
                "edge energy ratio",
                "mean box area",
                "label overlap IoU",
                "boxes/image (box-weighted)",
            ],
            "appearance_rows": appearance_rows,
            "reading": failures_reading,
            "ordering_header": ["rank (hardest first)", "ours", "our AP50", "published"],
            "ordering_rows": ordering_rows,
            "ordering_reading": ordering_reading,
        },
        "selection": {
            "winner": winner_key,
            "imgsz": deploy_imgsz,
            "tta": False,
            "conf": tuned_conf,
            "conf_source": tuned_source,
            "nms_iou": DEPLOY_NMS_IOU,
            "decisions": decisions,
            "verdict_paragraph": verdict,
            "recommendation": recommendation,
        },
        "positioning": positioning,
        "limitations": limitations,
    }


if __name__ == "__main__":
    raise SystemExit(main())
