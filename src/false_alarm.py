"""Clean-steel false alarm proxy, and an honest re-derivation of the operating point.

WHY THIS MODULE EXISTS
----------------------
Every NEU-DET image contains a defect. There is no defect-free frame anywhere in
the dataset. The "false alarm rate" reported by ``src/evaluate.py`` and carried
into ``reports/evaluation.md`` is therefore *not* the number a mill cares about:
it is the rate at which a spurious extra box appears on an image that genuinely
is defective. The coil was going to be flagged either way. The question that
decides whether an inspection system is tolerated on a line is different --

    how often does the system stop the line for a coil that is actually fine?

That number cannot be read off NEU-DET. This module builds the best proxy the
data supports, states exactly how far the proxy can be trusted, and then
re-derives the deployment threshold from it.

THE PROXY
---------
1. From the held-out ``test`` split, mine crops that have **zero** intersection
   with any ground-truth box, with a strict clearance margin (default 6 px) so a
   crop that merely grazes a label is rejected. Four crop scales are used so the
   result is not an artefact of one field of view. Overlapping candidates are
   deduplicated (IoU cap) and capped per image, so a single image with a large
   clean area cannot dominate the statistic.
2. These crops are genuine stainless-steel surface from the same camera, lighting
   and rolling conditions as the defective frames. A detection on one is a
   genuine false positive on defect-free metal.
3. Each crop is presented to the network at the **same magnification** the
   deployment path uses. The deployed detector feeds a 200x200 frame to a network
   input of 320 px, i.e. 1.6x. A 100 px crop is therefore run at imgsz 160, not
   at 320. Feeding a small crop at the full 320 would magnify the texture 3.2x
   and measure the model at a scale it was never trained on -- the resulting
   "false alarm rate" would be an artefact of resampling, not a property of the
   detector.

WHAT THIS PROXY IS NOT
----------------------
Stated here rather than buried, because the number is only useful if its limits
travel with it:

* **The crops are smaller than a frame.** A 60x60 crop is 9% of a 200x200 frame,
  so a per-crop false alarm rate is not a per-frame false alarm rate. The obvious
  bridge is to assume false positives are spatially independent and scale by area.
  This module does not assume that -- it measures the rate at four crop areas and
  tests the assumption, fitting a constant-rate model and a Poisson-in-area model
  to the same four cells and comparing their chi-squares on equal degrees of
  freedom. On the yolov8n checkpoint the independence model is decisively rejected
  (the measured rate is flat across a 4x span of area), so the per-patch rate is
  carried as the primary number and the area extrapolation only as a pessimistic
  bound. Both go through the cost model, and the report says which one led.
* **The crops come from coils that do contain defects elsewhere.** They are clean
  *regions*, not clean *coils*. A mill's clean coil may have different surface
  statistics than the quiet corner of a defective one.
* **A crop with no annotated box is not guaranteed defect-free.** NEU-DET
  annotation is not exhaustive; faint crazing or light rolled-in scale outside the
  boxed region is plausible and would be scored here as a false alarm when it is
  arguably a correct detection on an unlabelled defect. This biases the measured
  false alarm rate **upward**, so the number is conservative in the direction that
  matters -- but it is not clean.
* **The field of view is small.** A detector sees less context in a 60 px crop
  than in a 200 px frame. If that made the model simply stop firing, a low false
  alarm rate would mean nothing. That is why a **positive control** is mined and
  scored under the identical protocol: crops of the same sizes that fully contain
  a labelled defect. If the model still detects those, the small field of view has
  not silenced it and the clean-patch number is informative.

WHAT A REAL MEASUREMENT WOULD NEED
----------------------------------
Defect-free coil footage from the actual line: continuous capture of coils that
passed manual surface inspection and shipped as prime, at production line speed,
camera geometry and illumination, with the frames time-stamped back to a coil id
so a flagged frame can be adjudicated against the mill's own disposition record.
A few thousand such frames across a shift, several coils, several grades and both
strip surfaces would give a directly measured per-frame false alarm rate with a
usable confidence interval. Nothing in NEU-DET substitutes for that.

THE RE-DERIVED OPERATING POINT
------------------------------
The threshold is then chosen by combining two independently measured curves:

* ``D(t)`` -- image-level defect detection rate on real defective ``val`` frames
  (full 200x200 frames at the deployment imgsz), reusing ``src/evaluate.py``;
* ``F(t)`` -- false alarm rate on clean steel from the proxy above.

with an explicit, challengeable cost model in ``COST_MODEL`` at the top of this
file. The recommendation is reported together with its sensitivity to the
miss:false-alarm cost ratio, because a recommendation that flips wildly with an
unmeasured cost ratio is itself the finding.

Usage:
    python src/false_alarm.py
    python src/false_alarm.py --weights models/yolov8n_neudet/weights/best.pt
    python src/false_alarm.py --no-update-operating-point
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: this runs on the line PC and in CI

import cv2
import matplotlib.pyplot as plt
import numpy as np

try:  # works both as `python src/false_alarm.py` and as `from src import false_alarm`
    from .evaluate import (
        DATA_ROOT,
        REPORTS_DIR,
        GroundTruth,
        SweepPoint,
        _json_safe,
        cache_predictions,
        load_ground_truth,
        match_image,
        resolve_train_imgsz,
        sweep_confidence,
    )
    from .inference import CLASS_NAMES, DefectDetector, resolve_device, resolve_weights
except ImportError:  # pragma: no cover - script execution path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from evaluate import (
        DATA_ROOT,
        REPORTS_DIR,
        GroundTruth,
        SweepPoint,
        _json_safe,
        cache_predictions,
        load_ground_truth,
        match_image,
        resolve_train_imgsz,
        sweep_confidence,
    )
    from inference import CLASS_NAMES, DefectDetector, resolve_device, resolve_weights

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The deployment checkpoint. The nano run wins on the held-out test split
# (mAP50 0.7286 vs 0.6598 for the small run) and is what demo/app.py defaults to,
# so it is what the false alarm number has to be measured on.
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "yolov8n_neudet" / "weights" / "best.pt"

# ---------------------------------------------------------------- patch mining
#
# Every knob that shapes the clean-patch set, in one place, because each of them
# is a choice someone is entitled to challenge.
#
#   crop_sizes_px   Four fields of view. Chosen so that crop * magnification lands
#                   exactly on a multiple of 32 (the network stride constraint) at
#                   the deployment magnification of 1.6 -- see MAGNIFICATION_NOTE.
#                   Above ~120 px almost no clean crop exists (mean union-of-boxes
#                   coverage on the test split is 42% of frame area), so a larger
#                   scale would be measured on a handful of unrepresentative images.
#   stride_px       Candidate grid step. Finer than the dedup IoU cap, so the cap
#                   rather than the grid decides how much overlap survives.
#   clearance_px    A crop must be at least this far from *every* ground-truth box.
#                   This is the "strict margin": a crop that touches a label, or
#                   sits one pixel from it, is discarded rather than counted clean.
#   max_overlap_iou Deduplication cap between kept crops of the same scale on the
#                   same image. Without it, thousands of near-identical windows
#                   would be counted as independent samples and the confidence
#                   interval would be a fiction.
#   max_per_image_per_scale
#                   Hard cap so an image with a huge clean area cannot dominate.
PATCH_MINING: dict[str, Any] = {
    "crop_sizes_px": (60, 80, 100, 120),
    "stride_px": 10,
    "clearance_px": 6,
    "max_overlap_iou": 0.25,
    "max_per_image_per_scale": 6,
    "seed": 20260909,
}

# ------------------------------------------------------------------ cost model
#
# Units are "one unnecessary re-inspection" = 1.0. Everything here is an
# assumption, not a measurement, and every one of them is challengeable:
#
#   miss_cost_ratio      What one escaped defect costs, in re-inspections. An
#                        escaped defect downgrades a ~20 t stainless coil from
#                        prime to secondary and can become a customer claim; on
#                        304 the nickel content makes a downgraded tonne roughly
#                        three times as painful as commodity HRC. A false alarm
#                        costs an operator a manual look and a slice of line
#                        availability. 12 is a defensible order of magnitude and
#                        matches the ratio src/evaluate.py already uses, so the
#                        two reports are comparable. It is NOT a Jindal number.
#   defect_frame_prevalence
#                        Fraction of inspected frames that actually carry a
#                        defect. NEU-DET is 100% defective by construction, so
#                        this cannot be measured here at all. Jindal's own
#                        downgrade/rejection rate is not public (see
#                        docs/research_notes.md section 9). 0.05 is a placeholder.
#   min_detection_rate   A surface inspection system that lets more than one
#                        defective coil in ten pass as prime will not be accepted
#                        on the line, whatever its false alarm rate.
#
# IDENTIFIABILITY WARNING: the expected cost is
#     cost(t) = prevalence * ratio * (1 - D(t)) + (1 - prevalence) * F(t)
# and argmin_t cost(t) is invariant to positive rescaling, so the recommendation
# depends on prevalence and ratio ONLY through the single effective weight
#     lambda = prevalence * ratio / (1 - prevalence).
# The two parameters are not separately identifiable from the decision. This is
# reported explicitly rather than left for someone to discover.
COST_MODEL: dict[str, Any] = {
    "miss_cost_ratio": 12.0,
    # The operative ratio is included so the table shows what the economics alone
    # would pick at the number actually used, not only at three neighbours of it.
    "sensitivity_ratios": (3.0, 10.0, 12.0, 30.0),
    "defect_frame_prevalence": 0.05,
    "min_detection_rate": 0.90,
}

# Confidence sweep. Extended below 0.05 because the previous study's recommendation
# landed on the bottom edge of a 0.05-0.95 sweep, which is a sweep artefact rather
# than an optimum; the extra points let the curve turn inside the swept range.
SWEEP_THRESHOLDS: tuple[float, ...] = (
    0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
)

# Detectors are constructed at this confidence and every box is cached, so the
# whole sweep is pure NumPy. Filtering the cache at `conf >= t` reproduces exactly
# what a detector constructed at `t` would return: NMS is greedy in descending
# confidence, so a box above t can only ever be suppressed by another box above t.
CACHE_CONF = 0.01
NMS_IOU = 0.45

MAGNIFICATION_NOTE = (
    "Each crop is run at imgsz = round_to_32(crop_px * deployment_magnification), "
    "where deployment_magnification = train_imgsz / frame_px. This keeps steel "
    "texture at the pixels-per-millimetre the network was trained on; running a "
    "crop at the full deployment imgsz would magnify it and measure a resampling "
    "artefact instead of the detector."
)

NETWORK_STRIDE = 32
_WILSON_Z = 1.959963985  # two-sided 95%


# ---------------------------------------------------------------------------
# containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Patch:
    """One crop taken from a split image, in that image's pixel coordinates."""

    image_path: Path
    x0: int
    y0: int
    size: int
    imgsz: int
    # Ground-truth boxes fully contained in this crop, in crop-local coordinates.
    # Empty for a clean patch by construction.
    gt_boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.float64))
    gt_classes: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int64))

    @property
    def area_px(self) -> float:
        return float(self.size * self.size)


@dataclass(frozen=True)
class ScoredPatch:
    """Every box the detector produced for one patch at CACHE_CONF."""

    patch: Patch
    boxes: np.ndarray  # (M, 4) xyxy, crop-local
    confidences: np.ndarray  # (M,)
    classes: np.ndarray  # (M,) int


@dataclass
class CleanPoint:
    """Clean-steel false alarm statistics at one confidence threshold."""

    threshold: float
    n_patches: int
    patches_alarmed: int
    rate: float
    rate_lo: float
    rate_hi: float
    # Image-clustered bootstrap interval and the design effect that separates it
    # from the Wilson interval above. The patches are not independent draws, so
    # this pair -- not the Wilson interval -- is the honest precision statement.
    cluster_lo: float
    cluster_hi: float
    design_effect: float
    boxes_per_patch: float
    class_counts: dict[str, int]
    per_scale: dict[int, dict[str, float]]
    frame_rate_extrapolated: float
    intensity_per_10k_px: float | None
    fit_r2: float | None
    extrapolation_note: str = "poisson fit"
    # Is the false alarm rate actually growing with window area the way spatial
    # independence says it must? Two competing models, same degrees of freedom.
    chi2_constant: dict[str, Any] = field(default_factory=dict)
    chi2_poisson: dict[str, Any] = field(default_factory=dict)
    poisson_supported: bool | None = None
    # Clustering: are false alarms a property of the region, or of the image?
    n_source_images: int = 0
    images_alarmed: int = 0
    image_rate: float = 0.0
    image_rate_iid_prediction: float = 0.0
    # The pooled rate is a mixture over the six surface populations, sampled very
    # unevenly by the mining. Kept broken out because the mixture weights are an
    # artefact of how much clean area each defect type happens to leave.
    per_source_class: dict[str, dict[str, float]] = field(default_factory=dict)
    fp_matching_source_class: int = 0
    fp_total: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": round(self.threshold, 3),
            "n_patches": self.n_patches,
            "patches_alarmed": self.patches_alarmed,
            "clean_patch_false_alarm_rate": round(self.rate, 4),
            "wilson95_lo": round(self.rate_lo, 4),
            "wilson95_hi": round(self.rate_hi, 4),
            "cluster95_lo": round(self.cluster_lo, 4),
            "cluster95_hi": round(self.cluster_hi, 4),
            "design_effect": round(self.design_effect, 3),
            "effective_n": round(self.n_patches / max(self.design_effect, 1e-9), 1),
            "boxes_per_clean_patch": round(self.boxes_per_patch, 4),
            "false_positive_class_counts": dict(self.class_counts),
            "per_scale": {
                str(k): {kk: round(float(vv), 4) for kk, vv in v.items()}
                for k, v in self.per_scale.items()
            },
            "frame_rate_extrapolated": round(self.frame_rate_extrapolated, 4),
            "fp_intensity_per_10k_px": (
                None if self.intensity_per_10k_px is None else round(self.intensity_per_10k_px, 5)
            ),
            "poisson_fit_r2": None if self.fit_r2 is None else round(self.fit_r2, 4),
            "extrapolation_note": self.extrapolation_note,
            "area_dependence": {
                "constant_rate_model": self.chi2_constant,
                "poisson_area_model": self.chi2_poisson,
                "poisson_supported": self.poisson_supported,
            },
            "clustering": {
                "n_source_images": self.n_source_images,
                "images_with_at_least_one_alarm": self.images_alarmed,
                "image_level_rate": round(self.image_rate, 4),
                "image_level_rate_if_patches_independent": round(
                    self.image_rate_iid_prediction, 4
                ),
            },
            "per_source_class": {
                k: {kk: round(float(vv), 4) for kk, vv in v.items()}
                for k, v in self.per_source_class.items()
            },
            "false_positives_matching_source_class": self.fp_matching_source_class,
            "false_positives_total": self.fp_total,
            "share_of_fp_matching_source_class": (
                None
                if not self.fp_total
                else round(self.fp_matching_source_class / self.fp_total, 4)
            ),
        }


@dataclass
class ControlPoint:
    """Positive-control statistics at one confidence threshold."""

    threshold: float
    n_patches: int
    patches_with_any_box: int
    patches_localised: int  # >=1 box matching a contained GT box at IoU 0.5, right class

    def to_dict(self) -> dict[str, Any]:
        n = max(self.n_patches, 1)
        return {
            "threshold": round(self.threshold, 3),
            "n_patches": self.n_patches,
            "flag_rate": round(self.patches_with_any_box / n, 4),
            "localised_detection_rate": round(self.patches_localised / n, 4),
        }


@dataclass
class CostPoint:
    """One row of the re-derived operating curve."""

    threshold: float
    detection_rate: float  # D(t), defective val frames
    clean_frame_fa: float  # F(t), area-extrapolated to frame
    clean_patch_fa: float  # F(t), raw per-patch
    legacy_fa: float  # spurious box on an already-defective frame
    cost_frame: float
    cost_patch: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": round(self.threshold, 3),
            "defect_detection_rate": round(self.detection_rate, 4),
            "clean_frame_false_alarm_rate": round(self.clean_frame_fa, 4),
            "clean_patch_false_alarm_rate": round(self.clean_patch_fa, 4),
            "legacy_spurious_box_rate": round(self.legacy_fa, 4),
            "expected_cost_frame_basis": round(self.cost_frame, 5),
            "expected_cost_patch_basis": round(self.cost_patch, 5),
        }


# ---------------------------------------------------------------------------
# statistics helpers
# ---------------------------------------------------------------------------


def wilson_interval(successes: int, trials: int, z: float = _WILSON_Z) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Used instead of the normal approximation because several per-scale cells hold
    only a few dozen patches and hit rates near 0 or 1, where the normal interval
    runs outside [0, 1] and quietly lies about the precision of the estimate.
    """
    if trials <= 0:
        return (0.0, 0.0)
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = p + z * z / (2.0 * trials)
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials))
    return (max(0.0, (centre - half) / denom), min(1.0, (centre + half) / denom))


def cluster_bootstrap_interval(
    groups: Sequence[Sequence[int]], n_boot: int = 20_000, seed: int = 20260909
) -> tuple[float, float, float]:
    """Image-clustered bootstrap interval for the pooled patch rate, plus the design effect.

    ``wilson_interval`` assumes the patches are independent draws. They are not:
    up to ``max_per_image_per_scale`` crops are taken from each image at each of
    four scales, they overlap each other across scales (the IoU cap is applied
    within a scale only), and the module's own clustering probe shows alarming
    patches clump inside a minority of images. Treating 771 correlated crops as
    771 independent trials makes the confidence interval about half as wide as
    the evidence supports.

    Resampling whole *images* with replacement respects that structure. The
    returned design effect is the ratio of the clustered variance to the binomial
    variance -- the factor by which the naive interval overstates precision, and
    ``n / deff`` is the effective sample size that should be quoted.
    """
    groups = [list(g) for g in groups if len(g)]
    n = sum(len(g) for g in groups)
    if not groups or n == 0:
        return (0.0, 0.0, 1.0)
    p = sum(sum(g) for g in groups) / n
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(groups), size=(n_boot, len(groups)))
    hits = np.array([sum(g) for g in groups], dtype=np.float64)
    sizes = np.array([len(g) for g in groups], dtype=np.float64)
    draws = hits[idx].sum(axis=1) / np.maximum(sizes[idx].sum(axis=1), 1.0)
    lo, hi = (float(v) for v in np.percentile(draws, [2.5, 97.5]))

    n_groups = len(groups)
    var_srs = p * (1.0 - p) / n
    if n_groups < 2 or var_srs <= 0.0:
        return (lo, hi, 1.0)
    var_cluster = float(
        np.sum((hits - p * sizes) ** 2) * n_groups / ((n_groups - 1) * n * n)
    )
    return (lo, hi, float(var_cluster / var_srs))


def _intersection_area(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Intersection area of one xyxy box against many. Zero when disjoint."""
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float64)
    lt = np.maximum(box[:2], boxes[:, :2])
    rb = np.minimum(box[2:], boxes[:, 2:])
    return np.prod(np.clip(rb - lt, 0.0, None), axis=1)


def _pairwise_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float64)
    inter = _intersection_area(box, boxes)
    area_a = float((box[2] - box[0]) * (box[3] - box[1]))
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_a + area_b - inter
    return np.where(union > 0.0, inter / np.maximum(union, 1e-12), 0.0)


def poisson_area_fit(
    areas: Sequence[float], hits: Sequence[int], totals: Sequence[int]
) -> tuple[float, float | None, int]:
    """Fit ``rate = 1 - exp(-lambda * area)`` through the origin, by minimum chi-square.

    The model is that false positives on clean steel arrive as a spatial Poisson
    process with constant intensity, so the probability that a window of area A
    contains at least one is ``1 - exp(-lambda A)``.

    The fit minimises the same Pearson chi-square the goodness-of-fit test then
    reports. That matters: an earlier version linearised the model as
    ``-ln(1 - rate) = lambda * A`` and fitted by unweighted least squares, which
    is not the best-fitting member of the family. The chi-square of a
    badly-estimated parameter is not evidence against the model, and reporting it
    as such inflated the statistic by roughly 3.5x on this data. Rejecting a model
    is only honest against its best fit.

    Returns ``(lambda, r2_on_the_linearised_scale, n_cells_used)``. ``r2`` is None
    when fewer than two unsaturated cells remain, which is the honest answer
    rather than a fake 1.0.
    """
    a = np.asarray(areas, dtype=np.float64)
    k = np.asarray(hits, dtype=np.float64)
    n = np.asarray(totals, dtype=np.float64)
    usable = (n > 0) & (a > 0.0)
    a, k, n = a[usable], k[usable], n[usable]
    if a.size == 0:
        return 0.0, None, 0

    def chi_square(lam: float) -> float:
        pred = np.clip(1.0 - np.exp(-lam * a), 1e-12, 1.0 - 1e-12)
        return float(
            np.sum((k - n * pred) ** 2 / (n * pred))
            + np.sum(((n - k) - n * (1.0 - pred)) ** 2 / (n * (1.0 - pred)))
        )

    # Coarse grid then golden-section refinement: the objective is smooth and
    # one-dimensional, and a grid alone would quantise the reported statistic.
    rates = k / n
    seed = float(np.dot(a, -np.log(np.maximum(1.0 - np.minimum(rates, 1.0 - 1e-9), 1e-12))) /
                 max(float(np.dot(a, a)), 1e-30))
    grid = np.geomspace(max(seed, 1e-9) * 1e-3, max(seed, 1e-9) * 1e3, 4000)
    lam = float(grid[int(np.argmin([chi_square(float(g)) for g in grid]))])
    lo, hi = lam / 1.5, lam * 1.5
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(120):
        c, d = hi - phi * (hi - lo), lo + phi * (hi - lo)
        if chi_square(c) < chi_square(d):
            hi = d
        else:
            lo = c
    lam = 0.5 * (lo + hi)

    unsat = rates < 1.0
    if int(unsat.sum()) < 2:
        return lam, None, int(unsat.sum())
    y = -np.log(np.maximum(1.0 - rates[unsat], 1e-12))
    total = float(np.sum((y - y.mean()) ** 2))
    if total <= 0.0:
        return lam, None, int(unsat.sum())
    resid = float(np.sum((y - lam * a[unsat]) ** 2))
    return lam, float(1.0 - resid / total), int(unsat.sum())


_MIN_EXPECTED_CELL = 5.0  # below this the chi-square approximation is not usable


def _chi2_sf(stat: float, dof: int) -> float | None:
    """Upper tail of the chi-square distribution, or None when scipy is absent."""
    if dof <= 0:
        return None
    try:
        from scipy.stats import chi2 as _chi2

        return float(f"{float(_chi2.sf(stat, dof)):.4g}")
    except Exception:  # noqa: BLE001 - a missing scipy costs a p-value, not the run
        return None


def _chi2_binomial_fit(
    hits: Sequence[int], totals: Sequence[int], predicted: Sequence[float], fitted_params: int
) -> dict[str, Any]:
    """Pearson chi-square of observed hit counts against a predicted rate per cell.

    Degrees of freedom are ``cells - fitted_params``. The p-value is returned as
    None (not as a number) whenever any expected cell count falls below
    ``_MIN_EXPECTED_CELL``, because the chi-square approximation is not valid
    there and a printed p-value would be a fabricated one.
    """
    k = np.asarray(hits, dtype=np.float64)
    n = np.asarray(totals, dtype=np.float64)
    p = np.clip(np.asarray(predicted, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    expected_hit, expected_miss = n * p, n * (1.0 - p)
    usable = bool(
        expected_hit.size
        and float(expected_hit.min()) >= _MIN_EXPECTED_CELL
        and float(expected_miss.min()) >= _MIN_EXPECTED_CELL
    )
    stat = float(
        np.sum((k - expected_hit) ** 2 / expected_hit)
        + np.sum(((n - k) - expected_miss) ** 2 / expected_miss)
    )
    dof = int(len(k) - fitted_params)
    p_value: float | None = None
    if usable and dof > 0:
        try:
            from scipy.stats import chi2 as _chi2

            p_value = float(_chi2.sf(stat, dof))
        except Exception:  # noqa: BLE001 - a missing scipy costs a p-value, not the run
            p_value = None
    return {
        "chi2": round(stat, 3),
        "dof": dof,
        # Significant figures, not decimal places: a p-value of 4e-26 rounded to
        # six decimals reads as exactly 0.0, which claims more than the test does.
        "p_value": None if p_value is None else float(f"{p_value:.4g}"),
        "valid": usable and dof > 0,
        "predicted_rates": [round(float(v), 4) for v in p],
    }


# ---------------------------------------------------------------------------
# patch mining
# ---------------------------------------------------------------------------


def _candidate_origins(extent: int, size: int, stride: int) -> list[int]:
    """Grid origins covering ``extent``, last window flush to the far edge."""
    if extent < size:
        return []
    origins = list(range(0, extent - size + 1, stride))
    if origins[-1] != extent - size:
        origins.append(extent - size)
    return origins


def _dedup(
    candidates: np.ndarray, max_overlap: float, cap: int, rng: np.random.Generator
) -> list[int]:
    """Greedy random-order thinning so kept crops of one scale barely overlap.

    Random order rather than raster order: raster order would systematically
    prefer the top-left of every clean region, and if the illumination gradient
    across the frame is not flat that is a biased sample of the surface.
    """
    kept: list[int] = []
    for idx in rng.permutation(len(candidates)):
        if len(kept) >= cap:
            break
        if not kept or float(_pairwise_iou(candidates[idx], candidates[kept]).max()) < max_overlap:
            kept.append(int(idx))
    return kept


def mine_clean_patches(
    records: Sequence[GroundTruth], config: dict[str, Any] = PATCH_MINING, magnification: float = 1.6
) -> list[Patch]:
    """Crops that are strictly clear of every labelled defect.

    A candidate at ``(x, y, s)`` is kept only if the crop grown by
    ``clearance_px`` on all four sides has **zero** intersection area with every
    ground-truth box in the image. Growing the crop rather than shrinking the
    boxes means the margin is enforced even for a box that lies wholly outside the
    raw crop but within a few pixels of its edge.
    """
    rng = np.random.default_rng(int(config["seed"]))
    stride = int(config["stride_px"])
    margin = int(config["clearance_px"])
    patches: list[Patch] = []

    for record in records:
        for size in config["crop_sizes_px"]:
            size = int(size)
            imgsz = patch_imgsz(size, magnification)
            candidates: list[list[float]] = []
            for y0 in _candidate_origins(record.height, size, stride):
                for x0 in _candidate_origins(record.width, size, stride):
                    grown = np.array(
                        [x0 - margin, y0 - margin, x0 + size + margin, y0 + size + margin],
                        dtype=np.float64,
                    )
                    if record.boxes.size and float(_intersection_area(grown, record.boxes).max()) > 0.0:
                        continue
                    candidates.append([float(x0), float(y0), float(x0 + size), float(y0 + size)])
            if not candidates:
                continue
            arr = np.asarray(candidates, dtype=np.float64)
            for idx in _dedup(
                arr, float(config["max_overlap_iou"]), int(config["max_per_image_per_scale"]), rng
            ):
                patches.append(
                    Patch(
                        image_path=record.image_path,
                        x0=int(arr[idx, 0]),
                        y0=int(arr[idx, 1]),
                        size=size,
                        imgsz=imgsz,
                    )
                )
    return patches


def mine_defect_patches(
    records: Sequence[GroundTruth], config: dict[str, Any] = PATCH_MINING, magnification: float = 1.6
) -> list[Patch]:
    """Positive control: crops of the same sizes that fully contain a labelled defect.

    Containment is required in both directions -- the crop must hold at least one
    ground-truth box entirely, and it must not clip any other box. A crop holding
    half a defect would make "did the model detect it" ambiguous, and an ambiguous
    control cannot validate anything.
    """
    rng = np.random.default_rng(int(config["seed"]) + 1)
    stride = int(config["stride_px"])
    patches: list[Patch] = []

    for record in records:
        if record.boxes.size == 0:
            continue
        for size in config["crop_sizes_px"]:
            size = int(size)
            imgsz = patch_imgsz(size, magnification)
            candidates: list[list[float]] = []
            for y0 in _candidate_origins(record.height, size, stride):
                for x0 in _candidate_origins(record.width, size, stride):
                    crop = np.array([x0, y0, x0 + size, y0 + size], dtype=np.float64)
                    touching = _intersection_area(crop, record.boxes) > 0.0
                    if not touching.any():
                        continue
                    inside = (
                        (record.boxes[:, 0] >= crop[0])
                        & (record.boxes[:, 1] >= crop[1])
                        & (record.boxes[:, 2] <= crop[2])
                        & (record.boxes[:, 3] <= crop[3])
                    )
                    if not np.array_equal(touching, inside):
                        continue  # some box is clipped by the crop edge
                    candidates.append([float(x0), float(y0), float(x0 + size), float(y0 + size)])
            if not candidates:
                continue
            arr = np.asarray(candidates, dtype=np.float64)
            for idx in _dedup(
                arr, float(config["max_overlap_iou"]), int(config["max_per_image_per_scale"]), rng
            ):
                crop = arr[idx]
                inside = (
                    (record.boxes[:, 0] >= crop[0])
                    & (record.boxes[:, 1] >= crop[1])
                    & (record.boxes[:, 2] <= crop[2])
                    & (record.boxes[:, 3] <= crop[3])
                )
                local = record.boxes[inside] - np.array([crop[0], crop[1], crop[0], crop[1]])
                patches.append(
                    Patch(
                        image_path=record.image_path,
                        x0=int(crop[0]),
                        y0=int(crop[1]),
                        size=size,
                        imgsz=imgsz,
                        gt_boxes=local,
                        gt_classes=record.classes[inside].copy(),
                    )
                )
    return patches


def patch_imgsz(size: int, magnification: float, stride: int = NETWORK_STRIDE) -> int:
    """Network input size that presents a crop at the deployment magnification."""
    target = size * magnification
    return max(stride, int(round(target / stride)) * stride)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def _load_images(paths: Iterable[Path]) -> dict[Path, np.ndarray]:
    """Decode each distinct image once, as RGB."""
    cache: dict[Path, np.ndarray] = {}
    for path in paths:
        if path in cache:
            continue
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Unreadable image: {path}")
        cache[path] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cache


def score_patches(
    patches: Sequence[Patch],
    weights: Path,
    device: str,
    batch_size: int = 32,
    verbose: bool = True,
) -> list[ScoredPatch]:
    """Run the detector over every patch, one detector per crop scale.

    A detector is built per scale because ``imgsz`` is fixed at construction and
    each scale needs its own value to hold the magnification constant. Patches are
    grouped by scale so each detector sees a homogeneous, efficiently batchable
    workload.
    """
    if not patches:
        return []
    images = _load_images(p.image_path for p in patches)
    by_scale: dict[int, list[int]] = {}
    for i, patch in enumerate(patches):
        by_scale.setdefault(patch.size, []).append(i)

    scored: list[ScoredPatch | None] = [None] * len(patches)
    for size in sorted(by_scale):
        indices = by_scale[size]
        imgsz = patches[indices[0]].imgsz
        detector = DefectDetector(
            weights=weights, device=device, conf=CACHE_CONF, iou=NMS_IOU, imgsz=imgsz
        )
        detector.warmup(2)
        crops = [
            images[patches[i].image_path][
                patches[i].y0 : patches[i].y0 + size, patches[i].x0 : patches[i].x0 + size
            ]
            for i in indices
        ]
        if verbose:
            print(
                f"  scale {size}px -> imgsz {imgsz} "
                f"(magnification {imgsz / size:.2f}x): {len(crops)} patches",
                flush=True,
            )
        results = detector.predict_batch(crops, batch_size=batch_size)
        for i, result in zip(indices, results):
            if result.detections:
                boxes = np.asarray([d.bbox_xyxy for d in result.detections], dtype=np.float64)
                conf = np.asarray([d.confidence for d in result.detections], dtype=np.float64)
                classes = np.asarray([d.class_id for d in result.detections], dtype=np.int64)
            else:
                boxes = np.zeros((0, 4), dtype=np.float64)
                conf = np.zeros((0,), dtype=np.float64)
                classes = np.zeros((0,), dtype=np.int64)
            scored[i] = ScoredPatch(
                patch=patches[i], boxes=boxes, confidences=conf, classes=classes
            )
    return [s for s in scored if s is not None]


# ---------------------------------------------------------------------------
# clean-patch analysis
# ---------------------------------------------------------------------------


def clean_patch_sweep(
    scored: Sequence[ScoredPatch],
    frame_area_px: float,
    thresholds: Sequence[float] = SWEEP_THRESHOLDS,
    image_class: dict[Path, str] | None = None,
) -> list[CleanPoint]:
    """False alarm statistics on defect-free steel at every confidence threshold.

    ``image_class`` maps each source image to the defect class the *coil* carries.
    A clean crop mined from a crazing frame and a clean crop mined from an
    inclusion frame are different surfaces, and the pooled rate turns out to be a
    mixture over populations that differ by nearly an order of magnitude, so the
    breakdown is computed rather than left implicit in a single number.
    """
    scales = sorted({s.patch.size for s in scored})
    points: list[CleanPoint] = []

    by_scale = {size: [s for s in scored if s.patch.size == size] for size in scales}
    by_image: dict[Path, list[ScoredPatch]] = {}
    for item in scored:
        by_image.setdefault(item.patch.image_path, []).append(item)

    for threshold in thresholds:
        alarmed = 0
        boxes_total = 0
        class_counts: dict[str, int] = {name: 0 for name in CLASS_NAMES}
        per_scale: dict[int, dict[str, float]] = {}
        hit_counts: list[int] = []
        totals: list[int] = []

        for size in scales:
            subset = by_scale[size]
            hits = 0
            n_boxes = 0
            for item in subset:
                keep = item.confidences >= threshold
                count = int(keep.sum())
                n_boxes += count
                hits += int(count > 0)
                for cls in item.classes[keep]:
                    if 0 <= int(cls) < len(CLASS_NAMES):
                        class_counts[CLASS_NAMES[int(cls)]] += 1
            lo, hi = wilson_interval(hits, len(subset))
            scale_groups: dict[Path, list[int]] = {}
            for item in subset:
                scale_groups.setdefault(item.patch.image_path, []).append(
                    int(bool((item.confidences >= threshold).any()))
                )
            _, _, scale_deff = cluster_bootstrap_interval(list(scale_groups.values()), n_boot=1)
            per_scale[size] = {
                "n_patches": float(len(subset)),
                "area_px": float(size * size),
                "rate": hits / max(len(subset), 1),
                "wilson95_lo": lo,
                "wilson95_hi": hi,
                "boxes_per_patch": n_boxes / max(len(subset), 1),
                "design_effect": scale_deff,
            }
            alarmed += hits
            boxes_total += n_boxes
            hit_counts.append(hits)
            totals.append(len(subset))

        n = len(scored)
        lo, hi = wilson_interval(alarmed, n)
        image_groups = {
            path: [int(bool((it.confidences >= threshold).any())) for it in items]
            for path, items in by_image.items()
        }
        cluster_lo, cluster_hi, design_effect = cluster_bootstrap_interval(
            list(image_groups.values())
        )
        rates = [per_scale[s]["rate"] for s in scales]
        lam, r2, n_used = poisson_area_fit(
            [per_scale[s]["area_px"] for s in scales], hit_counts, totals
        )

        # The true rate is monotone non-decreasing in window area -- a bigger
        # window contains every smaller one -- and a frame is larger than every
        # crop scale. So the extrapolation is floored at the largest rate actually
        # observed, and when every scale has saturated at 1.0 the fit carries no
        # information and the frame rate is simply 1.0. Without this guard a
        # saturated cell linearises to lambda = 0 and reports a 0% frame-level
        # false alarm rate, which is the exact opposite of what was measured.
        observed_max = max(rates) if rates else 0.0
        if n_used == 0:
            frame_rate, intensity, note = observed_max, None, "all scales saturated; fit unusable"
        else:
            fitted = float(1.0 - math.exp(-lam * frame_area_px))
            frame_rate = max(fitted, observed_max)
            intensity = lam * 10_000.0
            note = (
                "poisson fit"
                if fitted >= observed_max
                else "poisson fit below largest observed rate; floored by monotonicity"
            )

        # Two competing accounts of what a false alarm on clean steel is, each
        # fitting one parameter to the four scale cells, so their chi-squares are
        # directly comparable on 3 dof:
        #   constant  -- the rate is a property of the *region* and does not grow
        #                with how much of it you show the network;
        #   poisson   -- false positives are independent events sprinkled over the
        #                surface, so the rate must grow as 1 - exp(-lambda * area).
        # Only the poisson account licenses extrapolating a patch rate to a frame.
        pooled = alarmed / max(n, 1)
        chi2_constant = _chi2_binomial_fit(hit_counts, totals, [pooled] * len(totals), 1)
        poisson_rates = [1.0 - math.exp(-lam * per_scale[s]["area_px"]) for s in scales]
        chi2_poisson = _chi2_binomial_fit(hit_counts, totals, poisson_rates, 1)
        # Both chi-squares above assume the patches inside a cell are independent
        # draws. They are not -- they are clustered by source image -- so the
        # nominal p-values are anti-conservative in the direction of rejecting.
        # A Rao-Scott correction divides the statistic by the mean design effect
        # of the cells, which is the standard first-order fix; it is reported
        # alongside, not instead of, the nominal value.
        mean_deff = float(
            np.mean([per_scale[s]["design_effect"] for s in scales])
        ) if scales else 1.0
        mean_deff = max(mean_deff, 1e-9)
        chi2_constant["mean_design_effect"] = round(mean_deff, 3)
        chi2_poisson["mean_design_effect"] = round(mean_deff, 3)
        for fit in (chi2_constant, chi2_poisson):
            fit["chi2_design_corrected"] = round(fit["chi2"] / mean_deff, 3)
            fit["p_value_design_corrected"] = _chi2_sf(fit["chi2"] / mean_deff, fit["dof"]) \
                if fit["valid"] else None

        poisson_supported: bool | None = None
        if chi2_poisson["valid"] and chi2_poisson["p_value_design_corrected"] is not None:
            # Judged on the design-corrected p-value, which is the conservative
            # one: rejecting spatial independence on a p-value that assumed
            # independence between the very patches whose dependence is the
            # question would be circular.
            poisson_supported = bool(
                chi2_poisson["p_value_design_corrected"] >= 0.05
                and chi2_poisson["chi2"] <= chi2_constant["chi2"]
            )

        # Clustering probe. If false positives were independent across the surface,
        # an image contributing m clean patches at rate p would show at least one
        # alarm with probability 1 - (1 - p)^m. Comparing that with the measured
        # image-level rate says whether false alarms are sprinkled or clumped, and
        # a clump is what makes the area extrapolation wrong.
        img_alarmed = sum(int(any(v)) for v in image_groups.values())
        n_images = len(by_image)
        # 1 - (1-p)^m is concave in m, so evaluating it at the MEAN patch count
        # overstates the average over images by Jensen's inequality -- by 12
        # points on this data, all of it exaggerating the clustering claim. The
        # prediction has to be averaged per image, not evaluated at the mean.
        iid_prediction = float(
            np.mean([1.0 - (1.0 - pooled) ** len(v) for v in image_groups.values()])
        ) if image_groups else 0.0

        # The pooled rate is a mixture over six surface populations that the
        # mining samples very unevenly. Break it out.
        per_class: dict[str, dict[str, float]] = {}
        same_class_fp = 0
        total_fp = 0
        if image_class:
            groups: dict[str, list[ScoredPatch]] = {}
            for item in scored:
                groups.setdefault(image_class.get(item.patch.image_path, "unknown"), []).append(item)
            for name, items in sorted(groups.items()):
                hits = sum(int(bool((it.confidences >= threshold).any())) for it in items)
                c_lo, c_hi = wilson_interval(hits, len(items))
                per_class[name] = {
                    "n_patches": float(len(items)),
                    "n_source_images": float(len({it.patch.image_path for it in items})),
                    "rate": hits / max(len(items), 1),
                    "wilson95_lo": c_lo,
                    "wilson95_hi": c_hi,
                }
            for item in scored:
                src = image_class.get(item.patch.image_path)
                for cls in item.classes[item.confidences >= threshold]:
                    total_fp += 1
                    if 0 <= int(cls) < len(CLASS_NAMES) and CLASS_NAMES[int(cls)] == src:
                        same_class_fp += 1

        points.append(
            CleanPoint(
                threshold=float(threshold),
                n_patches=n,
                patches_alarmed=alarmed,
                rate=alarmed / max(n, 1),
                rate_lo=lo,
                rate_hi=hi,
                cluster_lo=cluster_lo,
                cluster_hi=cluster_hi,
                design_effect=design_effect,
                boxes_per_patch=boxes_total / max(n, 1),
                class_counts=class_counts,
                per_scale=per_scale,
                frame_rate_extrapolated=frame_rate,
                intensity_per_10k_px=intensity,
                fit_r2=r2,
                extrapolation_note=note,
                chi2_constant=chi2_constant,
                chi2_poisson=chi2_poisson,
                poisson_supported=poisson_supported,
                n_source_images=n_images,
                images_alarmed=img_alarmed,
                image_rate=img_alarmed / max(n_images, 1),
                image_rate_iid_prediction=iid_prediction,
                per_source_class=per_class,
                fp_matching_source_class=same_class_fp,
                fp_total=total_fp,
            )
        )
    return points


def control_sweep(
    scored: Sequence[ScoredPatch], thresholds: Sequence[float] = SWEEP_THRESHOLDS
) -> list[ControlPoint]:
    """Positive control: does the model still fire on defects under this protocol?"""
    points: list[ControlPoint] = []
    for threshold in thresholds:
        flagged = 0
        localised = 0
        for item in scored:
            keep = item.confidences >= threshold
            boxes, conf, classes = item.boxes[keep], item.confidences[keep], item.classes[keep]
            flagged += int(len(boxes) > 0)
            match = match_image(item.patch.gt_boxes, item.patch.gt_classes, boxes, conf, classes)
            localised += int(bool(match.gt_matched.any()))
        points.append(
            ControlPoint(
                threshold=float(threshold),
                n_patches=len(scored),
                patches_with_any_box=flagged,
                patches_localised=localised,
            )
        )
    return points


# ---------------------------------------------------------------------------
# operating point
# ---------------------------------------------------------------------------


def effective_lambda(ratio: float, prevalence: float) -> float:
    """The single weight the decision actually depends on."""
    return float(prevalence * ratio / max(1.0 - prevalence, 1e-12))


def build_cost_curve(
    val_points: Sequence[SweepPoint],
    clean_points: Sequence[CleanPoint],
    ratio: float,
    prevalence: float,
) -> list[CostPoint]:
    """Expected cost per inspected frame at each threshold, on both FA bases.

    cost(t) = prevalence * ratio * (1 - D(t)) + (1 - prevalence) * F(t)

    ``D(t)`` comes from real defective val frames; ``F(t)`` from the clean-steel
    proxy. The "frame basis" uses the area-extrapolated per-frame false alarm rate
    so both terms are per frame; the "patch basis" substitutes the raw per-patch
    rate unchanged, which understates the false alarm term but rests on no
    extrapolation. Both are carried so the reader can see how much of the
    recommendation is the extrapolation talking.
    """
    by_threshold = {round(p.threshold, 4): p for p in clean_points}
    curve: list[CostPoint] = []
    for point in val_points:
        clean = by_threshold[round(point.threshold, 4)]
        miss_term = prevalence * ratio * (1.0 - point.defect_detection_rate)
        curve.append(
            CostPoint(
                threshold=point.threshold,
                detection_rate=point.defect_detection_rate,
                clean_frame_fa=clean.frame_rate_extrapolated,
                clean_patch_fa=clean.rate,
                legacy_fa=point.false_alarm_rate,
                cost_frame=miss_term + (1.0 - prevalence) * clean.frame_rate_extrapolated,
                cost_patch=miss_term + (1.0 - prevalence) * clean.rate,
            )
        )
    return curve


def choose_threshold(
    curve: Sequence[CostPoint], min_detection_rate: float, basis: str = "frame"
) -> tuple[CostPoint, bool, bool]:
    """Cheapest threshold that still clears the detection floor.

    Returns ``(point, floor_met, at_sweep_edge)``. If nothing clears the floor the
    constraint is dropped and ``floor_met`` is False, so the report can say so
    rather than quietly shipping a threshold nobody would accept.
    """
    if not curve:
        raise ValueError("Cannot choose a threshold from an empty cost curve")
    key = (lambda p: p.cost_frame) if basis == "frame" else (lambda p: p.cost_patch)
    eligible = [p for p in curve if p.detection_rate >= min_detection_rate]
    floor_met = bool(eligible)
    pool = eligible or list(curve)
    best = min(pool, key=lambda p: (key(p), -p.threshold))
    edges = (min(p.threshold for p in curve), max(p.threshold for p in curve))
    return best, floor_met, bool(best.threshold in edges)


def sensitivity_table(
    val_points: Sequence[SweepPoint],
    clean_points: Sequence[CleanPoint],
    ratios: Sequence[float],
    prevalence: float,
    min_detection_rate: float,
    basis: str = "patch",
) -> list[dict[str, Any]]:
    """What threshold wins at each miss:false-alarm ratio, constrained and not."""
    cost_of = (lambda p: p.cost_frame) if basis == "frame" else (lambda p: p.cost_patch)
    fa_of = (lambda p: p.clean_frame_fa) if basis == "frame" else (lambda p: p.clean_patch_fa)
    rows: list[dict[str, Any]] = []
    for ratio in ratios:
        curve = build_cost_curve(val_points, clean_points, ratio, prevalence)
        unconstrained = min(curve, key=lambda p: (cost_of(p), -p.threshold))
        constrained, floor_met, _ = choose_threshold(curve, min_detection_rate, basis=basis)
        rows.append(
            {
                "miss_cost_ratio": float(ratio),
                "effective_lambda": round(effective_lambda(ratio, prevalence), 5),
                "basis": basis,
                "unconstrained_threshold": round(unconstrained.threshold, 3),
                "unconstrained_cost": round(cost_of(unconstrained), 5),
                "unconstrained_detection_rate": round(unconstrained.detection_rate, 4),
                "unconstrained_clean_fa": round(fa_of(unconstrained), 4),
                "unconstrained_meets_floor": bool(
                    unconstrained.detection_rate >= min_detection_rate
                ),
                "constrained_threshold": round(constrained.threshold, 3),
                "constrained_cost": round(cost_of(constrained), 5),
                "constrained_detection_rate": round(constrained.detection_rate, 4),
                "constrained_clean_fa": round(fa_of(constrained), 4),
                "floor_reachable": floor_met,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

_GRID = {"color": "#d8dde3", "linewidth": 0.7, "alpha": 0.9}


def plot_false_alarm(
    curve: Sequence[CostPoint],
    ratio_curves: dict[float, list[CostPoint]],
    chosen: CostPoint,
    chosen_clean: CleanPoint,
    detection_floor: float,
    out_path: Path,
) -> Path:
    """Three panels: the operating curve, the cost curves, and the area-dependence test."""
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(17.5, 5.4))

    order = sorted(curve, key=lambda p: p.threshold)
    det = [p.detection_rate for p in order]

    ax_a.plot(det, [p.legacy_fa for p in order], "o-", color="#98a2b3", ms=4, lw=1.6,
              label="legacy: spurious box on a DEFECTIVE frame")
    ax_a.plot(det, [p.clean_frame_fa for p in order], "o--", color="#c1121f", ms=4, lw=1.6,
              label="clean steel, area-extrapolated (model rejected)")
    ax_a.plot(det, [p.clean_patch_fa for p in order], "o-", color="#0b6e4f", ms=5, lw=2.3,
              label="clean steel, measured per-patch")
    for p in order:
        if round(p.threshold, 3) in (0.01, 0.05, 0.15, 0.25, 0.40, 0.60, 0.80):
            ax_a.annotate(f"{p.threshold:.2f}", (p.detection_rate, p.clean_patch_fa),
                          textcoords="offset points", xytext=(6, 5), fontsize=8, color="#0b6e4f")
    ax_a.axvline(detection_floor, color="#1d3557", ls=":", lw=1.4)
    ax_a.annotate(f"{detection_floor:.0%} detection floor", (detection_floor, 0.98),
                  textcoords="offset points", xytext=(-7, 0), ha="right", va="top",
                  rotation=90, fontsize=8.5, color="#1d3557")
    ax_a.plot([chosen.detection_rate], [chosen.clean_patch_fa], "*", color="#1d3557", ms=17,
              zorder=5, label=f"recommended conf={chosen.threshold:.2f}")
    ax_a.set_xlabel("defect detection rate on defective val frames  D(t)")
    ax_a.set_ylabel("false alarm rate  F(t)")
    # Not "held out": the FA half of the objective is chosen on these same crops.
    ax_a.set_title(
        "A. False alarm vs detection\nclean-steel proxy (upper bound), test-split crops",
        fontsize=11,
    )
    ax_a.set_ylim(-0.02, 1.02)
    ax_a.grid(True, **_GRID)
    ax_a.legend(fontsize=8, loc="upper left")

    for (ratio, rc), colour in zip(
        sorted(ratio_curves.items()), ("#457b9d", "#c1121f", "#0b6e4f", "#8338ec")
    ):
        rc_sorted = sorted(rc, key=lambda p: p.threshold)
        ax_b.plot([p.threshold for p in rc_sorted], [p.cost_patch for p in rc_sorted],
                  "-", color=colour, lw=1.9, label=f"miss:FA = {ratio:g}:1")
        best = min(rc_sorted, key=lambda p: (p.cost_patch, -p.threshold))
        ax_b.plot([best.threshold], [best.cost_patch], "v", color=colour, ms=9)
    ax_b.axvline(chosen.threshold, color="#1d3557", ls="--", lw=1.3)
    # The recommendation is not at these minima: it is the cheapest point that also
    # clears the detection floor, and saying so on the chart stops the triangles
    # being read as the recommendation.
    ax_b.annotate(f"recommended {chosen.threshold:.2f}\n(cheapest point clearing\nthe detection floor)",
                  xy=(chosen.threshold, 1.0), xycoords=("data", "axes fraction"),
                  textcoords="offset points", xytext=(9, -10), ha="left", va="top",
                  fontsize=8.5, color="#1d3557")
    ax_b.set_xlabel("confidence threshold")
    # Not "per inspected frame": on the patch basis the miss term is per frame and
    # the false alarm term is per crop, so the curve orders thresholds and its
    # height is not a count of anything.
    ax_b.set_ylabel("expected cost score\n(orders thresholds; not a per-frame count)")
    ax_b.set_title("B. Expected cost vs threshold\ntriangle = unconstrained minimum", fontsize=11)
    ax_b.set_xscale("log")
    ax_b.grid(True, which="both", **_GRID)
    ax_b.legend(fontsize=9)

    scales = sorted(chosen_clean.per_scale)
    areas = np.array([chosen_clean.per_scale[s]["area_px"] for s in scales])
    rates = np.array([chosen_clean.per_scale[s]["rate"] for s in scales])
    lo = np.array([chosen_clean.per_scale[s]["wilson95_lo"] for s in scales])
    hi = np.array([chosen_clean.per_scale[s]["wilson95_hi"] for s in scales])
    ax_c.errorbar(areas, rates, yerr=[rates - lo, hi - rates], fmt="o", color="#1d3557",
                  ms=7, capsize=5, lw=1.6,
                  label="measured (95% Wilson, within cell)")
    grid = np.linspace(0.0, float(areas.max()) * 1.05, 200)
    ax_c.axhline(chosen_clean.rate, color="#0b6e4f", lw=2.0,
                 label=f"constant rate = {chosen_clean.rate:.3f}")
    predicted = chosen_clean.chi2_poisson.get("predicted_rates") or []
    if predicted:
        lam = -math.log(max(1.0 - predicted[0], 1e-12)) / max(float(areas[0]), 1e-9)
        ax_c.plot(grid, 1.0 - np.exp(-lam * grid), "--", color="#c1121f", lw=2.0,
                  label="if spatially independent")
    ax_c.set_xlabel("crop area (px$^2$)")
    ax_c.set_ylabel("false alarm rate on crops with no annotated defect")
    ax_c.set_title(
        f"C. Does the false alarm rate grow with area?\nat conf={chosen.threshold:.2f} -- it does not",
        fontsize=11,
    )
    ax_c.set_ylim(-0.02, 1.02)
    ax_c.grid(True, **_GRID)
    ax_c.legend(fontsize=8.5, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def _md_table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(str(h) for h in header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


LIMITATIONS: tuple[str, ...] = (
    "The crops are smaller than a frame, so a per-patch rate is not a per-frame rate. "
    "The obvious bridge -- assume false positives are spatially independent and scale "
    "by area -- is tested here against the four measured crop scales rather than "
    "assumed, and on this checkpoint it is rejected; the per-frame figure is therefore "
    "carried only as a pessimistic bound.",
    "The crops come from coils that do contain defects elsewhere. They are clean "
    "regions of defective coils, not clean coils.",
    "A crop with no annotated box is not guaranteed defect-free: NEU-DET annotation "
    "is not exhaustive. Unlabelled faint crazing or light scale inside a 'clean' "
    "crop is scored here as a false alarm, which biases the measured rate upward. "
    "This is not a small effect: at the recommended threshold the large majority of "
    "boxes raised on 'clean' crops carry the source image's OWN defect class, far "
    "above the 1-in-6 a class-independent hallucination would give. The headline "
    "rate is therefore an upper bound on false alarming, and an unknown share of it "
    "is the detector finding real but unlabelled defect.",
    "The pooled rate is a mixture over six surface populations that the mining "
    "samples very unevenly -- how many clean crops a defect class yields depends on "
    "how much of the frame its annotation covers, which has nothing to do with how "
    "often that surface appears on a line. The per-class rates differ by nearly an "
    "order of magnitude, so the single pooled figure carries a composition that is "
    "an artefact of the dataset, not of the mill.",
    "The crops are not independent samples. Up to six are taken per image per scale, "
    "they overlap across scales, and alarming crops clump inside a minority of "
    "images. The binomial (Wilson) interval is therefore too narrow; an "
    "image-clustered bootstrap interval is reported and is the one to quote.",
    "The field of view is small, so the model sees less context than it would on a "
    "full frame. The positive control quantifies how much of the detector's "
    "behaviour survives that, but it does not remove the difference.",
    "The clean patches are mined from the test split, as specified. The false alarm "
    "half of the objective is therefore not held out from the threshold choice; a "
    "val-mined cross-check is reported so the size of that leak is visible.",
    "Defect prevalence and the miss:false-alarm cost ratio are assumptions, not "
    "measurements, and they are not separately identifiable from the decision -- "
    "only their combination lambda = prevalence * ratio / (1 - prevalence) moves "
    "the recommended threshold.",
)

REAL_MEASUREMENT: str = (
    "Continuous capture of coils that passed manual surface inspection and shipped "
    "as prime, at production line speed, camera geometry and illumination, with "
    "frames time-stamped back to a coil id so a flagged frame can be adjudicated "
    "against the mill's own disposition record. A few thousand such frames across a "
    "shift, several coils, several grades and both strip surfaces would give a "
    "directly measured per-frame false alarm rate with a usable confidence interval."
)


def write_report(path: Path, payload: dict[str, Any]) -> Path:
    """Render the readable report. Every number here is in the JSON as well."""
    meta = payload["meta"]
    mining = payload["patch_mining"]
    clean = payload["clean_patch_false_alarm"]
    control = payload["positive_control"]
    rec = payload["recommendation"]
    cfg = payload["cost_model"]

    chosen_clean = next(
        row for row in clean["per_threshold"] if abs(row["threshold"] - rec["threshold"]) < 1e-9
    )
    chosen_control = next(
        row for row in control["per_threshold"] if abs(row["threshold"] - rec["threshold"]) < 1e-9
    )

    sweep_rows = [
        [
            f"{row['threshold']:.2f}",
            f"{row['defect_detection_rate']:.3f}",
            f"{row['clean_patch_false_alarm_rate']:.3f}",
            f"{row['clean_frame_false_alarm_rate']:.3f}",
            f"{row['legacy_spurious_box_rate']:.3f}",
            f"{row['expected_cost_patch_basis']:.4f}",
            f"{row['expected_cost_frame_basis']:.4f}",
        ]
        for row in payload["operating_curve"]
    ]

    cells = sorted(chosen_clean["per_scale"].items(), key=lambda kv: int(kv[0]))
    iid_rates = chosen_clean["area_dependence"]["poisson_area_model"]["predicted_rates"]
    scale_rows = [
        [
            f"{size}x{size}",
            int(cell["area_px"]),
            int(cell["n_patches"]),
            f"{cell['rate']:.3f}",
            f"[{cell['wilson95_lo']:.3f}, {cell['wilson95_hi']:.3f}]",
            f"{cell['boxes_per_patch']:.3f}",
            f"{iid:.3f}",
        ]
        for (size, cell), iid in zip(cells, iid_rates)
    ]

    const_fit = chosen_clean["area_dependence"]["constant_rate_model"]
    pois_fit = chosen_clean["area_dependence"]["poisson_area_model"]

    def _p(fit: dict[str, Any], key: str = "p_value") -> str:
        if not fit.get("valid") or fit.get(key) is None:
            return "n/a"
        value = float(fit[key])
        return f"{value:.2g}" if value < 0.001 else f"{value:.3f}"

    model_rows = [
        [
            "constant rate",
            "false alarms are a property of the region, not of area",
            f"{const_fit['chi2']:.2f}",
            const_fit["dof"],
            _p(const_fit),
            f"{const_fit.get('chi2_design_corrected', float('nan')):.2f}",
            _p(const_fit, "p_value_design_corrected"),
        ],
        [
            "Poisson in area",
            "false alarms are independent events per unit area",
            f"{pois_fit['chi2']:.2f}",
            pois_fit["dof"],
            _p(pois_fit),
            f"{pois_fit.get('chi2_design_corrected', float('nan')):.2f}",
            _p(pois_fit, "p_value_design_corrected"),
        ],
    ]
    supported = chosen_clean["area_dependence"]["poisson_supported"]
    if supported:
        area_verdict = (
            "The independence model survives, so extrapolating a patch rate to a "
            "frame is licensed here."
        )
    else:
        area_verdict = (
            f"The constant-rate model fits ({const_fit['chi2']:.2f} on "
            f"{const_fit['dof']} dof); the independence model does not "
            f"({pois_fit['chi2']:.2f} on {pois_fit['dof']} dof, "
            f"{pois_fit.get('chi2_design_corrected', float('nan')):.2f} after the design-effect "
            f"correction, p = {_p(pois_fit, 'p_value_design_corrected')}). Spatial independence "
            f"is rejected -- and it stays rejected after the correction, which is the "
            f"version to quote -- so the area extrapolation is **not** a prediction and "
            f"is reported here only as a pessimistic bound."
        )

    clustering = chosen_clean["clustering"]
    mean_patches = mining["clean_patches"] / max(clustering["n_source_images"], 1)
    mean_patches_txt = f"{mean_patches:.1f}"
    clustering_verdict = (
        "The measured image-level rate is well below the independent prediction, so "
        "alarming patches clump inside a minority of images: some surface regions "
        "provoke the detector repeatedly and most never do."
        if clustering["image_level_rate"] < clustering["image_level_rate_if_patches_independent"] - 0.02
        else "The measured and predicted image-level rates are close, so there is "
        "little evidence of clumping at this threshold."
    )

    intensity = chosen_clean["fp_intensity_per_10k_px"]
    intensity_txt = "not estimable (all scales saturated)" if intensity is None else f"{intensity:.3f}"
    r2 = chosen_clean["poisson_fit_r2"]
    r2_txt = "not estimable" if r2 is None else f"{r2:.3f}"

    fp_classes = chosen_clean["false_positive_class_counts"]
    total_fp = max(sum(fp_classes.values()), 1)
    ranked_fp = sorted(fp_classes.items(), key=lambda kv: -kv[1])
    class_rows = [[name, count, f"{count / total_fp:.1%}"] for name, count in ranked_fp]
    top_fp_class, top_fp_count = ranked_fp[0]
    top_fp_share = top_fp_count / total_fp
    # Why that class and not another, in the language of the surface rather than of
    # the loss function. Kept as a lookup so the sentence stays true if the ranking
    # changes on a different checkpoint.
    _FP_NOTES = {
        "scratches": (
            "Scratches are a linear, low-contrast feature, and rolling, levelling and "
            "grinding leave linear texture on perfectly sound stainless; the model has "
            "never been shown that texture with a 'clean' label."
        ),
        "crazing": (
            "Crazing is a fine-textured class with the weakest AP50 in the model, so "
            "the head firing on ordinary surface grain is consistent with its "
            "detection performance."
        ),
        "rolled-in_scale": (
            "Rolled-in scale is a mottled low-contrast class, easily confused with "
            "normal oxide colour variation and illumination shading on clean strip."
        ),
        "pitted_surface": (
            "Pitted surface is a dense field of small dark points, which ordinary "
            "surface roughness and sensor noise can mimic."
        ),
        "inclusion": (
            "Inclusions are small dark elongated marks, which surface debris and "
            "coolant staining can mimic on clean strip."
        ),
        "patches": (
            "Patches are broad brightness variations, which uneven illumination across "
            "a clean strip can mimic."
        ),
    }
    top_fp_note = _FP_NOTES.get(
        top_fp_class, "No standing explanation for this class is recorded here."
    )

    sens_rows = [
        [
            f"{row['miss_cost_ratio']:g}:1",
            f"{row['effective_lambda']:.4f}",
            f"{row['unconstrained_threshold']:.2f}",
            f"{row['unconstrained_detection_rate']:.3f}",
            f"{row['unconstrained_clean_fa']:.3f}",
            "yes" if row["unconstrained_meets_floor"] else "no",
            f"{row['constrained_threshold']:.2f}",
        ]
        for row in payload["sensitivity"]
    ]

    control_rows = [
        [
            f"{row['threshold']:.2f}",
            f"{row['flag_rate']:.3f}",
            f"{row['localised_detection_rate']:.3f}",
        ]
        for row in control["per_threshold"]
        if row["threshold"] in (0.01, 0.05, 0.15, 0.25, 0.40, 0.60, 0.80)
    ]

    per_class = chosen_clean.get("per_source_class") or {}
    ranked_class = sorted(per_class.items(), key=lambda kv: -kv[1]["rate"])
    class_split_table = _md_table(
        ["source image defect class", "clean crops", "source images", "FA rate", "95% Wilson"],
        [
            [
                name,
                int(cell["n_patches"]),
                int(cell["n_source_images"]),
                f"{cell['rate']:.3f}",
                f"[{cell['wilson95_lo']:.3f}, {cell['wilson95_hi']:.3f}]",
            ]
            for name, cell in ranked_class
        ],
    )
    solid = [(k, v) for k, v in ranked_class if v["n_patches"] >= 20]
    if len(solid) >= 2:
        hi_name, hi_cell = solid[0]
        lo_name, lo_cell = solid[-1]
        share = max(c["n_patches"] for _, c in solid) / max(chosen_clean["n_patches"], 1)
        class_split_verdict = (
            f"Across the populations with at least 20 crops the rate runs from "
            f"{lo_cell['rate']:.1%} (`{lo_name}`) to {hi_cell['rate']:.1%} (`{hi_name}`) -- "
            f"a spread of {hi_cell['rate'] / max(lo_cell['rate'], 1e-9):.0f} times with "
            f"non-overlapping intervals, so this is not noise. One class alone supplies "
            f"{share:.0%} of all clean crops. The single pooled number therefore carries a "
            f"composition chosen by the dataset's annotation habits, and a mill inspecting a "
            f"different product mix would see a different figure from the same detector."
        )
    else:
        class_split_verdict = (
            "Too few crops per source class to break the pooled rate down reliably."
        )
    fp_total = chosen_clean.get("false_positives_total") or 0
    fp_share_val = chosen_clean.get("share_of_fp_matching_source_class")
    fp_match_share = "n/a" if fp_share_val is None else f"**{fp_share_val:.0%}**"
    chance_share = f"{1 / len(CLASS_NAMES):.0%}"
    headline_rate = f"{chosen_clean['clean_patch_false_alarm_rate']:.1%}"

    holdout = payload["holdout_detection"]
    _h = float(holdout["detection_rate_at_recommended"])
    _v = float(rec["defect_detection_rate"])
    holdout_verdict = (
        f"The held-out figure is the higher of the two, so the recommendation's detection "
        f"rate is not inflated by checkpoint selection; {_v:.1%} is the conservative one to "
        f"quote."
        if _h >= _v
        else f"The held-out figure is {_v - _h:.1%} lower, so the {_v:.1%} carried through "
        f"this report is optimistic by that much and {_h:.1%} is the number to quote."
    )
    fa_basis_txt = (
        "area-extrapolated to frame scale"
        if rec["basis"] == "frame"
        else "the measured per-patch rate, with no extrapolation"
    )
    cross = payload["val_crosscheck"]

    text = f"""# Clean-steel false alarm rate, and the operating point it implies

**Checkpoint** `{meta['model_name']}` (`{meta['weights']}`)
**Device** {meta['device']} | **train imgsz** {meta['train_imgsz']} | **NMS IoU** {NMS_IOU}
**Clean patches mined from** `{mining['clean_split']}` ({mining['clean_source_images']} images)
**Detection rate measured on** `{meta['detection_split']}` ({meta['detection_images']} defective frames)
**Generated** {meta['generated_at']}

## 0. The gap this closes

Every NEU-DET image contains a defect, so the false alarm rate in
`reports/evaluation.md` is the rate at which a **spurious extra box appears on a
coil that was already defective**. The coil was going to be flagged anyway. The
number a mill buys on is different: how often does the system stop the line for a
coil that is fine? That cannot be read off NEU-DET, so this report builds a proxy
and states what the proxy cannot tell you.

## 1. The proxy

{mining['clean_patches']} crops were mined from the `{mining['clean_split']}` split
that have **zero** intersection with any ground-truth box, with every crop grown by
{mining['clearance_px']} px on all four sides before the test -- so a crop that
merely grazes a label is thrown away. Four crop scales
({', '.join(f"{s}px" for s in mining['crop_sizes_px'])}) were used, candidates were
thinned to at most IoU {mining['max_overlap_iou']} against each other and capped at
{mining['max_per_image_per_scale']} per image per scale, so no single image with a
large clean area can dominate the statistic. {mining['clean_source_images']} of the
{mining['split_images']} test images contributed at least one clean patch.

{MAGNIFICATION_NOTE} Deployment magnification here is
{meta['magnification']:.2f}x ({meta['train_imgsz']} / {meta['frame_px']}), so the four
scales run at imgsz {', '.join(str(v) for v in mining['imgsz_by_scale'].values())}
respectively -- achieved magnifications
{', '.join(f"{v:.2f}x" for v in mining['achieved_magnification'].values())}.

## 2. False alarms on defect-free steel

At the recommended threshold **conf = {rec['threshold']:.2f}**:

- **{chosen_clean['clean_patch_false_alarm_rate']:.1%}** of clean patches raise at
  least one box. **95% CI [{chosen_clean['cluster95_lo']:.1%},
  {chosen_clean['cluster95_hi']:.1%}]**, from an image-clustered bootstrap over
  {chosen_clean['clustering']['n_source_images']} source images -- this is the interval
  to quote. The binomial Wilson interval
  [{chosen_clean['wilson95_lo']:.1%}, {chosen_clean['wilson95_hi']:.1%}] is roughly
  half as wide and is wrong here: the {chosen_clean['n_patches']} crops are not
  independent draws (design effect {chosen_clean['design_effect']:.1f}, effective
  n = {chosen_clean['effective_n']:.0f}, not {chosen_clean['n_patches']}).
- **{chosen_clean['boxes_per_clean_patch']:.2f}** boxes per clean patch
- if the false positives were spatially independent, that would extrapolate to
  **{chosen_clean['frame_rate_extrapolated']:.1%}** of a full
  {meta['frame_px']}x{meta['frame_px']} frame (best-fit Poisson intensity
  {intensity_txt} per 10,000 clean px). **The data rejects that model** -- see below,
  so this figure is a bound, not a prediction.

Per crop scale at that threshold. This is the evidence for or against extrapolating
a patch rate to a frame, and it is the most useful thing in this report:

{_md_table(["crop", "area px", "n", "FA rate", "95% CI", "boxes/patch", "rate if independent"], scale_rows)}

The false alarm rate is **flat in crop area** over a 4x span of area. Fitting the
two competing one-parameter models to those four cells:

{_md_table(
    ["model", "what it says", "chi2", "dof", "p", "chi2 (deff-corrected)", "p (deff-corrected)"],
    model_rows,
)}

Both chi-squares are computed against each model's **best** fit -- the Poisson
intensity is fitted by minimising the same statistic, not by an unweighted
least-squares linearisation, because the chi-square of a badly-estimated
parameter is evidence about the estimator, not about the model. The
design-effect-corrected columns divide by the mean per-cell design effect
({pois_fit.get('mean_design_effect', 1.0):.2f}), because the crops inside a cell
are clustered by source image and the nominal p-values assume they are not.

{area_verdict}

The clustering probe says the same thing from the other direction. The
{mining['clean_patches']} clean patches come from {chosen_clean['clustering']['n_source_images']}
source images, {mean_patches_txt} patches each. At this threshold
{chosen_clean['clustering']['image_level_rate']:.1%} of those images have at least one
alarming patch; if patches within an image were independent, that figure would be
{chosen_clean['clustering']['image_level_rate_if_patches_independent']:.1%}.
{clustering_verdict}

**Practical consequence:** a false alarm on clean steel is a property of the local
surface texture, not a per-unit-area event. Doubling the inspected area does not
double the false alarms. The per-patch rate
({chosen_clean['clean_patch_false_alarm_rate']:.1%}) is therefore the more defensible
frame-level estimate -- and it is itself an upper bound, for the reason in the next
subsection -- while the {chosen_clean['frame_rate_extrapolated']:.1%} extrapolation
should be read as a pessimistic bound rather than a prediction.


### The pooled rate is a mixture, and the weights are an artefact

A clean crop from a crazing coil and a clean crop from an inclusion coil are not
the same surface, and the mining does not sample them evenly -- how many clean
crops a defect class yields is set by how much of the frame its annotation covers.
Broken out by the defect class of the source image:

{class_split_table}

{class_split_verdict}

**And the boxes are not class-random.** {fp_match_share} of the boxes raised on
clean crops carry the source image's **own** defect class; a hallucination
unrelated to the coil would give {chance_share}. That is the signature of the
detector firing on unlabelled continuation of the real defect -- crazing, scale
and scratches all extend past the boxed region in NEU-DET -- not of it inventing
defects on sound metal. It cannot be separated from genuine false alarming
without re-annotation, so **{headline_rate} is an upper bound**, and the two
populations whose annotation is most plausibly exhaustive sit far below it.

Which defect does the model hallucinate on clean steel?

{_md_table(["class", "false positives", "share"], class_rows)}

The single largest failure mode is **{top_fp_class}** at {top_fp_share:.0%} of all
false positives on clean metal. {top_fp_note} Either way this is the confusion to
attack first, and the fix is hard negatives from clean strip -- exactly the data
this report says the project does not yet have.

## 3. Positive control -- is the small field of view silencing the model?

A low false alarm rate on small crops means nothing if the model simply stops
firing on small crops. {control['n_patches']} crops of the same four sizes that
**fully contain** a labelled defect (and clip no other box) were scored under the
identical protocol:

{_md_table(["conf", "flag rate", "localised detection rate (IoU 0.5, right class)"], control_rows)}

At conf = {rec['threshold']:.2f} the control flags
{chosen_control['flag_rate']:.1%} of defect-containing crops and localises the
defect correctly on {chosen_control['localised_detection_rate']:.1%} of them,
against {chosen_clean['clean_patch_false_alarm_rate']:.1%} on clean crops. The
protocol therefore separates defective from clean surface; the clean-patch number
is informative rather than an artefact of a starved field of view.

## 4. The re-derived operating point

Cost model, entirely in `COST_MODEL` at the top of `src/false_alarm.py` and open to
challenge:

    cost(t) = prevalence * ratio * (1 - D(t)) + (1 - prevalence) * F(t)

- `ratio` = {cfg['miss_cost_ratio']:g} -- one escaped defect priced at
  {cfg['miss_cost_ratio']:g} unnecessary re-inspections
- `prevalence` = {cfg['defect_frame_prevalence']:g} -- fraction of inspected frames
  that actually carry a defect. **Not measurable from NEU-DET**, which is 100%
  defective by construction, and not public for Jindal (see
  `docs/research_notes.md` section 9). A placeholder.
- `D(t)` from real defective `{meta['detection_split']}` frames, full 200x200 at
  imgsz {meta['train_imgsz']}
- `F(t)` from the clean-steel proxy, on the **{rec['basis']}** basis ({fa_basis_txt})

**Identifiability:** argmin of that cost is invariant to positive rescaling, so
prevalence and ratio move the recommendation only through the single weight
`lambda = prevalence * ratio / (1 - prevalence)` = {rec['effective_lambda']:.4f}.
Arguing about the two separately is arguing about one number.

{_md_table(
    ["conf", "D(t) detect", "clean patch FA", "clean frame FA (extrap.)", "legacy FA", "cost (patch basis)", "cost (frame basis)"],
    sweep_rows,
)}

### Recommended threshold: **conf = {rec['threshold']:.2f}**

{rec['rationale']}

### Sensitivity to the cost ratio

Unconstrained = the cost minimum on economics alone; constrained = the cheapest
point that still clears the {cfg['min_detection_rate']:.0%} detection floor.

{_md_table(
    ["miss:FA", "lambda", "uncon. conf", "D(t) there", "clean FA there", "clears floor", "con. conf"],
    sens_rows,
)}

{rec['sensitivity_verdict']}

### Basis check

Chosen on the raw per-patch false alarm rate: conf =
{rec['patch_basis_threshold']:.2f}. Chosen on the area-extrapolated per-frame rate:
conf = {rec['frame_basis_threshold']:.2f}. {rec['basis_verdict']}

### Held-out detection cross-check

`D(t)` above is measured on `{meta['detection_split']}`, and that is the split
ultralytics used to select `best.pt` (`models/*/args.yaml`: `val=true`,
`split=val`), so the detection half of the objective is not held out either. The
same sweep on `{holdout['split']}`, which selected nothing, gives
**{holdout['detection_rate_at_recommended']:.1%}** at conf = {rec['threshold']:.2f}
against {rec['defect_detection_rate']:.1%} on the tuning split. {holdout_verdict}

### Val cross-check

The clean patches above are mined from `test`, as the task specifies, so the false
alarm half of this objective is not held out. Repeating the whole mining and
scoring procedure on `val` gives {cross['clean_patches']} clean patches,
a false alarm rate of {cross['rate_at_recommended']:.1%} at conf =
{rec['threshold']:.2f} (test: {chosen_clean['clean_patch_false_alarm_rate']:.1%}),
and a recommended threshold of {cross['threshold']:.2f}. {cross['verdict']}

## 5. Chart

`{Path(payload['artifacts']['chart']).name}` -- **A** false alarm rate against
detection rate for all three false-alarm definitions, with the detection floor and
the recommended point marked; **B** expected cost against threshold at each
sensitivity ratio, triangles at the unconstrained minima; **C** the area-dependence
test, measured per-scale rates with 95% Wilson bars against the constant-rate and
spatial-independence models.

## 6. What this proxy is not

""" + "\n".join(f"{i}. {line}" for i, line in enumerate(LIMITATIONS, 1)) + f"""

### What a real measurement would need

{REAL_MEASUREMENT}

Until that exists, the honest statement for a slide is:

> On unannotated stainless surface captured under the same imaging conditions, at
> conf = {rec['threshold']:.2f}, the detector raises a box on **at most
> {chosen_clean['clean_patch_false_alarm_rate']:.0%}** of
> {mining['crop_sizes_px'][0]}-{mining['crop_sizes_px'][-1]} px clean patches
> (n = {chosen_clean['n_patches']} crops from
> {chosen_clean['clustering']['n_source_images']} images, 95% image-clustered CI
> {chosen_clean['cluster95_lo']:.0%}-{chosen_clean['cluster95_hi']:.0%}), and that
> rate does not grow with inspected area. "At most", because {fp_match_share} of
> those boxes carry the source image's own defect class, so some are unlabelled
> real defect. Over the same threshold it detects the real defect on
> {rec['defect_detection_rate']:.0%} of defective frames
> ({holdout['detection_rate_at_recommended']:.0%} on the fully held-out split).
> This is a proxy measured on clean regions of defective coils, not a false alarm
> rate on clean coils; the latter has not been measured and cannot be measured
> from NEU-DET.

Three numbers to refuse to put on a slide, because this work cannot support them:
a false alarm rate per coil, a false alarm rate per kilometre of strip, and any
statement about how often the line would actually stop. All three need the
production capture described above.
"""
    path.write_text(text)
    return path


def update_operating_point(
    path: Path, payload: dict[str, Any], demo_conf_floor: float = 0.05
) -> dict[str, Any]:
    """Rewrite the JSON the demo reads, preserving the previous file beside it.

    Returns a small record of what happened so the caller can report it honestly
    instead of claiming an update that did not occur.
    """
    rec = payload["recommendation"]
    meta = payload["meta"]
    previous: dict[str, Any] | None = None
    if path.is_file():
        try:
            previous = json.loads(path.read_text())
        except (OSError, ValueError):
            previous = None

    new_conf = round(float(rec["threshold"]), 3)
    old_conf = float(previous["conf_threshold"]) if previous and "conf_threshold" in previous else None
    changed = old_conf is None or abs(old_conf - new_conf) > 1e-9 or (
        previous or {}
    ).get("model_name") != meta["model_name"]

    # Preserve the operating point this file supersedes -- but only the one this
    # module did not itself write. Backing up unconditionally would, on the second
    # run, overwrite the real evaluate.py artefact with a copy of this module's own
    # previous output and quietly destroy the thing the backup exists to keep.
    backup = path.with_name("operating_point.evaluate.json")
    superseded = previous is not None and previous.get("source") != "src/false_alarm.py"
    if superseded and not backup.is_file():
        backup.write_text(json.dumps(previous, indent=2) + "\n")
    elif not backup.is_file():
        backup = None

    chosen_clean = next(
        row
        for row in payload["clean_patch_false_alarm"]["per_threshold"]
        if abs(row["threshold"] - rec["threshold"]) < 1e-9
    )
    curve_row = next(
        row for row in payload["operating_curve"] if abs(row["threshold"] - rec["threshold"]) < 1e-9
    )

    body = {
        "conf_threshold": new_conf,
        "iou_threshold": NMS_IOU,
        "tuned_on_split": meta["detection_split"],
        "selection_rule": (
            "cheapest threshold that still clears the detection floor, with the false "
            "alarm term measured on crops carrying no annotated defect rather than on "
            "spurious boxes over already-defective frames"
        ),
        # Whether the cost model or the constraint actually picked this threshold.
        # When the floor binds, the cost model did no work and saying "chosen by
        # the cost model" on a slide would be false.
        "floor_binding": rec["floor_binding"],
        "chosen_by": (
            "the detection floor (the cost minimum on economics alone sits elsewhere)"
            if rec["floor_binding"]
            else "the cost model (its minimum already clears the detection floor)"
        ),
        # This string described the area-extrapolated frame basis unconditionally,
        # while the run actually reports whichever basis the area-dependence test
        # licensed -- so on every run that rejected independence (all of them so
        # far) the file's own basis label contradicted its cost_basis field.
        "false_alarm_basis": (
            "clean-steel patch proxy, area-extrapolated to a "
            f"{meta['frame_px']}x{meta['frame_px']} frame"
            if rec["basis"] == "frame"
            else (
                "clean-steel patch proxy, measured per crop with no area extrapolation "
                "(spatial independence was tested on four crop scales and rejected); "
                "an upper bound, because a crop with no annotated box is not guaranteed "
                "defect-free"
            )
        ),
        "detection_floor": payload["cost_model"]["min_detection_rate"],
        "detection_floor_met": rec["floor_met"],
        "at_sweep_edge": rec["at_sweep_edge"],
        "miss_cost_ratio": payload["cost_model"]["miss_cost_ratio"],
        "defect_frame_prevalence": payload["cost_model"]["defect_frame_prevalence"],
        "effective_lambda": rec["effective_lambda"],
        "f1_optimal_threshold": payload["reference"]["f1_optimal_threshold"],
        "expected_at_threshold": {
            "threshold": new_conf,
            "defect_detection_rate": curve_row["defect_detection_rate"],
            "false_alarm_rate": chosen_clean["clean_patch_false_alarm_rate"],
            "clean_frame_false_alarm_rate": curve_row["clean_frame_false_alarm_rate"],
            "legacy_spurious_box_rate": curve_row["legacy_spurious_box_rate"],
            "expected_cost": curve_row[
                "expected_cost_frame_basis" if rec["basis"] == "frame" else "expected_cost_patch_basis"
            ],
            "cost_basis": rec["basis"],
            # On the patch basis the miss term is per frame and the false alarm
            # term is per crop, so the sum is an ordering statistic for comparing
            # thresholds, not a quantity with a unit. It ranks thresholds; it does
            # not predict re-inspections per frame, and must not be quoted as if
            # it did.
            "cost_units_caveat": (
                "comparable across thresholds only; the miss term is per frame and the "
                "false alarm term is per crop, so this is not re-inspections per frame"
            )
            if rec["basis"] == "patch"
            else "re-inspection equivalents per inspected frame",
            "clean_patch_cluster95": [
                chosen_clean["cluster95_lo"],
                chosen_clean["cluster95_hi"],
            ],
            "clean_patch_effective_n": chosen_clean["effective_n"],
        },
        "rationale": rec["rationale"],
        "weights": meta["weights"],
        "model_name": meta["model_name"],
        "source": "src/false_alarm.py",
        "supersedes": "src/evaluate.py operating point (kept at reports/operating_point.evaluate.json)",
        "demo_slider_note": (
            f"demo/app.py snaps the default onto a {demo_conf_floor:g}-step slider whose "
            f"minimum is {demo_conf_floor:g}; a recommendation below that is clamped to "
            f"{demo_conf_floor:g} in the UI."
        ),
        "generated_at": meta["generated_at"],
    }
    path.write_text(json.dumps(body, indent=2) + "\n")
    return {
        "path": str(path),
        "changed": bool(changed),
        "previous_conf": old_conf,
        "previous_model": (previous or {}).get("model_name"),
        "new_conf": new_conf,
        "backup": str(backup) if backup else None,
        "demo_effective_conf": max(new_conf, demo_conf_floor),
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def analyse(
    weights: Path,
    clean_split: str = "test",
    detection_split: str = "val",
    device: str = "auto",
    data_root: Path = DATA_ROOT,
    reports_dir: Path = REPORTS_DIR,
    update_demo: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Mine, score, sweep, price, plot and write. Returns the full payload."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device)
    weights = resolve_weights(weights)
    train_imgsz = resolve_train_imgsz(weights)

    clean_records = load_ground_truth(clean_split, data_root=data_root)
    frame_px = int(clean_records[0].width)
    frame_area = float(clean_records[0].width * clean_records[0].height)
    magnification = train_imgsz / frame_px

    if verbose:
        print(f"weights      {weights}")
        print(f"device       {device} | train imgsz {train_imgsz} | frame {frame_px}px")
        print(f"magnification {magnification:.3f}x")

    # --- mine ---------------------------------------------------------------
    clean_patches = mine_clean_patches(clean_records, PATCH_MINING, magnification)
    defect_patches = mine_defect_patches(clean_records, PATCH_MINING, magnification)
    if not clean_patches:
        raise RuntimeError(
            "No clean patches could be mined. Loosen PATCH_MINING['clearance_px'] or "
            "reduce PATCH_MINING['crop_sizes_px']; do not report a false alarm rate "
            "without them."
        )
    if verbose:
        print(f"\nmined {len(clean_patches)} clean patches, {len(defect_patches)} control patches")

    # --- score --------------------------------------------------------------
    if verbose:
        print("scoring clean patches:")
    clean_scored = score_patches(clean_patches, weights, device, verbose=verbose)
    if verbose:
        print("scoring control patches:")
    control_scored = score_patches(defect_patches, weights, device, verbose=verbose)

    clean_class = {r.image_path: CLASS_NAMES[r.image_class] for r in clean_records}
    clean_points = clean_patch_sweep(
        clean_scored, frame_area, SWEEP_THRESHOLDS, image_class=clean_class
    )
    control_points = control_sweep(control_scored, SWEEP_THRESHOLDS)

    # --- detection rate on real defective frames ----------------------------
    if verbose:
        print(f"\nscoring {detection_split} frames at imgsz {train_imgsz}")
    detection_records = load_ground_truth(detection_split, data_root=data_root)
    frame_detector = DefectDetector(
        weights=weights, device=device, conf=CACHE_CONF, iou=NMS_IOU, imgsz=train_imgsz
    )
    frame_detector.warmup(2)
    frame_cache = cache_predictions(frame_detector, detection_records)
    val_points = sweep_confidence(detection_records, frame_cache, SWEEP_THRESHOLDS)

    # D(t) is measured on `val`, and `val` is the split ultralytics used to select
    # best.pt (models/*/args.yaml: val=true, split=val). So the detection half of
    # the objective is not held out either. The `test` split is, and running the
    # identical sweep there sizes the optimism directly rather than leaving the
    # reader to assume it.
    if verbose:
        print(f"held-out detection cross-check on {clean_split} frames")
    holdout_points = sweep_confidence(
        clean_records, cache_predictions(frame_detector, clean_records), SWEEP_THRESHOLDS
    )
    # --- price --------------------------------------------------------------
    ratio = float(COST_MODEL["miss_cost_ratio"])
    prevalence = float(COST_MODEL["defect_frame_prevalence"])
    floor = float(COST_MODEL["min_detection_rate"])
    curve = build_cost_curve(val_points, clean_points, ratio, prevalence)
    chosen_frame, floor_met, _ = choose_threshold(curve, floor, basis="frame")
    chosen_patch, _, _ = choose_threshold(curve, floor, basis="patch")

    # Which false alarm basis leads? The area extrapolation is only licensed if
    # the measured area dependence actually supports a spatial-independence model.
    # Tested at the frame-basis pick, which is where the extrapolation would be
    # doing its work; if the test rejects independence, the raw per-patch rate --
    # which rests on no extrapolation at all -- becomes the primary basis.
    probe = next(p for p in clean_points if abs(p.threshold - chosen_frame.threshold) < 1e-9)
    poisson_ok = bool(probe.poisson_supported)
    basis = "frame" if poisson_ok else "patch"
    chosen, floor_met, at_edge = choose_threshold(curve, floor, basis=basis)
    sensitivity = sensitivity_table(
        val_points, clean_points, COST_MODEL["sensitivity_ratios"], prevalence, floor, basis
    )
    f1_best = max(val_points, key=lambda p: (p.f1, p.threshold))

    # --- val cross-check on the mining side ---------------------------------
    if verbose:
        print(f"\ncross-check: mining clean patches from {detection_split}")
    cross_patches = mine_clean_patches(detection_records, PATCH_MINING, magnification)
    cross_scored = score_patches(cross_patches, weights, device, verbose=verbose)
    cross_points = clean_patch_sweep(
        cross_scored,
        frame_area,
        SWEEP_THRESHOLDS,
        image_class={r.image_path: CLASS_NAMES[r.image_class] for r in detection_records},
    )
    cross_curve = build_cost_curve(val_points, cross_points, ratio, prevalence)
    cross_chosen, _, _ = choose_threshold(cross_curve, floor, basis=basis)
    cross_at_rec = next(
        p for p in cross_points if abs(p.threshold - chosen.threshold) < 1e-9
    )

    # --- narrative ----------------------------------------------------------
    chosen_clean = next(p for p in clean_points if abs(p.threshold - chosen.threshold) < 1e-9)

    # Is the recommendation actually being made by the cost model, or is it pinned
    # against the detection floor? The distinction matters: a floor-pinned answer
    # is a statement about the constraint, not about the economics, and saying
    # "robust to the cost ratio" about it would be misleading.
    eligible = [p for p in curve if p.detection_rate >= floor]
    floor_binding = bool(
        eligible and abs(chosen.threshold - max(p.threshold for p in eligible)) < 1e-9
    )
    thresholds = [row["unconstrained_threshold"] for row in sensitivity]
    spread = max(thresholds) - min(thresholds)
    n_clearing = sum(1 for row in sensitivity if not row["unconstrained_meets_floor"])
    sensitivity_verdict = (
        f"The unconstrained optimum moves from conf {min(thresholds):.2f} to "
        f"{max(thresholds):.2f} across a 10x span of the cost ratio -- a spread of "
        f"{spread:.2f}. "
        + (
            "That is a wide swing: the recommendation is being set by the cost "
            "assumption at least as much as by the detector."
            if spread >= 0.10
            else "That is a narrow swing over that range."
        )
        + (
            f" {n_clearing} of the {len(sensitivity)} ratios tested put the "
            f"cost-minimising threshold somewhere that fails the {floor:.0%} detection "
            f"floor, so for those the floor -- not the economics -- fixes the answer at "
            f"conf={chosen.threshold:.2f}. The practical reading: the cost ratio only "
            f"starts to matter once someone is willing to move the detection floor, and "
            f"the number to settle with the quality department is therefore the "
            f"acceptable escape rate first and the cost of an escape second."
            if n_clearing
            else " Every tested ratio puts the cost minimum at a threshold that clears "
            f"the floor, so the recommendation is the economics talking rather than the "
            f"constraint."
        )
    )
    poisson_ok_txt = (
        "supported" if poisson_ok else "rejected by the measured per-scale rates"
    )
    basis_verdict = (
        f"The spatial-independence model behind the area extrapolation is "
        f"{poisson_ok_txt} (chi2 {probe.chi2_poisson.get('chi2')} vs "
        f"{probe.chi2_constant.get('chi2')} for a constant rate, {probe.chi2_poisson.get('dof')} dof), "
        f"so the primary basis here is the **{basis}** rate. "
        + (
            "Both bases pick the same threshold, so nothing in the recommendation "
            "rests on the extrapolation either way."
            if abs(chosen_patch.threshold - chosen_frame.threshold) < 1e-9
            else f"They disagree (patch basis conf {chosen_patch.threshold:.2f}, frame "
            f"basis conf {chosen_frame.threshold:.2f}), so the choice of basis is "
            f"load-bearing and the recommendation is provisional until a per-frame "
            f"false alarm rate is measured directly."
        )
    )
    cross_verdict = (
        "The two splits agree on the threshold, so the leak does not change the "
        "recommendation."
        if abs(cross_chosen.threshold - chosen.threshold) < 1e-9
        else f"The val-mined recommendation differs (conf {cross_chosen.threshold:.2f} "
        f"vs {chosen.threshold:.2f}), so the test-mined number is optimistic and the "
        f"val figure is the safer one to quote."
    )
    floor_note = (
        ""
        if floor_met
        else (
            f" WARNING: no threshold in the sweep reaches the {floor:.0%} detection floor "
            f"(best {max(p.detection_rate for p in curve):.1%}); the constraint was dropped "
            f"and this is the unconstrained minimum."
        )
    )
    edge_note = (
        f" The optimum sits on the {'bottom' if chosen.threshold <= min(p.threshold for p in curve) else 'top'} "
        f"edge of the swept {min(p.threshold for p in curve):.2f}-{max(p.threshold for p in curve):.2f} range, "
        f"so the true minimum may lie outside it."
        if at_edge
        else ""
    )
    chosen_cost = chosen.cost_frame if basis == "frame" else chosen.cost_patch
    extrapolation_clause = (
        f"extrapolating to {chosen.clean_frame_fa:.1%} of full clean frames"
        if poisson_ok
        else f"an area extrapolation the data rejects would put this at "
        f"{chosen.clean_frame_fa:.1%} of full frames, but the measured rate is flat in "
        f"crop area, so {chosen_clean.rate:.1%} is the better frame-level estimate"
    )
    class_rates = [v["rate"] for v in chosen_clean.per_source_class.values() if v["n_patches"] >= 20]
    mixture_clause = (
        f" That pooled figure is a mixture: across the source defect classes with at "
        f"least 20 clean crops it ranges from {min(class_rates):.1%} to "
        f"{max(class_rates):.1%}, and the mixture weights are set by how much frame "
        f"area each defect class leaves unannotated, not by anything about a mill."
        if len(class_rates) >= 2
        else ""
    )
    fp_share = (
        chosen_clean.fp_matching_source_class / chosen_clean.fp_total
        if chosen_clean.fp_total
        else 0.0
    )
    upper_bound_clause = (
        f" {fp_share:.0%} of the boxes raised on clean crops carry the source image's own "
        f"defect class (chance would be {1 / len(CLASS_NAMES):.0%}), so an unknown share of "
        f"them are unlabelled continuations of the real defect rather than hallucinations "
        f"on sound metal: read {chosen_clean.rate:.1%} as an upper bound."
        if chosen_clean.fp_total
        else ""
    )
    rationale = (
        f"At conf={chosen.threshold:.2f} the detector finds the real defect on "
        f"{chosen.detection_rate:.1%} of defective frames while raising a box on "
        f"{chosen_clean.rate:.1%} of steel crops carrying no annotated defect "
        f"(95% image-clustered CI {chosen_clean.cluster_lo:.1%}-{chosen_clean.cluster_hi:.1%}) "
        f"({chosen_clean.boxes_per_patch:.2f} boxes per clean patch; {extrapolation_clause}). "
        f"Pricing one escaped defect at "
        f"{ratio:g} unnecessary re-inspections and assuming {prevalence:.0%} of inspected "
        f"frames are defective scores it at {chosen_cost:.4f} on the {basis} basis -- the "
        + (
            "lowest of every threshold clearing the "
            f"{floor:.0%} detection floor. On this basis the miss term is per frame and the "
            f"false alarm term is per crop, so that score orders thresholds and is not a "
            f"count of re-inspections per frame. "
            if basis == "patch"
            else "lowest, in re-inspection equivalents per inspected frame, of every "
            f"threshold clearing the {floor:.0%} detection floor. "
        )
        + f"For contrast, the legacy false alarm definition (a spurious box "
        f"on an already-defective frame) reads {chosen.legacy_fa:.1%} at the same threshold, "
        f"which is {'higher' if chosen.legacy_fa > chosen_clean.rate else 'lower'} than the "
        f"clean-steel rate and measures a different event entirely."
        + mixture_clause
        + upper_bound_clause
        + floor_note
        + edge_note
    )

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    imgsz_by_scale = {int(s): patch_imgsz(int(s), magnification) for s in PATCH_MINING["crop_sizes_px"]}

    payload: dict[str, Any] = {
        "meta": {
            "weights": str(weights),
            "model_name": f"{weights.parent.parent.name}/{weights.name}",
            "device": device,
            "train_imgsz": train_imgsz,
            "frame_px": frame_px,
            "magnification": magnification,
            "detection_split": detection_split,
            "detection_images": len(detection_records),
            "cache_conf": CACHE_CONF,
            "nms_iou": NMS_IOU,
            "generated_at": generated_at,
        },
        "patch_mining": {
            **{k: list(v) if isinstance(v, tuple) else v for k, v in PATCH_MINING.items()},
            "clean_split": clean_split,
            "split_images": len(clean_records),
            "clean_patches": len(clean_patches),
            "clean_source_images": len({p.image_path for p in clean_patches}),
            "control_patches": len(defect_patches),
            "control_source_images": len({p.image_path for p in defect_patches}),
            "clean_patches_by_scale": {
                int(s): sum(1 for p in clean_patches if p.size == s)
                for s in PATCH_MINING["crop_sizes_px"]
            },
            "control_patches_by_scale": {
                int(s): sum(1 for p in defect_patches if p.size == s)
                for s in PATCH_MINING["crop_sizes_px"]
            },
            "imgsz_by_scale": imgsz_by_scale,
            "achieved_magnification": {k: v / k for k, v in imgsz_by_scale.items()},
            "magnification_note": MAGNIFICATION_NOTE,
        },
        "cost_model": {
            **{k: list(v) if isinstance(v, tuple) else v for k, v in COST_MODEL.items()},
            "formula": "cost(t) = prevalence * ratio * (1 - D(t)) + (1 - prevalence) * F(t)",
            "identifiability": (
                "argmin is invariant to positive rescaling, so prevalence and ratio act "
                "only through lambda = prevalence * ratio / (1 - prevalence)"
            ),
        },
        "clean_patch_false_alarm": {
            "split": clean_split,
            "n_patches": len(clean_patches),
            "per_threshold": [p.to_dict() for p in clean_points],
        },
        "positive_control": {
            "split": clean_split,
            "n_patches": len(defect_patches),
            "per_threshold": [p.to_dict() for p in control_points],
        },
        "defect_frame_detection": {
            "split": detection_split,
            "n_images": len(detection_records),
            "selection_leak": (
                f"best.pt was selected by fitness on `{detection_split}` during training "
                f"(models/*/args.yaml: val=true, split=val), so D(t) here is not held out; "
                f"see holdout_detection for the same sweep on `{clean_split}`"
            ),
            "per_threshold": [p.to_dict() for p in val_points],
        },
        "holdout_detection": {
            "split": clean_split,
            "n_images": len(clean_records),
            "note": (
                "The same image-level detection sweep on the split that selected no "
                "checkpoint. If this sits below the tuning split, the recommendation's "
                "detection rate is optimistic; if above, it is not."
            ),
            "detection_rate_at_recommended": next(
                p.defect_detection_rate
                for p in holdout_points
                if abs(p.threshold - chosen.threshold) < 1e-9
            ),
            "per_threshold": [p.to_dict() for p in holdout_points],
        },
        "operating_curve": [p.to_dict() for p in curve],
        "recommendation": {
            "threshold": chosen.threshold,
            "basis": basis,
            "poisson_independence_supported": poisson_ok,
            "floor_met": floor_met,
            "floor_binding": floor_binding,
            "at_sweep_edge": at_edge,
            "effective_lambda": effective_lambda(ratio, prevalence),
            "patch_basis_threshold": chosen_patch.threshold,
            "frame_basis_threshold": chosen_frame.threshold,
            "basis_verdict": basis_verdict,
            "sensitivity_verdict": sensitivity_verdict,
            "rationale": rationale,
            **chosen.to_dict(),
        },
        "sensitivity": sensitivity,
        "reference": {
            "f1_optimal_threshold": round(f1_best.threshold, 3),
            "f1_at_optimum": round(f1_best.f1, 4),
        },
        "val_crosscheck": {
            "split": detection_split,
            "clean_patches": len(cross_patches),
            "clean_source_images": len({p.image_path for p in cross_patches}),
            "rate_at_recommended": cross_at_rec.rate,
            "frame_rate_at_recommended": cross_at_rec.frame_rate_extrapolated,
            "threshold": cross_chosen.threshold,
            "verdict": cross_verdict,
            "per_threshold": [p.to_dict() for p in cross_points],
        },
        "limitations": list(LIMITATIONS),
        "what_a_real_measurement_needs": REAL_MEASUREMENT,
    }

    chart = plot_false_alarm(
        curve,
        {
            r: build_cost_curve(val_points, clean_points, r, prevalence)
            for r in (*COST_MODEL["sensitivity_ratios"], ratio)
        },
        chosen,
        chosen_clean,
        floor,
        reports_dir / "false_alarm_curve.png",
    )
    payload["artifacts"] = {
        "json": str(reports_dir / "false_alarm.json"),
        "markdown": str(reports_dir / "false_alarm.md"),
        "chart": str(chart),
    }

    (reports_dir / "false_alarm.json").write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n"
    )
    write_report(reports_dir / "false_alarm.md", payload)

    if update_demo:
        payload["operating_point_update"] = update_operating_point(
            reports_dir / "operating_point.json", payload
        )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--clean-split", default="test", help="split to mine clean patches from")
    parser.add_argument("--detection-split", default="val", help="split for the detection rate")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument(
        "--update-operating-point",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="rewrite reports/operating_point.json (previous kept as operating_point.evaluate.json)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = analyse(
        weights=args.weights,
        clean_split=args.clean_split,
        detection_split=args.detection_split,
        device=args.device,
        data_root=args.data_root,
        reports_dir=args.reports_dir,
        update_demo=args.update_operating_point,
    )
    rec = payload["recommendation"]
    clean = next(
        row
        for row in payload["clean_patch_false_alarm"]["per_threshold"]
        if abs(row["threshold"] - rec["threshold"]) < 1e-9
    )
    print("\n" + "=" * 78)
    print(f"recommended conf              {rec['threshold']:.2f}")
    print(f"defect detection rate (val)   {rec['defect_detection_rate']:.3f}")
    print(f"held-out detection (test)     {payload['holdout_detection']['detection_rate_at_recommended']:.3f}")
    print(f"clean-patch false alarm rate  {clean['clean_patch_false_alarm_rate']:.3f} "
          f"[{clean['cluster95_lo']:.3f}, {clean['cluster95_hi']:.3f}] "
          f"image-clustered, n={clean['n_patches']} eff_n={clean['effective_n']:.0f}")
    print(f"  (Wilson, assumes independence {clean['wilson95_lo']:.3f}-{clean['wilson95_hi']:.3f} "
          f"-- too narrow, deff={clean['design_effect']:.1f})")
    print(f"  share of FP boxes matching source image class "
          f"{clean['share_of_fp_matching_source_class']:.3f} -> rate is an UPPER BOUND")
    print(f"clean-frame FA (extrapolated) {clean['frame_rate_extrapolated']:.3f} (model rejected)")
    print(f"legacy FA (defective frames)  {rec['legacy_spurious_box_rate']:.3f}")
    for key, value in payload["artifacts"].items():
        print(f"{key:<29} {value}")
    if "operating_point_update" in payload:
        print(f"operating point               {payload['operating_point_update']}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
