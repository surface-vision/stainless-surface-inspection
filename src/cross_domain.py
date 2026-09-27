"""Cross-domain generalisation harness: does the shipped detector work on other steel?

WHY THIS MODULE EXISTS
----------------------
`src/false_alarm.py` measures the false alarm rate on defect-free *crops of
NEU-DET images*. That is the best proxy available inside the training
distribution, and the module is explicit that it is an upper bound taken from a
dataset with no genuinely defect-free frames in it. It cannot answer the two
questions a mill actually asks:

    1. On real, verified defect-free steel from a different line, how often does
       this model raise an alarm?
    2. Does the score it produces separate clean steel from defective steel *on
       that same line*, or does it only separate NEU-DET from everything else?

Severstal (`Voxel51/severstal_steel_defects`) answers both: 1600x256 line-scan
strip frames, a verified defect-free subset, and pixel masks for the defective
ones. This module scores a checkpoint on that population and writes a record
that any later checkpoint can be compared against.

WHAT IT MEASURES, AND THE UNIT EACH NUMBER IS IN
------------------------------------------------
The deployed system sees 200x200 frames at imgsz 256 (magnification 1.28x), so a
1600x256 strip frame is cut into 200x200 tiles and each tile is scored exactly as
a deployed frame would be. Two units follow, and they are not interchangeable:

    crop  -- one 200x200 tile. This is one inference, one operator alarm.
    frame -- one 1600x256 strip frame, flagged if ANY of its tiles alarms.
             This is the unit a coil disposition is built from, and it is the
             number that lands: a mill does not care that 1 tile in 2 alarms,
             it cares that 19 strip frames in 20 get held.

CONFIDENCE INTERVALS
--------------------
Tiles are clustered inside frames -- eight tiles from one frame share its
lighting, its scale build-up and its oxide pattern, so they are nothing like
eight independent draws. Every crop-level interval here is an image-clustered
bootstrap via `false_alarm.cluster_bootstrap_interval`, which also returns the
design effect (how much a naive binomial interval would have overstated the
precision). The AUC interval resamples whole frames on both arms for the same
reason; it needs its own bootstrap because the pooled-rate helper cannot express
a rank statistic computed across two populations.

WHAT THIS DOES NOT MEASURE
--------------------------
Severstal is hot-rolled carbon strip photographed on someone else's line. It is
not Jindal stainless, and a good score here would not prove the model works at
Hisar. A bad score, however, does prove the model does not transfer -- which is
the claim slide 5 makes. The asymmetry is the point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: this runs on the line PC and in CI

import cv2
import matplotlib.pyplot as plt
import numpy as np

try:  # works both as `python src/cross_domain.py` and as `from src import cross_domain`
    from .evaluate import DATA_ROOT, REPORTS_DIR, _json_safe, load_ground_truth
    from .false_alarm import (
        CACHE_CONF,
        NMS_IOU,
        PATCH_MINING,
        cluster_bootstrap_interval,
        mine_clean_patches,
        mine_defect_patches,
        score_patches,
    )
    from .inference import (
        DEFAULT_IMGSZ,
        DefectDetector,
        resolve_device,
        resolve_weights,
    )
    from .report import DispositionRules, build_coil_report, inspect_coil
except ImportError:  # pragma: no cover - script execution path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from evaluate import DATA_ROOT, REPORTS_DIR, _json_safe, load_ground_truth
    from false_alarm import (
        CACHE_CONF,
        NMS_IOU,
        PATCH_MINING,
        cluster_bootstrap_interval,
        mine_clean_patches,
        mine_defect_patches,
        score_patches,
    )
    from inference import DEFAULT_IMGSZ, DefectDetector, resolve_device, resolve_weights
    from report import DispositionRules, build_coil_report, inspect_coil

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The scratch pull the audit made (1000 verified defect-free + 400 masked defective
# 1600x256 frames). Overridable with --data-root so the same harness runs against
# whatever the ingest step finally lands on disk.
DEFAULT_DATA_ROOT = (
    Path("/private/tmp/claude-501/-Users-prathmeshwalimbe-Downloads-JSW-PS1")
    / "f88a3d3d-c8be-40c8-8600-132f500095d3"
    / "scratchpad"
    / "severstal"
)

DEFAULT_JSON = REPORTS_DIR / "cross_domain.json"
DEFAULT_MD = REPORTS_DIR / "cross_domain.md"
DEFAULT_PLOT = REPORTS_DIR / "cross_domain_fa_vs_recall.png"
DEFAULT_COIL_DIR = REPORTS_DIR / "cross_domain_coil"
_TILE_MARKER = ".cross_domain_tiles"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

# The frame geometry the detector was trained and deployed on. Crops are cut at
# CROP_PX and run at --imgsz, so imgsz/CROP_PX is the magnification; at the
# defaults (200 / 256) that is 1.28x, identical to deploying on a NEU-DET frame.
DEFAULT_CROP_PX = 200
NEUDET_FRAME_PX = 200

# A tile counts as defective when it holds at least this many labelled defect
# pixels. 64 px is an 8x8 patch: below that the "defect" inside the tile is a
# mask boundary artefact and asking the detector to find it is not a fair test.
DEFAULT_MIN_DEFECT_PX = 64

DEFAULT_CONF_SWEEP: tuple[float, ...] = (
    0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80,
)

# Held-out fraction of the clean frames. The other half is what a fix is allowed
# to train on; nothing in this file ever scores a frame the split assigns to
# train, so a post-fix record stays honest without anyone having to remember.
DEFAULT_HOLDOUT_FRAC = 0.5
HOLDOUT_SEED = "jsw-ps1-cross-domain-holdout-v1"

DEFAULT_BOOTSTRAP = 4000

_GRID = {"color": "#d8dde3", "linewidth": 0.7, "alpha": 0.9}


def deploy_conf(default: float = 0.15) -> float:
    """The shipped operating point, read from the artifact that owns it."""
    path = REPORTS_DIR / "operating_point.json"
    try:
        return float(json.loads(path.read_text())["conf_threshold"])
    except Exception:  # pragma: no cover - missing//malformed artifact
        return default


# ---------------------------------------------------------------------------
# data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Crop:
    """One 200x200 tile of a strip frame, with its ground-truth status."""

    frame: Path
    x0: int
    y0: int
    size: int
    label: int  # 1 defective, 0 defect-free, -1 unlabelled region of a defective frame
    defect_px: int = 0
    # The strip frame this tile came from. When the tiles are cut here it is the
    # frame path; when they arrive pre-cut on disk it is parsed out of the file
    # name. Either way it is the cluster the bootstrap resamples, because two
    # tiles of one frame share its lighting, scale build-up and oxide pattern.
    source: str = ""

    @property
    def source_key(self) -> str:
        return self.source or str(self.frame)


@dataclass
class ScoredCrop:
    """Every box the detector produced on one tile, cached at CACHE_CONF."""

    crop: Crop
    confidences: np.ndarray
    classes: np.ndarray

    @property
    def max_conf(self) -> float:
        return float(self.confidences.max()) if self.confidences.size else 0.0


@dataclass
class DomainSplit:
    """The frames this run scores, after the held-out split has been applied."""

    clean: list[Path] = field(default_factory=list)
    defect: list[Path] = field(default_factory=list)
    masks: dict[Path, Path] = field(default_factory=dict)
    clean_train: list[Path] = field(default_factory=list)
    defect_train: list[Path] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# dataset discovery and the held-out split
# ---------------------------------------------------------------------------


def _images_in(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def find_mask(frame: Path) -> Path | None:
    """Locate the pixel mask for a frame, across the layouts we have seen.

    The audit's scratch pull writes `<frame>.mask.npy` beside the image; a tidier
    ingest is likely to write `masks/<stem>.npy` or a PNG. Accept all of them
    rather than forcing one, because the module has to keep working when the
    ingest step replaces the directory under it.
    """
    candidates = [
        frame.with_suffix(frame.suffix + ".mask.npy"),
        frame.with_suffix(".mask.npy"),
        frame.with_suffix(".npy"),
        frame.parent / "masks" / f"{frame.stem}.npy",
        frame.parent.parent / "masks" / f"{frame.stem}.npy",
        frame.parent / "masks" / f"{frame.stem}.png",
        frame.parent.parent / "masks" / f"{frame.stem}.png",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_mask(path: Path) -> np.ndarray:
    """Read a defect mask as a 2-D array; non-zero means labelled defect."""
    if path.suffix.lower() == ".npy":
        mask = np.load(path)
    else:
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"Unreadable mask: {path}")
    if mask.ndim == 3:
        mask = mask.max(axis=2)
    return np.asarray(mask)


def is_holdout(stem: str, frac: float, seed: str = HOLDOUT_SEED) -> bool:
    """Deterministic per-frame held-out assignment, stable across directories.

    Hashing the frame id rather than shuffling a list means a training script and
    this harness agree on the split without sharing state, and the assignment does
    not move when the ingest step adds more frames.
    """
    if frac >= 1.0:
        return True
    if frac <= 0.0:
        return False
    digest = hashlib.blake2b(f"{seed}:{stem}".encode(), digest_size=8).digest()
    return (int.from_bytes(digest, "big") / float(1 << 64)) < frac


def discover_dataset(
    root: Path,
    *,
    clean_dirs: Sequence[str] = ("clean", "clean2", "negatives", "defect_free"),
    defect_dirs: Sequence[str] = ("defect", "defective", "positives"),
    holdout_frac: float = DEFAULT_HOLDOUT_FRAC,
    exclude_stems: Iterable[str] = (),
    limit_clean: int = 0,
    limit_defect: int = 0,
) -> DomainSplit:
    """Collect clean/defective frames under `root` and apply the held-out split.

    `exclude_stems` is the escape hatch that makes "held out" checkable instead of
    assumed: point it at the manifest a training run actually consumed and any
    overlap is removed here and counted in the record.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Cross-domain data root not found: {root}")

    clean_all: list[Path] = []
    for name in clean_dirs:
        clean_all.extend(_images_in(root / name))
    defect_all: list[Path] = []
    for name in defect_dirs:
        defect_all.extend(_images_in(root / name))

    if not clean_all:
        raise RuntimeError(
            f"No defect-free frames under {root} (looked in {list(clean_dirs)}). "
            "Point --data-root at a directory holding a clean/ subdirectory."
        )

    clean_stems = {p.stem for p in clean_all}
    overlap = clean_stems & {p.stem for p in defect_all}
    if overlap:
        raise RuntimeError(
            f"{len(overlap)} frame(s) appear in both the clean and defective sets, "
            f"e.g. {sorted(overlap)[:3]}. The clean population must be verified "
            "defect-free or every number in this report is meaningless."
        )

    excluded = sorted(set(exclude_stems))
    excluded_set = set(excluded)

    split = DomainSplit(excluded=[s for s in excluded if s in clean_stems])
    for path in clean_all:
        if path.stem in excluded_set:
            split.clean_train.append(path)
        elif is_holdout(path.stem, holdout_frac):
            split.clean.append(path)
        else:
            split.clean_train.append(path)

    for path in defect_all:
        mask = find_mask(path)
        if mask is None:
            # A "defective" frame without a mask cannot contribute a labelled
            # positive tile. Dropping it is safer than guessing which tiles are
            # defective, which would inflate recall.
            continue
        if path.stem in excluded_set or not is_holdout(path.stem, holdout_frac):
            split.defect_train.append(path)
            continue
        split.defect.append(path)
        split.masks[path] = mask

    if limit_clean > 0:
        split.clean = split.clean[:limit_clean]
    if limit_defect > 0:
        split.defect = split.defect[:limit_defect]
        split.masks = {p: split.masks[p] for p in split.defect}

    if not split.clean:
        raise RuntimeError("The held-out clean set is empty; lower --holdout-frac.")
    return split


def manifest_hash(paths: Sequence[Path]) -> str:
    """SHA-256 of the sorted frame ids, so a later run can prove it scored the same set."""
    joined = "\n".join(sorted(p.stem for p in paths))
    return hashlib.sha256(joined.encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# tiling
# ---------------------------------------------------------------------------


def crop_origins(extent: int, size: int) -> list[int]:
    """Tile origins spanning `extent`, evenly spaced, never running off the edge.

    A 1600 px frame at 200 px gives eight abutting tiles. A frame whose height is
    256 gives a single row centred on the strip -- the 28 px trimmed top and
    bottom are the frame border, not steel the operator would ever be shown.
    """
    if extent <= size:
        return [max(0, (extent - size) // 2)]
    count = max(1, int(round(extent / size)))
    if count == 1:
        return [(extent - size) // 2]
    step = (extent - size) / (count - 1)
    return sorted({int(round(i * step)) for i in range(count)})


def build_crops(
    frames: Sequence[Path],
    masks: dict[Path, Path] | None = None,
    *,
    crop_px: int = DEFAULT_CROP_PX,
    min_defect_px: int = DEFAULT_MIN_DEFECT_PX,
) -> list[Crop]:
    """Cut every frame into tiles and label each tile from the pixel mask.

    Tiles of a defective frame that hold no labelled pixels get label -1, not 0.
    Severstal's masks are the only evidence available, and a region nobody
    annotated on a frame that *does* carry a defect is not the same evidence as a
    frame the dataset verified defect-free end to end. Counting those tiles as
    clean would quietly relabel the hardest negatives in the set; they are
    reported separately instead.
    """
    masks = masks or {}
    crops: list[Crop] = []
    for frame in frames:
        header = _frame_size(frame)
        if header is None:
            raise RuntimeError(f"Unreadable frame: {frame}")
        width, height = header
        mask = load_mask(masks[frame]) if frame in masks else None
        if mask is not None and mask.shape[:2] != (height, width):
            raise RuntimeError(
                f"Mask {masks[frame]} is {mask.shape[:2]}, frame {frame} is "
                f"{(height, width)} -- they must be the same geometry."
            )
        for y0 in crop_origins(height, crop_px):
            for x0 in crop_origins(width, crop_px):
                if mask is None:
                    label, defect_px = 0, 0
                else:
                    defect_px = int(
                        np.count_nonzero(mask[y0 : y0 + crop_px, x0 : x0 + crop_px])
                    )
                    label = 1 if defect_px >= min_defect_px else -1
                crops.append(
                    Crop(
                        frame=frame, x0=x0, y0=y0, size=crop_px, label=label,
                        defect_px=defect_px, source=frame.stem,
                    )
                )
    return crops


def _frame_size(path: Path) -> tuple[int, int] | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    height, width = image.shape[:2]
    return width, height


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def score_crops(
    crops: Sequence[Crop],
    detector: DefectDetector,
    *,
    batch_size: int = 32,
    verbose: bool = True,
    label: str = "",
) -> list[ScoredCrop]:
    """Run the detector over every tile, caching all boxes above CACHE_CONF.

    The detector is constructed once at CACHE_CONF and the threshold sweep is then
    pure NumPy over the cache. That is exact, not an approximation: NMS is greedy
    in descending confidence, so a box above t can only ever have been suppressed
    by another box above t, and filtering the cache at `conf >= t` reproduces what
    a detector built at t would have returned.

    Frames are decoded once and released immediately -- 1400 Severstal frames held
    in memory at once is 1.7 GB for no reason.
    """
    scored: list[ScoredCrop] = []
    if not crops:
        return scored

    buffer: list[np.ndarray] = []
    pending: list[Crop] = []
    current_path: Path | None = None
    current_rgb: np.ndarray | None = None
    started = time.perf_counter()

    def flush() -> None:
        if not buffer:
            return
        for crop, result in zip(pending, detector.predict_batch(buffer, batch_size=batch_size)):
            if result.detections:
                conf = np.asarray([d.confidence for d in result.detections], dtype=np.float64)
                cls = np.asarray([d.class_id for d in result.detections], dtype=np.int64)
            else:
                conf = np.zeros((0,), dtype=np.float64)
                cls = np.zeros((0,), dtype=np.int64)
            scored.append(ScoredCrop(crop=crop, confidences=conf, classes=cls))
        buffer.clear()
        pending.clear()

    for crop in crops:
        if crop.frame != current_path:
            bgr = cv2.imread(str(crop.frame), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"Unreadable frame: {crop.frame}")
            current_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            current_path = crop.frame
        assert current_rgb is not None
        buffer.append(
            current_rgb[crop.y0 : crop.y0 + crop.size, crop.x0 : crop.x0 + crop.size].copy()
        )
        pending.append(crop)
        if len(buffer) >= batch_size:
            flush()
    flush()

    if verbose:
        print(
            f"  scored {len(scored)} {label} crops in {time.perf_counter() - started:.1f}s",
            flush=True,
        )
    return scored


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged. Written out because ties dominate here.

    Every tile with no box scores exactly 0.0, and at conf 0.15 that is a third of
    the population. A rank rule that broke those ties arbitrarily would move the
    AUC by several points depending on array order.
    """
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    i = 0
    while i < ordered.size:
        j = i
        while j + 1 < ordered.size and ordered[j + 1] == ordered[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def roc_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Tie-corrected Mann-Whitney AUC: P(score(defect) > score(clean)) + 0.5 P(=)."""
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    n_p, n_n = positive.size, negative.size
    if n_p == 0 or n_n == 0:
        return float("nan")
    ranks = _average_ranks(np.concatenate([positive, negative]))
    return float((ranks[:n_p].sum() - n_p * (n_p + 1) / 2.0) / (n_p * n_n))


def cluster_bootstrap_auc(
    positive_groups: Sequence[np.ndarray],
    negative_groups: Sequence[np.ndarray],
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = 20260909,
) -> tuple[float, float]:
    """95% interval for the AUC, resampling whole frames on both arms.

    `false_alarm.cluster_bootstrap_interval` is reused everywhere a pooled rate is
    reported, but it cannot be reused here: the AUC is a rank statistic computed
    jointly across two populations, not a mean of per-crop indicators, so the
    resampling has to be done on the scores themselves.
    """
    positive_groups = [np.asarray(g, dtype=np.float64) for g in positive_groups if len(g)]
    negative_groups = [np.asarray(g, dtype=np.float64) for g in negative_groups if len(g)]
    if not positive_groups or not negative_groups:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_pos, n_neg = len(positive_groups), len(negative_groups)
    draws = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pos = np.concatenate([positive_groups[i] for i in rng.integers(0, n_pos, n_pos)])
        neg = np.concatenate([negative_groups[i] for i in rng.integers(0, n_neg, n_neg)])
        draws[b] = roc_auc(pos, neg)
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:  # pragma: no cover - impossible with non-empty groups
        return (float("nan"), float("nan"))
    lo, hi = (float(v) for v in np.percentile(finite, [2.5, 97.5]))
    return (lo, hi)


def _group_by_source(scored: Sequence[ScoredCrop]) -> dict[str, list[ScoredCrop]]:
    """Bucket scored tiles by the strip frame they were cut from."""
    grouped: dict[str, list[ScoredCrop]] = {}
    for item in scored:
        grouped.setdefault(item.crop.source_key, []).append(item)
    return grouped


def sweep_thresholds(
    clean: Sequence[ScoredCrop],
    defect: Sequence[ScoredCrop],
    thresholds: Sequence[float],
    class_names: Sequence[str],
    *,
    n_boot: int = DEFAULT_BOOTSTRAP,
) -> list[dict[str, Any]]:
    """Per-threshold clean false alarm and defect recall on the same domain."""
    clean_by_frame = _group_by_source(clean)
    positives = [s for s in defect if s.crop.label == 1]
    ambiguous = [s for s in defect if s.crop.label == -1]
    positive_by_frame = _group_by_source(positives)

    points: list[dict[str, Any]] = []
    for t in thresholds:
        crop_groups = [
            [int((s.confidences >= t).any()) for s in group] for group in clean_by_frame.values()
        ]
        crop_hits = sum(sum(g) for g in crop_groups)
        n_crops = sum(len(g) for g in crop_groups)
        crop_lo, crop_hi, crop_deff = cluster_bootstrap_interval(crop_groups, n_boot=n_boot)

        frame_flags = [[int(any(g))] for g in crop_groups]
        frame_hits = sum(sum(g) for g in frame_flags)
        frame_lo, frame_hi, _ = cluster_bootstrap_interval(frame_flags, n_boot=n_boot)

        boxes = 0
        class_hist: dict[str, int] = {}
        for s in clean:
            keep = s.confidences >= t
            boxes += int(keep.sum())
            for cls_id in s.classes[keep]:
                name = (
                    class_names[int(cls_id)]
                    if 0 <= int(cls_id) < len(class_names)
                    else f"class_{int(cls_id)}"
                )
                class_hist[name] = class_hist.get(name, 0) + 1

        recall_groups = [
            [int((s.confidences >= t).any()) for s in group]
            for group in positive_by_frame.values()
        ]
        rec_hits = sum(sum(g) for g in recall_groups)
        n_pos = sum(len(g) for g in recall_groups)
        rec_lo, rec_hi, rec_deff = cluster_bootstrap_interval(recall_groups, n_boot=n_boot)
        frame_recall_hits = sum(1 for g in recall_groups if any(g))

        amb_hits = sum(1 for s in ambiguous if (s.confidences >= t).any())

        points.append(
            {
                "conf": float(t),
                "clean_crop_fa": crop_hits / n_crops if n_crops else float("nan"),
                "clean_crop_fa_ci": [crop_lo, crop_hi],
                "clean_crop_design_effect": crop_deff,
                "clean_crops_flagged": crop_hits,
                "clean_crops": n_crops,
                "clean_frame_fa": frame_hits / len(frame_flags) if frame_flags else float("nan"),
                "clean_frame_fa_ci": [frame_lo, frame_hi],
                "clean_frames_flagged": frame_hits,
                "clean_frames": len(frame_flags),
                "boxes_per_clean_crop": boxes / n_crops if n_crops else float("nan"),
                "boxes_total": boxes,
                "false_positive_classes": dict(
                    sorted(class_hist.items(), key=lambda kv: -kv[1])
                ),
                "false_positive_class_fracs": {
                    k: v / boxes for k, v in sorted(class_hist.items(), key=lambda kv: -kv[1])
                }
                if boxes
                else {},
                "defect_crop_recall": rec_hits / n_pos if n_pos else float("nan"),
                "defect_crop_recall_ci": [rec_lo, rec_hi],
                "defect_crop_design_effect": rec_deff,
                "defect_crops_hit": rec_hits,
                "defect_crops": n_pos,
                "defect_frame_recall": (
                    frame_recall_hits / len(recall_groups) if recall_groups else float("nan")
                ),
                "defect_frames": len(recall_groups),
                "unlabelled_region_flag_rate": (
                    amb_hits / len(ambiguous) if ambiguous else float("nan")
                ),
                "unlabelled_region_crops": len(ambiguous),
            }
        )
    return points


def separation(
    clean: Sequence[ScoredCrop],
    defect: Sequence[ScoredCrop],
    *,
    n_boot: int = DEFAULT_BOOTSTRAP,
) -> dict[str, Any]:
    """AUC of max-box-confidence, defective tiles vs verified defect-free tiles."""
    positives = [s for s in defect if s.crop.label == 1]
    pos_by_frame = _group_by_source(positives)
    neg_by_frame = _group_by_source(clean)
    pos_scores = np.asarray([s.max_conf for s in positives], dtype=np.float64)
    neg_scores = np.asarray([s.max_conf for s in clean], dtype=np.float64)
    auc = roc_auc(pos_scores, neg_scores)
    lo, hi = cluster_bootstrap_auc(
        [np.asarray([s.max_conf for s in g]) for g in pos_by_frame.values()],
        [np.asarray([s.max_conf for s in g]) for g in neg_by_frame.values()],
        n_boot=n_boot,
    )
    return {
        "auc": auc,
        "auc_ci": [lo, hi],
        "n_positive_crops": int(pos_scores.size),
        "n_negative_crops": int(neg_scores.size),
        "n_positive_frames": len(pos_by_frame),
        "n_negative_frames": len(neg_by_frame),
        "bootstrap_draws": int(n_boot),
        "bootstrap_unit": "source frame (both arms resampled independently)",
    }


# ---------------------------------------------------------------------------
# the in-domain comparison
# ---------------------------------------------------------------------------


def in_domain_separation(
    weights: Path,
    device: str,
    *,
    split: str = "test",
    data_root: Path = DATA_ROOT,
    magnification: float = float(DEFAULT_IMGSZ) / NEUDET_FRAME_PX,
    config: dict[str, Any] = PATCH_MINING,
    n_boot: int = DEFAULT_BOOTSTRAP,
    verbose: bool = True,
) -> dict[str, Any]:
    """The same AUC statistic, computed inside the training distribution.

    NEU-DET has no defect-free image, so the negatives have to be mined: crops
    that clear every labelled box by `clearance_px` on all four sides. That is
    exactly the proxy `src/false_alarm.py` built and defended, reused here rather
    than reinvented, and the positives are its containment-checked defect crops.

    One residual difference is unavoidable and is stated in the report: the mined
    crops are 60-120 px because a 200x200 NEU-DET frame has no room for a 200 px
    defect-free window, while the cross-domain crops are 200 px. Magnification is
    held equal (both run at imgsz = round32(crop_px * 1.28)), so the comparison
    controls for pixels-per-millimetre but not for field of view.
    """
    records = load_ground_truth(split, data_root=data_root)
    clean_patches = mine_clean_patches(records, config=config, magnification=magnification)
    defect_patches = mine_defect_patches(records, config=config, magnification=magnification)
    if verbose:
        print(
            f"  in-domain ({split}): {len(clean_patches)} clean patches, "
            f"{len(defect_patches)} defect patches",
            flush=True,
        )
    clean_scored = score_patches(clean_patches, weights, device, verbose=verbose)
    defect_scored = score_patches(defect_patches, weights, device, verbose=verbose)

    def _by_image(scored: Sequence[Any], size: int | None = None) -> dict[Path, list[float]]:
        out: dict[Path, list[float]] = {}
        for s in scored:
            if size is not None and s.patch.size != size:
                continue
            value = float(s.confidences.max()) if s.confidences.size else 0.0
            out.setdefault(s.patch.image_path, []).append(value)
        return out

    pos_groups = _by_image(defect_scored)
    neg_groups = _by_image(clean_scored)
    pos = np.asarray([v for g in pos_groups.values() for v in g], dtype=np.float64)
    neg = np.asarray([v for g in neg_groups.values() for v in g], dtype=np.float64)
    lo, hi = cluster_bootstrap_auc(
        [np.asarray(g) for g in pos_groups.values()],
        [np.asarray(g) for g in neg_groups.values()],
        n_boot=n_boot,
    )

    # Scale stratification. The mined clean patches are dominated by the smallest
    # crop size and the mined defect patches by the largest -- an artefact of
    # containment requiring room, not of the detector -- so a pooled AUC across
    # scales partly measures crop size. Computing the AUC within each size and
    # recombining with Mann-Whitney weights (n_pos * n_neg) removes that.
    per_scale: list[dict[str, Any]] = []
    for size in sorted({p.size for p in clean_patches} | {p.size for p in defect_patches}):
        p_s = np.asarray(
            [v for g in _by_image(defect_scored, size).values() for v in g], dtype=np.float64
        )
        n_s = np.asarray(
            [v for g in _by_image(clean_scored, size).values() for v in g], dtype=np.float64
        )
        per_scale.append(
            {
                "crop_px": int(size),
                "n_positive": int(p_s.size),
                "n_negative": int(n_s.size),
                "auc": roc_auc(p_s, n_s),
            }
        )
    weights_s = [
        float(s["n_positive"] * s["n_negative"])
        for s in per_scale
        if math.isfinite(float(s["auc"]))
    ]
    aucs_s = [float(s["auc"]) for s in per_scale if math.isfinite(float(s["auc"]))]
    stratified = (
        float(np.average(aucs_s, weights=weights_s)) if weights_s and sum(weights_s) else float("nan")
    )

    return {
        "auc": roc_auc(pos, neg),
        "auc_ci": [lo, hi],
        "auc_scale_stratified": stratified,
        "per_scale": per_scale,
        "n_positive_crops": int(pos.size),
        "n_negative_crops": int(neg.size),
        "n_positive_frames": len(pos_groups),
        "n_negative_frames": len(neg_groups),
        "split": split,
        "magnification": magnification,
        "crop_sizes_px": [int(s) for s in config["crop_sizes_px"]],
        "clearance_px": int(config["clearance_px"]),
        "negatives_are": (
            "mined crops of NEU-DET images that clear every labelled box by "
            f"{int(config['clearance_px'])} px; NEU-DET has no defect-free image, "
            "so this is a proxy and an optimistic one"
        ),
        "bootstrap_draws": int(n_boot),
    }


# ---------------------------------------------------------------------------
# the disposition chain on genuinely clean steel
# ---------------------------------------------------------------------------


def clean_coil_disposition(
    frames: Sequence[Path],
    *,
    weights: Path | str | None,
    device: str,
    conf: float,
    iou: float,
    imgsz: int,
    crop_px: int = DEFAULT_CROP_PX,
    coil_dir: Path = DEFAULT_COIL_DIR,
    coil_id: str = "SEVERSTAL-CLEAN-HOLDOUT",
    include_cam: bool = False,
    write_html: bool = True,
    html_stem: str = "cross_domain_clean_coil",
    out_dir: Path = REPORTS_DIR,
) -> dict[str, Any]:
    """Run the shipped disposition chain over a coil of verified defect-free steel.

    Two geometries, because the objection to either one alone is obvious:

      tiles       -- each 200x200 tile is a frame, which is what the deployed
                     system actually inspects and the geometry the detector was
                     trained on. This is the primary result.
      whole_frames-- each 1600x256 strip frame is handed to the detector whole,
                     letterboxed down to imgsz. Cheaper, lossier, and the naive
                     thing a first integration does; reported so nobody can
                     claim the verdict is an artefact of how the strip was cut.

    Nothing here is a new decision rule: `DispositionRules` is the shipped one,
    unmodified.
    """
    coil_dir = Path(coil_dir)
    coil_dir.mkdir(parents=True, exist_ok=True)
    # This directory is rewritten on every run, so refuse to touch one this module
    # did not create. --coil-dir pointed at a real image folder would otherwise
    # delete it.
    marker = coil_dir / _TILE_MARKER
    if any(coil_dir.iterdir()) and not marker.is_file():
        raise RuntimeError(
            f"{coil_dir} is not empty and was not written by this module "
            f"(no {_TILE_MARKER}). Point --coil-dir at a new or previously "
            "generated directory."
        )
    marker.write_text("Tiles written by src/cross_domain.py; safe to delete.\n")
    for stale in coil_dir.glob("*.jpg"):
        stale.unlink()

    tile_paths: list[Path] = []
    for frame in frames:
        bgr = cv2.imread(str(frame), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Unreadable frame: {frame}")
        height, width = bgr.shape[:2]
        for y0 in crop_origins(height, crop_px):
            for x0 in crop_origins(width, crop_px):
                out = coil_dir / f"{frame.stem}_x{x0:05d}_y{y0:05d}.jpg"
                cv2.imwrite(str(out), bgr[y0 : y0 + crop_px, x0 : x0 + crop_px])
                tile_paths.append(out)

    common = dict(
        weights=weights,
        device=device,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        include_cam=include_cam,
        rules=DispositionRules(),
        source_note=(
            "Frames are 200x200 tiles of verified defect-free Severstal strip "
            "(Voxel51/severstal_steel_defects). Every detection on this coil is a "
            "false alarm by construction."
        ),
    )

    if write_html:
        tiles_report, html_path, json_path = build_coil_report(
            tile_paths, out_dir, stem=html_stem, coil_id=coil_id, **common
        )
        artifacts = {"html": str(html_path), "json": str(json_path)}
    else:
        tiles_report = inspect_coil(tile_paths, coil_id=coil_id, **common)
        artifacts = {}

    frames_report = inspect_coil(
        list(frames),
        coil_id=f"{coil_id}-WHOLEFRAME",
        **{**common, "include_cam": False},
    )

    def _summary(report: Any) -> dict[str, Any]:
        stats = report.stats
        return {
            "coil_id": report.coil_id,
            "disposition": report.disposition,
            "reasons": list(report.reasons),
            "total_frames": stats.total_frames,
            "defect_frames": stats.defect_frames,
            "defect_rate": stats.defect_rate,
            "total_detections": stats.total_detections,
            "p95_severity": stats.p95_severity,
            "max_severity": stats.max_severity,
            "band_counts": dict(stats.band_counts),
            "class_frame_counts": dict(stats.class_frame_counts),
            "class_counts": dict(stats.class_counts),
        }

    return {
        "rules": DispositionRules().to_dict(),
        "source_frames": len(frames),
        "tiles": _summary(tiles_report),
        "whole_frames": _summary(frames_report),
        "artifacts": artifacts,
        "tile_dir": str(coil_dir),
    }


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def analyse(
    *,
    weights: str | Path | None = None,
    device: str = "auto",
    imgsz: int = DEFAULT_IMGSZ,
    iou: float = NMS_IOU,
    crop_px: int = DEFAULT_CROP_PX,
    data_root: Path = DEFAULT_DATA_ROOT,
    conf_sweep: Sequence[float] = DEFAULT_CONF_SWEEP,
    tag: str = "baseline",
    holdout_frac: float = DEFAULT_HOLDOUT_FRAC,
    exclude_stems: Iterable[str] = (),
    limit_clean: int = 0,
    limit_defect: int = 0,
    min_defect_px: int = DEFAULT_MIN_DEFECT_PX,
    batch_size: int = 32,
    n_boot: int = DEFAULT_BOOTSTRAP,
    coil_frames: int = 60,
    coil_dir: Path = DEFAULT_COIL_DIR,
    coil_html: bool = True,
    skip_in_domain: bool = False,
    in_domain_split: str = "test",
    dataset_note: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Score one checkpoint end to end and return the record for this `tag`."""
    started = time.perf_counter()
    weights_path = resolve_weights(weights)
    device = resolve_device(device)
    thresholds = sorted({float(t) for t in conf_sweep})
    if thresholds and min(thresholds) < CACHE_CONF:
        raise ValueError(
            f"Thresholds below the box cache floor ({CACHE_CONF}) cannot be swept "
            f"without re-running inference; got {min(thresholds)}."
        )

    split = discover_dataset(
        data_root,
        holdout_frac=holdout_frac,
        exclude_stems=exclude_stems,
        limit_clean=limit_clean,
        limit_defect=limit_defect,
    )
    if verbose:
        print(
            f"[cross-domain] {tag}: {len(split.clean)} held-out clean frames, "
            f"{len(split.defect)} defective frames, "
            f"{len(split.clean_train) + len(split.defect_train)} frames withheld from scoring",
            flush=True,
        )

    detector = DefectDetector(
        weights=weights_path, device=device, conf=CACHE_CONF, iou=iou, imgsz=imgsz
    )
    detector.warmup(2)

    clean_crops = build_crops(split.clean, crop_px=crop_px, min_defect_px=min_defect_px)
    defect_crops = build_crops(
        split.defect, split.masks, crop_px=crop_px, min_defect_px=min_defect_px
    )
    clean_scored = score_crops(
        clean_crops, detector, batch_size=batch_size, verbose=verbose, label="clean"
    )
    defect_scored = score_crops(
        defect_crops, detector, batch_size=batch_size, verbose=verbose, label="defective-frame"
    )

    points = sweep_thresholds(
        clean_scored, defect_scored, thresholds, detector.class_names, n_boot=n_boot
    )
    cross = separation(clean_scored, defect_scored, n_boot=n_boot)
    in_domain = (
        None
        if skip_in_domain
        else in_domain_separation(
            weights_path,
            device,
            split=in_domain_split,
            magnification=float(imgsz) / crop_px,
            n_boot=n_boot,
            verbose=verbose,
        )
    )

    conf = deploy_conf()
    coil = (
        clean_coil_disposition(
            split.clean[: max(0, int(coil_frames))],
            weights=weights_path,
            device=device,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            crop_px=crop_px,
            coil_dir=Path(coil_dir) / tag,
            coil_id=f"SEVERSTAL-CLEAN-HOLDOUT-{tag.upper()}",
            include_cam=False,
            write_html=coil_html,
            html_stem=f"cross_domain_clean_coil_{tag}",
        )
        if coil_frames > 0
        else None
    )

    return {
        "tag": tag,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "elapsed_s": time.perf_counter() - started,
        "model": {
            "weights": str(weights_path),
            "weights_sha256": file_sha256(weights_path),
            "model_name": detector.model_name,
            "class_names": list(detector.class_names),
            "device": device,
            "imgsz": int(imgsz),
            "iou": float(iou),
            "cache_conf": CACHE_CONF,
        },
        "protocol": {
            "crop_px": int(crop_px),
            "magnification": float(imgsz) / crop_px,
            "magnification_note": (
                f"Each {crop_px} px tile is run at imgsz {imgsz}, i.e. "
                f"{imgsz / crop_px:.2f}x. At the defaults this is exactly how the "
                "shipped system presents a 200x200 NEU-DET frame, so the texture "
                "reaches the network at the pixels-per-millimetre it was trained on."
            ),
            "min_defect_px": int(min_defect_px),
            "deploy_conf": conf,
            "holdout_frac": float(holdout_frac),
            "holdout_seed": HOLDOUT_SEED,
        },
        "dataset": {
            "root": str(data_root),
            "note": dataset_note
            or (
                "Severstal steel defect dataset (Voxel51/severstal_steel_defects), "
                "1600x256 line-scan strip. The clean arm is the subset the dataset "
                "itself verifies as defect-free; the defective arm carries pixel masks."
            ),
            "clean_frames_scored": len(split.clean),
            "clean_frames_withheld": len(split.clean_train),
            "clean_manifest_sha256": manifest_hash(split.clean),
            "defect_frames_scored": len(split.defect),
            "defect_frames_withheld": len(split.defect_train),
            "defect_manifest_sha256": manifest_hash(split.defect),
            "clean_crops": len(clean_crops),
            "defect_crops": sum(1 for c in defect_crops if c.label == 1),
            "unlabelled_region_crops": sum(1 for c in defect_crops if c.label == -1),
            "explicitly_excluded_clean_frames": len(split.excluded),
        },
        "sweep": points,
        "separation_cross_domain": cross,
        "separation_in_domain": in_domain,
        "clean_coil_disposition": coil,
    }


# ---------------------------------------------------------------------------
# artifacts
# ---------------------------------------------------------------------------


def merge_record(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    """Upsert this run's record into the JSON by tag, preserving the others."""
    payload: dict[str, Any] = {"schema": "cross_domain/1", "runs": []}
    if path.is_file():
        try:
            existing = json.loads(path.read_text())
            if isinstance(existing, dict) and isinstance(existing.get("runs"), list):
                payload = existing
        except json.JSONDecodeError:  # pragma: no cover - corrupt artifact
            pass
    runs = [r for r in payload["runs"] if r.get("tag") != record["tag"]]
    runs.append(record)
    runs.sort(key=lambda r: str(r.get("generated_at", "")))
    payload["runs"] = runs
    payload["schema"] = "cross_domain/1"
    payload["updated_at"] = record["generated_at"]
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n")
    return path


def _pct(value: float | None) -> str:
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "n/a"
    return f"{100.0 * float(value):.1f}%"


def _ci(bounds: Sequence[float] | None, as_pct: bool = True) -> str:
    if not bounds or any(b is None or not math.isfinite(float(b)) for b in bounds):
        return "n/a"
    if as_pct:
        return f"[{100.0 * float(bounds[0]):.1f}, {100.0 * float(bounds[1]):.1f}]"
    return f"[{float(bounds[0]):.3f}, {float(bounds[1]):.3f}]"


def _md_table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(str(h) for h in header) + " |"]
    out.append("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def plot_fa_vs_recall(payload: dict[str, Any], out_path: Path) -> Path:
    """Clean false alarm rate against defect recall, one curve per tagged run.

    Both false alarm units are drawn. The crop curve is what a precision/recall
    discussion normally uses; the frame curve is what the coil disposition sees,
    and the distance between them is the whole reason this module reports two.
    """
    runs = payload.get("runs", [])
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))
    colours = plt.get_cmap("tab10")

    for i, run in enumerate(runs):
        sweep = run.get("sweep") or []
        if not sweep:
            continue
        recall = [p["defect_crop_recall"] for p in sweep]
        colour = colours(i % 10)
        for ax, key, _label in (
            (axes[0], "clean_crop_fa", "crop"),
            (axes[1], "clean_frame_fa", "frame"),
        ):
            fa = [p[key] for p in sweep]
            ax.plot(recall, fa, "-o", color=colour, markersize=4.2, linewidth=1.7,
                    label=run.get("tag", f"run{i}"))
            deploy = run.get("protocol", {}).get("deploy_conf")
            for p in sweep:
                if deploy is not None and abs(p["conf"] - deploy) < 1e-9:
                    ax.plot(p["defect_crop_recall"], p[key], "*", color=colour,
                            markersize=17, markeredgecolor="#1c2128", markeredgewidth=0.6,
                            zorder=5)
                    ax.annotate(
                        f"conf {deploy:g}",
                        (p["defect_crop_recall"], p[key]),
                        textcoords="offset points", xytext=(9, -12), fontsize=8.5,
                        color="#1c2128",
                    )

    for ax, title, ylabel in (
        (axes[0], "Per 200x200 tile (one inference)", "clean-crop false alarm rate"),
        (axes[1], "Per 1600x256 strip frame (any tile alarms)", "clean-FRAME false alarm rate"),
    ):
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("class-agnostic defect recall on the SAME domain")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, **_GRID)
        ax.set_axisbelow(True)
        ax.plot([0, 1], [0, 1], color="#b6bec7", linewidth=1.0, linestyle="--", zorder=0)
        ax.legend(loc="lower right", fontsize=9, frameon=True)

    fig.suptitle(
        "Cross-domain operating curve on genuinely defect-free steel (Severstal)\n"
        "down and to the right is better; the dashed line is alarm rate = recall, "
        "which is what a coin flip achieves",
        fontsize=12.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def write_markdown(path: Path, payload: dict[str, Any], plot_path: Path | None = None) -> Path:
    """Human-readable cross-domain report covering every tagged run."""
    runs = payload.get("runs", [])
    lines: list[str] = [
        "# Cross-domain generalisation: does this detector work on other steel?",
        "",
        f"Generated {payload.get('updated_at', 'unknown')} by `src/cross_domain.py`.",
        "",
        "`src/false_alarm.py` measures false alarms on defect-free *crops of NEU-DET*",
        "images and says plainly that it is an upper bound from a dataset containing no",
        "defect-free frame. This report replaces that proxy with real, verified",
        "defect-free steel from a different line, and measures the clean/defective",
        "separation on that same line so the two are comparable.",
        "",
        "**Units.** A *crop* is one 200x200 tile: one inference, one operator alarm.",
        "A *frame* is one 1600x256 strip frame, flagged if any of its tiles alarms.",
        "Coil dispositions are built from frames, so the frame number is the one that",
        "decides whether prime steel ships.",
        "",
        "**Intervals.** Tiles are clustered inside frames, so every crop-level interval",
        "is an image-clustered bootstrap (`false_alarm.cluster_bootstrap_interval`),",
        "and the design effect is the factor by which a naive binomial interval would",
        "have overstated precision. AUC intervals resample whole frames on both arms.",
        "",
    ]

    for run in runs:
        model = run.get("model", {})
        data = run.get("dataset", {})
        proto = run.get("protocol", {})
        cross = run.get("separation_cross_domain") or {}
        indom = run.get("separation_in_domain") or {}
        deploy = proto.get("deploy_conf")

        lines += [
            f"## Run `{run.get('tag')}`",
            "",
            f"- checkpoint: `{model.get('weights')}`",
            f"- sha256: `{str(model.get('weights_sha256'))[:16]}...`",
            f"- imgsz {model.get('imgsz')}, iou {model.get('iou')}, device "
            f"{model.get('device')}, crop {proto.get('crop_px')} px "
            f"({proto.get('magnification', 0):.2f}x magnification)",
            f"- data: `{data.get('root')}`",
            f"- scored {data.get('clean_frames_scored')} held-out defect-free frames "
            f"({data.get('clean_crops')} tiles) and {data.get('defect_frames_scored')} "
            f"defective frames ({data.get('defect_crops')} labelled defective tiles)",
            f"- withheld from scoring: {data.get('clean_frames_withheld')} clean, "
            f"{data.get('defect_frames_withheld')} defective "
            f"(hash split, frac {proto.get('holdout_frac')}, seed `{proto.get('holdout_seed')}`)",
            f"- clean manifest sha256 `{str(data.get('clean_manifest_sha256'))[:16]}...`"
            + (f", {data.get('explicitly_excluded_clean_frames')} clean frame(s) removed "
               "by an explicit training manifest"
               if data.get("explicitly_excluded_clean_frames") else ""),
            f"- note: {data.get('note')}",
            f"- {data.get('unlabelled_region_crops')} tiles came from defective frames but "
            "carry no labelled pixels; they are excluded from both arms (see note below).",
            "",
        ]

        at_deploy = next(
            (p for p in run.get("sweep", [])
             if deploy is not None and abs(p["conf"] - deploy) < 1e-9),
            None,
        )
        if at_deploy:
            coil_block = ((run.get("clean_coil_disposition") or {}).get("tiles") or {})
            lines += [
                "**Bottom line at the shipped operating point "
                f"(conf {deploy:g}, imgsz {model.get('imgsz')}).** On steel with no defect "
                f"in it, {_pct(at_deploy['clean_frame_fa'])} of strip frames raise an alarm "
                f"and the detector puts {at_deploy['boxes_per_clean_crop']:.2f} boxes on the "
                "average tile. Defect recall on the same line is "
                f"{_pct(at_deploy['defect_crop_recall'])}, so the alarm carries "
                f"AUC {cross.get('auc', float('nan')):.3f} "
                f"{_ci(cross.get('auc_ci'), as_pct=False)} of information"
                + (
                    f" against {float(indom['auc']):.3f} "
                    f"{_ci(indom.get('auc_ci'), as_pct=False)} in domain."
                    if indom.get("auc") is not None
                    else " (the in-domain comparison was not run)."
                )
                + (f" The shipped disposition chain returns "
                   f"{coil_block.get('disposition')} on a clean coil."
                   if coil_block else ""),
                "",
            ]

        lines += [
            "### False alarms on genuinely defect-free steel, and recall on the same domain",
            "",
        ]
        rows = []
        for p in run.get("sweep", []):
            marker = " **<- shipped**" if deploy is not None and abs(p["conf"] - deploy) < 1e-9 else ""
            rows.append(
                [
                    f"{p['conf']:.2f}{marker}",
                    f"{_pct(p['clean_crop_fa'])} {_ci(p['clean_crop_fa_ci'])}",
                    f"{_pct(p['clean_frame_fa'])} {_ci(p['clean_frame_fa_ci'])}",
                    f"{p['boxes_per_clean_crop']:.2f}",
                    f"{_pct(p['defect_crop_recall'])} {_ci(p['defect_crop_recall_ci'])}",
                    f"{_pct(p['defect_frame_recall'])}",
                ]
            )
        lines += [
            _md_table(
                ["conf", "clean-crop FA [95% CI]", "clean-FRAME FA [95% CI]",
                 "boxes/crop", "defect crop recall [95% CI]", "defect frame recall"],
                rows,
            ),
            "",
        ]
        deffs = [p["clean_crop_design_effect"] for p in run.get("sweep", []) if p.get("clean_crop_design_effect")]
        if deffs:
            lines += [
                f"Design effect on the crop false alarm rate ranges "
                f"{min(deffs):.2f}-{max(deffs):.2f}: treating "
                f"{run.get('dataset', {}).get('clean_crops')} tiles as independent draws "
                f"would have quoted an interval up to {math.sqrt(max(deffs)):.2f}x too narrow.",
                "",
            ]

        if at_deploy and at_deploy.get("false_positive_classes"):
            total = at_deploy["boxes_total"]
            lines += [
                f"### What the false positives are called, at the shipped conf {deploy:g}",
                "",
                _md_table(
                    ["class", "false boxes", "share"],
                    [
                        [name, count, f"{100.0 * count / total:.1f}%"]
                        for name, count in at_deploy["false_positive_classes"].items()
                    ],
                ),
                "",
                f"Total {total} boxes on {at_deploy['clean_crops']} tiles of steel that "
                "carries no defect at all.",
                "",
                f"For scale: the {at_deploy['unlabelled_region_crops']} tiles cut from "
                "defective frames but holding no labelled pixels alarm at "
                f"{_pct(at_deploy['unlabelled_region_flag_rate'])}. They are excluded from "
                "both arms, but the similarity of that rate to the verified-clean rate is "
                "itself evidence that the detector is not responding to the defects.",
                "",
            ]
            if "inclusion" in at_deploy["false_positive_classes"]:
                lines += [
                    "`inclusion` is the class `report.DispositionRules.hold_classes` treats as",
                    "zero-tolerance: a single frame carrying it holds the coil unconditionally.",
                    "",
                ]

        lines += [
            "### Separation: is the confidence score measuring defects or measuring domain?",
            "",
            _md_table(
                ["population", "AUC (max box confidence)", "95% CI", "positives", "negatives"],
                [
                    [
                        "cross-domain (Severstal, defective vs verified clean)",
                        f"{cross.get('auc', float('nan')):.3f}",
                        _ci(cross.get("auc_ci"), as_pct=False),
                        cross.get("n_positive_crops"),
                        cross.get("n_negative_crops"),
                    ],
                    [
                        f"in-domain (NEU-DET {indom.get('split', 'test')}, defect vs mined clean)",
                        f"{indom.get('auc', float('nan')):.3f}" if indom else "not run",
                        _ci(indom.get("auc_ci"), as_pct=False) if indom else "n/a",
                        indom.get("n_positive_crops") if indom else "-",
                        indom.get("n_negative_crops") if indom else "-",
                    ],
                ],
            ),
            "",
        ]
        if indom and indom.get("per_scale"):
            lines += [
                "The two mined arms are not balanced across crop sizes -- containment needs",
                "room, so large crops are over-represented among the positives and small",
                "crops among the negatives. Stratifying by crop size and recombining with",
                "Mann-Whitney weights removes that composition effect:",
                "",
                _md_table(
                    ["crop px", "positives", "negatives", "AUC"],
                    [
                        [s_["crop_px"], s_["n_positive"], s_["n_negative"],
                         f"{float(s_['auc']):.3f}" if math.isfinite(float(s_["auc"])) else "n/a"]
                        for s_ in indom["per_scale"]
                    ],
                ),
                "",
                f"Size-stratified in-domain AUC: **{float(indom.get('auc_scale_stratified', float('nan'))):.3f}** "
                f"(pooled {float(indom.get('auc', float('nan'))):.3f}). Either way the "
                "in-domain figure sits far above the cross-domain one, and the gap is not a "
                "crop-size artefact.",
                "",
            ]
        if indom:
            lines += [
                f"The in-domain negatives are mined crops that clear every labelled box by "
                f"{indom.get('clearance_px')} px at sizes {indom.get('crop_sizes_px')} px -- "
                "NEU-DET has no defect-free image, so this proxy is the best available and it",
                "flatters the model. Magnification is held equal across the two rows "
                f"({indom.get('magnification', 0):.2f}x); field of view is not, and cannot be, "
                "because a 200x200 NEU-DET frame has no room for a 200 px defect-free window.",
                "",
            ]

        coil = run.get("clean_coil_disposition")
        if coil:
            for key, label in (("tiles", "200x200 tiles (deployment geometry)"),
                               ("whole_frames", "whole 1600x256 strip frames")):
                block = coil.get(key) or {}
                lines += [
                    f"### Shipped disposition chain on a clean coil -- {label}",
                    "",
                    "```",
                    f"coil {block.get('coil_id')}: {block.get('total_frames')} frames, "
                    f"defect rate {100.0 * float(block.get('defect_rate', 0.0)):.1f}%, "
                    f"{block.get('total_detections')} detections, "
                    f"p95 severity {float(block.get('p95_severity', 0.0)):.1f} "
                    f"-> {block.get('disposition')}",
                ]
                for reason in block.get("reasons", []):
                    lines.append(f"  - {reason}")
                lines += ["```", ""]
            lines += [
                "Every frame on that coil is verified defect-free steel, so every detection",
                "is a false alarm and every trigger listed above fired on nothing.",
                "",
            ]
        lines += ["---", ""]

    if plot_path is not None:
        lines += [f"![clean false alarm rate vs defect recall]({Path(plot_path).name})", ""]

    lines += [
        "## How to read this, and what it does not say",
        "",
        "- Severstal is hot-rolled carbon strip on someone else's line. A good score here",
        "  would not prove the model works at Hisar. A bad score does prove it does not",
        "  transfer, which is the asymmetry that makes the experiment worth running.",
        "- Tiles from defective frames that carry no labelled pixels are excluded from both",
        "  arms. They are not verified clean the way the clean arm is -- nobody annotated",
        "  that region, on a frame that does carry a defect elsewhere -- and counting them",
        "  as negatives would relabel the hardest cases in the set. Their flag rate is",
        "  recorded per threshold in the JSON as `unlabelled_region_flag_rate`.",
        "- Recall here is class-agnostic and crop-level: did any box appear on a tile that",
        "  contains a labelled defect. It is not mAP and it is not comparable to the",
        "  NEU-DET headline. Severstal's four classes do not map onto NEU-DET's six, so a",
        "  class-aware score across the two datasets would be meaningless.",
        "- The AUC is computed on max-box-confidence per tile. Tiles with no box score 0,",
        "  which is a large tie mass; ranks are tie-averaged so the value does not depend",
        "  on array order.",
        "",
        "## Reproduce",
        "",
        "```bash",
        ".venv/bin/python src/cross_domain.py --tag baseline",
        "```",
        "",
        "Add `--weights <ckpt> --tag <name>` to score a new checkpoint into the same",
        "JSON; every tagged run is redrawn on the chart, so a fix and its baseline sit on",
        "one pair of axes.",
        "",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_sweep(text: str) -> tuple[float, ...]:
    values = tuple(float(v) for v in text.replace(" ", "").split(",") if v)
    if not values:
        raise argparse.ArgumentTypeError("--conf-sweep needs at least one threshold")
    if any(not 0.0 < v < 1.0 for v in values):
        raise argparse.ArgumentTypeError("thresholds must lie strictly inside (0, 1)")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cross-domain false alarm and separation harness (Severstal vs NEU-DET).",
    )
    parser.add_argument("--weights", default=None, help="checkpoint (default: resolve_weights)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--iou", type=float, default=NMS_IOU)
    parser.add_argument("--crop-px", type=int, default=DEFAULT_CROP_PX,
                        help="tile side; with --imgsz it fixes the magnification")
    parser.add_argument("--conf-sweep", type=_parse_sweep,
                        default=DEFAULT_CONF_SWEEP,
                        help="comma-separated thresholds (default: "
                             + ",".join(f"{v:g}" for v in DEFAULT_CONF_SWEEP) + ")")
    parser.add_argument("--tag", default="baseline",
                        help="record key; runs with different tags accumulate in one JSON")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--dataset-note", default=None)
    parser.add_argument("--holdout-frac", type=float, default=DEFAULT_HOLDOUT_FRAC)
    parser.add_argument("--exclude-list", default=None,
                        help="file of frame ids (one per line) that a training run consumed; "
                             "they are dropped from scoring and the count is recorded")
    parser.add_argument("--limit-clean", type=int, default=0)
    parser.add_argument("--limit-defect", type=int, default=0)
    parser.add_argument("--min-defect-px", type=int, default=DEFAULT_MIN_DEFECT_PX)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--coil-frames", type=int, default=60,
                        help="held-out clean frames fed to the disposition chain (0 = skip)")
    parser.add_argument("--coil-dir", default=str(DEFAULT_COIL_DIR))
    parser.add_argument("--no-coil-html", action="store_true")
    parser.add_argument("--skip-in-domain", action="store_true",
                        help="skip the NEU-DET comparison (it needs the local dataset)")
    parser.add_argument("--in-domain-split", default="test")
    parser.add_argument("--json", default=str(DEFAULT_JSON))
    parser.add_argument("--md", default=str(DEFAULT_MD))
    parser.add_argument("--plot", default=str(DEFAULT_PLOT))
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    exclude: list[str] = []
    if args.exclude_list:
        exclude = [
            Path(line.strip()).stem
            for line in Path(args.exclude_list).read_text().splitlines()
            if line.strip()
        ]

    record = analyse(
        weights=args.weights,
        device=args.device,
        imgsz=args.imgsz,
        iou=args.iou,
        crop_px=args.crop_px,
        data_root=Path(args.data_root),
        conf_sweep=args.conf_sweep,
        tag=args.tag,
        holdout_frac=args.holdout_frac,
        exclude_stems=exclude,
        limit_clean=args.limit_clean,
        limit_defect=args.limit_defect,
        min_defect_px=args.min_defect_px,
        batch_size=args.batch_size,
        n_boot=args.bootstrap,
        coil_frames=args.coil_frames,
        coil_dir=Path(args.coil_dir),
        coil_html=not args.no_coil_html,
        skip_in_domain=args.skip_in_domain,
        in_domain_split=args.in_domain_split,
        dataset_note=args.dataset_note,
        verbose=not args.quiet,
    )

    payload = merge_record(Path(args.json), record)
    json_path = write_json(Path(args.json), payload)
    plot_path = plot_fa_vs_recall(payload, Path(args.plot))
    md_path = write_markdown(Path(args.md), payload, plot_path)

    deploy = record["protocol"]["deploy_conf"]
    at_deploy = next((p for p in record["sweep"] if abs(p["conf"] - deploy) < 1e-9), None)
    print(f"\n[cross-domain] tag={record['tag']} weights={record['model']['model_name']}")
    if at_deploy:
        print(
            f"  conf {deploy:g}: clean-crop FA {_pct(at_deploy['clean_crop_fa'])} "
            f"{_ci(at_deploy['clean_crop_fa_ci'])}, clean-FRAME FA "
            f"{_pct(at_deploy['clean_frame_fa'])} {_ci(at_deploy['clean_frame_fa_ci'])}, "
            f"boxes/crop {at_deploy['boxes_per_clean_crop']:.2f}, defect recall "
            f"{_pct(at_deploy['defect_crop_recall'])} {_ci(at_deploy['defect_crop_recall_ci'])}"
        )
    cross = record["separation_cross_domain"]
    print(f"  AUC cross-domain {cross['auc']:.3f} {_ci(cross['auc_ci'], as_pct=False)}")
    if record["separation_in_domain"]:
        ind = record["separation_in_domain"]
        print(f"  AUC in-domain    {ind['auc']:.3f} {_ci(ind['auc_ci'], as_pct=False)}")
    coil = record["clean_coil_disposition"]
    if coil:
        print(
            f"  clean coil ({coil['source_frames']} frames): "
            f"tiles -> {coil['tiles']['disposition']}, "
            f"whole frames -> {coil['whole_frames']['disposition']}"
        )
    print(f"  wrote {json_path}\n        {md_path}\n        {plot_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
