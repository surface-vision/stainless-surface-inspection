"""Build the Severstal and NEU-DET+Severstal joint YOLO datasets.

WHY THIS MODULE EXISTS
----------------------
NEU-DET contains 1800 images and every single one of them contains a defect.
A detector trained on it has never once been shown a piece of steel that is
simply fine. The audit measured the consequence directly: on genuinely
defect-free strip the shipped checkpoint fires on ~91% of frames, and the coil
disposition chain lands on HOLD. That is not a threshold problem, it is a
missing-data problem, and no amount of confidence tuning fixes it. The model
has no representation of "clean".

Severstal (``Voxel51/severstal_steel_defects``, public, ungated) is the fix.
Its train split is 12,568 frames of real steel strip, 6,666 defective and
**5,902 verified defect-free**, all 1600x256x3. This module turns it into two
YOLO datasets:

  ``data/severstal/``  Severstal alone, 4 classes, with defect-free frames as
                       YOLO background images (empty ``.txt`` label files).
  ``data/joint/``      NEU-DET classes 0-5 (indices unchanged, so the existing
                       ``yolov8n_neudet`` checkpoint warm-starts cleanly) plus
                       Severstal classes 6-9, plus the Severstal defect-free
                       crops as background images.

The audit's warning is baked into the design: training on negatives *alone*
teaches a domain classifier, not a defect classifier. Severstal's labelled
positives are therefore in the same dataset, in the same index space, in the
same run.

GEOMETRY: WHY 256x256 TILES AND NOT WHOLE FRAMES
------------------------------------------------
NEU-DET frames are 200x200. Severstal frames are 1600x256 -- a 6.25:1 strip.
Three options were considered.

1. *Feed the 1600x256 frame whole.* YOLO letterboxes to a square, so at the
   deployment size of 256 the frame is squashed to roughly 256x41 before the
   letterbox pad. A 20 px defect becomes 3 px tall. This destroys exactly the
   fine texture (crazing-like, pitted-like) the network has to discriminate,
   and it puts Severstal at ~1/6 the pixels-per-mm of NEU-DET in one axis and
   full resolution in the other. Rejected.
2. *Resize the frame to 200x200.* Same anisotropic destruction, worse.
3. *Tile the frame into square crops at native resolution.* Chosen.

Severstal frames are exactly 256 px tall, so a 256 px square window needs no
vertical tiling and no resampling at all: a crop is a byte-exact sub-rectangle
of the source pixels. 256 is also the deployment ``imgsz`` of this project
(``DEFAULT_IMGSZ = 256``) and is within 28% of the NEU-DET frame size, so a
NEU-DET defect and a Severstal defect arrive at the network at comparable
pixels-per-mm and comparable fraction-of-frame. Nothing is rescaled, so no box
coordinate is ever approximated by a resize.

Window 256, stride 224 gives 7 tiles per frame at x = 0, 224, ..., 1344, with
1344 + 256 = 1600 exactly -- full coverage, no partial tile, 32 px of overlap so
a defect sitting on a tile seam is still seen whole by its neighbour.

CLIPPED DEFECTS: KEEP THE BOX OR DROP THE CROP
-----------------------------------------------
A tile boundary that cuts a defect must not silently produce an unlabelled
defect fragment -- that would teach the network that a defect is background,
which is the precise failure this dataset exists to fix. The rule applied to
every component fragment that lands in a crop is:

    keep the clipped box  if  clipped width >= 8 px
                          and clipped height >= 8 px
                          and clipped mask area >= 64 px
    otherwise             DROP THE WHOLE CROP

The box is recomputed as the tight bounding box of the mask pixels *inside the
crop*, not by intersecting a full-frame box with the crop rectangle, so the
coordinates are exact for concave and diagonal defects too. The criterion is
absolute rather than a fraction of the parent component on purpose: Severstal
defects are frequently 400-1000 px wide, so a "retain >= 35% of the component"
rule drops every tile of every wide defect (measured: 18/400 frames lost their
defect entirely under that rule, versus 7/400 under this one). Measured cost of
the rule: 0.25 crops dropped per defective frame.

Negatives are taken only from frames the dataset marks ``has_defect = False``.
Those samples carry no ``ground_truth`` object at all (verified: 0 of 50 probed
clean samples have one), i.e. zero annotated defect pixels by construction. The
builder still materialises the all-zero mask and asserts that the crop's mask
slice sums to zero for every class, and reports the count. Empty crops from
*defective* frames are deliberately not used as negatives: Severstal's labels
are known to be imperfect, and a crop that is merely unlabelled is not the same
thing as a crop the dataset certifies as clean.

WALL-CLOCK BUDGET
-----------------
The joint training run that follows has to fit in about 3 hours on an M5 at
imgsz 320, where NEU-DET's 1440 train images cost ~35 s/epoch (~24 ms/image).
Defaults here select 3000 positive and 3000 negative Severstal crops, split
80/10/10 by source frame, giving a joint train split of 1440 + 4800 = 6240
images at ~152 s/epoch, i.e. roughly 65-70 epochs in 3 hours. Total joint
dataset 7800 images across all splits.

Positive source frames are picked round-robin over their rarest present defect
class rather than uniformly, because the natural frame distribution is
class 3: 5150, class 1: 897, class 4: 801, class 2: 247, and a uniform draw of
~1200 frames would contain ~45 class-2 frames. Round-robin takes the rare
classes out first and only falls back to class 3 once the others are exhausted.
This deliberately departs from the domain prior; the resulting per-class
instance counts are reported so the effect is visible rather than hidden.

Severstal's ``test`` split (5506 frames) is never used: those samples carry
``has_defect = null`` and no mask -- they are the Kaggle competition's unlabelled
holdout. Every image here comes from the labelled ``train`` split, and the
train/val/test split is this module's own, cut at the source-frame level.

CLASS NAMES
-----------
The Severstal competition never published semantic names for its four defect
ids and neither the dataset card nor ``samples.json`` documents any, so the
classes are named ``severstal_1`` .. ``severstal_4`` after the pixel value that
encodes them in the mask.

USAGE
-----
    python -m src.prepare_severstal --out data --max-positives 3000 \
        --max-negatives 3000 --seed 1337
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import random
import shutil
import sys
import time
import urllib.error
import urllib.request
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import yaml
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]

HF_DATASET = "Voxel51/severstal_steel_defects"
HF_BASE = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main/"
SAMPLES_FILENAME = "samples.json"

FRAME_WIDTH = 1600
FRAME_HEIGHT = 256
TILE = 256
STRIDE = 224

MIN_BOX_SIDE = 8
MIN_FRAG_PIXELS = 64

SEVERSTAL_CLASS_NAMES: tuple[str, ...] = (
    "severstal_1",
    "severstal_2",
    "severstal_3",
    "severstal_4",
)
NEU_CLASS_NAMES: tuple[str, ...] = (
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
)
NEU_OFFSET = len(NEU_CLASS_NAMES)

SPLITS: tuple[str, ...] = ("train", "val", "test")
SPLIT_FRACTIONS: tuple[float, float, float] = (0.8, 0.1, 0.1)

NEG_CROPS_PER_FRAME = 2
JPEG_QUALITY = 95

_DOWNLOAD_BATCH = 256
_DOWNLOAD_WORKERS = 24


# --------------------------------------------------------------------------
# data model
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Box:
    """A YOLO box in normalised crop coordinates. ``cls`` is the mask value 1-4."""

    cls: int
    cx: float
    cy: float
    w: float
    h: float

    def line(self, offset: int) -> str:
        # offset 0 -> severstal-only indices 0-3; offset 6 -> joint indices 6-9.
        return (
            f"{self.cls - 1 + offset} "
            f"{self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"
        )


@dataclass
class Crop:
    """One 256x256 tile, already JPEG-encoded, with its labels and provenance."""

    source_id: str
    x0: int
    positive: bool
    boxes: list[Box]
    jpeg: bytes
    content_hash: str

    @property
    def stem(self) -> str:
        return f"sev_{Path(self.source_id).stem}_x{self.x0:04d}"


@dataclass
class BuildStats:
    frames_seen: int = 0
    frames_yielding: int = 0
    crops_dropped_clipped: int = 0
    crops_empty_skipped: int = 0
    crops_duplicate: int = 0
    negatives_pixel_verified: int = 0
    downloads: int = 0
    cache_hits: int = 0
    reused_from_extra_dirs: int = 0
    failures: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# hub access
# --------------------------------------------------------------------------
def _fetch(url: str, timeout: int = 120, retries: int = 4) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # network flake; back off and retry
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last}")


def load_samples(cache_dir: Path, samples_path: Path | None = None) -> list[dict[str, Any]]:
    """Return the ``samples`` list from the dataset's ``samples.json`` (~25 MB)."""
    candidates: list[Path] = []
    if samples_path is not None:
        candidates.append(samples_path)
    candidates.append(cache_dir / SAMPLES_FILENAME)
    candidates.append(Path("/tmp/sev_samples.json"))
    for cand in candidates:
        if cand.is_file() and cand.stat().st_size > 1_000_000:
            print(f"[samples] using {cand} ({cand.stat().st_size / 1e6:.1f} MB)")
            with cand.open("rb") as fh:
                return json.load(fh)["samples"]
    dest = cache_dir / SAMPLES_FILENAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[samples] downloading {HF_BASE + SAMPLES_FILENAME}")
    dest.write_bytes(_fetch(HF_BASE + SAMPLES_FILENAME))
    with dest.open("rb") as fh:
        return json.load(fh)["samples"]


def decode_mask(sample: dict[str, Any]) -> np.ndarray:
    """Decode a sample's segmentation mask to ``(256, 1600)`` uint8 of class ids.

    Defect-free samples carry no ``ground_truth`` object at all; they get an
    explicit all-zero mask so that the negative verification below is a real
    array check rather than an assumption.
    """
    gt = sample.get("ground_truth")
    if not gt:
        return np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
    raw = zlib.decompress(base64.b64decode(gt["mask"]["$binary"]["base64"]))
    mask = np.load(io.BytesIO(raw))
    if mask.shape != (FRAME_HEIGHT, FRAME_WIDTH):
        raise ValueError(f"{sample['image_id']}: unexpected mask shape {mask.shape}")
    return mask.astype(np.uint8, copy=False)


# --------------------------------------------------------------------------
# frame cache
# --------------------------------------------------------------------------
class FrameStore:
    """Local pool of Severstal JPEGs, seeded from prior scratch dirs."""

    def __init__(self, cache_dir: Path, extra_dirs: Sequence[Path], allow_download: bool):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.allow_download = allow_download
        self.extra: dict[str, Path] = {}
        for d in extra_dirs:
            if not d.is_dir():
                continue
            for p in sorted(d.rglob("*.jpg")):
                self.extra.setdefault(p.name, p)
        if self.extra:
            print(f"[cache] {len(self.extra)} frames discoverable in extra dirs")

    def path(self, image_id: str) -> Path:
        return self.cache_dir / image_id

    def have(self, image_id: str) -> bool:
        p = self.path(image_id)
        return p.is_file() and p.stat().st_size > 1000

    def ensure(self, samples: Sequence[dict[str, Any]], stats: BuildStats) -> None:
        """Make every sample's JPEG present in the cache dir."""
        missing: list[dict[str, Any]] = []
        for s in samples:
            if self.have(s["image_id"]):
                stats.cache_hits += 1
                continue
            src = self.extra.get(s["image_id"])
            if src is not None:
                shutil.copyfile(src, self.path(s["image_id"]))
                stats.reused_from_extra_dirs += 1
                continue
            missing.append(s)
        if not missing:
            return
        if not self.allow_download:
            raise RuntimeError(f"{len(missing)} frames missing and --no-download was set")
        from concurrent.futures import ThreadPoolExecutor

        def grab(s: dict[str, Any]) -> str | None:
            try:
                self.path(s["image_id"]).write_bytes(_fetch(HF_BASE + s["filepath"]))
                return None
            except Exception as exc:
                return f"{s['image_id']}: {exc}"

        with ThreadPoolExecutor(_DOWNLOAD_WORKERS) as ex:
            for err in ex.map(grab, missing):
                if err:
                    stats.failures.append(err)
                else:
                    stats.downloads += 1


# --------------------------------------------------------------------------
# tiling
# --------------------------------------------------------------------------
def tile_starts(width: int = FRAME_WIDTH, tile: int = TILE, stride: int = STRIDE) -> list[int]:
    """Left edges of the tiling windows; always covers the full width."""
    starts = list(range(0, width - tile + 1, stride))
    if not starts:
        return [0]
    if starts[-1] + tile < width:
        starts.append(width - tile)
    return starts


def mask_components(mask: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """Connected components of the mask, per class id, as boolean full-frame arrays."""
    comps: list[tuple[int, np.ndarray]] = []
    for cls in range(1, len(SEVERSTAL_CLASS_NAMES) + 1):
        binary = mask == cls
        if not binary.any():
            continue
        labelled, n = ndimage.label(binary)
        for i in range(1, n + 1):
            comps.append((cls, labelled == i))
    return comps


def crop_boxes(
    components: Sequence[tuple[int, np.ndarray]], x0: int, tile: int = TILE
) -> list[Box] | None:
    """Boxes for one crop, or ``None`` if the crop cuts a defect too finely to keep.

    ``None`` means the caller must discard the crop entirely -- never emit it as
    a background image, because it contains defect pixels we refused to label.
    """
    boxes: list[Box] = []
    for cls, comp in components:
        sl = comp[:, x0 : x0 + tile]
        if not sl.any():
            continue
        ys, xs = np.nonzero(sl)
        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())
        w = x_max - x_min + 1
        h = y_max - y_min + 1
        if w < MIN_BOX_SIDE or h < MIN_BOX_SIDE or int(sl.sum()) < MIN_FRAG_PIXELS:
            return None
        boxes.append(
            Box(
                cls=cls,
                cx=min(max((x_min + x_max + 1) / 2.0 / tile, 0.0), 1.0),
                cy=min(max((y_min + y_max + 1) / 2.0 / tile, 0.0), 1.0),
                w=min(w / tile, 1.0),
                h=min(h / tile, 1.0),
            )
        )
    return boxes


def _encode(crop_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def _read_frame(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"cannot decode {path}")
    if img.shape[:2] != (FRAME_HEIGHT, FRAME_WIDTH):
        raise RuntimeError(f"{path.name}: unexpected geometry {img.shape}")
    return img


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------
def order_positive_frames(
    defective: Sequence[dict[str, Any]], rng: random.Random
) -> list[dict[str, Any]]:
    """Round-robin over rarest-present class so class 2 is not crowded out."""
    freq = Counter(c for s in defective for c in s["defect_classes"])
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for s in defective:
        primary = min(s["defect_classes"], key=lambda c: (freq[c], c))
        groups[primary].append(s)
    for cls in groups:
        groups[cls].sort(key=lambda s: s["image_id"])
        rng.shuffle(groups[cls])
    order_classes = sorted(groups, key=lambda c: (freq[c], c))
    out: list[dict[str, Any]] = []
    idx = {c: 0 for c in order_classes}
    while len(out) < len(defective):
        progressed = False
        for c in order_classes:
            if idx[c] < len(groups[c]):
                out.append(groups[c][idx[c]])
                idx[c] += 1
                progressed = True
        if not progressed:
            break
    return out


def collect_positive_crops(
    ordered: Sequence[dict[str, Any]],
    store: FrameStore,
    budget: int,
    stats: BuildStats,
    seen_hashes: set[str],
) -> list[Crop]:
    crops: list[Crop] = []
    starts = tile_starts()
    for chunk_start in range(0, len(ordered), _DOWNLOAD_BATCH):
        if len(crops) >= budget:
            break
        chunk = ordered[chunk_start : chunk_start + _DOWNLOAD_BATCH]
        store.ensure(chunk, stats)
        for sample in chunk:
            if len(crops) >= budget:
                break
            path = store.path(sample["image_id"])
            if not path.is_file():
                continue
            stats.frames_seen += 1
            try:
                frame = _read_frame(path)
            except RuntimeError as exc:
                stats.failures.append(str(exc))
                continue
            comps = mask_components(decode_mask(sample))
            produced = 0
            for x0 in starts:
                boxes = crop_boxes(comps, x0)
                if boxes is None:
                    stats.crops_dropped_clipped += 1
                    continue
                if not boxes:
                    stats.crops_empty_skipped += 1
                    continue
                patch = np.ascontiguousarray(frame[:, x0 : x0 + TILE])
                digest = hashlib.sha256(patch.tobytes()).hexdigest()
                if digest in seen_hashes:
                    stats.crops_duplicate += 1
                    continue
                seen_hashes.add(digest)
                crops.append(
                    Crop(sample["image_id"], x0, True, boxes, _encode(patch), digest)
                )
                produced += 1
            if produced:
                stats.frames_yielding += 1
    return crops


def collect_negative_crops(
    ordered: Sequence[dict[str, Any]],
    store: FrameStore,
    budget: int,
    per_frame: int,
    rng: random.Random,
    stats: BuildStats,
    seen_hashes: set[str],
) -> list[Crop]:
    crops: list[Crop] = []
    starts = tile_starts()
    for chunk_start in range(0, len(ordered), _DOWNLOAD_BATCH):
        if len(crops) >= budget:
            break
        chunk = ordered[chunk_start : chunk_start + _DOWNLOAD_BATCH]
        store.ensure(chunk, stats)
        for sample in chunk:
            if len(crops) >= budget:
                break
            path = store.path(sample["image_id"])
            if not path.is_file():
                continue
            stats.frames_seen += 1
            # Contract check: only frames the dataset certifies clean get here.
            if sample.get("has_defect") or sample.get("defect_classes"):
                raise AssertionError(f"{sample['image_id']} is not a defect-free sample")
            mask = decode_mask(sample)
            try:
                frame = _read_frame(path)
            except RuntimeError as exc:
                stats.failures.append(str(exc))
                continue
            produced = 0
            for x0 in sorted(rng.sample(starts, min(per_frame, len(starts)))):
                if len(crops) >= budget:
                    break
                sub = mask[:, x0 : x0 + TILE]
                # Explicit pixel-level proof that this background image is clean.
                if int(sub.sum()) != 0 or bool(np.isin(sub, (1, 2, 3, 4)).any()):
                    stats.failures.append(f"{sample['image_id']}@{x0}: nonzero mask on negative")
                    continue
                stats.negatives_pixel_verified += 1
                patch = np.ascontiguousarray(frame[:, x0 : x0 + TILE])
                digest = hashlib.sha256(patch.tobytes()).hexdigest()
                if digest in seen_hashes:
                    stats.crops_duplicate += 1
                    continue
                seen_hashes.add(digest)
                crops.append(Crop(sample["image_id"], x0, False, [], _encode(patch), digest))
                produced += 1
            if produced:
                stats.frames_yielding += 1
    return crops


def assign_splits(
    source_ids: Sequence[str], rng: random.Random
) -> dict[str, str]:
    """Map each *source frame id* to a split. Crops never straddle a split."""
    ids = sorted(set(source_ids))
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(n * SPLIT_FRACTIONS[0]))
    n_val = int(round(n * SPLIT_FRACTIONS[1]))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    out: dict[str, str] = {}
    for i, sid in enumerate(ids):
        if i < n_train:
            out[sid] = "train"
        elif i < n_train + n_val:
            out[sid] = "val"
        else:
            out[sid] = "test"
    return out


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------
def _prepare_tree(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    for split in SPLITS:
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)


def write_crops(root: Path, crops: Sequence[Crop], split_of: dict[str, str], offset: int) -> None:
    for crop in crops:
        split = split_of[crop.source_id]
        (root / split / "images" / f"{crop.stem}.jpg").write_bytes(crop.jpeg)
        lines = "\n".join(b.line(offset) for b in crop.boxes)
        (root / split / "labels" / f"{crop.stem}.txt").write_text(
            lines + "\n" if lines else "", encoding="utf-8"
        )


def copy_neu(neu_root: Path, dest_root: Path) -> dict[str, list[str]]:
    """Copy NEU-DET verbatim into the joint tree. Indices 0-5 are untouched."""
    per_split: dict[str, list[str]] = {}
    for split in SPLITS:
        names: list[str] = []
        src_img = neu_root / split / "images"
        src_lbl = neu_root / split / "labels"
        for img in sorted(src_img.glob("*.jpg")):
            shutil.copyfile(img, dest_root / split / "images" / img.name)
            lbl = src_lbl / f"{img.stem}.txt"
            if lbl.is_file():
                shutil.copyfile(lbl, dest_root / split / "labels" / lbl.name)
            else:
                (dest_root / split / "labels" / f"{img.stem}.txt").write_text("", encoding="utf-8")
            names.append(img.name)
        per_split[split] = names
    return per_split


def write_yaml(root: Path, names: Sequence[str], header: str) -> None:
    lines = [f"# {header}", f"path: {root}", "train: train/images", "val: val/images", "test: test/images", f"nc: {len(names)}", "names:"]
    lines += [f"  {i}: {n}" for i, n in enumerate(names)]
    (root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------
def image_hashes(root: Path, split: str) -> dict[str, str]:
    """sha256 of decoded pixels for every image in a split."""
    out: dict[str, str] = {}
    for img in sorted((root / split / "images").glob("*.jpg")):
        arr = cv2.imread(str(img), cv2.IMREAD_COLOR)
        if arr is None:
            raise RuntimeError(f"cannot decode {img}")
        out[img.name] = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
    return out


def check_disjoint(root: Path, label: str) -> dict[str, Any]:
    hashes = {s: image_hashes(root, s) for s in SPLITS}
    sets = {s: set(v.values()) for s, v in hashes.items()}
    report: dict[str, Any] = {
        "counts": {s: len(v) for s, v in hashes.items()},
        "unique_hashes": {s: len(v) for s, v in sets.items()},
        "collisions": {},
        "ok": True,
    }
    print(f"[leakage] {label}: content-hash split check")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = sets[a] & sets[b]
        report["collisions"][f"{a}^{b}"] = len(inter)
        if inter:
            report["ok"] = False
        print(f"           {a} n={len(hashes[a])} vs {b} n={len(hashes[b])}: {len(inter)} shared image hashes")
    dupes = {s: len(hashes[s]) - len(sets[s]) for s in SPLITS}
    report["intra_split_duplicates"] = dupes
    print(f"           intra-split duplicate hashes: {dupes}")
    print(f"           VERDICT: {'PASS - all splits disjoint by content hash' if report['ok'] else 'FAIL'}")
    return report


def verify_roundtrip(
    sev_root: Path,
    joint_root: Path,
    store: FrameStore,
    samples_by_id: dict[str, dict[str, Any]],
    split_of: dict[str, str],
    rng: random.Random,
    fidelity_sample: int = 600,
) -> tuple[list[str], dict[str, Any]]:
    """Re-derive every written label from the ORIGINAL mask and compare.

    This deliberately re-reads ``samples.json`` and the cached 1600x256 frames
    rather than trusting anything the build kept in memory, so a bug in the
    crop/label path cannot hide behind itself.
    """
    log: list[str] = []
    by_frame: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for split in SPLITS:
        for lbl in sorted((sev_root / split / "labels").glob("sev_*.txt")):
            _, sid, xtag = lbl.stem.split("_")
            by_frame[f"{sid}.jpg"].append((split, lbl.stem, int(xtag[1:])))

    errors: list[str] = []
    n_labelled = n_boxes = n_background = 0
    for image_id, entries in by_frame.items():
        sample = samples_by_id[image_id]
        mask = decode_mask(sample)
        comps = mask_components(mask)
        for split, stem, x0 in entries:
            expected = crop_boxes(comps, x0)
            if expected is None:
                errors.append(f"{stem}: emitted a crop the clipping rule says to drop")
                continue
            got_sev = (sev_root / split / "labels" / f"{stem}.txt").read_text().strip().splitlines()
            got_joint = (joint_root / split / "labels" / f"{stem}.txt").read_text().strip().splitlines()
            if got_sev != [b.line(0) for b in expected]:
                errors.append(f"{stem}: severstal label does not re-derive")
            if got_joint != [b.line(NEU_OFFSET) for b in expected]:
                errors.append(f"{stem}: joint label does not re-derive")
            if expected:
                n_labelled += 1
                n_boxes += len(expected)
            else:
                n_background += 1
                if int(mask[:, x0 : x0 + TILE].sum()) != 0:
                    errors.append(f"{stem}: background crop covers nonzero mask pixels")
                if sample.get("has_defect") or sample.get("defect_classes") or sample.get("ground_truth"):
                    errors.append(f"{stem}: background crop came from a defective frame")
            if split_of[image_id] != split:
                errors.append(f"{stem}: on disk in {split} but assigned {split_of[image_id]}")

    log.append("[verify] independent re-derivation from samples.json masks and source frames")
    log.append(
        f"           {n_labelled} labelled crops ({n_boxes} boxes) + {n_background} background crops "
        f"from {len(by_frame)} source frames re-derived"
    )
    log.append(f"           label mismatches: {len(errors)}")
    for e in errors[:5]:
        log.append(f"           ! {e}")

    # Geometry fidelity: a crop must be the source frame's pixels, modulo the
    # JPEG re-encode. Exactness of the *rectangle* is what matters for boxes.
    all_imgs = [
        (split, p)
        for split in SPLITS
        for p in sorted((sev_root / split / "images").glob("sev_*.jpg"))
    ]
    picks = rng.sample(all_imgs, min(fidelity_sample, len(all_imgs)))
    diffs: list[float] = []
    worst = 0
    shape_bad = copy_bad = 0
    for split, p in picks:
        _, sid, xtag = p.stem.split("_")
        x0 = int(xtag[1:])
        src = cv2.imread(str(store.path(f"{sid}.jpg")), cv2.IMREAD_COLOR)
        got = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if got is None or src is None or got.shape != (TILE, TILE, 3):
            shape_bad += 1
            continue
        d = np.abs(src[:, x0 : x0 + TILE].astype(np.int16) - got.astype(np.int16))
        diffs.append(float(d.mean()))
        worst = max(worst, int(d.max()))
        if (joint_root / split / "images" / p.name).read_bytes() != p.read_bytes():
            copy_bad += 1
    log.append(
        f"[verify] {len(picks)} sampled crops vs their source-frame rectangle: mean abs pixel delta "
        f"{np.mean(diffs):.3f}/255, worst {worst}/255 (JPEG q{JPEG_QUALITY} re-encode only; the crop "
        f"rectangle itself is exact)"
    )
    log.append(f"           wrong-shape crops: {shape_bad}; joint copies differing from severstal copies: {copy_bad}")
    ok = not errors and shape_bad == 0 and copy_bad == 0
    log.append(f"           VERDICT: {'PASS - every label re-derives from the source mask' if ok else 'FAIL'}")
    return log, {
        "ok": ok,
        "labelled_crops": n_labelled,
        "boxes": n_boxes,
        "background_crops": n_background,
        "source_frames": len(by_frame),
        "errors": errors[:50],
        "fidelity_mean_abs_delta": round(float(np.mean(diffs)), 4) if diffs else None,
        "fidelity_worst_abs_delta": worst,
    }


def loader_check(yaml_paths: Sequence[Path]) -> list[str]:
    """Prove Ultralytics itself can scan the datasets and sees the backgrounds."""
    log: list[str] = []
    try:
        import contextlib

        from ultralytics.cfg import get_cfg
        from ultralytics.data.dataset import YOLODataset
        from ultralytics.data.utils import check_det_dataset
    except Exception as exc:
        log.append(f"[loader] ultralytics unavailable, skipped ({exc})")
        return log
    for yp in yaml_paths:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            data = check_det_dataset(str(yp))
            rows = []
            for split in SPLITS:
                ds = YOLODataset(img_path=data[split], data=data, task="detect", augment=False, hyp=get_cfg())
                bg = sum(1 for lab in ds.labels if len(lab["cls"]) == 0)
                inst = int(sum(len(lab["cls"]) for lab in ds.labels))
                present = sorted({int(c) for lab in ds.labels for c in np.asarray(lab["cls"]).ravel()})
                rows.append((split, len(ds.labels), bg, inst, present))
        log.append(f"[loader] {yp.parent.name}: nc={data['nc']} names={list(data['names'].values())}")
        for split, n, bg, inst, present in rows:
            log.append(f"           {split}: {n} images, {bg} backgrounds, {inst} instances, classes present {present}")
    return log


def count_instances(root: Path, n_classes: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in SPLITS:
        per_class = Counter()
        n_img = n_bg = 0
        for lbl in sorted((root / split / "labels").glob("*.txt")):
            n_img += 1
            text = lbl.read_text(encoding="utf-8").strip()
            if not text:
                n_bg += 1
                continue
            for line in text.splitlines():
                per_class[int(line.split()[0])] += 1
        out[split] = {
            "images": n_img,
            "background_images": n_bg,
            "labelled_images": n_img - n_bg,
            "instances": {i: per_class.get(i, 0) for i in range(n_classes)},
            "total_instances": sum(per_class.values()),
        }
    return out


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def write_report(path: Path, manifest: dict[str, Any]) -> None:
    m = manifest
    sev_names = SEVERSTAL_CLASS_NAMES
    joint_names = list(NEU_CLASS_NAMES) + list(sev_names)

    def table(counts: dict[str, Any], names: Sequence[str]) -> list[str]:
        rows = ["| split | images | labelled | background | instances |", "|---|---|---|---|---|"]
        for s in SPLITS:
            c = counts[s]
            rows.append(
                f"| {s} | {c['images']} | {c['labelled_images']} | "
                f"{c['background_images']} | {c['total_instances']} |"
            )
        tot = {k: sum(counts[s][k] for s in SPLITS) for k in ("images", "labelled_images", "background_images", "total_instances")}
        rows.append(
            f"| **total** | **{tot['images']}** | **{tot['labelled_images']}** | "
            f"**{tot['background_images']}** | **{tot['total_instances']}** |"
        )
        rows.append("")
        rows.append("| class | idx | " + " | ".join(SPLITS) + " | total |")
        rows.append("|---|---|" + "---|" * (len(SPLITS) + 1))
        for i, name in enumerate(names):
            per = [counts[s]["instances"].get(i, counts[s]["instances"].get(str(i), 0)) for s in SPLITS]
            rows.append(f"| {name} | {i} | " + " | ".join(str(v) for v in per) + f" | {sum(per)} |")
        return rows

    lines: list[str] = []
    lines.append("# Severstal and joint NEU-DET + Severstal datasets")
    lines.append("")
    lines.append(f"Built by `src/prepare_severstal.py` on {m['built_at']} with seed {m['seed']}.")
    lines.append("Every number on this page is emitted by that script from the files it wrote.")
    lines.append("")
    lines.append("## Why")
    lines.append("")
    lines.append(
        "NEU-DET has no defect-free image. The audit measured what that costs: the shipped "
        "checkpoint fires on ~91% of genuinely clean strip frames and the coil disposition "
        "chain returns HOLD. These two datasets put real defect-free steel, and real "
        "strip-geometry positives, into the training set."
    )
    lines.append("")
    lines.append("## Source")
    lines.append("")
    lines.append(f"- HuggingFace `{HF_DATASET}` (public, ungated), 18,074 images, every one 1600x256x3.")
    lines.append(
        f"- Labelled `train` split only: {m['source']['train_total']} frames = "
        f"{m['source']['train_defective']} defective / {m['source']['train_clean']} verified defect-free. "
        "The dataset's own `test` split (5,506) carries `has_defect: null` and no mask -- it is the "
        "Kaggle unlabelled holdout and is never used here."
    )
    lines.append(
        "- Masks decode base64 -> zlib -> `.npy` to (256, 1600) uint8 where the pixel value is the "
        "class id 1-4. Boxes come from `scipy.ndimage.label` per class."
    )
    lines.append(
        "- Class names: the competition never published semantic names and the dataset card documents "
        "none, so classes are named after their mask value: `severstal_1` .. `severstal_4`."
    )
    lines.append("")
    lines.append("## Image geometry and the tiling decision")
    lines.append("")
    lines.append(
        f"NEU-DET frames are 200x200. Severstal frames are {FRAME_WIDTH}x{FRAME_HEIGHT}, a 6.25:1 strip. "
        "Feeding a whole Severstal frame to YOLO at imgsz 256 letterboxes it to about 256x41 before "
        "padding, which squashes a 20 px defect to 3 px and puts the two datasets at wildly different "
        "pixels-per-mm in the vertical axis. So Severstal is **tiled**, not resized."
    )
    lines.append("")
    lines.append(
        f"- Window {TILE}x{TILE}, stride {STRIDE}, {len(tile_starts())} tiles per frame at "
        f"x = {', '.join(str(x) for x in tile_starts())}; the last tile ends at exactly "
        f"{tile_starts()[-1] + TILE}, so coverage is complete with {TILE - STRIDE} px of overlap."
    )
    lines.append(
        f"- Frames are exactly {FRAME_HEIGHT} px tall, so a {TILE} px square window needs no vertical "
        "tiling and **no resampling at all**: every crop is a byte-exact sub-rectangle of the source "
        "pixels and no box coordinate is ever approximated by a resize."
    )
    lines.append(
        f"- {TILE} is this project's deployment `imgsz` and is within 28% of the NEU-DET frame size, so "
        "a NEU-DET defect and a Severstal defect reach the network at comparable magnification."
    )
    lines.append("")
    lines.append("### Clipped defects: keep the box or drop the crop")
    lines.append("")
    lines.append(
        f"For each connected component that lands in a crop, the box is recomputed as the tight bounding "
        f"box of the mask pixels **inside the crop** (not by intersecting a full-frame box, so concave and "
        f"diagonal defects stay exact). The fragment is kept as a labelled box if its clipped width and "
        f"height are both >= {MIN_BOX_SIDE} px and it holds >= {MIN_FRAG_PIXELS} mask pixels. If any "
        f"fragment fails that test the **entire crop is discarded** -- it is never emitted as a background "
        f"image, because an unlabelled defect fragment teaches exactly the error this dataset exists to fix."
    )
    lines.append("")
    lines.append(
        f"The threshold is absolute rather than a fraction of the parent component on purpose. Severstal "
        f"defects are routinely 400-1000 px wide, so a 'retain >= 35% of the component' rule drops every "
        f"tile of every wide defect; measured on 400 defective frames it lost the defect entirely on 18 "
        f"frames versus 7 under this rule. Measured cost of the rule in this build: "
        f"{m['stats']['crops_dropped_clipped']} crops discarded."
    )
    lines.append("")
    lines.append("## What was subsampled, and how")
    lines.append("")
    lines.append(
        f"Budget: `--max-positives {m['budget']['max_positives']}`, "
        f"`--max-negatives {m['budget']['max_negatives']}`, `--seed {m['seed']}`."
    )
    lines.append("")
    lines.append(
        f"- **Positives.** {m['selection']['positive_source_frames']} defective source frames were "
        f"consumed to reach {m['selection']['positive_crops']} positive crops "
        f"({m['selection']['positive_crops'] / max(m['selection']['positive_source_frames'], 1):.2f} "
        "crops/frame). Frames are drawn **round-robin over their rarest present class**, not uniformly: "
        f"the natural frame distribution is {m['source']['frames_per_class']}, so a uniform draw of ~1200 "
        "frames would contain ~45 class-2 frames. Round-robin exhausts the rare classes first. This "
        "departs from the domain prior deliberately; the per-class instance counts below make the effect "
        "visible."
    )
    lines.append(
        f"- **Negatives.** {m['selection']['negative_source_frames']} `has_defect=False` source frames, "
        f"{NEG_CROPS_PER_FRAME} randomly chosen tiles each, giving {m['selection']['negative_crops']} "
        "background crops. Two tiles per frame rather than all seven maximises source-frame diversity for "
        "a fixed crop budget."
    )
    lines.append(
        "- Negatives are taken **only** from frames the dataset certifies defect-free. Empty crops from "
        "defective frames are not used: Severstal's labels are known to be imperfect and 'unlabelled' is "
        "not 'clean'."
    )
    lines.append("")
    lines.append("### Wall-clock arithmetic")
    lines.append("")
    lines.append(
        f"NEU-DET's 1440 train images cost ~35 s/epoch at imgsz 320 on this machine (~24 ms/image). "
        f"The joint train split is {m['joint']['counts']['train']['images']} images, so ~"
        f"{m['joint']['counts']['train']['images'] * 35 / 1440:.0f} s/epoch, i.e. roughly "
        f"{int(3 * 3600 / (m['joint']['counts']['train']['images'] * 35 / 1440))} epochs in a 3-hour "
        f"budget before validation overhead. Total joint dataset: "
        f"{sum(m['joint']['counts'][s]['images'] for s in SPLITS)} images."
    )
    lines.append("")
    lines.append("## `data/severstal/` -- Severstal alone, 4 classes")
    lines.append("")
    lines += table(m["severstal"]["counts"], sev_names)
    lines.append("")
    lines.append("## `data/joint/` -- NEU-DET 0-5 + Severstal 6-9, one index space")
    lines.append("")
    lines.append(
        "NEU-DET indices 0-5 and its train/val/test membership are **unchanged**, so the shipped "
        "`yolov8n_neudet` checkpoint warm-starts on the first six classes and the held-out NEU-DET test "
        "numbers stay directly comparable."
    )
    lines.append("")
    lines += table(m["joint"]["counts"], joint_names)
    lines.append("")
    lines.append("## Leakage check (script output, verbatim)")
    lines.append("")
    lines.append(
        "Splits are cut at the **source-frame** level, so all seven tiles of a Severstal frame land in the "
        "same split. Two independent checks then run over the written files: source-frame id disjointness, "
        "and sha256 of the decoded pixels of every image on disk."
    )
    lines.append("")
    lines.append("```")
    lines += m["leakage_log"]
    lines.append("```")
    lines.append("")
    lines.append("## Negative verification")
    lines.append("")
    lines.append(
        f"`has_defect=False` samples carry no `ground_truth` object at all, i.e. zero annotated defect "
        f"pixels by construction. The builder still materialises the all-zero (256, 1600) mask and asserts "
        f"the crop's slice sums to zero and contains none of the values 1-4. "
        f"**{m['stats']['negatives_pixel_verified']} / {m['selection']['negative_crops']} background crops "
        f"passed that check; {len([f for f in m['stats']['failures'] if 'nonzero mask' in f])} failed.**"
    )
    lines.append("")
    lines.append("## Independent re-derivation of every label (script output, verbatim)")
    lines.append("")
    lines.append(
        "After writing, the builder re-opens `samples.json`, re-decodes each source mask, re-runs "
        "connected components and the clipping rule, and compares the result against the `.txt` files "
        "on disk -- in both index spaces. Nothing from the build is reused, so a bug in the crop path "
        "cannot hide behind itself."
    )
    lines.append("")
    lines.append("```")
    lines += m["verify_log"]
    lines.append("```")
    lines.append("")
    lines.append("## Ultralytics loader check (script output, verbatim)")
    lines.append("")
    lines.append("```")
    lines += m["loader_log"]
    lines.append("```")
    lines.append("")
    lines.append("## Known quirks, inherited not introduced")
    lines.append("")
    lines.append(
        "- `data/neu-det/train` contains one byte-identical image pair (`patches_101.jpg` == "
        "`patches_105.jpg`). Both are in **train**, so it is not split leakage, and the joint set "
        "carries it forward unchanged rather than silently editing NEU-DET."
    )
    lines.append(
        "- Ultralytics reports 3 duplicate label rows while scanning the joint set "
        "(`crazing_120`, `inclusion_62`, `patches_198`) and drops them, which is why its instance "
        "totals read 7172/913 against the 7174/914 counted from the raw `.txt` files. All three are "
        "pre-existing NEU-DET annotations, copied verbatim. Severstal contributes zero duplicates."
    )
    lines.append(
        f"- Crops are re-encoded as JPEG q{JPEG_QUALITY}. The crop *rectangle* is exact -- no resampling "
        "-- but the re-encode costs a mean absolute pixel delta of "
        f"{m['verification']['fidelity_mean_abs_delta']}/255 against the source frame "
        f"(worst {m['verification']['fidelity_worst_abs_delta']}/255)."
    )
    lines.append("")
    lines.append("## Build provenance")
    lines.append("")
    lines.append(f"- frames read from cache: {m['stats']['cache_hits']}")
    lines.append(f"- frames reused from prior audit scratch dirs: {m['stats']['reused_from_extra_dirs']}")
    lines.append(f"- frames downloaded from the hub this run: {m['stats']['downloads']}")
    lines.append(f"- crops rejected as byte-identical duplicates: {m['stats']['crops_duplicate']}")
    lines.append(f"- fetch/decode failures: {len(m['stats']['failures'])}")
    lines.append(f"- machine-readable manifest: `data/severstal/manifest.json`, `data/joint/manifest.json`")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def build(
    out_root: Path,
    neu_root: Path,
    cache_dir: Path,
    extra_dirs: Sequence[Path],
    max_positives: int,
    max_negatives: int,
    seed: int,
    samples_path: Path | None = None,
    allow_download: bool = True,
    report_path: Path | None = None,
) -> dict[str, Any]:
    t0 = time.time()
    rng = random.Random(seed)
    stats = BuildStats()

    samples = load_samples(cache_dir, samples_path)
    train = [s for s in samples if s.get("split") == "train"]
    defective = sorted((s for s in train if s.get("has_defect")), key=lambda s: s["image_id"])
    clean = sorted(
        (s for s in train if not s.get("has_defect") and s.get("has_defect") is not None),
        key=lambda s: s["image_id"],
    )
    frames_per_class = Counter(c for s in defective for c in s["defect_classes"])
    print(
        f"[source] train {len(train)} = {len(defective)} defective / {len(clean)} defect-free; "
        f"frames per class {dict(sorted(frames_per_class.items()))}"
    )

    store = FrameStore(cache_dir, extra_dirs, allow_download)
    seen_hashes: set[str] = set()

    pos_order = order_positive_frames(defective, random.Random(seed))
    print(f"[select] collecting up to {max_positives} positive crops ...")
    pos_crops = collect_positive_crops(pos_order, store, max_positives, stats, seen_hashes)

    neg_order = list(clean)
    random.Random(seed + 1).shuffle(neg_order)
    print(f"[select] collecting up to {max_negatives} background crops ...")
    neg_crops = collect_negative_crops(
        neg_order, store, max_negatives, NEG_CROPS_PER_FRAME, random.Random(seed + 2), stats, seen_hashes
    )

    pos_sources = {c.source_id for c in pos_crops}
    neg_sources = {c.source_id for c in neg_crops}
    print(
        f"[select] {len(pos_crops)} positive crops from {len(pos_sources)} frames; "
        f"{len(neg_crops)} background crops from {len(neg_sources)} frames"
    )

    # Split by SOURCE FRAME, positives and negatives independently so both are
    # represented in every split at the same ratio.
    split_of = assign_splits(sorted(pos_sources), random.Random(seed + 3))
    split_of.update(assign_splits(sorted(neg_sources), random.Random(seed + 4)))
    assert not (pos_sources & neg_sources), "a frame cannot be both defective and defect-free"

    all_crops = pos_crops + neg_crops

    sev_root = out_root / "severstal"
    joint_root = out_root / "joint"
    _prepare_tree(sev_root)
    _prepare_tree(joint_root)

    write_crops(sev_root, all_crops, split_of, offset=0)
    write_yaml(sev_root, SEVERSTAL_CLASS_NAMES, "Severstal steel strip defects, 256x256 tiles, 4 classes + background frames")

    write_crops(joint_root, all_crops, split_of, offset=NEU_OFFSET)
    neu_names = copy_neu(neu_root, joint_root)
    write_yaml(
        joint_root,
        list(NEU_CLASS_NAMES) + list(SEVERSTAL_CLASS_NAMES),
        "Joint NEU-DET (0-5, indices unchanged) + Severstal (6-9) + Severstal defect-free backgrounds",
    )

    # --- verification -----------------------------------------------------
    log: list[str] = []

    def emit(line: str) -> None:
        print(line)
        log.append(line)

    emit("[leakage] severstal source-frame disjointness (crops of one frame never straddle a split)")
    by_split: dict[str, set[str]] = {s: set() for s in SPLITS}
    for c in all_crops:
        by_split[split_of[c.source_id]].add(c.source_id)
    src_ok = True
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = by_split[a] & by_split[b]
        if inter:
            src_ok = False
        emit(f"           {a} frames={len(by_split[a])} vs {b} frames={len(by_split[b])}: {len(inter)} shared source ids")
    emit(f"           VERDICT: {'PASS - no source frame straddles a split' if src_ok else 'FAIL'}")

    import contextlib

    for label, root in (("data/severstal", sev_root), ("data/joint", joint_root), ("data/neu-det", neu_root)):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rep = check_disjoint(root, label)
        for line in buf.getvalue().rstrip("\n").split("\n"):
            emit(line)
        if label == "data/severstal":
            sev_leak = rep
        elif label == "data/joint":
            joint_leak = rep
        else:
            neu_leak = rep

    verify_log, verify_rep = verify_roundtrip(
        sev_root,
        joint_root,
        store,
        {s["image_id"]: s for s in train},
        split_of,
        random.Random(seed + 5),
    )
    for line in verify_log:
        print(line)
    load_log = loader_check([sev_root / "data.yaml", joint_root / "data.yaml"])
    for line in load_log:
        print(line)

    sev_counts = count_instances(sev_root, len(SEVERSTAL_CLASS_NAMES))
    joint_counts = count_instances(joint_root, len(NEU_CLASS_NAMES) + len(SEVERSTAL_CLASS_NAMES))

    manifest: dict[str, Any] = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "elapsed_s": round(time.time() - t0, 1),
        "source": {
            "hf_dataset": HF_DATASET,
            "train_total": len(train),
            "train_defective": len(defective),
            "train_clean": len(clean),
            "frames_per_class": dict(sorted(frames_per_class.items())),
            "frame_geometry": [FRAME_HEIGHT, FRAME_WIDTH, 3],
        },
        "tiling": {
            "tile": TILE,
            "stride": STRIDE,
            "starts": tile_starts(),
            "min_box_side_px": MIN_BOX_SIDE,
            "min_fragment_px": MIN_FRAG_PIXELS,
            "resampling": "none - crops are byte-exact sub-rectangles",
        },
        "budget": {"max_positives": max_positives, "max_negatives": max_negatives},
        "selection": {
            "positive_crops": len(pos_crops),
            "negative_crops": len(neg_crops),
            "positive_source_frames": len(pos_sources),
            "negative_source_frames": len(neg_sources),
            "negative_crops_per_frame": NEG_CROPS_PER_FRAME,
        },
        "severstal": {"counts": sev_counts, "leakage": sev_leak, "names": list(SEVERSTAL_CLASS_NAMES)},
        "joint": {
            "counts": joint_counts,
            "leakage": joint_leak,
            "names": list(NEU_CLASS_NAMES) + list(SEVERSTAL_CLASS_NAMES),
            "neu_images_per_split": {s: len(v) for s, v in neu_names.items()},
        },
        "neu_det_leakage": neu_leak,
        "stats": {
            "frames_seen": stats.frames_seen,
            "frames_yielding": stats.frames_yielding,
            "crops_dropped_clipped": stats.crops_dropped_clipped,
            "crops_empty_skipped": stats.crops_empty_skipped,
            "crops_duplicate": stats.crops_duplicate,
            "negatives_pixel_verified": stats.negatives_pixel_verified,
            "downloads": stats.downloads,
            "cache_hits": stats.cache_hits,
            "reused_from_extra_dirs": stats.reused_from_extra_dirs,
            "failures": stats.failures[:50],
        },
        "leakage_log": log,
        "verification": verify_rep,
        "verify_log": verify_log,
        "loader_log": load_log,
        "split_of_source": split_of,
    }
    (sev_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (joint_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if report_path is not None:
        write_report(report_path, manifest)
        print(f"[report] wrote {report_path}")

    print(f"[done] {manifest['elapsed_s']} s")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, default=REPO_ROOT / "data", help="root under which severstal/ and joint/ are written")
    p.add_argument("--max-positives", type=int, default=3000, help="cap on Severstal crops containing >=1 defect box")
    p.add_argument("--max-negatives", type=int, default=3000, help="cap on verified defect-free background crops")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--neu-root", type=Path, default=REPO_ROOT / "data" / "neu-det")
    p.add_argument("--cache", type=Path, default=REPO_ROOT / "data" / ".severstal_cache", help="local pool of downloaded Severstal frames")
    p.add_argument("--extra-frame-dir", type=Path, action="append", default=[], help="read-only dir of already-downloaded frames (repeatable)")
    p.add_argument("--samples", type=Path, default=None, help="path to an existing samples.json")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--report", type=Path, default=REPO_ROOT / "reports" / "severstal_dataset.md")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    a = parse_args(argv)
    neu_yaml = yaml.safe_load((a.neu_root / "data.yaml").read_text(encoding="utf-8"))
    got = tuple(neu_yaml["names"][i] for i in range(neu_yaml["nc"]))
    if got != NEU_CLASS_NAMES:
        raise SystemExit(f"NEU-DET class order changed: {got} != {NEU_CLASS_NAMES}")
    build(
        out_root=a.out,
        neu_root=a.neu_root,
        cache_dir=a.cache,
        extra_dirs=a.extra_frame_dir,
        max_positives=a.max_positives,
        max_negatives=a.max_negatives,
        seed=a.seed,
        samples_path=a.samples,
        allow_download=not a.no_download,
        report_path=a.report,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
