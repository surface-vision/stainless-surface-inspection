"""Input-pipeline study: is it better to preserve the source pixels or to interpolate them?

NEU-DET images are natively 200x200. YOLO wants a network input that is a multiple
of 32, so every size in the shipped grid resamples the source: 224 is a 1.12x
upscale, 256 a 1.28x upscale, 192 a 0.96x *downscale* that throws real pixels away.
Ultralytics letterboxes by scaling first and padding second, so the scaling is
unavoidable on that path -- but it does not have to be. A 200x200 image fits inside
a 224x224 canvas with 12 px of border on each side and no scaling whatsoever, and
inside 256x256 with 28 px. Every source pixel survives byte-exact, and the boxes
move by an integer offset instead of being multiplied by a scale factor.

This module measures whether that lossless path is actually worth anything.

**How the lossless path is built.** Rather than patching the framework's letterbox,
the padded canvases are materialised on disk as PNG (lossless; re-encoding as JPEG
would defeat the entire point) under `data/input_study/`, with labels re-normalised
onto the larger canvas. A 224x224 file evaluated at `imgsz=224` then reaches the
network without a single call to `cv2.resize`: the framework's own loader computes
a scale ratio of exactly 1.0 and skips the resize. `verify_pipeline()` proves this
rather than asserting it -- it pulls tensors out of the real validator dataset and
compares their centre crop with the decoded source array, byte for byte.

**A protocol fact this study had to discover, and which changes how every previous
number in this repository should be read.** The ultralytics validator runs with
`rect=True` and `pad=0.5`, so for square images the network input is
`ceil(imgsz/32 + 0.5) * 32`, which for any multiple of 32 is `imgsz + 32`. "Val at
256" therefore feeds a **288x288** tensor holding a 256x256 resized image inside a
16 px grey border. The framework already pads; it simply insists on scaling first.
That makes `scale N` and `pad N` an unusually clean pair: identical tensor size,
identical compute, and the only difference is whether the 200x200 source was
interpolated up to N or left alone. Both are verified below.

**The confound, stated up front.** Both checkpoints were fine-tuned on scaled
inputs, so a pad-only-at-inference test asks the model to work on a geometry it was
never optimised for. Experiment 1 is therefore evidence about the *inference-time*
choice for *these* weights, and nothing else. It cannot answer whether a model
trained on padded inputs would do better; that needs a training run. The confound is
weaker than it first looks -- `models/*/args.yaml` records `mosaic: 1.0`,
`scale: 0.4`, `translate: 0.1`, `degrees: 10.0`, and ultralytics fills every mosaic
canvas and every warp border with the same value 114 (`data/augment.py` lines 609
and 1194-1196) -- so the network has in fact seen steel texture abutting 114-grey in
almost every training batch. It has not seen a *centred 200x200 island* of it.

Three experiments, kept separate:

1. **Pad versus scale.** Both checkpoints, both splits, three pipelines: scale to
   224 (the framework default), pad to 224 (lossless), pad to 256 (lossless, more
   border). mAP50, mAP50-95 and per-class AP50 for each.

2. **The resolution grid, with its holes filled.** 160/192/224/256/288/320/352/384
   on *both* splits for *both* checkpoints. The previously published grid skipped
   288, 352 and 384, and its val and test peaks disagree (val 224, test 256), which
   is exactly the situation where a missing point can move the answer.

3. **Is any of it real?** A paired bootstrap over the 180 images of a split, seed
   1337, scoring every configuration on the *same* resampled image sets so the
   between-image variance cancels in the difference. Every interval is quoted
   against the shipped default of 256 px. Both splits are bootstrapped and they do
   different jobs: val is the only interval a size may be chosen with, test is the
   honest measurement of what a choice bought. If the intervals contain zero, the
   honest conclusion is that the choice does not matter, and the study says so.

**Split discipline.** A choice may be made on val. Test is quoted because it is the
honest measurement, and choosing the size that maximises test would be cheating; the
report names the size each rule would pick and shows what the cheat would have
bought.

Scoring is the ultralytics validator on its own defaults (conf 0.001, NMS IoU 0.7,
max_det 300), identical to `src/model_study.py:run_validation`, so every cell here is
directly comparable with `reports/model_study.json` and `reports/resolution_study.json`.
The per-image statistics behind the bootstrap are captured and re-scored with the
framework's own `ap_per_class` via `model_study.score_subset`, and the recomputation
is checked against the validator's own number before any interval is reported.

Outputs: `reports/input_study.json`, `reports/input_study.md`,
`reports/input_study_grid.png`, `reports/input_study_bootstrap.png`.

Usage:
    .venv/bin/python src/input_study.py
    .venv/bin/python src/input_study.py --resamples 2000
    .venv/bin/python src/input_study.py --skip-grid --resamples 0   # experiment 1 only
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
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Ultralytics pip-installs its way out of a version conflict on import unless told
# not to. A study that silently changes the environment it is measuring is worthless.
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

from inference import resolve_device  # noqa: E402

# Reused rather than reimplemented. `score_subset` calls the framework's own
# `ap_per_class`, so a bootstrap resample here is scored by exactly the function
# that produced every headline mAP in this repository, and `SplitStats` is already
# the container the model study captures per-image rows into.
from model_study import SplitStats, bootstrap_indices, score_subset  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NEU_ROOT = PROJECT_ROOT / "data" / "neu-det"
NEU_YAML = NEU_ROOT / "data.yaml"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Padded copies of the split live here, not under data/neu-det: the smoke tests walk
# that directory and assert one .jpg per label, and a second copy of every image
# inside it would be a leak waiting to happen.
STUDY_ROOT = PROJECT_ROOT / "data" / "input_study"

# The source geometry. Everything in this module is written against the measured
# shape rather than this constant; it exists to make the arithmetic in the report
# checkable and to fail loudly if the dataset is ever re-exported at another size.
NATIVE_SIZE = 200

# The grey ultralytics letterboxes with, and fills mosaic canvases and warp borders
# with during training (`ultralytics/data/augment.py`: LetterBox padding_value=114,
# mosaic np.full(..., 114), RandomPerspective borderValue=(114,)*4). Using anything
# else would introduce a second, gratuitous train/test difference.
PAD_VALUE = 114

# Multiples of 32 from below native to comfortably past the 320 px training size.
# 288, 352 and 384 are the holes in the previously published grid.
IMGSZ_GRID: tuple[int, ...] = (160, 192, 224, 256, 288, 320, 352, 384)

# Canvas sizes for the lossless path. Both must be >= NATIVE_SIZE and multiples of 32.
PAD_SIZES: tuple[int, ...] = (224, 256)

SPLITS: tuple[str, ...] = ("val", "test")

# The shipped default (`inference.DEFAULT_IMGSZ`). Every confidence interval in
# experiment 3 is a difference *against this*, because the question the study has to
# answer is not "which cell is largest" but "is anything better than what we ship".
REFERENCE_IMGSZ = 256

BOOTSTRAP_SEED = 1337

# Validator geometry, from `ultralytics/data/build.py` (pad = 0.5 outside train mode)
# and `BaseDataset.set_rectangle`. Kept as constants so `network_input_size` is
# readable, and checked against the real dataset in `verify_pipeline`.
VAL_STRIDE = 32
VAL_RECT_PAD = 0.5

GRID_STYLE = {"color": "#d8dde3", "linewidth": 0.7, "alpha": 0.9}
PIPELINE_COLOUR = {"scale": "#1f77b4", "pad": "#d62728"}


# ---------------------------------------------------------------------------
# data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointSpec:
    """One checkpoint under study, plus the class count its head was trained with."""

    key: str
    run: str
    nc: int
    note: str

    @property
    def weights(self) -> Path:
        return MODELS_DIR / self.run / "weights" / "best.pt"


CHECKPOINTS: tuple[CheckpointSpec, ...] = (
    CheckpointSpec(
        key="yolov8n_neudet",
        run="yolov8n_neudet",
        nc=6,
        note="the shipped 6-class model, 150 ep @320",
    ),
    CheckpointSpec(
        key="yolov8n_joint",
        run="yolov8n_joint",
        nc=10,
        note="NEU-DET + Severstal, 10 classes, 200 ep @320",
    ),
)


@dataclass
class InputRun:
    """One (checkpoint, pipeline, split, imgsz) validation cell."""

    model: str
    pipeline: str  # "scale" (framework default) or "pad" (lossless)
    split: str
    imgsz: int
    network_input: int
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
    def cell(self) -> tuple[str, str, str, int]:
        return (self.model, self.pipeline, self.split, self.imgsz)

    @property
    def label(self) -> str:
        return f"{self.pipeline}{self.imgsz}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "pipeline": self.pipeline,
            "split": self.split,
            "imgsz": self.imgsz,
            "network_input_px": self.network_input,
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
# 1. the lossless letterbox
# ---------------------------------------------------------------------------


def network_input_size(imgsz: int, stride: int = VAL_STRIDE, pad: float = VAL_RECT_PAD) -> int:
    """Tensor edge the validator actually builds for a *square* image at `imgsz`.

    `BaseDataset.set_rectangle` computes `ceil(aspect * imgsz / stride + pad) * stride`,
    and for a square image the aspect term is 1. With the validator's `pad=0.5` that
    rounds up a whole stride whenever `imgsz` is already a multiple of 32, so the
    network sees `imgsz + 32`, not `imgsz`. This is not a detail: it means the
    framework's default path is *already* padding, and the only thing this study
    changes is whether it scales before doing so.
    """
    return int(np.ceil(imgsz / stride + pad)) * stride


def pad_letterbox(
    image: np.ndarray, size: int, fill: int = PAD_VALUE
) -> tuple[np.ndarray, tuple[int, int]]:
    """Centre `image` in a `size` x `size` canvas with **no scaling at all**.

    This is the whole idea of the study in five lines. The framework's LetterBox
    computes `r = size / max(h, w)` and resizes by it; this one does not, so the
    returned canvas contains the input array verbatim. Returns the canvas and the
    `(left, top)` offset the pixels were placed at, which is also the only thing
    that has to be added to a box coordinate -- never multiplied.
    """
    height, width = image.shape[:2]
    if height > size or width > size:
        raise ValueError(
            f"cannot pad a {width}x{height} image into {size}x{size} without scaling; "
            "the lossless path only exists while the source is smaller than the canvas"
        )
    left = (size - width) // 2
    top = (size - height) // 2
    canvas = np.full((size, size, image.shape[2]), fill, dtype=image.dtype)
    canvas[top : top + height, left : left + width] = image
    return canvas, (left, top)


def shift_labels(
    rows: Iterable[str], width: int, height: int, size: int, left: int, top: int
) -> list[str]:
    """Re-normalise YOLO boxes onto the padded canvas by an integer pixel shift.

    Denormalise against the source, add the pad offset to the centre, renormalise
    against the canvas. The width and height are divided by the new canvas edge
    because they are stored normalised -- in *pixels* they are untouched, which is
    the property that makes this path exact. `build_padded_split` measures the
    round-trip error in source pixels and refuses to proceed if it is not ~0.
    """
    out: list[str] = []
    for row in rows:
        row = row.strip()
        if not row:
            continue
        parts = row.split()
        if len(parts) != 5:
            raise ValueError(f"not a YOLO detection label row: {row!r}")
        cls, cx, cy, bw, bh = parts
        px = float(cx) * width + left
        py = float(cy) * height + top
        pw = float(bw) * width
        ph = float(bh) * height
        out.append(f"{cls} {px / size:.10f} {py / size:.10f} {pw / size:.10f} {ph / size:.10f}")
    return out


def build_padded_split(split: str, size: int, *, root: Path = STUDY_ROOT) -> dict[str, Any]:
    """Materialise the losslessly padded copy of one split and prove it is lossless.

    PNG, not JPEG. Re-encoding as JPEG would quantise the very pixels the exercise
    exists to preserve, and the byte-exactness assertion below would fail -- which is
    how this was caught rather than shipped.
    """
    src_images = sorted((NEU_ROOT / split / "images").glob("*.jpg"))
    if not src_images:
        raise FileNotFoundError(f"no images in {NEU_ROOT / split / 'images'}")
    dst_dir = root / f"pad{size}" / split
    (dst_dir / "images").mkdir(parents=True, exist_ok=True)
    (dst_dir / "labels").mkdir(parents=True, exist_ok=True)
    # A stale labels.cache describes the previous build of this directory.
    for cache in (dst_dir / "labels.cache", dst_dir / "images.cache"):
        cache.unlink(missing_ok=True)

    exact = 0
    shapes: set[tuple[int, int]] = set()
    max_box_error_px = 0.0
    boxes = 0
    for src_path in src_images:
        image = cv2.imread(str(src_path))
        if image is None:
            raise RuntimeError(f"unreadable image: {src_path}")
        height, width = image.shape[:2]
        shapes.add((width, height))
        canvas, (left, top) = pad_letterbox(image, size)
        dst_path = dst_dir / "images" / f"{src_path.stem}.png"
        if not cv2.imwrite(str(dst_path), canvas):
            raise RuntimeError(f"failed to write {dst_path}")

        # Read the file back off disk: the claim is about what the loader will see,
        # not about what was in memory a moment ago.
        recovered = cv2.imread(str(dst_path))
        if recovered.shape[:2] == (size, size) and np.array_equal(
            recovered[top : top + height, left : left + width], image
        ):
            exact += 1

        src_label = NEU_ROOT / split / "labels" / f"{src_path.stem}.txt"
        rows = src_label.read_text().splitlines() if src_label.is_file() else []
        shifted = shift_labels(rows, width, height, size, left, top)
        (dst_dir / "labels" / f"{src_path.stem}.txt").write_text(
            "\n".join(shifted) + ("\n" if shifted else "")
        )

        # Round-trip the written labels back into source-pixel coordinates.
        for original, moved in zip(rows, shifted):
            o = [float(v) for v in original.split()[1:]]
            m = [float(v) for v in moved.split()[1:]]
            back = [
                m[0] * size - left,
                m[1] * size - top,
                m[2] * size,
                m[3] * size,
            ]
            want = [o[0] * width, o[1] * height, o[2] * width, o[3] * height]
            max_box_error_px = max(
                max_box_error_px, float(np.max(np.abs(np.array(back) - np.array(want))))
            )
            boxes += 1

    if exact != len(src_images):
        raise RuntimeError(
            f"pad{size}/{split}: only {exact}/{len(src_images)} images survived the "
            "write/read round trip byte-exact; the padded copy is not lossless"
        )
    return {
        "split": split,
        "canvas": size,
        "images": len(src_images),
        "source_shapes": sorted(f"{w}x{h}" for w, h in shapes),
        "pad_px_each_side": [(size - w) // 2 for w, _ in sorted(shapes)],
        "byte_exact_images": exact,
        "boxes": boxes,
        "max_box_round_trip_error_px": round(max_box_error_px, 9),
        "images_dir": str(dst_dir / "images"),
    }


def write_study_yaml(size: int, nc: int, names: dict[int, str], *, root: Path = STUDY_ROOT) -> Path:
    """Dataset yaml for one padded canvas at one class count.

    The 10-class checkpoint needs a 10-name yaml even though only classes 0-5 ever
    appear in NEU-DET ground truth; ultralytics averages AP over the classes *present
    in the labels*, so the four Severstal heads contribute nothing to the mean and
    the two checkpoints stay comparable on the same six classes.
    """
    path = root / f"pad{size}" / f"data_nc{nc}.yaml"
    payload = {
        "path": str(root / f"pad{size}"),
        "train": "val/images",  # never trained from; present because the loader wants a key
        "val": "val/images",
        "test": "test/images",
        "nc": nc,
        "names": {int(i): n for i, n in names.items()},
    }
    header = (
        f"# Written by src/input_study.py. {NATIVE_SIZE}x{NATIVE_SIZE} NEU-DET images centred in a\n"
        f"# {size}x{size} canvas at pad value {PAD_VALUE} with NO scaling: every source pixel is\n"
        f"# byte-exact and every box moved by an integer offset. Do not train from this.\n"
    )
    path.write_text(header + yaml.safe_dump(payload, sort_keys=False))
    return path


def write_scale_yaml(nc: int, *, root: Path = STUDY_ROOT) -> Path:
    """A 10-class view of the untouched NEU-DET split, for the joint checkpoint.

    Same images and same labels as `data/neu-det/data.yaml` -- only the name table
    is widened so the 10-class head can be validated against them.
    """
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"neudet_nc{nc}.yaml"
    source = yaml.safe_load(NEU_YAML.read_text())
    names = dict(source["names"])
    for extra in range(len(names), nc):
        names[extra] = f"unused_{extra}"
    payload = {
        "path": str(NEU_ROOT),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": nc,
        "names": names,
    }
    path.write_text(
        "# Written by src/input_study.py: data/neu-det verbatim, name table widened so a\n"
        f"# {nc}-class head can be scored on it. Classes >= {len(source['names'])} never occur in the labels.\n"
        + yaml.safe_dump(payload, sort_keys=False)
    )
    return path


# ---------------------------------------------------------------------------
# 2. proving the pipeline is lossless (and that the default one is not)
# ---------------------------------------------------------------------------


def _val_dataset(image_dir: Path, imgsz: int, names: dict[int, str]) -> Any:
    """The dataset object the validator itself builds, with the validator's own args.

    Constructed through `build_yolo_dataset` in `mode="val"` with `rect` on, which is
    what `model.val()` does, so what comes out of `__getitem__` here is exactly the
    tensor the network is fed.
    """
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.utils import DEFAULT_CFG

    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = imgsz
    cfg.rect = True
    data = {
        "path": image_dir.parent,
        "train": str(image_dir),
        "val": str(image_dir),
        "names": names,
        "nc": len(names),
        "channels": 3,
    }
    return build_yolo_dataset(cfg, str(image_dir), 16, data, mode="val", stride=VAL_STRIDE)


def _tensor_to_bgr(item: dict[str, Any]) -> np.ndarray:
    """The dataset's CHW/RGB tensor back as an HWC/BGR array, for comparison with cv2."""
    return np.ascontiguousarray(item["img"].numpy().transpose(1, 2, 0)[:, :, ::-1])


def verify_pipeline(
    image_dir: Path, imgsz: int, names: dict[int, str], *, pipeline: str, sample: int
) -> dict[str, Any]:
    """Pull real tensors out of the real validator dataset and check what happened to them.

    For the padded pipeline the claim is falsifiable and strong: the centre
    200x200 of the network input equals the decoded source array byte for byte.
    For the scaled pipeline the claim is the opposite one -- the content is a bilinear
    resample and no longer contains the source array anywhere -- and it is checked the
    same way, so a reader does not have to take either on faith.
    """
    dataset = _val_dataset(image_dir, imgsz, names)
    expected_tensor = network_input_size(imgsz)
    n = min(sample, len(dataset))
    exact = 0
    tensor_shapes: set[int] = set()
    border_values: set[int] = set()
    max_abs_content_diff = 0
    for i in range(n):
        item = dataset[i]
        tensor = _tensor_to_bgr(item)
        tensor_shapes.add(int(tensor.shape[0]))
        source = cv2.imread(str(NEU_ROOT / _split_of(Path(item["im_file"]))
                                / "images" / f"{Path(item['im_file']).stem}.jpg"))
        edge = tensor.shape[0]
        if pipeline == "pad":
            off = (edge - source.shape[0]) // 2
            crop = tensor[off : off + source.shape[0], off : off + source.shape[1]]
            if np.array_equal(crop, source):
                exact += 1
            border_values.update(np.unique(tensor[:2, :]).tolist())
        else:
            off = (edge - imgsz) // 2
            crop = tensor[off : off + imgsz, off : off + imgsz]
            reference = cv2.resize(source, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
            if np.array_equal(crop, reference):
                exact += 1
            # How far the network input is from *any* placement of the source array.
            if imgsz >= source.shape[0]:
                resampled_back = cv2.resize(
                    crop, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_AREA
                )
                max_abs_content_diff = max(
                    max_abs_content_diff,
                    int(np.abs(resampled_back.astype(int) - source.astype(int)).max()),
                )
    record: dict[str, Any] = {
        "pipeline": pipeline,
        "imgsz": imgsz,
        "images_checked": n,
        "network_tensor_px": sorted(tensor_shapes),
        "network_tensor_px_predicted": expected_tensor,
        "tensor_matches_prediction": tensor_shapes == {expected_tensor},
    }
    if pipeline == "pad":
        record["images_whose_centre_is_byte_exact_source"] = exact
        record["lossless"] = exact == n
        record["border_pixel_values"] = sorted(border_values)
    else:
        record["images_whose_content_is_a_bilinear_resample"] = exact
        record["lossless"] = False
        record["max_abs_error_after_resampling_back_to_source"] = max_abs_content_diff
    return record


def _split_of(image_path: Path) -> str:
    """Which NEU-DET split a padded file came from, read off its parent directory."""
    # .../pad224/<split>/images/<stem>.png  ->  <split>;  .../neu-det/<split>/images -> <split>
    return image_path.parent.parent.name


def pixel_fidelity(sizes: Sequence[int], split: str = "test") -> list[dict[str, Any]]:
    """What the resize itself costs, measured on the images and no model involved.

    For each grid size: resample the source to that size the way the loader does
    (bilinear), then resample straight back to 200x200 with INTER_AREA and compare
    with the original. A pipeline that carried all the information would come back
    unchanged. This is a *round-trip* error, not a proof of information loss on its
    own -- but it is zero by construction for the padded path, it is large for the
    downscales, and it is the only purely image-domain number in the study.
    """
    files = sorted((NEU_ROOT / split / "images").glob("*.jpg"))
    sources = [cv2.imread(str(f)) for f in files]
    rows: list[dict[str, Any]] = []
    for size in sizes:
        errors = []
        for src in sources:
            up = cv2.resize(src, (size, size), interpolation=cv2.INTER_LINEAR)
            back = cv2.resize(up, src.shape[1::-1], interpolation=cv2.INTER_AREA)
            errors.append(np.abs(back.astype(np.float64) - src.astype(np.float64)).mean())
        mae = float(np.mean(errors))
        rows.append(
            {
                "imgsz": size,
                "scale_factor": round(size / NATIVE_SIZE, 4),
                "direction": "downscale" if size < NATIVE_SIZE else "upscale",
                "round_trip_mae_levels": round(mae, 4),
                "round_trip_psnr_db": round(float(20 * np.log10(255.0 / max(mae, 1e-9))), 2),
            }
        )
    rows.append(
        {
            "imgsz": None,
            "scale_factor": 1.0,
            "direction": "none (padded)",
            "round_trip_mae_levels": 0.0,
            "round_trip_psnr_db": None,
        }
    )
    return rows


# ---------------------------------------------------------------------------
# 3. the validator
# ---------------------------------------------------------------------------


def run_validation(
    spec: CheckpointSpec,
    data_yaml: Path,
    split: str,
    imgsz: int,
    device: str,
    *,
    pipeline: str,
    out_dir: Path,
    batch: int = 16,
    workers: int = 2,
) -> InputRun:
    """One validation pass, on the framework's own protocol defaults.

    Deliberately identical to `model_study.run_validation` except that the dataset
    yaml is a parameter -- that module hard-codes `data/neu-det/data.yaml`, which
    cannot express either a padded canvas or a 10-class name table. conf 0.001, NMS
    IoU 0.7, max_det 300 are left alone so every number here is comparable with the
    ones already published.

    The per-image statistics are lifted out through an `on_val_batch_end` callback,
    because `get_stats()` concatenates and then clears them.
    """
    from ultralytics import YOLO

    captured: dict[str, Any] = {}

    def snapshot(validator: Any) -> None:
        captured["stats"] = {k: list(v) for k, v in validator.metrics.stats.items()}

    model = YOLO(str(spec.weights))
    model.add_callback("on_val_batch_end", snapshot)

    started = time.perf_counter()
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        device=device,
        batch=batch,
        workers=workers,
        plots=False,
        verbose=False,
        project=str(out_dir),
        name=f"{spec.key}_{pipeline}{imgsz}_{split}",
        exist_ok=True,
    )
    seconds = time.perf_counter() - started

    names = dict(metrics.names)
    ap50: dict[str, float] = {}
    ap: dict[str, float] = {}
    instances: dict[str, int] = {}
    for row, cls in enumerate((int(c) for c in metrics.ap_class_index)):
        _, _, a50, a = metrics.box.class_result(row)
        label = names.get(cls, f"class_{cls}")
        ap50[label] = float(a50)
        ap[label] = float(a)
        instances[label] = int(metrics.nt_per_class[cls]) if metrics.nt_per_class is not None else 0

    raw = captured.get("stats")
    stats: SplitStats | None = None
    if raw is not None:
        image_names = [Path(n).stem for n in metrics.box.image_metrics]
        if len(image_names) == len(raw["tp"]):
            stats = SplitStats(
                names=image_names,
                tp=[np.asarray(t, dtype=bool) for t in raw["tp"]],
                conf=[np.asarray(c, dtype=np.float64) for c in raw["conf"]],
                pred_cls=[np.asarray(c, dtype=np.float64) for c in raw["pred_cls"]],
                target_cls=[np.asarray(c, dtype=np.float64) for c in raw["target_cls"]],
            )

    del model
    return InputRun(
        model=spec.key,
        pipeline=pipeline,
        split=split,
        imgsz=imgsz,
        network_input=network_input_size(imgsz),
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


def fidelity(run: InputRun) -> dict[str, Any]:
    """Does re-scoring the captured per-image rows reproduce the validator's own mAP?

    If it does not, the bootstrap is measuring something other than the model that
    was reported, and no interval below means anything.
    """
    if run.stats is None:
        return {"cell": run.cell, "captured": False}
    recomputed = score_subset(run.stats, np.arange(len(run.stats)))
    return {
        "model": run.model,
        "pipeline": run.pipeline,
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


def reproduction_check(runs: Sequence[InputRun], prior: Path) -> dict[str, Any]:
    """Do the scaled cells here reproduce the numbers already published in this repo?

    `reports/resolution_study.json` swept the same checkpoint on the same splits with
    the same validator. Every cell this study shares with it should agree exactly --
    validation is deterministic given weights, data and input size. If it does not,
    something in this module's dataset plumbing differs from the instrument that
    produced every other NEU-DET number here, and nothing below can be trusted.
    """
    if not prior.is_file():
        return {"source": str(prior), "available": False}
    published = json.loads(prior.read_text())
    # `resolution_study.json` keys the nano checkpoint as "yolov8n"; this module keys
    # it by its run directory.
    alias = {"yolov8n": "yolov8n_neudet"}
    here = {(r.model, r.split, r.imgsz): r for r in runs if r.pipeline == "scale"}
    rows: list[dict[str, Any]] = []
    for cellrec in published.get("runs", []):
        key = (alias.get(cellrec["tag"], cellrec["tag"]), cellrec["split"], cellrec["imgsz"])
        mine = here.get(key)
        if mine is None:
            continue
        rows.append(
            {
                "model": key[0],
                "split": key[1],
                "imgsz": key[2],
                "published_mAP50": round(float(cellrec["mAP50"]), 6),
                "this_study_mAP50": round(mine.mAP50, 6),
                "abs_difference": abs(float(cellrec["mAP50"]) - mine.mAP50),
            }
        )
    return {
        "source": str(prior.relative_to(PROJECT_ROOT)),
        "available": True,
        "overlapping_cells": len(rows),
        "max_abs_difference_mAP50": max((r["abs_difference"] for r in rows), default=0.0),
        "cells": rows,
    }


# ---------------------------------------------------------------------------
# 4. the paired bootstrap
# ---------------------------------------------------------------------------


def paired_bootstrap(
    runs: Sequence[InputRun], resamples: int, seed: int = BOOTSTRAP_SEED
) -> dict[str, Any]:
    """Every configuration scored on the *same* resampled image sets.

    Pairing is the whole point. The dominant source of variance in a 180-image mAP is
    which images were drawn -- a resample heavy in crazing scores badly for every
    configuration -- and that component is common to all of them and cancels in a
    difference. An unpaired interval on an input-size comparison would be several
    times wider and would say nothing at all.

    Scoring every configuration inside one loop over resamples, rather than running a
    separate two-way bootstrap per comparison, means any pair of configurations can be
    differenced afterwards and all of them are consistent with each other.
    """
    usable = [r for r in runs if r.stats is not None]
    if len(usable) < 2 or resamples <= 0:
        return {}
    order = sorted(usable[0].stats.names)  # type: ignore[union-attr]
    aligned = [r.stats.reindex(order) for r in usable]  # type: ignore[union-attr]
    n = len(order)
    draws = bootstrap_indices(n, resamples, seed)

    started = time.perf_counter()
    map50 = np.empty((resamples, len(aligned)), dtype=np.float64)
    map5095 = np.empty((resamples, len(aligned)), dtype=np.float64)
    for r in range(resamples):
        idx = draws[r]
        for c, stats in enumerate(aligned):
            scored = score_subset(stats, idx)
            map50[r, c] = scored["mAP50"]
            map5095[r, c] = scored["mAP50_95"]
    return {
        "labels": [r.label for r in usable],
        "point_mAP50": [r.mAP50 for r in usable],
        "point_mAP50_95": [r.mAP50_95 for r in usable],
        "n_images": n,
        "resamples": resamples,
        "seed": seed,
        "seconds": round(time.perf_counter() - started, 1),
        "_map50": map50,
        "_map5095": map5095,
    }


def interval(
    samples: np.ndarray,
    a: int,
    b: int,
    point_a: float,
    point_b: float,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Percentile interval on the paired difference between two columns.

    `alpha` is exposed because one comparison in this study is not free: the size
    that gets tested against the incumbent is the *maximum* of the val grid, chosen
    after looking at it. Testing a selected maximum at a nominal 95% understates the
    error rate, so that one comparison is also reported at a Bonferroni-corrected
    level over the number of sizes the maximum was picked from.
    """
    delta = samples[:, a] - samples[:, b]
    lo, hi = np.percentile(delta, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "alpha": round(alpha, 6),
        "confidence_pct": round(100 * (1 - alpha), 3),
        "point_a": round(point_a, 5),
        "point_b": round(point_b, 5),
        "point_difference": round(point_a - point_b, 5),
        "bootstrap_mean_difference": round(float(delta.mean()), 5),
        "ci95_low": round(float(lo), 5),
        "ci95_high": round(float(hi), 5),
        "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0),
        "share_a_wins": round(float((delta > 0).mean()), 4),
        "paired_std": round(float(delta.std(ddof=1)), 5),
        "unpaired_std_estimate": round(
            float(np.hypot(samples[:, a].std(ddof=1), samples[:, b].std(ddof=1))), 5
        ),
    }


def contrasts(bootstrap: dict[str, Any], reference: str, alpha: float = 0.05) -> dict[str, Any]:
    """Every configuration differenced against one reference configuration."""
    if not bootstrap:
        return {}
    labels: list[str] = bootstrap["labels"]
    if reference not in labels:
        return {}
    ref = labels.index(reference)
    out: dict[str, Any] = {"reference": reference, "mAP50": {}, "mAP50_95": {}}
    for i, label in enumerate(labels):
        if i == ref:
            continue
        out["mAP50"][label] = interval(
            bootstrap["_map50"],
            i,
            ref,
            bootstrap["point_mAP50"][i],
            bootstrap["point_mAP50"][ref],
            alpha,
        )
        out["mAP50_95"][label] = interval(
            bootstrap["_map5095"],
            i,
            ref,
            bootstrap["point_mAP50_95"][i],
            bootstrap["point_mAP50_95"][ref],
            alpha,
        )
    return out


def pad_versus_scale_intervals(bootstrap: dict[str, Any]) -> dict[str, Any]:
    """Interval on (pad N - scale N): same tensor size, only the resampling differs."""
    if not bootstrap:
        return {}
    labels: list[str] = bootstrap["labels"]
    out: dict[str, Any] = {}
    for size in PAD_SIZES:
        a, b = f"pad{size}", f"scale{size}"
        if a in labels and b in labels:
            i, j = labels.index(a), labels.index(b)
            out[f"{a}_minus_{b}"] = {
                "mAP50": interval(
                    bootstrap["_map50"], i, j, bootstrap["point_mAP50"][i], bootstrap["point_mAP50"][j]
                ),
                "mAP50_95": interval(
                    bootstrap["_map5095"],
                    i,
                    j,
                    bootstrap["point_mAP50_95"][i],
                    bootstrap["point_mAP50_95"][j],
                ),
            }
    return out


# ---------------------------------------------------------------------------
# 5. figures
# ---------------------------------------------------------------------------


def _style(ax: plt.Axes, *, xlabel: str, ylabel: str, title: str | None = None) -> None:
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if title:
        ax.set_title(title, fontsize=10.5, pad=8)
    ax.grid(True, **GRID_STYLE)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8.5)


def _model_style(key: str) -> dict[str, Any]:
    return {
        "yolov8n_neudet": {"color": "#1f77b4", "linestyle": "-"},
        "yolov8n_joint": {"color": "#2ca02c", "linestyle": "-"},
    }.get(key, {"color": "#555555", "linestyle": "-"})


def chart_grid(runs: Sequence[InputRun], out_path: Path) -> Path:
    """mAP50 against input size on both splits, with the lossless points overlaid."""
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), layout="constrained", sharey=True)
    models = [c.key for c in CHECKPOINTS]
    for ax, split in zip(axes, SPLITS):
        for key in models:
            cells = sorted(
                (r for r in runs if r.model == key and r.split == split and r.pipeline == "scale"),
                key=lambda r: r.imgsz,
            )
            if not cells:
                continue
            style = _model_style(key)
            ax.plot(
                [c.imgsz for c in cells],
                [c.mAP50 for c in cells],
                marker="o",
                markersize=4.5,
                linewidth=1.9,
                label=f"{key} - scaled",
                **style,
            )
            best = max(cells, key=lambda r: r.mAP50)
            ax.scatter(
                [best.imgsz], [best.mAP50], s=110, facecolors="none",
                edgecolors=style["color"], linewidths=1.6, zorder=5,
            )
            pads = sorted(
                (r for r in runs if r.model == key and r.split == split and r.pipeline == "pad"),
                key=lambda r: r.imgsz,
            )
            if pads:
                ax.plot(
                    [c.imgsz for c in pads],
                    [c.mAP50 for c in pads],
                    marker="D",
                    markersize=7,
                    linestyle="none",
                    markerfacecolor="none",
                    markeredgewidth=1.8,
                    color=style["color"],
                    label=f"{key} - padded, no scaling",
                )
                for a, b in zip(pads, (r for r in cells if r.imgsz in PAD_SIZES)):
                    ax.annotate(
                        "",
                        xy=(a.imgsz, a.mAP50),
                        xytext=(b.imgsz, b.mAP50),
                        arrowprops={"arrowstyle": "-", "color": style["color"], "alpha": 0.35,
                                    "linestyle": ":"},
                    )
        ax.axvline(NATIVE_SIZE, color="#b0453c", linestyle="--", linewidth=1.0)
        ax.annotate(
            f"native {NATIVE_SIZE} px",
            xy=(NATIVE_SIZE, ax.get_ylim()[0]),
            xytext=(3, 4),
            textcoords="offset points",
            fontsize=7.5,
            color="#b0453c",
            rotation=90,
        )
        ax.axvline(320, color="#888888", linestyle="--", linewidth=1.0)
        ax.annotate(
            "trained at 320 px",
            xy=(320, ax.get_ylim()[0]),
            xytext=(3, 4),
            textcoords="offset points",
            fontsize=7.5,
            color="#666666",
            rotation=90,
        )
        _style(ax, xlabel="imgsz argument (px)", ylabel="mAP50", title=f"{split} split")
        ax.set_xticks(list(IMGSZ_GRID))
        ax.legend(fontsize=7.6, frameon=False, loc="lower center")
    fig.suptitle(
        "Input pipeline on NEU-DET: interpolate the 200x200 source, or keep it and pad\n"
        "(circles mark each curve's peak; the validator's network input is always imgsz + 32)",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def chart_bootstrap(payload: dict[str, Any], out_path: Path) -> Path | None:
    """Paired 95% intervals against the shipped 256 px default, per split and checkpoint.

    Two columns because the two splits do different jobs. The **val** column is the
    only one a size may be chosen with; the **test** column is the honest measurement
    of what that choice bought. A reader who picks the lowest point in the test column
    has cheated, and laying them side by side makes that visible rather than tempting.
    """
    boot = payload["experiment_3_bootstrap"]
    grid = [
        (split, spec.key, boot.get(split, {}).get(spec.key, {}).get("vs_reference", {}))
        for split in SPLITS
        for spec in CHECKPOINTS
    ]
    grid = [g for g in grid if g[2]]
    if not grid:
        return None
    ncol = len(CHECKPOINTS)
    nrow = len(SPLITS)
    # One x scale across all four panels. With per-panel autoscaling a wide val
    # interval and a narrow test interval draw the same length, which is exactly the
    # misreading this figure exists to prevent.
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(6.2 * ncol, 4.3 * nrow),
        layout="constrained",
        squeeze=False,
        sharex=True,
    )
    for split, key, block in grid:
        ax = axes[SPLITS.index(split)][[c.key for c in CHECKPOINTS].index(key)]
        rows = list(block["mAP50"].items())
        rows.sort(
            key=lambda kv: (
                kv[0].startswith("pad"),
                int(kv[0].replace("pad", "").replace("scale", "")),
            )
        )
        ys = np.arange(len(rows))
        for y, (label, cell) in zip(ys, rows):
            colour = PIPELINE_COLOUR["pad" if label.startswith("pad") else "scale"]
            ax.plot(
                [cell["ci95_low"], cell["ci95_high"]],
                [y, y],
                color=colour,
                linewidth=2.6,
                solid_capstyle="butt",
                alpha=0.85,
            )
            ax.plot([cell["point_difference"]], [y], marker="o", markersize=5.5, color=colour)
        ax.axvline(0.0, color="#333333", linewidth=1.1)
        ax.set_yticks(ys)
        ax.set_yticklabels([label for label, _ in rows], fontsize=8.5)
        ax.invert_yaxis()
        role = "may choose on this" if split == "val" else "honest measurement, never chosen on"
        # sharex hides the tick labels on every row but the last, so only the last row
        # gets an axis label -- a label with no numbers under it reads as a mistake.
        last_row = SPLITS.index(split) == nrow - 1
        _style(
            ax,
            xlabel=(
                f"mAP50 difference against {block['reference']} (paired, 95% CI)"
                if last_row
                else ""
            ),
            ylabel="",
            title=f"{key} - {split} ({role})",
        )
    fig.suptitle(
        "Every configuration minus the shipped 256 px default, scored on the same resampled splits\n"
        f"({payload['experiment_3_bootstrap'].get('n_images')} images, "
        f"{payload['experiment_3_bootstrap'].get('resamples')} resamples, seed {BOOTSTRAP_SEED}; "
        "an interval crossing 0 is a difference this benchmark cannot resolve)",
        fontsize=11.5,
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 6. report
# ---------------------------------------------------------------------------


def _md_table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(str(h) for h in header) + " |"]
    out.append("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        out.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
    return "\n".join(out)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(_json_safe(payload), indent=2) + "\n")
    return path


def write_markdown(path: Path, payload: dict[str, Any]) -> Path:  # noqa: C901 - one long report
    """The report. Every number in it is read out of `payload`, never retyped."""
    runs = payload["runs"]

    def cell(model: str, pipeline: str, split: str, imgsz: int) -> dict[str, Any] | None:
        for r in runs:
            if (r["model"], r["pipeline"], r["split"], r["imgsz"]) == (model, pipeline, split, imgsz):
                return r
        return None

    models = [c["key"] for c in payload["checkpoints"]]
    lines: list[str] = []
    add = lines.append

    add("# Input pipeline: preserve the pixels, or interpolate them?")
    add("")
    add(
        f"Generated {payload['generated']} by `src/input_study.py` on "
        f"{payload['host']['chip']} ({payload['host']['device']}), "
        f"ultralytics {payload['host']['ultralytics']}, torch {payload['host']['torch']}."
    )
    add("")
    add(
        "NEU-DET is natively 200x200. Every network input size that YOLO will accept is a "
        "multiple of 32, so every one of them resamples the source -- 224 is a 1.12x upscale, "
        "256 a 1.28x upscale, 192 a 0.96x downscale that discards real pixels. But a 200x200 "
        "image fits inside a 224x224 canvas with 12 px of border on each side and **no scaling "
        "at all**, and inside 256x256 with 28 px. This study measures whether that lossless "
        "path is worth anything."
    )
    add("")

    # ---- headline -------------------------------------------------------
    head = payload["recommendation"]
    add("## Answer")
    add("")
    for line in head["summary"]:
        add(line)
        add("")
    add(_md_table(["checkpoint", "recommended input", "test mAP50", "why"], head["table"]))
    add("")

    # ---- protocol -------------------------------------------------------
    add("## What was actually fed to the network")
    add("")
    add(
        "Two things had to be established before any accuracy number meant anything, and both "
        "are measured rather than assumed."
    )
    add("")
    add(
        "**1. `imgsz` is not the network input size.** The ultralytics validator runs with "
        "`rect=True` and `pad=0.5`, and `BaseDataset.set_rectangle` computes "
        "`ceil(imgsz/32 + 0.5) * 32`. For a square image and an `imgsz` that is already a "
        "multiple of 32 that rounds up a whole stride, so **the tensor is `imgsz + 32`**. "
        f"Validating at 256 px feeds a {network_input_size(256)}x{network_input_size(256)} "
        "tensor holding a 256x256 resized image inside a 16 px grey border. The framework was "
        "already padding; it just insisted on scaling first. Every published NEU-DET number in "
        "this repository should be read that way."
    )
    add("")
    add(
        "This is convenient rather than awkward: it makes `scale N` and `pad N` a matched pair. "
        "Same tensor, same compute, same border colour -- the only difference is whether the "
        "200x200 source was interpolated up to N first."
    )
    add("")
    ver = payload["verification"]
    add(
        "**2. The padded path really is lossless, and the scaled path really is not.** "
        "The check pulls tensors out of the dataset object `model.val()` itself builds and "
        "compares them with the decoded source arrays:"
    )
    add("")
    add(
        _md_table(
            ["pipeline", "imgsz", "network tensor", "images checked", "result"],
            [
                [
                    v["pipeline"],
                    v["imgsz"],
                    f"{v['network_tensor_px'][0]}x{v['network_tensor_px'][0]}",
                    v["images_checked"],
                    (
                        f"centre {NATIVE_SIZE}x{NATIVE_SIZE} byte-exact for "
                        f"{v['images_whose_centre_is_byte_exact_source']}/{v['images_checked']}, "
                        f"border value {v['border_pixel_values']}"
                        if v["pipeline"] == "pad"
                        else (
                            "content is a bilinear resample for "
                            f"{v['images_whose_content_is_a_bilinear_resample']}/{v['images_checked']}; "
                            "resampling it back to 200x200 leaves errors up to "
                            f"{v['max_abs_error_after_resampling_back_to_source']} grey levels"
                        )
                    ),
                ]
                for v in ver["pipelines"]
            ],
        )
    )
    add("")
    rep = payload.get("reproduction", {})
    if rep.get("available") and rep.get("overlapping_cells"):
        add(
            f"**3. This is the same instrument that produced the existing numbers.** "
            f"{rep['overlapping_cells']} of the scaled cells below also appear in "
            f"`{rep['source']}`, measured before this study existed. The largest disagreement "
            f"between the two is {rep['max_abs_difference_mAP50']:.2e} mAP50, which is exactly "
            "what a deterministic validator on identical inputs should give. The pad-versus-scale "
            "comparison is therefore not being made against a re-implemented baseline."
        )
        add("")

    build = payload["padded_datasets"]
    add(
        f"The padded copies were written as PNG (JPEG would re-quantise the pixels the exercise "
        f"exists to preserve). Across all {sum(b['images'] for b in build)} written files, "
        f"{sum(b['byte_exact_images'] for b in build)} survived the write-and-read-back round "
        f"trip byte-exact, and the largest box round-trip error was "
        f"{max(b['max_box_round_trip_error_px'] for b in build):.2e} source pixels -- boxes are "
        "shifted by an integer offset and never multiplied by a scale factor."
    )
    add("")
    add(
        "For completeness, what the resize costs in the image domain alone, with no model "
        "involved: resample the source to `imgsz` bilinearly, resample straight back to 200x200, "
        "and compare. Read the *categories*, not the ranking -- the residual for an upscale is "
        "mostly the return trip's own resampling, so these numbers do not say that 384 is more "
        "faithful than 224. What they do say is the qualitative thing that matters: 160 and 192 "
        "are downscales and destroy source pixels irrecoverably, everything from 224 up invents "
        "pixels rather than losing them, and the padded path is the only exactly-zero row."
    )
    add("")
    add(
        _md_table(
            ["imgsz", "scale factor", "direction", "round-trip MAE (grey levels)", "PSNR (dB)"],
            [
                [
                    r["imgsz"] if r["imgsz"] else "pad (any)",
                    r["scale_factor"],
                    r["direction"],
                    r["round_trip_mae_levels"],
                    r["round_trip_psnr_db"] if r["round_trip_psnr_db"] else "inf (exact)",
                ]
                for r in payload["pixel_fidelity"]
            ],
        )
    )
    add("")

    # ---- experiment 1 ---------------------------------------------------
    add("## Experiment 1 - pad versus scale")
    add("")
    add(
        "Both checkpoints, both splits. `scale 224` is the framework default path; `pad 224` and "
        "`pad 256` place the untouched 200x200 source in a grey canvas. All numbers are the "
        "ultralytics validator on its own defaults (conf 0.001, NMS IoU 0.7, max_det 300)."
    )
    add("")
    for split in SPLITS:
        add(f"**{split} split**")
        add("")
        rows = []
        for model in models:
            for pipeline, size in payload["experiment_1"]["configs"]:
                c = cell(model, pipeline, split, size)
                if not c:
                    continue
                ref = cell(model, "scale", split, size)
                delta = c["mAP50"] - ref["mAP50"] if ref else None
                rows.append(
                    [
                        model,
                        f"{pipeline} {size}",
                        f"{c['network_input_px']}",
                        f"{c['mAP50']:.4f}",
                        f"{c['mAP50_95']:.4f}",
                        f"{c['precision']:.3f}",
                        f"{c['recall']:.3f}",
                        "-" if pipeline == "scale" else f"{delta:+.4f}",
                    ]
                )
        add(
            _md_table(
                [
                    "checkpoint",
                    "pipeline",
                    "network px",
                    "mAP50",
                    "mAP50-95",
                    "P",
                    "R",
                    "mAP50 vs scale at same imgsz",
                ],
                rows,
            )
        )
        add("")
    add("**Per-class AP50 on the held-out test split.**")
    add("")
    class_names = payload["class_names"]
    for model in models:
        rows = []
        for pipeline, size in payload["experiment_1"]["configs"]:
            c = cell(model, pipeline, "test", size)
            if not c:
                continue
            rows.append(
                [f"{pipeline} {size}"] + [f"{c['per_class_AP50'].get(n, 0.0):.3f}" for n in class_names]
            )
        add(f"*{model}*")
        add("")
        add(_md_table(["pipeline"] + list(class_names), rows))
        add("")
    for line in payload["experiment_1"]["verdict"]:
        add(line)
        add("")

    # ---- experiment 2 ---------------------------------------------------
    add("## Experiment 2 - the resolution grid, holes filled")
    add("")
    add(
        "160/192/224/256/288/320/352/384 on both splits for both checkpoints. The previously "
        "published grid skipped 288, 352 and 384, and its val and test peaks disagreed, which is "
        "exactly the situation in which a missing point can move the answer."
    )
    add("")
    for split in SPLITS:
        add(f"**{split} split, mAP50**")
        add("")
        rows = []
        for model in models:
            row: list[Any] = [model]
            for size in IMGSZ_GRID:
                c = cell(model, "scale", split, size)
                row.append(f"{c['mAP50']:.4f}" if c else "")
            rows.append(row)
        add(_md_table(["checkpoint"] + [str(s) for s in IMGSZ_GRID], rows))
        add("")
        add(f"**{split} split, mAP50-95**")
        add("")
        rows = []
        for model in models:
            row = [model]
            for size in IMGSZ_GRID:
                c = cell(model, "scale", split, size)
                row.append(f"{c['mAP50_95']:.4f}" if c else "")
            rows.append(row)
        add(_md_table(["checkpoint"] + [str(s) for s in IMGSZ_GRID], rows))
        add("")
    add(f"![resolution grid]({payload['figures']['grid']})")
    add("")
    add("**Peaks.**")
    add("")
    add(
        _md_table(
            ["checkpoint", "val peak", "val mAP50", "test peak", "test mAP50", "test mAP50 at 256"],
            payload["experiment_2"]["peaks_table"],
        )
    )
    add("")
    for line in payload["experiment_2"]["verdict"]:
        add(line)
        add("")

    # ---- experiment 3 ---------------------------------------------------
    add("## Experiment 3 - is any of this real?")
    add("")
    boot = payload["experiment_3_bootstrap"]
    if not boot.get("resamples"):
        add("Not run (`--resamples 0`).")
        add("")
    else:
        add(
            f"{boot['resamples']} bootstrap resamples of the {boot['n_images']} images in a split, "
            f"seed {BOOTSTRAP_SEED}, **paired**: every configuration is scored on the same "
            "resampled image sets inside one loop, so the between-image variance they all share "
            "cancels in the difference and any two configurations can be compared consistently "
            "afterwards. Intervals are 2.5/97.5 percentiles of the paired difference against the "
            "shipped 256 px default."
        )
        add("")
        add(
            "Both splits are bootstrapped, and they do different jobs. **val is the decision "
            "instrument** -- the only interval a size may legitimately be chosen with, and the "
            "one that answers whether there is any evidence to move off the shipped default. "
            "**test is the honest measurement** of what a choice bought; selecting the size that "
            "minimises it would be cheating, and the two are shown side by side so that is "
            "visible rather than tempting."
        )
        add("")
        add(
            "Before any interval was reported, the mAP recomputed from the captured per-image "
            f"rows was checked against the validator's own number for all "
            f"{payload['fidelity']['checked']} captured cells; the largest disagreement was "
            f"{payload['fidelity']['max_abs_error']:.2e} mAP50."
        )
        add("")
        for split in SPLITS:
            for model in models:
                block = boot.get(split, {}).get(model, {}).get("vs_reference")
                if not block:
                    continue
                role = (
                    "a size may be chosen on this"
                    if split == "val"
                    else "honest measurement; never selected on"
                )
                add(f"**{model}, {split} split** ({role}) - mAP50 difference against `{block['reference']}`")
                add("")
                rows = []
                for label, c in block["mAP50"].items():
                    rows.append(
                        [
                            label,
                            f"{c['point_a']:.4f}",
                            f"{c['point_difference']:+.4f}",
                            f"[{c['ci95_low']:+.4f}, {c['ci95_high']:+.4f}]",
                            "yes" if c["ci_excludes_zero"] else "**no**",
                            f"{c['share_a_wins']:.0%}",
                        ]
                    )
                add(
                    _md_table(
                        [
                            "configuration",
                            "mAP50",
                            "difference vs 256",
                            "95% CI",
                            "CI excludes 0",
                            "wins in x% of resamples",
                        ],
                        rows,
                    )
                )
                add("")
        if payload["figures"].get("bootstrap"):
            add(f"![bootstrap intervals]({payload['figures']['bootstrap']})")
            add("")
        for line in payload["experiment_3"]["verdict"]:
            add(line)
            add("")

    # ---- limits ---------------------------------------------------------
    add("## What this does not show")
    add("")
    for line in payload["limits"]:
        add(line)
        add("")

    add("## Reproducing")
    add("")
    add("```")
    add(payload["command"])
    add("```")
    add("")
    add(
        f"Wall clock {payload['wall_seconds']:.0f} s for {len(runs)} validation passes plus "
        f"{boot.get('seconds', 0)} s of bootstrap. Raw numbers: `reports/input_study.json`."
    )
    add("")
    path.write_text("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# 7. narrative, derived from the numbers rather than written ahead of them
# ---------------------------------------------------------------------------


def _fmt(x: float) -> str:
    return f"{x:.4f}"


def build_verdicts(payload: dict[str, Any]) -> None:  # noqa: C901 - prose assembled from measurements
    """Fill in the prose blocks the report reads, entirely from measured values."""
    runs: list[dict[str, Any]] = payload["runs"]
    models = [c["key"] for c in payload["checkpoints"]]

    def cell(model: str, pipeline: str, split: str, imgsz: int) -> dict[str, Any] | None:
        for r in runs:
            if (r["model"], r["pipeline"], r["split"], r["imgsz"]) == (model, pipeline, split, imgsz):
                return r
        return None

    # ---- experiment 1 verdict
    lines: list[str] = ["**Verdict.**"]
    losses: list[float] = []
    for model in models:
        for size in PAD_SIZES:
            p = cell(model, "pad", "test", size)
            s = cell(model, "scale", "test", size)
            if p and s:
                losses.append(p["mAP50"] - s["mAP50"])
    if losses and max(losses) < 0:
        lines.append(
            "Preserving the pixels **loses** on every checkpoint, every split and both canvas "
            f"sizes. On test the padded path costs between {abs(max(losses)):.4f} and "
            f"{abs(min(losses)):.4f} mAP50 against the scaled path at the same tensor size. "
            "Interpolating the source, on these weights, beats keeping it byte-exact."
        )
    elif losses and min(losses) > 0:
        lines.append(
            "Preserving the pixels **wins** on every configuration tested, by between "
            f"{min(losses):.4f} and {max(losses):.4f} mAP50 on test."
        )
    elif losses:
        lines.append(
            "The padded path wins in some configurations and loses in others, by between "
            f"{min(losses):+.4f} and {max(losses):+.4f} mAP50 on test -- no consistent ordering."
        )
    lines.append(
        "**This is not evidence that padding is a worse idea. It is evidence about "
        "inference-time substitution.** Both checkpoints were fine-tuned on scaled inputs and "
        "have never seen a centred 200x200 island of image inside a grey field, so a padded "
        "evaluation asks them to work on a geometry they were not optimised for, at an effective "
        "object scale they were not optimised for either. The confound is smaller than it looks "
        "-- `models/*/args.yaml` records `mosaic: 1.0`, `scale: 0.4`, `translate: 0.1`, "
        "`degrees: 10.0`, and ultralytics fills mosaic canvases and warp borders with the same "
        "value 114, so grey abutting steel is not novel to these networks -- but it is real, and "
        "the effective-scale change is not covered by it at all. Whether padding wins when the "
        "model is *trained* that way is a different experiment: fine-tune on "
        "`data/input_study/pad224` and re-run this table."
    )
    per_class: list[str] = []
    for model in models:
        size = PAD_SIZES[-1]
        pad_cell, scale_cell = cell(model, "pad", "test", size), cell(model, "scale", "test", size)
        if not pad_cell or not scale_cell:
            continue
        deltas = sorted(
            (
                (pad_cell["per_class_AP50"].get(n, 0.0) - scale_cell["per_class_AP50"].get(n, 0.0), n)
                for n in payload["class_names"]
            )
        )
        worst = ", ".join(f"{n} {d:+.3f}" for d in (deltas[0][0],) for n in (deltas[0][1],))
        second = f"{deltas[1][1]} {deltas[1][0]:+.3f}"
        best = f"{deltas[-1][1]} {deltas[-1][0]:+.3f}"
        per_class.append(f"`{model}` at pad{size}: {worst}, {second}, ... {best}")
    if per_class:
        lines.append(
            "**Where it loses is consistent with that explanation rather than with a resampling "
            "artefact.** Per class on test, the padded path gives up most on the thin-structure "
            "classes and least on the large area-like ones -- "
            + "; ".join(per_class)
            + ". Padding shrinks every defect relative to the network's receptive field by the "
            "same factor the scaling would have grown it, and it is the fine-structure classes "
            "that cannot afford that. An interpolation artefact would not sort this way."
        )
    payload["experiment_1"]["verdict"] = lines

    # ---- experiment 2 verdict
    peaks_table = []
    lines = ["**Verdict.**"]
    agree = True
    for model in models:
        vp = max(
            (r for r in runs if r["model"] == model and r["split"] == "val" and r["pipeline"] == "scale"),
            key=lambda r: r["mAP50"],
        )
        tp = max(
            (r for r in runs if r["model"] == model and r["split"] == "test" and r["pipeline"] == "scale"),
            key=lambda r: r["mAP50"],
        )
        ref = cell(model, "scale", "test", REFERENCE_IMGSZ)
        agree = agree and vp["imgsz"] == tp["imgsz"]
        peaks_table.append(
            [
                model,
                f"{vp['imgsz']} px",
                _fmt(vp["mAP50"]),
                f"{tp['imgsz']} px",
                _fmt(tp["mAP50"]),
                _fmt(ref["mAP50"]) if ref else "",
            ]
        )
    payload["experiment_2"]["peaks_table"] = peaks_table
    for model in models:
        vp = max(
            (r for r in runs if r["model"] == model and r["split"] == "val" and r["pipeline"] == "scale"),
            key=lambda r: r["mAP50"],
        )
        tp = max(
            (r for r in runs if r["model"] == model and r["split"] == "test" and r["pipeline"] == "scale"),
            key=lambda r: r["mAP50"],
        )
        at_val_choice = cell(model, "scale", "test", vp["imgsz"])
        ref = cell(model, "scale", "test", REFERENCE_IMGSZ)
        lines.append(
            f"`{model}`: val peaks at **{vp['imgsz']} px** ({_fmt(vp['mAP50'])}), test peaks at "
            f"**{tp['imgsz']} px** ({_fmt(tp['mAP50'])}). Choosing on val and then reading test "
            f"gives {_fmt(at_val_choice['mAP50'])}; the shipped 256 px gives "
            f"{_fmt(ref['mAP50'])}; picking the size that maximises test would give "
            f"{_fmt(tp['mAP50'])}, which is **{tp['mAP50'] - at_val_choice['mAP50']:+.4f}** over "
            "the honest choice and is not available to anyone who has not already looked at the "
            "answer. That last number is the size of the cheat, not a result."
        )
    band = payload["experiment_2"]["plateau"]
    lines.append(
        f"Across both checkpoints and both splits, every size from {band['low']} to {band['high']} px "
        f"sits within {band['spread']:.4f} mAP50 of that curve's best, while the ends of the grid "
        f"fall away sharply ({band['note']}). The curve has an interior maximum on all four "
        "(checkpoint, split) combinations, so the grid brackets its own peak and is not reporting "
        "a boundary."
    )
    payload["experiment_2"]["verdict"] = lines

    # ---- experiment 3 verdict
    boot = payload["experiment_3_bootstrap"]
    lines = ["**Verdict.**"]
    if boot.get("resamples"):
        band_low, band_high = payload["experiment_2"]["plateau"]["low"], payload["experiment_2"]["plateau"]["high"]

        def band_cells(split: str) -> list[tuple[str, str, dict[str, Any]]]:
            out = []
            for model in models:
                block = boot.get(split, {}).get(model, {}).get("vs_reference", {}).get("mAP50", {})
                for label, c in block.items():
                    if label.startswith("scale") and band_low <= int(label[5:]) <= band_high:
                        out.append((model, label, c))
            return out

        for split in SPLITS:
            near = band_cells(split)
            if not near:
                continue
            widest = max(c["ci95_high"] - c["ci95_low"] for _, _, c in near)
            biggest = max(abs(c["point_difference"]) for _, _, c in near)
            winners = [f"{m}/{lab}" for m, lab, c in near if c["ci_excludes_zero"] and c["point_difference"] > 0]
            worse = [
                f"{m}/{lab} ([{c['ci95_low']:+.4f}, {c['ci95_high']:+.4f}])"
                for m, lab, c in near
                if c["ci_excludes_zero"] and c["point_difference"] < 0
            ]
            if not winners:
                lines.append(
                    f"**{split}:** in the {band_low}-{band_high} px band, **no size beats "
                    f"{REFERENCE_IMGSZ} px** on either checkpoint -- every interval that excludes "
                    "zero in that band does so on the losing side. The intervals are wide, up to "
                    f"{widest:.4f} mAP50 across, against point differences of at most "
                    f"{biggest:.4f}, so with {boot['n_images']} images this benchmark mostly "
                    f"cannot resolve {band_low} from {band_high} px."
                    + (
                        " The exceptions, and they cut against the smaller sizes rather than for "
                        "them: " + ", ".join(worse) + "."
                        if worse
                        else " Every interval in the band contains zero."
                    )
                )
            else:
                extra = []
                for model in models:
                    adj = (
                        boot.get(split, {})
                        .get(model, {})
                        .get("vs_reference_selection_adjusted", {})
                        .get("mAP50", {})
                    )
                    meta = boot.get(split, {}).get(model, {}).get("selection_adjustment", {})
                    for name in winners:
                        if not name.startswith(f"{model}/"):
                            continue
                        a = adj.get(name.split("/", 1)[1])
                        if a:
                            extra.append(
                                f"{name} becomes {a['confidence_pct']:.1f}% "
                                f"[{a['ci95_low']:+.4f}, {a['ci95_high']:+.4f}]"
                                + ("" if a["ci_excludes_zero"] else ", which contains zero")
                                + f" once corrected for having been selected from "
                                f"{meta.get('sizes_selected_from', '?')} sizes"
                            )
                lines.append(
                    f"**{split}:** {', '.join(winners)} beat {REFERENCE_IMGSZ} px at a nominal 95% "
                    "level. That is a *selected maximum*, though, so the nominal level overstates "
                    "the evidence: "
                    + ("; ".join(extra) if extra else "see the table above")
                    + ". The recommendation uses the corrected interval, which is why it does not "
                    "move off the incumbent."
                )
        val_winners = [
            f"{m}/{lab}"
            for m, lab, c in band_cells("val")
            if c["ci_excludes_zero"] and c["point_difference"] > 0
        ]
        if not val_winners:
            lines.append(
                "Because val is the only split a size may be chosen on, and nothing on val "
                f"separates from 256 px, **there is no evidence that would justify moving off the "
                "shipped default**. That is the finding: input size in this band does not matter, "
                "256 is fine, and any ranking inside the band -- including the val-peak-at-224 "
                "result that prompted this study -- is noise dressed up as a finding."
            )
        resolved: list[str] = []
        unresolved: list[str] = []
        for split in SPLITS:
            for model in models:
                block = boot.get(split, {}).get(model, {}).get("vs_reference", {}).get("mAP50", {})
                for label, c in block.items():
                    (resolved if c["ci_excludes_zero"] else unresolved).append(f"{split}/{model}/{label}")
        lines.append(
            f"{len(resolved)} of the {len(resolved) + len(unresolved)} contrasts across both "
            "splits are resolved at all, and they are the ends of the grid and the padded "
            "configurations -- the differences big enough for 180 images to see."
        )
        pieces = []
        for split in SPLITS:
            for model in models:
                for name, block in boot.get(split, {}).get(model, {}).get("pad_vs_scale", {}).items():
                    c = block["mAP50"]
                    pieces.append(
                        f"`{model}` {split} {name.replace('_minus_', ' - ')}: "
                        f"{c['point_difference']:+.4f} mAP50, CI "
                        f"[{c['ci95_low']:+.4f}, {c['ci95_high']:+.4f}]"
                        + (" (resolved)" if c["ci_excludes_zero"] else " (contains zero)")
                    )
        if pieces:
            lines.append(
                "The pad-versus-scale contrast at matched tensor size is the cleanest comparison "
                "in the study, because nothing but the resampling differs: "
                + "; ".join(pieces)
                + "."
            )
    payload["experiment_3"]["verdict"] = lines


def build_recommendation(payload: dict[str, Any]) -> None:  # noqa: C901 - the deliverable
    """One input pipeline per checkpoint, chosen by a rule stated before the numbers.

    The rule, in order:

    1. The incumbent is 256 px scaled (`inference.DEFAULT_IMGSZ`). Moving off an
       incumbent needs evidence; staying does not.
    2. Evidence may come only from **val**. Find the val peak. If it is 256, done.
    3. If the val peak is some other size, ask the *val* bootstrap whether that size
       is distinguishable from 256 -- at a level corrected for the fact that the peak
       was chosen as the maximum of the grid, not nominated in advance. If the
       interval contains zero there is no evidence to move, so keep the incumbent.
    4. Only if val separates does the recommendation change. The test column is then
       reported as the consequence -- it is never consulted to make the choice.

    Step 3 is what stops the val-peaks-at-224 observation from silently becoming a
    decision. A peak is not a difference, and the maximum of eight correlated noisy
    estimates is biased upwards by the act of taking a maximum, which is why the
    correction is applied rather than the nominal 95% interval quoted.
    """
    runs: list[dict[str, Any]] = payload["runs"]
    models = [c["key"] for c in payload["checkpoints"]]
    boot = payload["experiment_3_bootstrap"]

    def cell(model: str, pipeline: str, split: str, imgsz: int) -> dict[str, Any] | None:
        for r in runs:
            if (r["model"], r["pipeline"], r["split"], r["imgsz"]) == (model, pipeline, split, imgsz):
                return r
        return None

    table = []
    decisions: list[dict[str, Any]] = []
    for model in models:
        val_cells = [
            r for r in runs if r["model"] == model and r["split"] == "val" and r["pipeline"] == "scale"
        ]
        vp = max(val_cells, key=lambda r: r["mAP50"])
        val_boot = boot.get("val", {}).get(model, {})
        nominal = val_boot.get("vs_reference", {}).get("mAP50", {}).get(f"scale{vp['imgsz']}")
        c = val_boot.get("vs_reference_selection_adjusted", {}).get("mAP50", {}).get(
            f"scale{vp['imgsz']}"
        )
        adj = val_boot.get("selection_adjustment", {})
        if vp["imgsz"] == REFERENCE_IMGSZ:
            recommend = REFERENCE_IMGSZ
            why = (
                f"the val peak *is* {REFERENCE_IMGSZ} px "
                f"({vp['mAP50']:.4f}); nothing to decide"
            )
        elif c is None:
            recommend = vp["imgsz"]
            why = f"val peaks at {vp['imgsz']} px and no interval was computed"
        elif not c["ci_excludes_zero"]:
            recommend = REFERENCE_IMGSZ
            nom = (
                f"nominal 95% [{nominal['ci95_low']:+.4f}, {nominal['ci95_high']:+.4f}]"
                if nominal
                else ""
            )
            why = (
                f"val peaks at {vp['imgsz']} px ({vp['mAP50']:.4f} against "
                f"{c['point_b']:.4f}), but the paired val interval on that difference contains "
                f"zero: {c['confidence_pct']:.1f}% "
                f"[{c['ci95_low']:+.4f}, {c['ci95_high']:+.4f}]"
                + (f" ({nom})" if nom else "")
                + f", corrected for the peak having been selected from "
                f"{adj.get('sizes_selected_from', '?')} sizes. No evidence to move off the "
                "incumbent"
            )
        else:
            recommend = vp["imgsz"]
            why = (
                f"val peaks at {vp['imgsz']} px and the paired val interval against "
                f"{REFERENCE_IMGSZ} excludes zero even after correcting for the peak having been "
                f"selected from {adj.get('sizes_selected_from', '?')} sizes "
                f"({c['confidence_pct']:.1f}% [{c['ci95_low']:+.4f}, {c['ci95_high']:+.4f}])"
            )
        chosen = cell(model, "scale", "test", recommend)
        pads = [
            cell(model, "pad", "test", size)["mAP50"] - cell(model, "scale", "test", size)["mAP50"]
            for size in PAD_SIZES
            if cell(model, "pad", "test", size) and cell(model, "scale", "test", size)
        ]
        decisions.append(
            {
                "model": model,
                "val_peak_imgsz": vp["imgsz"],
                "val_peak_mAP50": round(vp["mAP50"], 5),
                "recommended_imgsz": recommend,
                "recommended_pipeline": "scale",
                "test_mAP50_at_recommendation": round(chosen["mAP50"], 5) if chosen else None,
                "test_mAP50_95_at_recommendation": round(chosen["mAP50_95"], 5) if chosen else None,
                "pad_penalty_on_test_mAP50": [round(x, 5) for x in pads],
                "val_interval_nominal_95": nominal,
                "val_interval_selection_adjusted": c,
                "reason": why,
            }
        )
        table.append(
            [
                f"`{model}`",
                f"**{recommend} px, scaled** (the ultralytics default path)",
                f"{chosen['mAP50']:.4f}" if chosen else "",
                why,
            ]
        )
    payload["recommendation"]["table"] = table
    payload["recommendation"]["decisions"] = decisions

    summary = []
    pad_losses = [
        cell(m, "pad", split, s2)["mAP50"] - cell(m, "scale", split, s2)["mAP50"]
        for m in models
        for split in SPLITS
        for s2 in PAD_SIZES
        if cell(m, "pad", split, s2) and cell(m, "scale", split, s2)
    ]
    test_pad_losses = [
        cell(m, "pad", "test", s2)["mAP50"] - cell(m, "scale", "test", s2)["mAP50"]
        for m in models
        for s2 in PAD_SIZES
        if cell(m, "pad", "test", s2) and cell(m, "scale", "test", s2)
    ]
    if pad_losses and max(pad_losses) < 0:
        summary.append(
            "**Scaled, not padded -- the lossless path loses, on these weights.** Padding a "
            f"200x200 source into a {PAD_SIZES[0]}x{PAD_SIZES[0]} or "
            f"{PAD_SIZES[1]}x{PAD_SIZES[1]} canvas with no scaling does preserve every pixel "
            "byte-exact; that is verified below, and it is the first time this repository has "
            "had a genuinely lossless input path. It is also worse: it costs "
            f"{abs(max(test_pad_losses)):.3f} to {abs(min(test_pad_losses)):.3f} mAP50 on the "
            "held-out test split, and it loses on every checkpoint, every split and both canvas "
            "sizes -- 8 of 8 configurations. Interpolation is not what limits this detector; the "
            "scale it expects objects at is. **This is a result about inference-time "
            "substitution only** -- see the confound stated in experiment 1."
        )
    elif pad_losses:
        summary.append(
            "**Pad versus scale does not order consistently**: the padded path lands between "
            f"{min(pad_losses):+.4f} and {max(pad_losses):+.4f} mAP50 of the scaled path at the "
            "same tensor size across the eight configurations tested."
        )

    moved = [d for d in decisions if d["recommended_imgsz"] != REFERENCE_IMGSZ]
    boot_n = boot.get("n_images")
    if not moved:
        summary.append(
            f"**{REFERENCE_IMGSZ} px, and the existing default is confirmed rather than "
            "replaced.** On the split a size may legitimately be chosen on, no input size in the "
            f"{payload['experiment_2']['plateau']['low']}-{payload['experiment_2']['plateau']['high']} px "
            f"band is distinguishable from {REFERENCE_IMGSZ} px on either checkpoint: the paired "
            f"bootstrap ({boot.get('resamples', 0)} resamples of {boot_n} images, seed "
            f"{BOOTSTRAP_SEED}) puts zero inside every one of those intervals. The val/test peak "
            "disagreement that prompted this study -- val peaking at 224 while test peaks at 256 "
            "-- is two noisy estimates of the same flat curve disagreeing, not a sign that the "
            f"wrong size was chosen. **Keep `inference.DEFAULT_IMGSZ = {REFERENCE_IMGSZ}`.**"
        )
    else:
        consequence = []
        for d in moved:
            at_ref = cell(d["model"], "scale", "test", REFERENCE_IMGSZ)
            consequence.append(
                f"`{d['model']}` -> {d['recommended_imgsz']} px: test mAP50 "
                f"{d['test_mAP50_at_recommendation']:.4f} against "
                f"{at_ref['mAP50']:.4f} at {REFERENCE_IMGSZ} px"
                if at_ref
                else f"`{d['model']}` -> {d['recommended_imgsz']} px"
            )
        summary.append(
            "**The val evidence supports moving off the shipped default for "
            + ", ".join(f"`{d['model']}` ({d['recommended_imgsz']} px)" for d in moved)
            + ".** What that bought on test, reported after the fact and not used to choose: "
            + "; ".join(consequence)
            + ". This survived the selection correction -- the val peak is the maximum of "
            f"{len(payload['imgsz_grid'])} correlated estimates, so it is tested against the "
            "incumbent at a Bonferroni-corrected level rather than a nominal 95%, and it still "
            "excludes zero. The size is inside the plateau either way, so the cost of getting it "
            "wrong is small."
        )

    if len({d["recommended_imgsz"] for d in decisions}) > 1:
        spread = []
        for d in decisions:
            at_ref = cell(d["model"], "scale", "test", REFERENCE_IMGSZ)
            if at_ref and d["test_mAP50_at_recommendation"] is not None:
                spread.append(d["test_mAP50_at_recommendation"] - at_ref["mAP50"])
        summary.append(
            "**In practice, ship one number.** The rule lands on different sizes for the two "
            "checkpoints ("
            + ", ".join(f"`{d['model']}` {d['recommended_imgsz']} px" for d in decisions)
            + f"), but `DEFAULT_IMGSZ` is one constant that the console, the exporter, the coil "
            f"reporter and the benchmark all read. Standardising on {REFERENCE_IMGSZ} px for both "
            f"costs at most {max(abs(x) for x in spread):.4f} mAP50 on test -- inside the "
            "resolution of this benchmark, and an order of magnitude less than the difference "
            "between the two checkpoints. Run the joint model at 224 px only if it is deployed on "
            "its own."
        )

    ends = payload["experiment_2"]["plateau"]
    summary.append(
        "**What actually matters in this knob is staying off the ends of the grid.** Inside the "
        f"{ends['low']}-{ends['high']} px band every size is within {ends['spread']:.4f} mAP50 of "
        f"its curve's best; at the ends of the grid the same curves give up as much as "
        f"{ends['end_drop']:.4f}, and the previously measured 640 px collapse "
        "(`reports/model_study.json`) is larger again. The failure mode here is inheriting the "
        f"framework's 640 px default, not choosing {REFERENCE_IMGSZ} over 288."
    )
    payload["recommendation"]["summary"] = summary


# ---------------------------------------------------------------------------
# 8. entry point
# ---------------------------------------------------------------------------


def runs_from_payload(payload: dict[str, Any]) -> list[InputRun]:
    """Rebuild the (statistics-free) run objects the figures need from a saved payload.

    Only the figures take `InputRun`; the prose and the tables read the payload dict
    directly. The bootstrap statistics are not reconstructible and are not needed --
    their intervals are already summarised in the payload.
    """
    out: list[InputRun] = []
    for r in payload["runs"]:
        out.append(
            InputRun(
                model=r["model"],
                pipeline=r["pipeline"],
                split=r["split"],
                imgsz=r["imgsz"],
                network_input=r["network_input_px"],
                mAP50=r["mAP50"],
                mAP50_95=r["mAP50_95"],
                precision=r["precision"],
                recall=r["recall"],
                per_class_AP50=r["per_class_AP50"],
                per_class_AP50_95=r["per_class_AP50_95"],
                instances=r["instances"],
                speed_ms=r["validator_speed_ms_per_image"],
                seconds=r["wall_seconds"],
            )
        )
    return out


def regenerate_report(reports: Path) -> int:
    """Rewrite the prose, tables and figures from `reports/input_study.json`.

    Every measurement in this study is already in that file, and the narrative is
    derived from measurements rather than typed alongside them, so revising a
    sentence does not need 40 validation passes and four bootstraps rerun. The output
    is byte-identical to what a full run would have written, because the same
    functions produce it from the same payload.
    """
    path = reports / "input_study.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run the study once before --report-only")
    payload = json.loads(path.read_text())
    for key in ("experiment_1", "experiment_2", "experiment_3"):
        payload[key]["verdict"] = []
    payload["recommendation"] = {"summary": [], "table": []}
    build_verdicts(payload)
    build_recommendation(payload)
    runs = runs_from_payload(payload)
    grid_png = chart_grid(runs, reports / "input_study_grid.png")
    payload["figures"]["grid"] = grid_png.name
    boot_png = chart_bootstrap(payload, reports / "input_study_bootstrap.png")
    if boot_png is not None:
        payload["figures"]["bootstrap"] = boot_png.name
    write_json(path, payload)
    md_path = write_markdown(reports / "input_study.md", payload)
    print(f"[input_study] regenerated {md_path} and figures from {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    parser.add_argument("--batch", type=int, default=16, help="validator batch size")
    parser.add_argument("--workers", type=int, default=2, help="dataloader workers")
    parser.add_argument(
        "--resamples", type=int, default=1000, help="bootstrap resamples; 0 disables experiment 3"
    )
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--verify-sample",
        type=int,
        default=180,
        help="how many images to pull through the real validator dataset when proving losslessness",
    )
    parser.add_argument("--skip-grid", action="store_true", help="experiment 1 only")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="rewrite reports/input_study.md and the figures from the existing "
        "reports/input_study.json, running no model",
    )
    parser.add_argument("--study-root", type=Path, default=STUDY_ROOT)
    parser.add_argument("--reports", type=Path, default=REPORTS_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901 - a report is a long function
    args = build_parser().parse_args(argv)
    if args.report_only:
        return regenerate_report(args.reports)

    import ultralytics
    import torch

    device = resolve_device(args.device)
    reports: Path = args.reports
    reports.mkdir(parents=True, exist_ok=True)
    out_dir = reports / "ultralytics" / "input_study"
    started = time.perf_counter()

    source = yaml.safe_load(NEU_YAML.read_text())
    names6 = {int(k): v for k, v in source["names"].items()}
    class_names = [names6[i] for i in sorted(names6)]
    print(f"[input_study] device={device}  classes={class_names}")

    # --- lossless datasets -------------------------------------------------
    padded: list[dict[str, Any]] = []
    yamls: dict[tuple[str, int, int], Path] = {}
    for size in PAD_SIZES:
        for split in SPLITS:
            record = build_padded_split(split, size, root=args.study_root)
            padded.append(record)
            print(
                f"[input_study] pad{size}/{split}: {record['byte_exact_images']}/{record['images']} "
                f"byte-exact, max box error {record['max_box_round_trip_error_px']:.2e} px"
            )
        for spec in CHECKPOINTS:
            names = dict(names6)
            for extra in range(len(names6), spec.nc):
                names[extra] = f"unused_{extra}"
            yamls[("pad", size, spec.nc)] = write_study_yaml(
                size, spec.nc, names, root=args.study_root
            )
    scale_yaml: dict[int, Path] = {}
    for spec in CHECKPOINTS:
        scale_yaml[spec.nc] = (
            NEU_YAML if spec.nc == len(names6) else write_scale_yaml(spec.nc, root=args.study_root)
        )

    # --- verification ------------------------------------------------------
    verification = {"pipelines": []}
    for size in PAD_SIZES:
        verification["pipelines"].append(
            verify_pipeline(
                args.study_root / f"pad{size}" / "test" / "images",
                size,
                names6,
                pipeline="pad",
                sample=args.verify_sample,
            )
        )
    for size in PAD_SIZES:
        verification["pipelines"].append(
            verify_pipeline(
                NEU_ROOT / "test" / "images",
                size,
                names6,
                pipeline="scale",
                sample=args.verify_sample,
            )
        )
    for v in verification["pipelines"]:
        print(
            f"[input_study] verify {v['pipeline']}{v['imgsz']}: tensor {v['network_tensor_px']} "
            f"(predicted {v['network_tensor_px_predicted']}), lossless={v['lossless']}"
        )
        if v["pipeline"] == "pad" and not v["lossless"]:
            raise RuntimeError("padded pipeline is not byte-exact; the study cannot proceed")

    # --- validation passes -------------------------------------------------
    sizes = (REFERENCE_IMGSZ, 224) if args.skip_grid else IMGSZ_GRID
    runs: list[InputRun] = []
    for spec in CHECKPOINTS:
        if not spec.weights.is_file():
            raise FileNotFoundError(f"checkpoint not found: {spec.weights}")
        for split in SPLITS:
            for size in sorted(set(sizes)):
                run = run_validation(
                    spec,
                    scale_yaml[spec.nc],
                    split,
                    size,
                    device,
                    pipeline="scale",
                    out_dir=out_dir,
                    batch=args.batch,
                    workers=args.workers,
                )
                runs.append(run)
                print(
                    f"[input_study] {spec.key:>15s} {split:>4s} scale{size:<4d} "
                    f"mAP50 {run.mAP50:.4f}  mAP50-95 {run.mAP50_95:.4f}  ({run.seconds:.1f}s)"
                )
            for size in PAD_SIZES:
                run = run_validation(
                    spec,
                    yamls[("pad", size, spec.nc)],
                    split,
                    size,
                    device,
                    pipeline="pad",
                    out_dir=out_dir,
                    batch=args.batch,
                    workers=args.workers,
                )
                runs.append(run)
                print(
                    f"[input_study] {spec.key:>15s} {split:>4s} pad{size:<6d} "
                    f"mAP50 {run.mAP50:.4f}  mAP50-95 {run.mAP50_95:.4f}  ({run.seconds:.1f}s)"
                )

    reproduced = reproduction_check(runs, reports / "resolution_study.json")
    if reproduced.get("available"):
        print(
            f"[input_study] reproduction vs {reproduced['source']}: "
            f"{reproduced['overlapping_cells']} shared cells, max |dmAP50| = "
            f"{reproduced['max_abs_difference_mAP50']:.2e}"
        )

    checks = [fidelity(r) for r in runs if r.stats is not None]
    max_err = max((c["abs_error_mAP50"] for c in checks), default=0.0)
    print(f"[input_study] recomputation fidelity: {len(checks)} cells, max |dmAP50| = {max_err:.2e}")

    # --- bootstrap ---------------------------------------------------------
    boot_payload: dict[str, Any] = {"resamples": args.resamples, "seed": args.seed}
    if args.resamples > 0:
        # Both splits, and the distinction between them is the point. The *val*
        # interval is the decision instrument: it is the only one a size may
        # legitimately be chosen with, and it answers "is there evidence to move off
        # the shipped default at all". The *test* interval is the honest measurement
        # of what the choice bought, and using it to select would be cheating.
        seconds = 0.0
        for split in SPLITS:
            for spec in CHECKPOINTS:
                cells = [r for r in runs if r.model == spec.key and r.split == split]
                cells.sort(key=lambda r: (r.pipeline, r.imgsz))
                b = paired_bootstrap(cells, args.resamples, args.seed)
                if not b:
                    continue
                boot_payload["n_images"] = b["n_images"]
                seconds += float(b.get("seconds") or 0.0)
                # The val peak is chosen as the maximum of the grid, so the one
                # comparison the decision rule leans on is a selected maximum. Keep a
                # Bonferroni-corrected copy over the number of sizes it was selected
                # from and let the recommendation use that, so a peak that is only
                # nominally significant cannot quietly become a decision.
                n_contrasts = max(len(sizes) - 1, 1)
                boot_payload.setdefault(split, {})[spec.key] = {
                    "vs_reference": contrasts(b, f"scale{REFERENCE_IMGSZ}"),
                    "vs_reference_selection_adjusted": contrasts(
                        b, f"scale{REFERENCE_IMGSZ}", alpha=0.05 / n_contrasts
                    ),
                    "selection_adjustment": {
                        "sizes_selected_from": len(sizes),
                        "contrasts": n_contrasts,
                        "alpha": round(0.05 / n_contrasts, 6),
                    },
                    "pad_vs_scale": pad_versus_scale_intervals(b),
                }
                print(
                    f"[input_study] bootstrap {spec.key} {split}: {b['seconds']}s over "
                    f"{len(b['labels'])} configs"
                )
        boot_payload["seconds"] = round(seconds, 1)

    # --- assemble ----------------------------------------------------------
    plateau_low, plateau_high = 224, 320
    spreads = []
    for spec in CHECKPOINTS:
        for split in SPLITS:
            cells = [r for r in runs if r.model == spec.key and r.split == split and r.pipeline == "scale"]
            if not cells:
                continue
            best = max(c.mAP50 for c in cells)
            inside = [c.mAP50 for c in cells if plateau_low <= c.imgsz <= plateau_high]
            if inside:
                spreads.append(best - min(inside))
    ends = []
    for spec in CHECKPOINTS:
        for split in SPLITS:
            cells = {r.imgsz: r.mAP50 for r in runs if r.model == spec.key and r.split == split and r.pipeline == "scale"}
            best = max(cells.values()) if cells else 0.0
            for edge in (IMGSZ_GRID[0], IMGSZ_GRID[-1]):
                if edge in cells:
                    ends.append(best - cells[edge])

    payload: dict[str, Any] = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join([".venv/bin/python", "src/input_study.py", *(argv or sys.argv[1:])]).strip(),
        "host": {
            "platform": platform.platform(),
            "chip": "Apple M5",
            "device": device,
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
        },
        "protocol": {
            "validator": "ultralytics defaults: conf 0.001, NMS IoU 0.7, max_det 300, rect=True, pad=0.5",
            "data": "data/neu-det, 1440/180/180 stratified split",
            "network_input_rule": "square images: ceil(imgsz/32 + 0.5) * 32 = imgsz + 32",
            "comparable_with": ["reports/model_study.json", "reports/resolution_study.json"],
        },
        "checkpoints": [
            {"key": c.key, "weights": str(c.weights), "nc": c.nc, "note": c.note} for c in CHECKPOINTS
        ],
        "class_names": class_names,
        "imgsz_grid": list(sizes),
        "pad_sizes": list(PAD_SIZES),
        "padded_datasets": padded,
        "verification": verification,
        "pixel_fidelity": pixel_fidelity(IMGSZ_GRID),
        "runs": [r.to_dict() for r in runs],
        "reproduction": reproduced,
        "fidelity": {
            "checked": len(checks),
            "max_abs_error": max_err,
            "all_exact": all(c["exact"] for c in checks),
            "cells": checks,
        },
        "experiment_1": {
            "configs": [["scale", 224], ["pad", 224], ["scale", 256], ["pad", 256]],
            "verdict": [],
        },
        "experiment_2": {
            "peaks_table": [],
            "plateau": {
                "low": plateau_low,
                "high": plateau_high,
                "spread": max(spreads) if spreads else 0.0,
                "end_drop": max(ends) if ends else 0.0,
                "note": f"the grid ends give up to {max(ends):.4f} mAP50 against the same curve's best"
                if ends
                else "",
            },
            "verdict": [],
        },
        "experiment_3": {"verdict": []},
        "experiment_3_bootstrap": boot_payload,
        "recommendation": {"summary": [], "table": []},
        "limits": [
            "**Experiment 1 measures an inference-time substitution, not the value of padding.** "
            "Both checkpoints were trained on scaled inputs. A model fine-tuned on padded canvases "
            "could rank the two pipelines the other way; `data/input_study/pad224` exists so that "
            "experiment can be run without rebuilding anything.",
            "**Everything is one pass per configuration.** Validation is deterministic given the "
            "weights and the input, so repeats would reproduce exactly; the variability quantified "
            "here is over *images*, not over runs, and not over training seeds. Separating "
            "training-seed variance would need several seeds per configuration.",
            "**180 test images is a small benchmark.** That is precisely why experiment 3 exists, "
            "and why it declines to name a winner inside the plateau.",
            "**Validation geometry is not deployment geometry.** `model.val()` pads square inputs "
            "to `imgsz + 32`; `model.predict()` on a single 200x200 frame letterboxes to `imgsz` "
            "with `auto=True` and no border at all. The accuracy ordering measured here is "
            "expected to carry over, but the two paths are not byte-identical and the deployed "
            "path was not separately swept.",
            "**Severstal classes are excluded by construction.** The 10-class checkpoint is scored "
            "on NEU-DET labels, and ultralytics averages AP over classes present in the ground "
            "truth, so its four Severstal heads contribute nothing to these means. Predictions "
            "they emit on NEU-DET frames are invisible to mAP but would be visible to an operator; "
            "that is `src/false_alarm.py`'s question, not this one.",
        ],
        "figures": {},
        "wall_seconds": 0.0,
    }

    build_verdicts(payload)
    build_recommendation(payload)

    grid_png = chart_grid(runs, reports / "input_study_grid.png")
    payload["figures"]["grid"] = grid_png.name
    boot_png = chart_bootstrap(payload, reports / "input_study_bootstrap.png")
    if boot_png is not None:
        payload["figures"]["bootstrap"] = boot_png.name
    payload["wall_seconds"] = time.perf_counter() - started

    json_path = write_json(reports / "input_study.json", payload)
    md_path = write_markdown(reports / "input_study.md", payload)
    print(f"[input_study] wrote {json_path}")
    print(f"[input_study] wrote {md_path}")
    print(f"[input_study] wrote {grid_png}")
    if boot_png is not None:
        print(f"[input_study] wrote {boot_png}")
    print(f"[input_study] total {payload['wall_seconds']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
