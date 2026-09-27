"""Does resampling *damage* the image, or does magnification *help* the detector?

`src/input_study.py` established that the lossless padded path (a 200x200 NEU-DET
frame dropped verbatim into a 224 or 256 canvas, zero interpolation) loses 0.024 to
0.052 mAP50 against the scaled path at the same tensor size, on 8 of 8
configurations. That contrast is clean but it moves **two** things at once:

1. *fidelity* -- pad keeps every source pixel byte-exact, scale runs a bilinear
   kernel over them and invents intermediate grey levels;
2. *magnification* -- pad presents the defect at 1.00x, scale presents it at 1.28x
   relative to the network's fixed stride grid.

The report attributes the loss to (2) on per-class grounds (thin structure loses
0.11, area-like classes lose 0.006) but does not separate them, and the whole
project premise -- "resampling damages the source, so a non-resampling path should
win" -- is a claim about (1). This module separates them with one arm the study did
not have:

    **nearest-neighbour magnification.** `cv2.resize(src, (256, 256),
    INTER_NEAREST)` on a 200x200 source is an *exactly invertible* transform:
    the scale factor is > 1 so every one of the 200 source indices is hit by at
    least one of the 256 output indices, and `verify_nearest_is_invertible`
    recovers the source byte-for-byte by pure indexing. No grey level is invented,
    no detail is smoothed away, nothing is thrown out. It is every bit as lossless
    as padding -- and it magnifies by 1.28x, exactly like the bilinear path.

So the three-way comparison at one fixed tensor size (288 px) reads:

    pad256       lossless, 1.00x     <- input_study's candidate
    nearest256   lossless, 1.28x     <- this module
    bilinear256  lossy,    1.28x     <- the shipped path

If fidelity is what matters, `nearest256` and `pad256` should sit together above
`bilinear256`. If magnification is what matters, `nearest256` and `bilinear256`
should sit together above `pad256`.

Two controls keep the instrument honest:

* `bilinear256` is pre-computed on disk with the identical `cv2.resize` call
  `BaseDataset.load_image` makes for an upscale, so validating it at `imgsz=256`
  (where the loader's own scale factor becomes exactly 1.0 and it does nothing)
  must reproduce the framework's own `scale 256` number to the last digit. If it
  does not, the on-disk substitution is not neutral and nothing else here counts.
* `decimated256` throws real information away -- INTER_AREA down to 160x160, then
  bilinear back up to 256x256 -- at the *same* 1.28x magnification as the other
  two. It is what genuine pixel destruction looks like to this benchmark, and it
  exists so that a null result on `nearest256` cannot be waved away as "180 images
  cannot see a fidelity difference at all".

Everything shares `src/input_study.py`'s instrument: its `run_validation` (the
framework's own validator on its own protocol defaults), its per-image statistics
capture, its paired bootstrap, its seed. Numbers from here are directly
comparable with `reports/input_study.json`.

    .venv/bin/python src/input_pipeline_probe.py --resamples 1000

Writes `reports/input_pipeline_probe.json`. Conclusions live in
`reports/input_pipeline_decision.md`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from input_study import (  # noqa: E402
    CHECKPOINTS,
    NEU_ROOT,
    PAD_VALUE,
    REPORTS_DIR,
    STUDY_ROOT,
    CheckpointSpec,
    InputRun,
    interval,
    network_input_size,
    paired_bootstrap,
    run_validation,
    _val_dataset,  # noqa: F401  (re-exported for the tensor check below)
    _tensor_to_bgr,
)

# The one size everything is compared at. 256 is `inference.DEFAULT_IMGSZ`, and at
# `imgsz=256` a 256x256 file on disk passes through `load_image` untouched (its
# scale factor is 1.0), so the network tensor is the file plus a 16 px grey border.
PROBE_IMGSZ = 256

NATIVE_SIZE = 200

# Same seed and resample count as src/input_study.py, so an interval from here and
# an interval from there are drawn from the same bootstrap design.
BOOTSTRAP_SEED = 1337
DEFAULT_RESAMPLES = 1000

NEU_NAMES: dict[int, str] = {
    0: "crazing",
    1: "inclusion",
    2: "patches",
    3: "pitted_surface",
    4: "rolled-in_scale",
    5: "scratches",
}


def _nearest(image: np.ndarray, size: int) -> np.ndarray:
    """Lossless magnification: pixel replication, exactly invertible for an upscale."""
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_NEAREST)


def _bilinear(image: np.ndarray, size: int) -> np.ndarray:
    """Exactly the call `BaseDataset.load_image` makes when its scale factor is > 1."""
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)


def _decimated(image: np.ndarray, size: int) -> np.ndarray:
    """Same magnification, real information destroyed: 200 -> 160 (AREA) -> 256.

    The down leg is a 0.8x INTER_AREA decimation, which is the one operation in this
    module that genuinely cannot be undone -- 36% of the source samples stop existing.
    The up leg restores the geometry so that the network sees defects at the same
    1.28x as the other two arms and nothing but fidelity differs.
    """
    small = cv2.resize(image, (160, 160), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (size, size), interpolation=cv2.INTER_LINEAR)


KERNELS: dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "nearest": _nearest,
    "bilinear": _bilinear,
    "decimated": _decimated,
}

KERNEL_NOTE: dict[str, str] = {
    "nearest": "lossless magnification: pixel replication, source exactly recoverable",
    "bilinear": "the shipped path, pre-computed on disk; lossy, smooths detail",
    "decimated": "control: 1.28x magnification with 36% of the source samples destroyed",
}


# ---------------------------------------------------------------------------
# 1. materialise the derived splits
# ---------------------------------------------------------------------------


def verify_nearest_is_invertible(image: np.ndarray, size: int) -> dict[str, Any]:
    """Prove, on a real frame, that the nearest-neighbour upscale loses nothing.

    OpenCV's INTER_NEAREST maps output index `j` to source index `floor(j * src/dst)`.
    For an upscale that map is surjective onto the source indices, so picking the first
    output index for each source index inverts it exactly. This is checked rather than
    argued, because the entire value of the arm rests on it.
    """
    src = image.shape[0]
    grown = _nearest(image, size)
    forward = np.floor(np.arange(size) * src / size).astype(int)
    convention_holds = bool(np.array_equal(grown, image[forward][:, forward]))
    first: dict[int, int] = {}
    for out_index, source_index in enumerate(forward):
        first.setdefault(int(source_index), out_index)
    covered = len(first) == src
    inverse = np.array([first[i] for i in range(src)]) if covered else np.arange(src)
    recovered = grown[inverse][:, inverse]
    return {
        "opencv_index_convention_is_floor_j_times_src_over_dst": convention_holds,
        "every_source_index_survives": covered,
        "source_recovered_byte_exact_by_indexing": bool(np.array_equal(recovered, image)),
        "max_abs_error_after_inversion": int(
            np.abs(recovered.astype(int) - image.astype(int)).max()
        ),
    }


def build_resampled_split(
    split: str, kernel: str, size: int = PROBE_IMGSZ, *, root: Path = STUDY_ROOT
) -> dict[str, Any]:
    """Write one derived copy of a NEU-DET split, and check the write did not alter it.

    PNG, for the same reason `input_study.build_padded_split` uses PNG: a JPEG round
    trip would re-quantise exactly the pixels this study is about, and the `nearest`
    arm would stop being lossless somewhere other than where it is being measured.

    Labels are copied verbatim and that is not laziness: YOLO labels are normalised
    to the image, and a uniform resize of a square image leaves every normalised
    coordinate unchanged. Unlike the padded path there is no offset to add, so there
    is no arithmetic here to get wrong.
    """
    if kernel not in KERNELS:
        raise KeyError(f"unknown kernel {kernel!r}; have {sorted(KERNELS)}")
    src_images = sorted((NEU_ROOT / split / "images").glob("*.jpg"))
    if not src_images:
        raise FileNotFoundError(f"no images in {NEU_ROOT / split / 'images'}")

    dst_dir = root / f"{kernel}{size}" / split
    img_dir, lbl_dir = dst_dir / "images", dst_dir / "labels"
    for d in (img_dir, lbl_dir):
        d.mkdir(parents=True, exist_ok=True)
    for stale in dst_dir.glob("*.cache"):
        stale.unlink()

    fn = KERNELS[kernel]
    written = 0
    round_trip_exact = 0
    native_only = True
    for path in src_images:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"cannot decode {path}")
        if image.shape[0] != NATIVE_SIZE or image.shape[1] != NATIVE_SIZE:
            native_only = False
        out = fn(image, size)
        out_path = img_dir / f"{path.stem}.png"
        cv2.imwrite(str(out_path), out)
        if np.array_equal(cv2.imread(str(out_path)), out):
            round_trip_exact += 1
        label = NEU_ROOT / split / "labels" / f"{path.stem}.txt"
        if label.exists():
            shutil.copyfile(label, lbl_dir / f"{path.stem}.txt")
        written += 1

    record: dict[str, Any] = {
        "kernel": kernel,
        "note": KERNEL_NOTE[kernel],
        "split": split,
        "size": size,
        "images_written": written,
        "png_round_trip_byte_exact": round_trip_exact,
        "all_sources_were_native_200px": native_only,
        "dir": str(dst_dir),
    }
    if kernel == "nearest":
        record["invertibility"] = verify_nearest_is_invertible(
            cv2.imread(str(src_images[0])), size
        )
    return record


def write_probe_yaml(kernel: str, nc: int, size: int = PROBE_IMGSZ, *, root: Path = STUDY_ROOT) -> Path:
    """Dataset yaml for one derived copy, in 6- or 10-class flavour.

    The 10-class flavour exists because a 10-class head cannot be validated against a
    6-class name table; indices 6-9 never occur in NEU-DET labels, so widening the
    table changes nothing about the metric (ultralytics averages AP over classes
    present in the ground truth).
    """
    names = dict(NEU_NAMES)
    for extra in range(len(NEU_NAMES), nc):
        names[extra] = f"unused_{extra}"
    root_dir = root / f"{kernel}{size}"
    lines = [
        f"# Written by src/input_pipeline_probe.py: NEU-DET resampled 200 -> {size} with",
        f"# the {kernel} kernel and written as PNG. {KERNEL_NOTE[kernel]}.",
        "# Validated at imgsz=256, where the loader's own scale factor is 1.0 and it",
        "# therefore does nothing -- the resampling under test is the one done here.",
        "# Do not train from this.",
        f"path: {root_dir}",
        "train: val/images",
        "val: val/images",
        "test: test/images",
        f"nc: {nc}",
        "names:",
    ]
    lines += [f"  {i}: {n}" for i, n in sorted(names.items())]
    out = root_dir / f"data_nc{nc}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def verify_tensor(image_dir: Path, size: int = PROBE_IMGSZ, sample: int = 180) -> dict[str, Any]:
    """Confirm the network really is fed the file on disk, unscaled, in a grey border.

    Pulled out of the dataset object the validator itself builds, exactly as
    `input_study.verify_pipeline` does. The claim being checked is narrow and
    falsifiable: the centre `size` x `size` of the network tensor equals the decoded
    PNG byte for byte, so whatever resampling happened, happened in
    `build_resampled_split` and not afterwards.
    """
    dataset = _val_dataset(image_dir, size, NEU_NAMES)
    expected = network_input_size(size)
    n = min(sample, len(dataset))
    exact = 0
    shapes: set[int] = set()
    borders: set[int] = set()
    for i in range(n):
        item = dataset[i]
        tensor = _tensor_to_bgr(item)
        shapes.add(int(tensor.shape[0]))
        disk = cv2.imread(item["im_file"])
        edge = tensor.shape[0]
        off = (edge - size) // 2
        crop = tensor[off : off + size, off : off + size]
        if np.array_equal(crop, disk):
            exact += 1
        borders.update(np.unique(tensor[:2, :]).tolist())
    return {
        "images_checked": n,
        "network_tensor_px": sorted(shapes),
        "network_tensor_px_predicted": expected,
        "tensor_matches_prediction": shapes == {expected},
        "images_whose_content_is_the_file_byte_exact": exact,
        "content_is_unscaled": exact == n,
        "border_pixel_values": sorted(borders),
        "border_is_pad_value": borders == {PAD_VALUE},
    }


# ---------------------------------------------------------------------------
# 2. the arms
# ---------------------------------------------------------------------------


def yaml_for(spec: CheckpointSpec, arm: str) -> Path:
    """Dataset yaml for one (checkpoint, arm) pair, in the head's own class count."""
    if arm == "scale":
        return (
            NEU_ROOT / "data.yaml" if spec.nc == 6 else STUDY_ROOT / "neudet_nc10.yaml"
        )
    if arm == "pad":
        return STUDY_ROOT / f"pad{PROBE_IMGSZ}" / f"data_nc{spec.nc}.yaml"
    return STUDY_ROOT / f"{arm}{PROBE_IMGSZ}" / f"data_nc{spec.nc}.yaml"


ARMS: tuple[str, ...] = ("scale", "bilinear", "nearest", "decimated", "pad")


def run_arms(
    spec: CheckpointSpec,
    split: str,
    device: str,
    out_dir: Path,
    arms: Sequence[str] = ARMS,
) -> list[InputRun]:
    """One validation pass per arm, all at `PROBE_IMGSZ`, all on the same 180 images."""
    runs: list[InputRun] = []
    for arm in arms:
        yaml_path = yaml_for(spec, arm)
        if not yaml_path.exists():
            raise FileNotFoundError(f"{yaml_path} is missing; build the split first")
        run = run_validation(
            spec,
            yaml_path,
            split,
            PROBE_IMGSZ,
            device,
            pipeline=arm,
            out_dir=out_dir,
        )
        runs.append(run)
        print(
            f"  {spec.key:<15} {split:<5} {arm:<10} "
            f"mAP50 {run.mAP50:.4f}  mAP50-95 {run.mAP50_95:.4f}  "
            f"({run.seconds:.1f}s, tensor {run.network_input}px)",
            flush=True,
        )
    return runs


def contrast_table(bootstrap: dict[str, Any], pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """Paired percentile intervals on the differences this probe was built to read."""
    if not bootstrap:
        return {}
    labels: list[str] = bootstrap["labels"]
    out: dict[str, Any] = {}
    for a, b in pairs:
        la, lb = f"{a}{PROBE_IMGSZ}", f"{b}{PROBE_IMGSZ}"
        if la not in labels or lb not in labels:
            continue
        i, j = labels.index(la), labels.index(lb)
        out[f"{la}_minus_{lb}"] = {
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


CONTRASTS: tuple[tuple[str, str], ...] = (
    ("bilinear", "scale"),  # instrument identity: must be exactly zero
    ("nearest", "bilinear"),  # fidelity, magnification held fixed
    ("nearest", "pad"),  # magnification, fidelity held (both lossless)
    ("decimated", "bilinear"),  # sensitivity: what real pixel loss costs here
    ("pad", "scale"),  # input_study's headline, re-measured here
)


# ---------------------------------------------------------------------------
# 3. driver
# ---------------------------------------------------------------------------


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
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu by default: this probe is small and the GPU may be training",
    )
    parser.add_argument("--splits", nargs="+", default=["test", "val"])
    parser.add_argument(
        "--joint-splits",
        nargs="+",
        default=["test"],
        help="the 10-class head is a confirmation arm, so it is run on fewer splits",
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--out", type=Path, default=REPORTS_DIR / "input_pipeline_probe.json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "imgsz": PROBE_IMGSZ,
        "network_tensor_px": network_input_size(PROBE_IMGSZ),
        "device": args.device,
        "seed": BOOTSTRAP_SEED,
        "resamples": args.resamples,
        "arms": {k: KERNEL_NOTE[k] for k in KERNELS},
        "builds": [],
        "tensor_checks": {},
        "runs": [],
        "contrasts": {},
    }

    wanted = sorted({s for s in args.splits} | {s for s in args.joint_splits})
    if not args.skip_build:
        for kernel in KERNELS:
            for split in wanted:
                record = build_resampled_split(split, kernel)
                payload["builds"].append(record)
                print(
                    f"built {kernel}{PROBE_IMGSZ}/{split}: {record['images_written']} images, "
                    f"png round trip exact {record['png_round_trip_byte_exact']}",
                    flush=True,
                )
        for kernel in KERNELS:
            for nc in (6, 10):
                write_probe_yaml(kernel, nc)

    for kernel in KERNELS:
        for split in wanted:
            image_dir = STUDY_ROOT / f"{kernel}{PROBE_IMGSZ}" / split / "images"
            if image_dir.exists():
                payload["tensor_checks"][f"{kernel}/{split}"] = verify_tensor(image_dir)
    for key, check in payload["tensor_checks"].items():
        print(
            f"tensor check {key}: {check['images_whose_content_is_the_file_byte_exact']}"
            f"/{check['images_checked']} unscaled, tensor {check['network_tensor_px']}, "
            f"border {check['border_pixel_values']}",
            flush=True,
        )

    out_dir = PROJECT_ROOT / "runs" / "input_pipeline_probe"
    for spec in CHECKPOINTS:
        splits = args.splits if spec.nc == 6 else args.joint_splits
        for split in splits:
            runs = run_arms(spec, split, args.device, out_dir)
            payload["runs"].extend(r.to_dict() | {"checkpoint": spec.key} for r in runs)
            boot = paired_bootstrap(runs, args.resamples, BOOTSTRAP_SEED)
            payload["contrasts"][f"{spec.key}/{split}"] = contrast_table(boot, CONTRASTS)

    payload["wall_seconds"] = round(time.perf_counter() - started, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    print(f"\nwrote {args.out} in {payload['wall_seconds']}s")

    for key, table in payload["contrasts"].items():
        print(f"\n{key}")
        for name, entry in table.items():
            row = entry["mAP50"]
            print(
                f"  {name:<34} {row['point_difference']:+.4f} mAP50  "
                f"95% [{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}]  "
                f"{'resolved' if row['ci_excludes_zero'] else 'contains zero'}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
