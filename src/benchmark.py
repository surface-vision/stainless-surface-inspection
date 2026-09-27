"""Production-readiness speed study for the Jindal Stainless defect detector.

This answers one question: *is the detector fast enough to sit on a rolling line?*

It has two halves.

1. A measurement half that sweeps every checkpoint under `models/` across image
   size, batch size and device, timing the **shipping code path** (the public
   `DefectDetector` API in `inference.py`) rather than a bare forward pass, so the
   numbers include letterboxing, host/device transfer, NMS and record building --
   everything the line actually pays for. Warm-up runs are always discarded; the
   first inference on MPS pays lazy kernel compilation and is never representative.
   The per-stage split is only reported as trustworthy when the framework's stage
   clocks are accelerator-synchronised (`protocol.stage_timing_synchronised`);
   unsynchronised, MPS bills the forward pass to postprocess.

2. A translation half that turns milliseconds into mill engineering: given a line
   speed, a strip width, a camera field of view and the frame overlap needed for
   gap-free coverage, how many frames per second must the vision system sustain,
   and therefore how many camera streams can one accelerator serve? Every
   assumption lives in `MILL_CONFIG` at the top of this file so a process engineer
   can challenge it by editing one dict.

   Two feeding modes are costed. "Tiled" cuts each camera frame into windows the
   size of the network input, so one source pixel is one network pixel and the
   stated optical resolution actually survives -- which means the tile count is a
   function of the input size and rises quadratically as the input shrinks.
   "Downscaled" squashes the whole frame into one inference and trades resolution
   for speed. The tiled figures across input sizes are therefore a cost comparison
   at *equal* resolution, not a speed comparison.

Outputs: `reports/benchmark.json` plus two charts named after it,
`reports/benchmark_latency_vs_imgsz.png` and
`reports/benchmark_streams_vs_line_speed.png`. A sweep written to a different
`--out` names its charts after that file, so a spot check cannot overwrite the
charts belonging to the signed-off report.

Checkpoints are snapshotted to a temporary directory before they are measured: a
training job may be rewriting `models/<run>/weights/last.pt` while this runs, and a
benchmark that cannot say exactly which bytes it measured is worthless. The SHA-256
of every snapshot is recorded in the JSON.

Usage:
    .venv/bin/python src/benchmark.py                    # full sweep
    .venv/bin/python src/benchmark.py --devices mps --iters 30
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Ultralytics pip-installs its way out of a version conflict on import unless told
# not to. A benchmark that silently changes the environment it is measuring is
# worthless, and it has already downgraded numpy once in this venv.
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

from inference import DefectDetector, resolve_device, resolve_weights  # noqa: E402
# Deliberately the *same* tiling geometry the runtime uses, so the capacity plan
# below cannot drift away from what predict_tiled() actually does.
from inference import _tile_origins as tile_origins  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
TEST_IMAGES_DIR = PROJECT_ROOT / "data" / "neu-det" / "test" / "images"


# ---------------------------------------------------------------------------
# Mill assumptions. Everything downstream is derived from this dict -- challenge
# the numbers here, not the arithmetic.
# ---------------------------------------------------------------------------

MILL_CONFIG: dict[str, Any] = {
    # --- product and line geometry -------------------------------------------------
    # Widest coil the line runs. Jindal Stainless hot-rolled coil is typically
    # 1000-1250 mm; 1280 mm is the planning worst case.
    "strip_width_m": 1.28,
    # --- camera and optics ---------------------------------------------------------
    # Required optical resolution. The smallest defect the quality department cares
    # about is roughly 1 mm across, and a CNN needs about 5 px across a feature to
    # classify it reliably, so 0.20 mm/px is the coarsest defensible sampling.
    "optical_resolution_mm_per_px": 0.20,
    # Square area-scan sensor, pixels per side (a 4 MP global-shutter machine-vision
    # camera). Area scan rather than line scan, because the detector is a 2-D
    # bounding-box model and needs a 2-D frame.
    "camera_sensor_px": 2048,
    # Frame overlap. Across the web it guarantees the seams between adjacent cameras
    # are covered even with mechanical drift; along the web it guarantees no strip
    # passes between two exposures unseen, and gives a defect crossing a frame
    # boundary a chance to be whole in one of them.
    "overlap_across_web": 0.08,
    "overlap_along_web": 0.15,
    # --- how a camera frame is fed to the network ----------------------------------
    # A 2048 px frame is far larger than the network input. Two honest options:
    #   "tiled"      -- cut the frame into overlapping tiles and run each one
    #                   (this is DefectDetector.predict_tiled). To actually keep
    #                   full optical resolution the tile must be cut at the
    #                   *network input size*, so one source pixel maps to one
    #                   network pixel; the tile count is therefore a function of
    #                   imgsz, not a constant. Cutting 640 px tiles and letting the
    #                   network resize them to 320 is downscaling wearing a tiled
    #                   costume: it samples at 0.4 mm/px, not 0.2.
    #   "downscaled" -- squash the whole frame to imgsz and run once. One inference
    #                   per frame, but the effective resolution degrades by
    #                   sensor_px / imgsz and fine crazing texture is destroyed.
    # Both are reported; tiled is the one to plan capacity against.
    "tile_overlap": 0.20,
    # --- engineering headroom ------------------------------------------------------
    # Never size an accelerator above this utilisation: leaves room for jitter, OS
    # scheduling, camera bursts and the day the line runs 10% faster than the plan.
    "utilisation_ceiling": 0.70,
    # Wall-clock budget from photon to verdict at the HMI / marking gun. Used to
    # report how much strip passes the camera before the defect is flagged, which
    # sets how far downstream a marker or diverter has to sit.
    "decision_latency_budget_ms": 200.0,
    # --- line speeds to cover ------------------------------------------------------
    # Representative stainless hot and cold line speeds, slowest to fastest.
    "line_speeds_m_per_min": {
        "Bright annealing line": 45.0,
        "Hot-band anneal and pickle line": 90.0,
        "Skin-pass / inspection line": 250.0,
        "20-hi reversing cold mill": 420.0,
        "Continuous pickling line exit": 500.0,
        "Hot strip mill finishing exit": 900.0,
        "Tandem cold mill exit, thin gauge": 1500.0,
    },
    # Continuous sweep for the capacity chart.
    "speed_sweep_m_per_min": (40.0, 1600.0),
}


# ---------------------------------------------------------------------------
# Chart palette (validated categorical slots 1-3, light surface)
# ---------------------------------------------------------------------------

PALETTE = {
    "surface": "#fcfcfb",
    "page": "#f9f9f7",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "ink_muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    # Four categorical slots, not three: the input-size sweep now carries 256 px as
    # well, and a modulo-wrapped palette would draw two of the four series in the
    # same colour and quietly make the chart lie.
    "series": ("#2a78d6", "#eb6834", "#1baf7a", "#8a5cd6"),
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Checkpoint:
    """One distinct set of weights under test, pinned by content hash."""

    key: str
    source_path: str
    snapshot_path: str
    sha256: str
    size_mb: float
    aliases: list[str]
    params: int = 0
    gflops_by_imgsz: dict[str, float] = field(default_factory=dict)


@dataclass
class LatencyStats:
    """Distribution of a timed call, over `n` samples after warm-up."""

    n: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    std_ms: float
    min_ms: float
    max_ms: float

    @classmethod
    def from_samples(cls, samples: Sequence[float]) -> "LatencyStats":
        arr = np.asarray(samples, dtype=np.float64)
        return cls(
            n=int(arr.size),
            mean_ms=float(arr.mean()),
            median_ms=float(np.median(arr)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            # Sample standard deviation: these are a sample of the run's behaviour.
            std_ms=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            min_ms=float(arr.min()),
            max_ms=float(arr.max()),
        )


@dataclass
class Measurement:
    """One (checkpoint, device, imgsz, batch) cell of the sweep."""

    checkpoint: str
    device: str
    imgsz: int
    batch: int
    ok: bool
    batch_latency: dict[str, float] | None = None
    per_frame_ms: float = 0.0
    throughput_fps: float = 0.0
    throughput_fps_p95: float = 0.0
    stage_ms: dict[str, float] = field(default_factory=dict)
    mean_detections: float = 0.0
    error: str | None = None
    # Populated only when the sweep is run more than once. `repeats` is the
    # per-frame cost each independent pass measured for this cell; the fields
    # above are the *median* pass, not the best one.
    repeats_per_frame_ms: list[float] = field(default_factory=list)
    across_repeats: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_checkpoints(
    models_dir: Path,
    staging: Path,
    min_age_s: float = 45.0,
    runs: Sequence[str] | None = None,
    files: Sequence[str] | None = None,
) -> list[Checkpoint]:
    """Snapshot every distinct checkpoint under `models_dir`.

    Files younger than `min_age_s` are skipped: a training job writes `last.pt`
    every epoch and half a checkpoint is not a checkpoint. Identical files (a
    `best.pt` that is a byte copy of `last.pt`, or a run cloned to another
    directory) are measured once and their aliases recorded, because benchmarking
    the same bytes four times is padding, not evidence.

    `runs` restricts the sweep to named run directories and `files` to named
    weight files. `models/` accumulates scratch and stopgap runs that will never be
    served, and every run carries a `last.pt` alongside its `best.pt`; measuring
    those spends the per-cell budget on weights nobody will deploy and pads the
    report with rows a reader has to learn to ignore.
    """
    now = time.time()
    by_hash: dict[str, Checkpoint] = {}
    wanted_runs = set(runs) if runs else None
    wanted_files = set(files) if files else None

    for path in sorted(models_dir.glob("*/weights/*.pt")):
        run_name = path.parent.parent.name
        if wanted_runs is not None and run_name not in wanted_runs:
            print(f"[skip] {path.relative_to(models_dir)} not in --runs")
            continue
        if wanted_files is not None and path.name not in wanted_files:
            print(f"[skip] {path.relative_to(models_dir)} not in --files")
            continue
        stat_before = path.stat()
        age = now - stat_before.st_mtime
        if age < min_age_s:
            print(f"[skip] {path.relative_to(models_dir)} written {age:.0f}s ago (in flight)")
            continue

        # Copy first, then hash the copy. Hashing the source and copying afterwards
        # would record a digest for bytes that are not the ones measured: the
        # training job can rewrite the file between the two reads. The source mtime
        # and size are re-checked afterwards to catch a copy that raced a rewrite.
        alias = f"{path.parent.parent.name}/{path.name}"
        pending = staging / f"pending_{path.parent.parent.name}_{path.name}"
        try:
            shutil.copy2(path, pending)
            digest = _sha256(pending)
            stat_after = path.stat()
        except OSError as exc:  # pragma: no cover - transient FS race
            print(f"[skip] {path}: {exc}")
            pending.unlink(missing_ok=True)
            continue
        if (stat_after.st_mtime_ns, stat_after.st_size) != (
            stat_before.st_mtime_ns,
            stat_before.st_size,
        ):
            print(f"[skip] {path.relative_to(models_dir)} was rewritten while being copied")
            pending.unlink(missing_ok=True)
            continue

        existing = by_hash.get(digest)
        if existing is not None:
            existing.aliases.append(alias)
            pending.unlink(missing_ok=True)
            continue

        snapshot = staging / f"{digest[:12]}.pt"
        pending.replace(snapshot)
        by_hash[digest] = Checkpoint(
            key=alias,
            source_path=str(path),
            snapshot_path=str(snapshot),
            sha256=digest,
            size_mb=round(snapshot.stat().st_size / 1e6, 2),
            aliases=[alias],
        )

    return list(by_hash.values())


def measure_complexity(checkpoint: Checkpoint, image_sizes: Sequence[int]) -> None:
    """Fill in parameter count and GFLOPs (which scale with input size)."""
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops, get_num_params

    model = YOLO(checkpoint.snapshot_path)
    checkpoint.params = int(get_num_params(model.model))
    checkpoint.gflops_by_imgsz = {
        str(sz): round(float(get_flops(model.model, sz)), 3) for sz in image_sizes
    }
    del model
    gc.collect()


# ---------------------------------------------------------------------------
# Frame pool
# ---------------------------------------------------------------------------


def build_frame_pool(count: int, frame_px: int) -> list[np.ndarray]:
    """Real test frames, resized to the camera tile size, held in RAM.

    In the mill the frames arrive over GigE straight into memory, so JPEG decode
    from disk is not part of the measured loop. Images are sampled evenly across
    the sorted test split so all six defect classes are represented: NMS cost
    depends on how many boxes survive, so a pool of one class would flatter or
    punish the postprocess number.
    """
    paths = sorted(TEST_IMAGES_DIR.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No test images under {TEST_IMAGES_DIR}")
    count = max(1, int(count))
    # Evenly spaced *indices*, not a stride truncated to `count`: striding then
    # slicing overshoots and drops the tail of the split, which is one whole class
    # (the split is sorted by class name). NMS cost tracks how many boxes survive,
    # so a pool missing scratches would understate postprocess.
    chosen = [paths[i] for i in np.unique(np.linspace(0, len(paths) - 1, count).round().astype(int))]

    pool: list[np.ndarray] = []
    for path in chosen:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (frame_px, frame_px), interpolation=cv2.INTER_LINEAR)
        pool.append(np.ascontiguousarray(rgb))
    if not pool:
        raise RuntimeError(f"Could not decode any image under {TEST_IMAGES_DIR}")
    return pool


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def time_case(
    detector: DefectDetector,
    pool: list[np.ndarray],
    batch: int,
    iters: int,
    min_iters: int,
    budget_s: float,
    warmup: int,
) -> Measurement:
    """Warm up, then time `iters` calls of the shipping API, budget permitting.

    The warm-up is run at the exact batch shape, not just any shape: MPS compiles
    and caches kernels per shape, so warming at batch 1 and timing at batch 16
    would charge the first timed iteration for a compile. The loop stops early once
    `budget_s` is exceeded (and at least `min_iters` samples are in hand) so a slow
    CPU cell cannot monopolise a machine that is sharing its GPU with training.
    """
    def window(offset: int) -> list[np.ndarray]:
        return [pool[(offset + i) % len(pool)] for i in range(batch)]

    def call(frames: list[np.ndarray]) -> list:
        return (
            [detector.predict(frames[0])]
            if batch == 1
            else detector.predict_batch(frames, batch_size=batch)
        )

    for w in range(max(1, warmup)):
        call(window(w * batch))

    samples: list[float] = []
    stages = np.zeros(3, dtype=np.float64)
    detections = 0
    frames_seen = 0
    loop_start = time.perf_counter()

    for iteration in range(max(1, int(iters))):
        # Rotate through the pool so postprocess cost is averaged over frames with
        # different defect loads, not measured on one lucky image.
        frames = window(iteration * batch)
        t0 = time.perf_counter()
        results = call(frames)
        samples.append((time.perf_counter() - t0) * 1000.0)
        for res in results:
            stages += (res.preprocess_ms, res.inference_ms, res.postprocess_ms)
            detections += res.defect_count
            frames_seen += 1
        if len(samples) >= min_iters and (time.perf_counter() - loop_start) > budget_s:
            break

    stats = LatencyStats.from_samples(samples)
    stages /= max(1, frames_seen)
    return Measurement(
        checkpoint=detector.model_name,
        device=detector.device,
        imgsz=detector.imgsz,
        batch=batch,
        ok=True,
        batch_latency=asdict(stats),
        per_frame_ms=stats.mean_ms / batch,
        throughput_fps=1000.0 * batch / stats.mean_ms,
        throughput_fps_p95=1000.0 * batch / stats.p95_ms,
        stage_ms={
            "preprocess": round(float(stages[0]), 3),
            "inference": round(float(stages[1]), 3),
            "postprocess": round(float(stages[2]), 3),
        },
        mean_detections=detections / max(1, frames_seen),
    )


def cpu_thread_state() -> dict[str, Any]:
    """The intra-op thread budget the CPU cells actually ran under.

    Three things fight over this number and the loser is silent: OMP_NUM_THREADS
    from the shell, torch's own default, and ultralytics, which raises the intra-op
    count on the first predict. A CPU latency quoted without it is not reproducible
    -- the same script in a different shell measures a different machine.
    """
    return {
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS"),
        "cpu_count": os.cpu_count(),
    }


def load_snapshot() -> dict[str, Any]:
    """Kernel load average and core count, so contention is evidence not a claim.

    A previous edition of this report carried a hand-written sentence saying the
    sweep had shared the machine with a training job. That sentence is only true
    of the run that wrote it, and it survived into re-runs where it was false. The
    load average is read from the kernel at the start and end of the sweep
    instead: it cannot go stale, and a reader can decide for themselves how much
    the CPU cells were contended.
    """
    try:
        one, five, fifteen = os.getloadavg()
    except OSError:  # pragma: no cover - not available on every platform
        return {"available": False}
    return {
        "available": True,
        "load_avg_1min": round(one, 2),
        "load_avg_5min": round(five, 2),
        "load_avg_15min": round(fifteen, 2),
        "cpu_count": os.cpu_count(),
    }


def cpu_batch_note(measurements: Sequence[Measurement]) -> str:
    """The CPU large-batch cliff, quoted from this run's own numbers.

    The previous edition of this sentence carried two hand-typed millisecond
    figures. They were true of the sweep that first observed the effect and
    survived unchanged into later sweeps that measured something else, which is
    the same failure the host-contention claim had. The effect is stable and worth
    naming; the numbers attached to it have to come from the run being described.
    """
    cells = {
        (m.imgsz, m.batch): m
        for m in measurements
        if m.ok and m.device.startswith("cpu")
    }
    sizes = sorted({sz for sz, _ in cells})
    for imgsz in sizes:
        good, bad = cells.get((imgsz, 8)), cells.get((imgsz, 16))
        if not good or not bad or bad.per_frame_ms <= good.per_frame_ms:
            continue
        return (
            "CPU throughput peaks at batch 8 and collapses at batch 16 and above "
            f"-- in this run, {good.per_frame_ms:.2f} ms/frame at batch 8 against "
            f"{bad.per_frame_ms:.2f} ms/frame at batch 16 at {imgsz} px, a "
            f"{bad.per_frame_ms / good.per_frame_ms:.1f}x regression from asking "
            "for twice the work. This is not contention and not this codebase: it "
            "reproduces on a bare two-layer torch Conv2d/SiLU stack at the same "
            "batch boundary and is unaffected by the intra-op thread count. Treat "
            "batch 8 as the CPU ceiling on Apple silicon and never submit 16 "
            "frames at once."
        )
    return "No CPU cells in this sweep, so the large-batch cliff was not exercised."


def release(device: str) -> None:
    """Hand memory back between cells; the GPU is shared with a training run."""
    gc.collect()
    if device.startswith("mps"):
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.empty_cache()


def run_sweep(
    checkpoints: Sequence[Checkpoint],
    pool: list[np.ndarray],
    devices: Sequence[str],
    image_sizes: Sequence[int],
    batches: Sequence[int],
    args: argparse.Namespace,
) -> list[Measurement]:
    """Full cross product, ordered so each detector is constructed exactly once."""
    results: list[Measurement] = []
    total = len(checkpoints) * len(devices) * len(image_sizes) * len(batches)
    done = 0

    # Per-cell warm-up pays for kernel compilation at that shape, but not for the
    # one-off process-level cost of bringing the Metal stack up, and that lands
    # entirely on whichever cell happens to be measured first. It is worth roughly
    # 35% on this host: yolov8n at 256 px batch 1 times 10.2/11.0/9.9 ms/frame when
    # it leads the sweep and 8.0/8.3/6.4 ms/frame when one cell precedes it, which
    # is enough to make the smallest input look slower than the largest and to
    # mis-order a table anyone reads as latency-versus-input-size. Burning one
    # throwaway detector before the grid starts moves that cost off the first cell.
    for device in devices:
        try:
            DefectDetector(
                weights=checkpoints[0].snapshot_path,
                device=device,
                conf=args.conf,
                iou=args.iou,
                imgsz=int(image_sizes[0]),
            ).warmup(3)
        except Exception:  # noqa: BLE001 - a failed warm-up costs accuracy, not correctness
            pass
        release(device)

    for ckpt in checkpoints:
        for device in devices:
            for imgsz in image_sizes:
                try:
                    detector = DefectDetector(
                        weights=ckpt.snapshot_path,
                        device=device,
                        conf=args.conf,
                        iou=args.iou,
                        imgsz=imgsz,
                    )
                except Exception as exc:  # noqa: BLE001 - one bad cell must not stop the sweep
                    for batch in batches:
                        done += 1
                        results.append(
                            Measurement(ckpt.key, device, imgsz, batch, ok=False, error=str(exc))
                        )
                    print(f"[fail] {ckpt.key} {device} imgsz={imgsz}: {exc}")
                    continue

                for batch in batches:
                    done += 1
                    try:
                        measurement = time_case(
                            detector,
                            pool,
                            batch=batch,
                            iters=args.iters,
                            min_iters=args.min_iters,
                            budget_s=args.budget_s,
                            warmup=args.warmup,
                        )
                        measurement.checkpoint = ckpt.key
                        print(
                            f"[{done:>3}/{total}] {ckpt.key:<28} {device:<3} "
                            f"imgsz={imgsz:<3} batch={batch:<2} "
                            f"n={measurement.batch_latency['n']:<3} "
                            f"{measurement.per_frame_ms:7.2f} ms/frame "
                            f"{measurement.throughput_fps:7.1f} FPS"
                        )
                    except Exception as exc:  # noqa: BLE001
                        measurement = Measurement(
                            ckpt.key, device, imgsz, batch, ok=False, error=str(exc)
                        )
                        print(f"[fail] {ckpt.key} {device} imgsz={imgsz} batch={batch}: {exc}")
                    results.append(measurement)

                del detector
                release(device)
    return results


def fold_repeats(passes: Sequence[Sequence[Measurement]]) -> list[Measurement]:
    """Collapse independent passes of the same sweep to one median Measurement each.

    Why the median and not the minimum: quoting the fastest pass would be quoting
    the luckiest thermal and GPU-clock state the sweep happened to catch, and a
    capacity plan built on it would be wrong by more than the utilisation ceiling
    covers.

    What this does NOT buy, and the docstring used to imply that it did: these
    passes are not independent draws from the between-process distribution. They
    run back to back in one process and inherit its thermal and GPU-clock state,
    so folding them tightens the number without making it reproducible. Two
    independent three-pass invocations of this script on an idle machine put the
    yolov8n 320 px peak at 233.9 and 282.6 FPS -- a 21% gap between two medians of
    three. Treat across_repeats as within-invocation precision only, and read
    protocol.repeatability_note for the figure that actually bounds the number.
    """
    if len(passes) == 1:
        return list(passes[0])

    by_cell: dict[tuple[str, str, int, int], list[Measurement]] = {}
    for single in passes:
        for m in single:
            by_cell.setdefault((m.checkpoint, m.device, m.imgsz, m.batch), []).append(m)

    folded: list[Measurement] = []
    for cell, group in by_cell.items():
        healthy = [m for m in group if m.ok]
        if not healthy:
            folded.append(group[0])
            continue
        ordered = sorted(healthy, key=lambda m: m.per_frame_ms)
        # Lower median for an even count: the pessimistic half of the pair, since
        # this number is about to be multiplied out into a hardware count.
        chosen = ordered[(len(ordered) - 1) // 2]
        samples = [round(m.per_frame_ms, 4) for m in ordered]
        chosen.repeats_per_frame_ms = samples
        chosen.across_repeats = {
            "passes": len(samples),
            "min_ms": samples[0],
            "median_ms": round(chosen.per_frame_ms, 4),
            "max_ms": samples[-1],
            "spread_ratio": round(samples[-1] / samples[0], 3) if samples[0] > 0 else 0.0,
        }
        folded.append(chosen)

    order = {}
    for single in passes[0]:
        order.setdefault((single.checkpoint, single.device, single.imgsz, single.batch), len(order))
    folded.sort(key=lambda m: order.get((m.checkpoint, m.device, m.imgsz, m.batch), 1 << 30))
    return folded


# ---------------------------------------------------------------------------
# Mill translation
# ---------------------------------------------------------------------------


def tiles_per_frame(sensor_px: int, tile_px: int, overlap: float) -> int:
    """Overlapping tiles needed to cover one square sensor frame.

    Uses `inference._tile_origins`, so the plan counts exactly the windows
    `predict_tiled` would run -- including its flush-to-the-edge last window.
    """
    tile = max(1, min(int(tile_px), int(sensor_px)))
    stride = max(1, int(round(tile * (1.0 - float(overlap)))))
    return len(tile_origins(int(sensor_px), tile, stride)) ** 2


@dataclass
class LineGeometry:
    """Everything MILL_CONFIG implies about the camera installation."""

    strip_width_m: float
    fov_m: float
    mm_per_px: float
    cameras_across: int
    advance_per_frame_m: float
    sensor_px: int
    tile_overlap: float
    coverage_width_m: float

    def tiles_for_input(self, imgsz: int) -> int:
        """Tiles per camera frame when the source is cut at the network input size.

        This is the only tiling that preserves the stated optical resolution: one
        source pixel per network pixel. It is why the tile count is always quoted
        against an imgsz and never as a single number.
        """
        return tiles_per_frame(self.sensor_px, imgsz, self.tile_overlap)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strip_width_m": self.strip_width_m,
            "camera_fov_m": round(self.fov_m, 4),
            "optical_resolution_mm_per_px": self.mm_per_px,
            "cameras_across_width": self.cameras_across,
            "installed_coverage_width_m": round(self.coverage_width_m, 3),
            "strip_advance_per_frame_m": round(self.advance_per_frame_m, 4),
            "sensor_px": self.sensor_px,
            "tile_overlap": self.tile_overlap,
        }


def derive_geometry(config: dict[str, Any]) -> LineGeometry:
    """Turn the optics assumptions into camera count and per-frame strip advance.

    Field of view follows from the sensor and the required sampling:
        fov = sensor_px * mm_per_px.
    Cameras must tile the width with `overlap_across_web` between neighbours, so
    each contributes fov * (1 - overlap) of *new* width. Along the web, a frame is
    only allowed to advance fov * (1 - overlap) between exposures, otherwise strip
    passes the camera unphotographed -- that single line is what sets the frame
    rate, and everything else is bookkeeping.
    """
    sensor_px = int(config["camera_sensor_px"])
    mm_per_px = float(config["optical_resolution_mm_per_px"])
    fov_m = sensor_px * mm_per_px / 1000.0

    effective_width = fov_m * (1.0 - float(config["overlap_across_web"]))
    cameras_across = max(1, math.ceil(float(config["strip_width_m"]) / effective_width))

    advance = fov_m * (1.0 - float(config["overlap_along_web"]))

    return LineGeometry(
        strip_width_m=float(config["strip_width_m"]),
        fov_m=fov_m,
        mm_per_px=mm_per_px,
        cameras_across=cameras_across,
        advance_per_frame_m=advance,
        sensor_px=sensor_px,
        tile_overlap=float(config["tile_overlap"]),
        coverage_width_m=cameras_across * effective_width,
    )


def demand_at_speed(
    geometry: LineGeometry, speed_m_per_min: float, imgsz: int
) -> dict[str, float]:
    """Frame-rate demand the vision system must sustain at this line speed.

    The camera frame rate follows from optics and line speed alone, but the
    inference rate does not: the tiled figures depend on `imgsz`, because that is
    what fixes how many full-resolution tiles one frame decomposes into.
    """
    speed_m_per_s = speed_m_per_min / 60.0
    fps_per_camera = speed_m_per_s / geometry.advance_per_frame_m
    tiles = geometry.tiles_for_input(imgsz)
    return {
        "line_speed_m_per_min": speed_m_per_min,
        "line_speed_m_per_s": round(speed_m_per_s, 3),
        "fps_per_camera": round(fps_per_camera, 2),
        "fps_line_total": round(fps_per_camera * geometry.cameras_across, 2),
        "network_input_px": int(imgsz),
        "tiles_per_frame": tiles,
        "inferences_per_s_tiled": round(fps_per_camera * geometry.cameras_across * tiles, 1),
        "inferences_per_s_downscaled": round(fps_per_camera * geometry.cameras_across, 2),
        "inferences_per_s_per_camera_tiled": round(fps_per_camera * tiles, 2),
    }


def capacity_plan(
    geometry: LineGeometry,
    imgsz: int,
    sustainable_fps: float,
    speed_m_per_min: float,
    latency_ms: float,
    sustainable_fps_low: float | None = None,
    peak_batch: int = 1,
    p95_batch_ms: float = 0.0,
) -> dict[str, Any]:
    """How many camera streams one accelerator serves at this speed, and the
    distance of strip that passes before a verdict exists.

    `sustainable_fps_low` is the same rate derived from the slowest measured pass.
    The accelerator count it produces is the one to budget against: the median
    pass is a plausible day, the slowest pass is a day that has already happened.
    """
    demand = demand_at_speed(geometry, speed_m_per_min, imgsz)
    per_stream_tiled = demand["inferences_per_s_per_camera_tiled"]
    per_stream_down = demand["fps_per_camera"]

    plan = dict(demand)
    plan["streams_per_accelerator_tiled"] = int(sustainable_fps // per_stream_tiled)
    plan["streams_per_accelerator_downscaled"] = int(sustainable_fps // per_stream_down)
    plan["accelerators_for_full_width_tiled"] = math.ceil(
        demand["inferences_per_s_tiled"] / sustainable_fps
    )
    plan["accelerators_for_full_width_downscaled"] = math.ceil(
        demand["inferences_per_s_downscaled"] / sustainable_fps
    )
    if sustainable_fps_low and sustainable_fps_low > 0:
        plan["accelerators_for_full_width_tiled_slowest_pass"] = math.ceil(
            demand["inferences_per_s_tiled"] / sustainable_fps_low
        )
        plan["accelerators_for_full_width_downscaled_slowest_pass"] = math.ceil(
            demand["inferences_per_s_downscaled"] / sustainable_fps_low
        )
    plan["strip_travel_during_decision_m"] = round(
        demand["line_speed_m_per_s"] * latency_ms / 1000.0, 3
    )
    # `decision_latency_budget_ms` is an assumption, and the peak throughput it sits
    # beside was reached by batching. A frame cannot be scored until its batch is
    # full, so batch fill is part of photon-to-verdict and the two numbers have to
    # be checked against each other: a plan that quotes a 200 ms budget while
    # batching 16 frames off a slow line is quoting a latency the batching forbids.
    if peak_batch > 1 and plan["accelerators_for_full_width_tiled"] > 0:
        per_accelerator_tiles_s = (
            demand["inferences_per_s_tiled"] / plan["accelerators_for_full_width_tiled"]
        )
        if per_accelerator_tiles_s > 0:
            fill_ms = 1000.0 * peak_batch / per_accelerator_tiles_s
            plan["batch_fill_ms_at_peak_batch"] = round(fill_ms, 1)
            plan["decision_latency_ms_tiled_estimate"] = round(fill_ms + p95_batch_ms, 1)
            plan["meets_latency_budget_at_peak_batch"] = bool(
                fill_ms + p95_batch_ms <= latency_ms
            )
    return plan


# Accuracy is allowed to fall this far below the best measured input size before
# an input size is disqualified as a capacity-planning reference. Planning against
# an input the detector cannot actually see at is buying throughput that does no
# work: at 640 px this checkpoint scores mAP50 0.344 against 0.752 at 256 px.
ACCURACY_TOLERANCE = 0.95


def accuracy_by_imgsz(checkpoint_key: str, split: str = "val") -> dict[int, float]:
    """mAP50 per input size on `split`, read from `reports/model_study.json`.

    The default is **val**, not test, and that is the whole point of the argument.
    Choosing the reference input size is a decision, and a decision made on the
    test split turns the test split into a selection split: every accuracy figure
    reported next to it afterwards is then optimistically biased by the choice.
    Val is the split that already carries that bias (`best.pt` is the max-val-
    fitness epoch), so spending it again costs nothing that has not been spent.
    The test curve is still read and published beside the plan for disclosure --
    it is reported, not selected on.

    The throughput sweep on its own cannot tell a good input size from a bad one --
    it only knows that smaller is faster -- so left alone it will always nominate
    the largest input as the cheapest way to tile a sensor frame, because that
    needs the fewest tiles. For this model that answer is 640 px, where accuracy
    has already collapsed to less than half its peak. The accuracy curve therefore
    has to enter the capacity plan, and it is read from the artefact that measured
    it rather than restated here, so it cannot go stale.

    Returns an empty mapping when the study has not been run; the caller then
    plans on cost alone and says so.
    """
    study = REPORTS_DIR / "model_study.json"
    if not study.is_file():
        return {}
    try:
        runs = json.loads(study.read_text()).get("runs", [])
    except (OSError, ValueError):  # pragma: no cover - unreadable artefact
        return {}
    # `checkpoint_key` is "<run>/<file>"; the study keys models as "yolov8n"/"yolov8s".
    run_dir = checkpoint_key.split("/")[0]
    out: dict[int, float] = {}
    for row in runs:
        if row.get("split") != split or row.get("augment"):
            continue
        if not run_dir.startswith(str(row.get("model", "\0"))):
            continue
        try:
            out[int(row["imgsz"])] = float(row["mAP50"])
        except (KeyError, TypeError, ValueError):
            continue
    return out


def build_mill_section(
    measurements: Sequence[Measurement],
    config: dict[str, Any],
    deployment_device: str,
    deployment_checkpoint: str,
) -> dict[str, Any]:
    """Join the measured throughput to the line arithmetic."""
    geometry = derive_geometry(config)
    ceiling = float(config["utilisation_ceiling"])
    latency_ms = float(config["decision_latency_budget_ms"])

    # Peak sustainable rate per image size: the best batch size wins, then derated
    # by the utilisation ceiling. Planning at 100% of a measured peak is how vision
    # systems end up dropping frames on the day the line runs hot.
    peak: dict[int, dict[str, Any]] = {}
    for m in measurements:
        if not m.ok or m.device != deployment_device or m.checkpoint != deployment_checkpoint:
            continue
        best = peak.get(m.imgsz)
        if best is None or m.throughput_fps > best["measured_fps"]:
            # The slowest pass of the repeat set, derated the same way. Between
            # independent invocations of this script the same cell moves by tens of
            # percent (see protocol.repeatability_note), so a single derated median
            # is not a number to buy hardware against. Every accelerator count is
            # therefore also reported against the slowest pass observed.
            slowest_ms = m.across_repeats.get("max_ms") or m.per_frame_ms
            slowest_fps = 1000.0 / slowest_ms if slowest_ms > 0 else m.throughput_fps
            peak[m.imgsz] = {
                "measured_fps": round(m.throughput_fps, 2),
                "at_batch": m.batch,
                "per_frame_ms": round(m.per_frame_ms, 3),
                "p95_batch_ms": round(m.batch_latency["p95_ms"], 3),
                "sustainable_fps": round(m.throughput_fps * ceiling, 2),
                "measured_fps_slowest_pass": round(slowest_fps, 2),
                "sustainable_fps_slowest_pass": round(slowest_fps * ceiling, 2),
                "passes": m.across_repeats.get("passes", 1),
            }

    # Which input size is the cheapest way to inspect a whole frame at full optical
    # resolution? Smaller inputs are faster per inference but need quadratically
    # more tiles to cover the same sensor, so the answer is not obvious and is not
    # always the fastest cell in the sweep. Cost is tiles / sustainable rate, in
    # seconds of accelerator time per camera frame.
    tiled_cost = {
        sz: geometry.tiles_for_input(sz) / info["sustainable_fps"]
        for sz, info in peak.items()
        if info["sustainable_fps"] > 0
    }
    # Cost alone always nominates the largest input, because tiles fall as the
    # square of the input size while latency rises much more slowly. That answer is
    # only usable if the detector can still see at that input. Sizes whose measured
    # test mAP50 has fallen below ACCURACY_TOLERANCE of the best measured size are
    # struck out of the candidate set before the cheapest is chosen.
    # Selected on val; test is read only so the plan can publish the curve it did
    # not choose on.
    selection_accuracy = accuracy_by_imgsz(deployment_checkpoint, split="val")
    accuracy = accuracy_by_imgsz(deployment_checkpoint, split="test")
    usable = dict(tiled_cost)
    accuracy_basis = "none: reports/model_study.json absent, reference chosen on cost alone"
    if selection_accuracy:
        scored = {sz: selection_accuracy[sz] for sz in tiled_cost if sz in selection_accuracy}
        if scored:
            best_map = max(scored.values())
            kept = {
                sz: cost
                for sz, cost in tiled_cost.items()
                if scored.get(sz, 0.0) >= ACCURACY_TOLERANCE * best_map
            }
            if kept:
                usable = kept
            accuracy_basis = (
                f"VAL mAP50 from reports/model_study.json (the test split is "
                f"reported below, never selected on); best {best_map:.4f}, "
                f"sizes kept within {ACCURACY_TOLERANCE:.0%} of it: "
                f"{sorted(usable)} of {sorted(tiled_cost)}"
            )
    reference_imgsz = min(usable, key=usable.get) if usable else 0

    scenarios: list[dict[str, Any]] = []
    for name, speed in config["line_speeds_m_per_min"].items():
        entry: dict[str, Any] = {"line": name}
        # Scenario-level demand is quoted at the reference input size; every input
        # size measured is spelled out under by_imgsz.
        entry.update(demand_at_speed(geometry, speed, reference_imgsz or 640))
        entry["by_imgsz"] = {
            str(sz): capacity_plan(
                geometry,
                sz,
                info["sustainable_fps"],
                speed,
                latency_ms,
                sustainable_fps_low=info.get("sustainable_fps_slowest_pass"),
                peak_batch=int(info.get("at_batch", 1)),
                p95_batch_ms=float(info.get("p95_batch_ms", 0.0)),
            )
            for sz, info in sorted(peak.items())
        }
        scenarios.append(entry)

    lo, hi = config["speed_sweep_m_per_min"]
    # Log-spaced: the decisions that matter (does one GPU still cover one camera?)
    # all happen at the slow end, and a linear sweep spends 90% of its points where
    # the answer has already saturated at zero.
    sweep_speeds = np.geomspace(float(lo), float(hi), 64)
    demands = [demand_at_speed(geometry, float(v), reference_imgsz or 640) for v in sweep_speeds]
    curve = {}
    for sz, info in sorted(peak.items()):
        capacity = info["sustainable_fps"]
        demands_sz = [demand_at_speed(geometry, float(v), sz) for v in sweep_speeds]
        tiled = [capacity / d["inferences_per_s_per_camera_tiled"] for d in demands_sz]
        down = [capacity / d["fps_per_camera"] for d in demands]
        curve[str(sz)] = {
            # `exact` is the capacity ratio and can be fractional: 0.5 means one
            # camera needs two accelerators. `floor` is what you can actually deploy.
            "tiled_exact": [round(v, 4) for v in tiled],
            "tiled_floor": [int(v) for v in tiled],
            "downscaled_exact": [round(v, 4) for v in down],
            "downscaled_floor": [int(v) for v in down],
        }

    return {
        "geometry": geometry.to_dict(),
        "assumptions": {
            **{k: v for k, v in config.items() if k != "line_speeds_m_per_min"},
            "note": (
                "sustainable_fps = measured peak throughput * utilisation_ceiling. "
                "Tiled mode preserves the full optical resolution because the tile "
                "is cut at the network input size, so its cost is "
                "tiles_per_frame(imgsz) inferences per camera frame and rises "
                "quadratically as imgsz falls. Downscaled mode costs 1 inference "
                f"per frame but degrades resolution by {geometry.sensor_px}/imgsz. "
                "Comparing the two at the same imgsz is the real architecture "
                "decision; comparing tiled figures across imgsz is not a like-for-"
                "like speed comparison, it is a cost comparison at equal resolution."
            ),
        },
        "deployment": {
            "device": deployment_device,
            "checkpoint": deployment_checkpoint,
            "peak_by_imgsz": {str(k): v for k, v in sorted(peak.items())},
            "tiled_reference_imgsz": reference_imgsz,
            "test_mAP50_by_imgsz": {str(sz): round(accuracy[sz], 4) for sz in sorted(accuracy)},
            "val_mAP50_by_imgsz": {
                str(sz): round(selection_accuracy[sz], 4) for sz in sorted(selection_accuracy)
            },
            "reference_imgsz_basis": accuracy_basis,
            "accuracy_note": (
                "Input size is an accuracy axis before it is a throughput axis. "
                "This checkpoint was trained at 320 px and its accuracy collapses "
                "above that -- measured test mAP50 by input size: "
                + (
                    ", ".join(f"{sz} px {accuracy[sz]:.3f}" for sz in sorted(accuracy))
                    or "not measured (reports/model_study.json absent)"
                )
                + ". Every throughput figure in this report is real, but a plan "
                "built on the fastest input rather than a usable one would be "
                "sizing hardware to run a detector that no longer detects. The "
                "reference input above is the cheapest one still within "
                f"{ACCURACY_TOLERANCE:.0%} of the best measured accuracy, and it "
                "is chosen on val -- the test curve is published here, not "
                "selected on. The transfer this plan cannot check: that accuracy "
                "curve was measured on 200x200 NEU-DET crops, where the defect "
                "fills much of the frame. A tile cut at the network input from a "
                f"{geometry.sensor_px} px sensor at {geometry.mm_per_px} mm/px "
                f"covers {geometry.mm_per_px * reference_imgsz:.0f} mm of strip "
                "and presents the same defect at a different magnification and a "
                "different area fraction. Nothing here measures that, because no "
                "labelled strip capture exists in this project; the capacity "
                "arithmetic is sound, but the accuracy it is paired with is a "
                "transfer assumption until it is re-measured on real line frames."
            ),
            "tiles_per_frame_by_input": {
                str(sz): geometry.tiles_for_input(sz) for sz in sorted(peak)
            },
            "accelerator_seconds_per_frame_tiled": {
                str(sz): round(cost, 4) for sz, cost in sorted(tiled_cost.items())
            },
            "effective_resolution_mm_per_px_tiled": {
                str(sz): geometry.mm_per_px for sz in sorted(peak)
            },
            "effective_resolution_mm_per_px_downscaled": {
                str(sz): round(geometry.mm_per_px * geometry.sensor_px / sz, 3)
                for sz in sorted(peak)
            },
        },
        "scenarios": scenarios,
        "streams_curve": {
            "line_speed_m_per_min": [round(float(v), 1) for v in sweep_speeds],
            "streams_per_accelerator": curve,
        },
    }


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def _style_axes(ax, *, xlabel: str, ylabel: str, title: str | None = None) -> None:
    """Recessive chrome: hairline solid grid, no top/right spines, muted tick ink."""
    ax.set_facecolor(PALETTE["surface"])
    ax.grid(True, which="major", color=PALETTE["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PALETTE["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=PALETTE["ink_muted"], labelsize=8, length=3, width=0.8)
    ax.set_xlabel(xlabel, color=PALETTE["ink_secondary"], fontsize=9)
    ax.set_ylabel(ylabel, color=PALETTE["ink_secondary"], fontsize=9)
    if title:
        ax.set_title(title, color=PALETTE["ink"], fontsize=10, loc="left", pad=8)


def _plain_log_axis(ax, axis: str = "y", subs: tuple[float, ...] = (1.0, 2.0, 5.0)) -> None:
    """Log scale with readable numbers: 20, 50, 100 rather than 2 x 10^1."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_locator(LogLocator(base=10.0, subs=subs, numticks=14))
    target.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    target.set_minor_formatter(NullFormatter())


def _figure_legend(fig, ax, ncol: int) -> None:
    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(
        handles, labels, loc="lower center", frameon=False, fontsize=8, ncol=ncol,
        bbox_to_anchor=(0.5, 0.002),
    )
    for text in legend.get_texts():
        text.set_color(PALETTE["ink_secondary"])


def chart_latency_vs_imgsz(report: dict[str, Any], out_path: Path) -> bool:
    """Small multiples: one panel per checkpoint, one line per device.

    Single-frame (batch 1) latency is what decides whether a defect can be flagged
    before the strip has moved on, so that is what is plotted; batch throughput
    lives in the JSON and drives the capacity chart.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [m for m in report["measurements"] if m["ok"] and m["batch"] == 1]
    if not rows:
        return False
    checkpoints = sorted({r["checkpoint"] for r in rows})
    devices = sorted({r["device"] for r in rows}, reverse=True)  # accelerator first
    sizes = sorted({r["imgsz"] for r in rows})

    fig, axes = plt.subplots(
        1, len(checkpoints), figsize=(max(9.0, 4.8 * len(checkpoints)), 4.6),
        squeeze=False, sharey=True,
    )
    fig.patch.set_facecolor(PALETTE["page"])

    for col, ckpt in enumerate(checkpoints):
        ax = axes[0][col]
        for idx, device in enumerate(devices):
            pts = sorted(
                (r for r in rows if r["checkpoint"] == ckpt and r["device"] == device),
                key=lambda r: r["imgsz"],
            )
            if not pts:
                continue
            xs = [p["imgsz"] for p in pts]
            ys = [p["batch_latency"]["median_ms"] for p in pts]
            hi = [p["batch_latency"]["p95_ms"] for p in pts]
            colour = PALETTE["series"][idx % len(PALETTE["series"])]
            ax.fill_between(xs, ys, hi, color=colour, alpha=0.13, linewidth=0, zorder=1)
            ax.plot(
                xs, ys, color=colour, linewidth=2.0, marker="o", markersize=5,
                markeredgecolor=PALETTE["surface"], markeredgewidth=2.0,
                label=device.upper(), zorder=3,
            )
            ax.annotate(
                f"{ys[-1]:.0f} ms",
                xy=(xs[-1], ys[-1]), xytext=(7, 0), textcoords="offset points",
                color=PALETTE["ink_secondary"], fontsize=8, va="center",
            )
        ax.set_yscale("log")
        ax.set_xticks(sizes)
        _plain_log_axis(ax, "y")
        _style_axes(
            ax,
            xlabel="network input size (px)",
            ylabel="single-frame latency, p50 (ms)" if col == 0 else "",
            title=ckpt,
        )
        ax.margins(x=0.20)

    _figure_legend(fig, axes[0][0], ncol=len(devices))
    fig.suptitle(
        "Single-frame latency vs network input size   (shaded band: p50 to p95)",
        color=PALETTE["ink"], fontsize=11.5, x=0.012, ha="left", y=0.985,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(out_path, dpi=160, facecolor=PALETTE["page"])
    plt.close(fig)
    return True


def chart_streams_vs_speed(report: dict[str, Any], out_path: Path) -> bool:
    """Three panels answering the capacity question.

    Left and centre are the same axes for the two ways of feeding a 2048 px camera
    frame to the network, so the 16x cost of keeping full optical resolution is a
    horizontal shift the reader can measure by eye. The curve is the *exact*
    capacity ratio -- 0.4 means one camera needs three accelerators -- because the
    deployable integer floors to zero above modest speeds and a floor of zero draws
    a flat, uninformative line. The right panel names the lines and puts their
    demand against the measured capacity of one GPU.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mill = report["mill"]
    curve = mill["streams_curve"]
    speeds = curve["line_speed_m_per_min"]
    series = curve["streams_per_accelerator"]
    scenarios = sorted(mill["scenarios"], key=lambda s: s["line_speed_m_per_min"])
    if not series or not scenarios:
        return False
    sizes = sorted(series, key=int)
    geometry = mill["geometry"]

    fig, (ax_tiled, ax_down, ax_demand) = plt.subplots(
        1, 3, figsize=(16.0, 5.2), gridspec_kw={"width_ratios": [1.0, 1.0, 1.25]}
    )
    fig.patch.set_facecolor(PALETTE["page"])

    tiles_by_input = mill["deployment"]["tiles_per_frame_by_input"]
    modes = (
        (ax_tiled, "tiled_exact",
         f"Full-resolution tiled  (tile cut at the network input: "
         f"{geometry['optical_resolution_mm_per_px']} mm/px at every input size)"),
        (ax_down, "downscaled_exact",
         "Downscaled single-shot  (1 inference/frame, resolution traded for speed)"),
    )
    for ax, field_name, title in modes:
        traces = [(sz, series[sz][field_name]) for sz in sizes]
        # Stagger the direct labels by where each line actually sits, not by the
        # order the sizes were measured, so near-parallel curves do not stack
        # their labels on top of one another.
        rank = {
            sz: pos
            for pos, (sz, _) in enumerate(sorted(traces, key=lambda t: -t[1][0]))
        }
        for idx, (sz, ys) in enumerate(traces):
            colour = PALETTE["series"][idx % len(PALETTE["series"])]
            label = (
                f"{sz} px input, {tiles_by_input.get(sz, '?')} tiles/frame"
                if ax is ax_tiled
                else f"{sz} px input"
            )
            ax.plot(speeds, ys, color=colour, linewidth=2.0, label=label, zorder=3)
            if ax is not ax_tiled:
                continue
            ax.annotate(
                f"{sz} px", xy=(speeds[0], ys[0]),
                xytext=(7, 11 - 11 * rank[sz]), textcoords="offset points",
                color=PALETTE["ink_secondary"], fontsize=8,
            )
        ax.axhline(1.0, color=PALETTE["axis"], linewidth=1.2, zorder=2)
        ax.annotate(
            "one camera per accelerator",
            xy=(speeds[-1], 1.0), xytext=(-4, 5), textcoords="offset points",
            color=PALETTE["ink_muted"], fontsize=7.5, ha="right",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        _plain_log_axis(ax, "x")
        _plain_log_axis(ax, "y")
        _style_axes(
            ax,
            xlabel="line speed (m/min)",
            ylabel="camera streams per accelerator" if ax is ax_tiled else "",
            title=title,
        )

    ys = np.arange(len(scenarios))
    ax_demand.scatter(
        [s["inferences_per_s_tiled"] for s in scenarios], ys, s=62,
        color=PALETTE["ink_secondary"], zorder=4,
        edgecolor=PALETTE["surface"], linewidth=2.0,
        label=f"demand, tiled @{mill['deployment']['tiled_reference_imgsz']} px",
    )
    ax_demand.scatter(
        [s["inferences_per_s_downscaled"] for s in scenarios], ys, s=62,
        facecolor=PALETTE["surface"], zorder=4,
        edgecolor=PALETTE["ink_secondary"], linewidth=1.6, label="demand, downscaled",
    )
    for y, scenario in zip(ys, scenarios):
        ax_demand.annotate(
            f"{scenario['inferences_per_s_tiled']:,.0f}/s",
            xy=(scenario["inferences_per_s_tiled"], y), xytext=(10, 0),
            textcoords="offset points", color=PALETTE["ink_secondary"], fontsize=8,
            va="center",
        )

    capacities = []
    for idx, sz in enumerate(sizes):
        cap = mill["deployment"]["peak_by_imgsz"][sz]["sustainable_fps"]
        capacities.append(cap)
        colour = PALETTE["series"][idx % len(PALETTE["series"])]
        ax_demand.axvline(cap, color=colour, linewidth=1.6, zorder=2)
        # The capacities sit within a factor of a few of each other, so the
        # labels are stepped down the panel instead of overprinting.
        ax_demand.annotate(
            f"{sz} px: {cap:.0f}/s",
            xy=(cap, len(scenarios) - 0.6 - 1.6 * idx), xytext=(-5, 0),
            textcoords="offset points", color=PALETTE["ink_secondary"], fontsize=7.5,
            rotation=90, va="top", ha="right",
        )

    ax_demand.set_yticks(ys)
    ax_demand.set_yticklabels(
        [f"{s['line']}  ({s['line_speed_m_per_min']:.0f} m/min)" for s in scenarios],
        fontsize=8, color=PALETTE["ink_secondary"],
    )
    ax_demand.set_xscale("log")
    lo = min(min(s["inferences_per_s_downscaled"] for s in scenarios), min(capacities))
    hi = max(max(s["inferences_per_s_tiled"] for s in scenarios), max(capacities))
    ax_demand.set_xlim(lo * 0.45, hi * 2.6)
    ax_demand.margins(y=0.12)
    _plain_log_axis(ax_demand, "x", subs=(1.0,))
    _style_axes(
        ax_demand,
        xlabel="required inferences per second, whole strip width",
        ylabel="",
        title="Line demand vs capacity of one accelerator",
    )
    ax_demand.grid(axis="y", visible=False)
    demand_legend = ax_demand.legend(
        loc="lower right", frameon=False, fontsize=7.5, handletextpad=0.3
    )
    for text in demand_legend.get_texts():
        text.set_color(PALETTE["ink_secondary"])

    _figure_legend(fig, ax_tiled, ncol=len(sizes))
    fig.suptitle(
        f"Camera streams one accelerator can serve   "
        f"({geometry['cameras_across_width']} cameras across "
        f"{geometry['strip_width_m']} m strip, one frame every "
        f"{geometry['strip_advance_per_frame_m']:.2f} m of travel; "
        f"tiled demand quoted at the cheapest full-resolution input, "
        f"{mill['deployment']['tiled_reference_imgsz']} px)",
        color=PALETTE["ink"], fontsize=11.5, x=0.008, ha="left", y=0.985,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.92))
    fig.savefig(out_path, dpi=160, facecolor=PALETTE["page"])
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--iters", type=int, default=50, help="timed iterations per cell")
    ap.add_argument("--min-iters", type=int, default=8, help="floor before the budget applies")
    ap.add_argument("--budget-s", type=float, default=6.0, help="wall-clock budget per cell")
    ap.add_argument("--warmup", type=int, default=3, help="discarded warm-up calls per cell")
    ap.add_argument(
        "--repeats", type=int, default=1,
        help="independent passes over the whole sweep. Every cell is then reported "
             "at its median pass with the across-pass spread recorded beside it. "
             "1 is a spot check; use 3 for a number anyone is going to size "
             "hardware against.",
    )
    ap.add_argument("--devices", nargs="+", default=None, help="default: mps (or cuda) and cpu")
    # 256 is in the default grid because it is the input size `src/model_study.py`
    # selected on val and the size the system is meant to ship at. A sweep that
    # starts at 320 cannot cost the shipping configuration at all.
    ap.add_argument("--imgsz", nargs="+", type=int, default=[256, 320, 416, 640])
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 4, 8, 16])
    ap.add_argument("--frame-px", type=int, default=640, help="camera tile size fed to the model")
    ap.add_argument("--pool", type=int, default=32, help="distinct frames held in RAM")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--min-age-s", type=float, default=45.0, help="ignore newer checkpoints")
    ap.add_argument(
        "--runs", nargs="+", default=None,
        help="restrict the sweep to these run directories under models/ "
             "(default: every run found)",
    )
    ap.add_argument(
        "--files", nargs="+", default=None,
        help="restrict the sweep to these weight filenames, e.g. best.pt "
             "(default: every .pt in each run's weights/ directory)",
    )
    ap.add_argument("--out", type=Path, default=REPORTS_DIR / "benchmark.json")
    ap.add_argument("--no-charts", action="store_true")
    ap.add_argument(
        "--replot", action="store_true",
        help="re-render the charts from an existing --out report without re-measuring",
    )
    return ap.parse_args()


def _stage_timing_trustworthy(devices: Sequence[str]) -> bool:
    """True when every device's per-stage clocks are accelerator-synchronised.

    CPU is synchronous by construction and CUDA profilers already synchronise; MPS
    only does once `inference._install_synchronised_profiler` has swapped in a
    syncing Profile. Without it the forward pass is billed to postprocess and the
    pipeline reads as NMS-bound when it is compute-bound.
    """
    from inference import _install_synchronised_profiler

    return all(_install_synchronised_profiler(resolve_device(d)) for d in devices)


# Fields the charts read that older reports do not carry. Checked up front so
# --replot on a stale artefact says what is wrong instead of dying on a KeyError
# halfway through, having already overwritten one of the two charts.
_CHART_REQUIRED = (
    ("measurements",),
    ("mill", "deployment", "tiles_per_frame_by_input"),
    ("mill", "deployment", "peak_by_imgsz"),
    ("mill", "streams_curve", "streams_per_accelerator"),
    ("mill", "scenarios"),
)


def _require_chartable(report: dict[str, Any], source: Path) -> None:
    """Fail loudly if `report` predates the current schema."""
    for keys in _CHART_REQUIRED:
        node: Any = report
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                raise SystemExit(
                    f"{source} does not carry '{'.'.join(keys)}' and was written by an "
                    "older version of this script; re-run the sweep "
                    f"(python src/benchmark.py --out {source}) rather than replotting it."
                )
            node = node[key]


def render_charts(report: dict[str, Any], report_path: Path) -> list[Path]:
    """Render both charts beside `report_path`, named after it.

    The names are derived from the report rather than fixed, because a sweep
    written to a second `--out` (a spot check, a re-run with different flags) would
    otherwise silently overwrite the charts belonging to the canonical report and
    leave a JSON and two PNGs on disk that disagree with each other.
    """
    stem = report_path.stem
    out_dir = report_path.parent
    written: list[Path] = []
    for chart, suffix in (
        (chart_latency_vs_imgsz, "latency_vs_imgsz"),
        (chart_streams_vs_speed, "streams_vs_line_speed"),
    ):
        target = out_dir / f"{stem}_{suffix}.png"
        if chart(report, target):
            written.append(target)
        else:
            print(f"[skip] {target.name}: no data in the report to plot")
    return written


def main() -> int:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.replot:
        # A styling change should not cost another sweep, and re-measuring would
        # silently replace the numbers the report was signed off against.
        report = json.loads(args.out.read_text())
        _require_chartable(report, args.out)
        for path in render_charts(report, args.out):
            print(f"[write] {path}")
        print_headline(report)
        return 0

    accelerator = resolve_device("auto")
    if args.devices:
        devices = list(dict.fromkeys(args.devices))
    else:
        devices = list(dict.fromkeys([accelerator, "cpu"]))

    staging = Path(tempfile.mkdtemp(prefix="jsl_bench_"))
    started = time.perf_counter()
    load_start = load_snapshot()
    try:
        checkpoints = discover_checkpoints(
            MODELS_DIR, staging, args.min_age_s, args.runs, args.files
        )
        if not checkpoints:
            print(f"No stable checkpoint found under {MODELS_DIR}.", file=sys.stderr)
            return 1
        for ckpt in checkpoints:
            measure_complexity(ckpt, args.imgsz)
            print(
                f"[model] {ckpt.key}  {ckpt.params/1e6:.2f}M params  "
                f"{ckpt.size_mb:.1f} MB  GFLOPs {ckpt.gflops_by_imgsz}  "
                f"aliases={ckpt.aliases}"
            )

        stage_sync = _stage_timing_trustworthy(devices)
        pool = build_frame_pool(args.pool, args.frame_px)
        print(
            f"[pool] {len(pool)} frames at {args.frame_px}x{args.frame_px} "
            f"from {TEST_IMAGES_DIR.relative_to(PROJECT_ROOT)}"
        )

        passes = []
        for attempt in range(max(1, int(args.repeats))):
            if args.repeats > 1:
                print(f"\n[pass {attempt + 1}/{args.repeats}]")
            passes.append(
                run_sweep(checkpoints, pool, devices, args.imgsz, args.batches, args)
            )
        measurements = fold_repeats(passes)
        # Read after the sweep, not before: ultralytics raises the intra-op count on
        # its first CPU predict, so a reading taken up front reports a state that no
        # timed iteration ever ran under.
        threads = cpu_thread_state()

        # The deployment configuration is the accelerator plus the checkpoint the
        # runtime would actually serve. That is resolve_weights(), not whichever
        # cell happened to time fastest: two checkpoints of the same architecture
        # differ only by noise, so picking the maximum would silently promote a
        # scratch checkpoint the app will never load and quote its luckiest run.
        healthy = [m for m in measurements if m.ok and m.device == accelerator]
        if not healthy:
            healthy = [m for m in measurements if m.ok]
        if not healthy:
            print("Every cell failed; nothing to plan against.", file=sys.stderr)
            return 1
        deployment_device = healthy[0].device

        served = resolve_weights()
        served_key = f"{served.parent.parent.name}/{served.name}"
        served_aliases = {
            alias for c in checkpoints if served_key in c.aliases for alias in [c.key]
        }
        measured_keys = {m.checkpoint for m in healthy}
        if served_aliases & measured_keys:
            deployment_checkpoint = next(iter(served_aliases & measured_keys))
            selection = f"resolve_weights() -> {served_key}"
        else:
            # The served checkpoint was in flight (skipped) or absent. Say so
            # rather than quietly planning against something else.
            deployment_checkpoint = max(healthy, key=lambda m: m.throughput_fps).checkpoint
            selection = (
                f"resolve_weights() -> {served_key}, which was not measured "
                f"(in flight or missing); fell back to the fastest measured "
                f"checkpoint {deployment_checkpoint}"
            )
        print(f"[deploy] {selection}")

        mill = build_mill_section(
            measurements, MILL_CONFIG, deployment_device, deployment_checkpoint
        )
        mill["deployment"]["served_checkpoint_path"] = str(served)
        mill["deployment"]["selection"] = selection

        report: dict[str, Any] = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "elapsed_s": round(time.perf_counter() - started, 1),
            "host": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor() or platform.machine(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "accelerator": accelerator,
            },
            "protocol": {
                "timed_call": "DefectDetector.predict (batch 1) / predict_batch (batch > 1)",
                "warmup_calls_discarded": args.warmup,
                "target_iterations": args.iters,
                "sweep_passes": max(1, int(args.repeats)),
                "per_cell_budget_s": args.budget_s,
                "min_iterations": args.min_iters,
                "source_frame_px": args.frame_px,
                "frame_pool": len(pool),
                "conf": args.conf,
                "iou": args.iou,
                "stage_timing_synchronised": stage_sync,
                # Read before quoting any number from this file.
                "note": (
                    "Percentiles are of the observed sample; cells whose n is small "
                    "were cut short by the per-cell time budget, so read p99 there as "
                    "the observed worst case rather than a true tail estimate. This "
                    "stage split (preprocess/inference/postprocess) is only "
                    "meaningful when stage_timing_synchronised is true: the "
                    "framework's own stage clocks do not synchronise on MPS, which "
                    "bills the forward pass to postprocess and makes the pipeline "
                    "look NMS-bound when it is compute-bound. Wall-clock totals and "
                    "throughput are unaffected either way. CPU cells share the "
                    "performance cores with everything else on the host, so read "
                    "host_contention below before treating them as a floor; MPS "
                    "cells own the GPU. These are Apple M-series numbers on a "
                    "laptop, not a claim about the industrial PC or accelerator "
                    "that would sit in the pulpit -- they bound the model's cost, "
                    "not the mill's hardware."
                ),
                "cpu_threads": threads,
                "cpu_batch_note": cpu_batch_note(measurements),
                "repeatability_note": (
                    "MPS latency on this host is not stable between processes, and "
                    "repeating the sweep inside one process does not fix that. "
                    "Every cell here carries across_repeats: the spread over the "
                    "passes of THIS invocation. That spread is the smaller of the "
                    "two, because the passes run back to back and share whatever "
                    "thermal and GPU-clock state the process started in. The "
                    "between-invocation spread is larger and was measured: two "
                    "independent three-pass runs of this script, same weights, same "
                    "frames, idle machine, reported peak yolov8n throughput at 320 "
                    "px of 233.9 and 282.6 FPS, and at 640 px batch 1 a median pass "
                    "of 11.74 and 7.37 ms/frame -- 21% and 59% apart, both of them "
                    "medians of three. macOS clocks the GPU against thermal and "
                    "power state and the window server shares it. Read every "
                    "throughput figure here as carrying roughly +/-25% between "
                    "invocations, propagate that into any accelerator count (the "
                    "slowest-pass counts in mill.scenarios are the budgeting "
                    "figures), and re-measure on the actual target accelerator "
                    "before ordering hardware. These figures bound the model's "
                    "cost, not the deployment's."
                ),
                "host_contention": {
                    "at_start": load_start,
                    "at_end": load_snapshot(),
                    "note": (
                        "Kernel load average, read at the start and end of the "
                        "sweep. The sweep itself contributes to it; what matters "
                        "is whether the start figure shows another heavy job "
                        "already resident."
                    ),
                },
            },
            "checkpoints": [asdict(c) for c in checkpoints],
            "measurements": [asdict(m) for m in measurements],
            "mill": mill,
        }
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\n[write] {args.out}")

        if not args.no_charts:
            for path in render_charts(report, args.out):
                print(f"[write] {path}")

        print_headline(report)
        return 0
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def print_headline(report: dict[str, Any]) -> None:
    """The three sentences a plant engineer actually reads."""
    mill = report["mill"]
    geo = mill["geometry"]
    dep = mill["deployment"]
    print("\n" + "=" * 78)
    print("MILL READINESS")
    print("=" * 78)
    print(
        f"Installation: {geo['cameras_across_width']} cameras across "
        f"{geo['strip_width_m']} m of strip, {geo['camera_fov_m']:.3f} m field of view "
        f"at {geo['optical_resolution_mm_per_px']} mm/px, one frame every "
        f"{geo['strip_advance_per_frame_m']:.3f} m of travel; at full optical "
        f"resolution one frame is "
        + ", ".join(
            f"{tiles} tiles at {sz} px"
            for sz, tiles in dep["tiles_per_frame_by_input"].items()
        )
        + "."
    )
    accuracy = dep.get("test_mAP50_by_imgsz") or {}
    for sz, info in dep["peak_by_imgsz"].items():
        accuracy_bit = f" | test mAP50 {accuracy[sz]:.3f}" if sz in accuracy else ""
        print(
            f"  {sz:>3} px on {dep['device']}: {info['measured_fps']:.1f} FPS peak "
            f"(batch {info['at_batch']}), plan at {info['sustainable_fps']:.1f} FPS "
            f"({info.get('sustainable_fps_slowest_pass', info['sustainable_fps']):.1f} on "
            f"the slowest of {info.get('passes', 1)} pass(es)), "
            f"{info['per_frame_ms']:.2f} ms/frame{accuracy_bit}."
        )
    print(f"  Reference input for tiled capacity: {dep.get('reference_imgsz_basis')}")
    reference = str(dep.get("tiled_reference_imgsz") or "")
    for scenario in mill["scenarios"]:
        if not scenario["by_imgsz"]:
            continue
        # The reference input is the cheapest way to cover a frame at full optical
        # resolution. Quoting the smallest input instead would report the tiled
        # plan of a configuration that does not deliver that resolution at all.
        sz, plan = (
            (reference, scenario["by_imgsz"][reference])
            if reference in scenario["by_imgsz"]
            else min(scenario["by_imgsz"].items(), key=lambda kv: int(kv[0]))
        )
        print(
            f"  {scenario['line']:<36} {scenario['line_speed_m_per_min']:>6.0f} m/min "
            f"{scenario['fps_per_camera']:>6.1f} FPS/cam | "
            f"tiled {plan['inferences_per_s_tiled']:>7.0f} inf/s -> "
            f"{plan['accelerators_for_full_width_tiled']:>3}"
            f"-{plan.get('accelerators_for_full_width_tiled_slowest_pass', plan['accelerators_for_full_width_tiled']):<3} GPU | "
            f"downscaled {scenario['inferences_per_s_downscaled']:>6.1f} inf/s -> "
            f"{plan['streams_per_accelerator_downscaled']:>2} cam/GPU, "
            f"{plan['accelerators_for_full_width_downscaled']:>2} GPU  "
            f"(@{sz}px)"
        )
    print(
        "  Tiled preserves "
        f"{mill['geometry']['optical_resolution_mm_per_px']} mm/px; downscaled trades it "
        "for speed at " + ", ".join(
            f"{sz}px -> {mm} mm/px"
            for sz, mm in dep["effective_resolution_mm_per_px_downscaled"].items()
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
