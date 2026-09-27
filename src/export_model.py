"""Deployment export path for the Jindal Stainless surface-defect detector.

A `.pt` checkpoint is a research artefact: it needs the full PyTorch stack, it
carries optimizer and EMA state, and it will not run on the industrial PC or the
Jetson-class box that ends up in the pulpit. This script produces the artefacts
that will, and -- more importantly -- **proves they are the same model**:

* export the checkpoint to ONNX (opset-pinned, statically shaped by default) and,
  on Apple silicon, to CoreML as well. A CoreML failure is reported and survived:
  it is a convenience target on this machine, not the deployment target;
* structurally validate the ONNX graph with `onnx.checker`;
* run the exported graph in ONNX Runtime and the original PyTorch model on the
  *same* preprocessed tensor from a real NEU-DET test image, and report the raw
  head-tensor deviation -- max/mean absolute error and cosine similarity;
* run the full detection pipeline through both backends and match the resulting
  boxes, because a tensor that agrees to 1e-4 is only interesting if the boxes
  and classes it decodes to also agree;
* time both backends so the latency cost of the portable artefact is on the table;
* report file sizes, separating the inference payload from training baggage.

Nothing under `models/` is written or modified: the checkpoint is staged into
`export/` first and every artefact is produced there.

Usage:
    .venv/bin/python src/export_model.py
    .venv/bin/python src/export_model.py --weights models/<run>/weights/best.pt --imgsz 320
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Ultralytics will pip-install its way out of a version conflict on import of a
# submodule if allowed to. A deployment script must never silently mutate the
# environment it is certifying, so the auto-updater is switched off before the
# deferred `ultralytics` imports below ever run.
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

from inference import CLASS_NAMES, DEFAULT_IMGSZ, resolve_weights  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPORT_DIR = PROJECT_ROOT / "export"
REPORTS_DIR = PROJECT_ROOT / "reports"
TEST_IMAGES_DIR = PROJECT_ROOT / "data" / "neu-det" / "test" / "images"

# Opset 17 is the sweet spot for this graph: new enough for the ops YOLOv8 emits
# without a shim, old enough that TensorRT 8.6, OpenVINO 2023+ and the Jetson
# JetPack runtimes all accept it without a rebuild.
DEFAULT_OPSET = 17


@dataclass
class ArtefactReport:
    """One exported file (or bundle) and everything measured about it."""

    format: str
    ok: bool
    path: str | None = None
    size_mb: float = 0.0
    seconds: float = 0.0
    error: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sha256(path: Path) -> str:
    """Content hash of the staged checkpoint.

    `models/<run>/weights/best.pt` is a moving target while a training job runs, so
    a report that names only the path does not say which weights were exported.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def path_size_mb(path: Path) -> float:
    """Size in MB; CoreML exports are directories (.mlpackage), not files."""
    if path.is_dir():
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    else:
        total = path.stat().st_size
    return round(total / 1e6, 2)


def pick_test_image(preferred: Path | None) -> Path:
    """A real test frame -- verification on synthetic noise proves nothing, because
    a random tensor never lights up the detection head the way steel texture does."""
    if preferred is not None:
        if not preferred.is_file():
            raise FileNotFoundError(f"Test image not found: {preferred}")
        # Decode now rather than after a two-minute export: a truncated or
        # mislabelled file otherwise sails past every check until cv2 raises a raw
        # assertion deep inside the verification step.
        if cv2.imread(str(preferred), cv2.IMREAD_COLOR) is None:
            raise ValueError(f"Test image could not be decoded as an image: {preferred}")
        return preferred
    candidates = sorted(TEST_IMAGES_DIR.glob("*.jpg"))
    if not candidates:
        raise FileNotFoundError(f"No test images under {TEST_IMAGES_DIR}")
    return candidates[len(candidates) // 2]


def letterbox_tensor(image_path: Path, imgsz: int) -> np.ndarray:
    """Build the exact NCHW float32 input the detector would see.

    Ultralytics' own `LetterBox` is used rather than a hand-rolled resize so the
    padding, scaling and stride alignment match the runtime byte for byte.
    """
    from ultralytics.data.augment import LetterBox

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not decode {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    padded = LetterBox((imgsz, imgsz), auto=False, scaleup=True, stride=32)(image=rgb)
    chw = np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32) / 255.0
    return chw[None, ...]


def timed(fn, n: int, warmup: int = 3) -> dict[str, float]:
    """Median-of-n wall clock, warm-up discarded. Median rather than mean: this
    runs on a machine that is sharing its CPU with a training job."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(samples)
    return {
        "n": int(arr.size),
        "median_ms": round(float(np.median(arr)), 3),
        "mean_ms": round(float(arr.mean()), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def stage_checkpoint(weights: Path, export_dir: Path) -> Path:
    """Copy the checkpoint into `export/` before exporting.

    Ultralytics writes export artefacts beside the source weights. Staging keeps
    `models/` untouched (a training run may be writing into it right now) and puts
    every deployable artefact in one directory that can be shipped as a unit.
    """
    export_dir.mkdir(parents=True, exist_ok=True)
    staged = export_dir / f"{weights.parent.parent.name}_{weights.stem}.pt"
    shutil.copy2(weights, staged)
    return staged


def export_format(
    staged: Path, fmt: str, imgsz: int, opset: int | None, extra: dict[str, Any] | None = None
) -> ArtefactReport:
    """Run one ultralytics export, catching failure rather than aborting the run."""
    from ultralytics import YOLO

    kwargs: dict[str, Any] = {
        "format": fmt,
        "imgsz": imgsz,
        "device": "cpu",  # exporters trace on CPU; MPS tracing is not supported
        "verbose": False,
    }
    if opset is not None:
        kwargs["opset"] = opset
    kwargs.update(extra or {})

    t0 = time.perf_counter()
    try:
        out = Path(YOLO(str(staged)).export(**kwargs))
        elapsed = time.perf_counter() - t0
        return ArtefactReport(
            format=fmt,
            ok=True,
            path=str(out),
            size_mb=path_size_mb(out),
            seconds=round(elapsed, 2),
            detail={"kwargs": {k: v for k, v in kwargs.items() if k != "verbose"}},
        )
    except Exception as exc:  # noqa: BLE001 - CoreML failing must not sink the run
        return ArtefactReport(
            format=fmt,
            ok=False,
            seconds=round(time.perf_counter() - t0, 2),
            error=f"{type(exc).__name__}: {exc}",
            detail={"traceback_tail": traceback.format_exc().strip().splitlines()[-4:]},
        )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def check_onnx_graph(onnx_path: Path) -> dict[str, Any]:
    """Structural validation plus the metadata a deployment engineer will ask for."""
    import onnx

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)

    def shape_of(value) -> list[Any]:
        return [
            d.dim_value if d.HasField("dim_value") else (d.dim_param or "?")
            for d in value.type.tensor_type.shape.dim
        ]

    return {
        "checker": "pass",
        "ir_version": int(model.ir_version),
        "opset": [{"domain": o.domain or "ai.onnx", "version": int(o.version)}
                  for o in model.opset_import],
        "producer": f"{model.producer_name} {model.producer_version}".strip(),
        "inputs": [{"name": i.name, "shape": shape_of(i)} for i in model.graph.input],
        "outputs": [{"name": o.name, "shape": shape_of(o)} for o in model.graph.output],
        "nodes": len(model.graph.node),
    }


def compare_raw_outputs(
    staged: Path, onnx_path: Path, tensor: np.ndarray
) -> dict[str, Any]:
    """Run both backends on one identical tensor and quantify the deviation.

    This is the check that actually catches a broken export. Detection-level
    agreement can hide a real numeric drift (NMS is a step function and absorbs
    small errors); the raw head tensor cannot.
    """
    import onnxruntime as ort
    from ultralytics import YOLO

    torch_model = YOLO(str(staged)).model.float().eval()
    with torch.inference_mode():
        torch_out = torch_model(torch.from_numpy(tensor))
    if isinstance(torch_out, (list, tuple)):
        torch_out = torch_out[0]
    reference = torch_out.detach().cpu().numpy().astype(np.float64)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    candidate = np.asarray(ort_out, dtype=np.float64)

    if candidate.shape != reference.shape:
        return {
            "match": False,
            "reason": f"shape mismatch: torch {reference.shape} vs onnx {candidate.shape}",
        }

    diff = np.abs(candidate - reference)
    scale = float(np.abs(reference).max()) or 1.0
    flat_a, flat_b = reference.ravel(), candidate.ravel()
    cosine = float(
        flat_a @ flat_b / (np.linalg.norm(flat_a) * np.linalg.norm(flat_b) + 1e-12)
    )
    return {
        "match": True,
        "output_shape": list(reference.shape),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "max_rel_diff_vs_peak": float(diff.max() / scale),
        "cosine_similarity": cosine,
        "allclose_atol_1e-3": bool(np.allclose(reference, candidate, atol=1e-3, rtol=0)),
        "allclose_atol_1e-4": bool(np.allclose(reference, candidate, atol=1e-4, rtol=0)),
        "provider": session.get_providers()[0],
    }


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two (N,4) and (M,4) xyxy box sets."""
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def compare_detections(
    staged: Path, onnx_path: Path, image_path: Path, imgsz: int, conf: float, iou: float
) -> dict[str, Any]:
    """End-to-end agreement: same image, same thresholds, both backends.

    Boxes are matched greedily by IoU in descending confidence order, which is how
    a quality engineer would compare two annotated images by eye.

    The confidence threshold walks down a ladder until the PyTorch side actually
    produces boxes. An early-epoch checkpoint finds nothing at 0.25, and "0 boxes
    versus 0 boxes" agrees perfectly while proving nothing: it never exercises the
    decode, the coordinate transform or the NMS. The threshold that was actually
    used is reported, so nobody mistakes a relaxed check for the operating point.
    """
    from ultralytics import YOLO

    ref_model, cand_model = YOLO(str(staged)), YOLO(str(onnx_path))

    def run(model, threshold: float) -> np.ndarray:
        # rect=False on both sides. With the default rect=True the framework
        # letterboxes a PyTorch model to a rectangular stride multiple but a
        # fixed-shape ONNX graph to the full square, so a non-square verification
        # frame would be run at 224x640 through one backend and 640x640 through the
        # other -- and the "do the boxes agree?" check would be comparing two
        # different computations. Square on both sides also matches the static
        # input shape this graph was exported with.
        result = model.predict(
            str(image_path), imgsz=imgsz, conf=threshold, iou=iou,
            device="cpu", rect=False, verbose=False,
        )[0]
        boxes = result.boxes.data.detach().cpu().numpy().astype(np.float64)
        return boxes[boxes[:, 4].argsort()[::-1]] if boxes.size else boxes

    ladder = [c for c in (conf, 0.10, 0.05, 0.02, 0.01) if c <= conf] or [conf]
    used = ladder[-1]
    ref = np.zeros((0, 6))
    for threshold in ladder:
        ref = run(ref_model, threshold)
        if len(ref):
            used = threshold
            break
    cand = run(cand_model, used)

    ious = _iou_matrix(ref[:, :4], cand[:, :4])
    matched: list[tuple[int, int]] = []
    taken: set[int] = set()
    for i in range(len(ref)):
        for j in (np.argsort(-ious[i]) if ious.size else []):
            if int(j) in taken:
                continue
            if ious[i, j] < 0.9 or int(ref[i, 5]) != int(cand[j, 5]):
                break
            taken.add(int(j))
            matched.append((i, int(j)))
            break

    summary: dict[str, Any] = {
        "image": image_path.name,
        "letterbox": "square, rect=False on both backends",
        "conf_requested": conf,
        "conf_used": used,
        "iou": iou,
        "pytorch_detections": int(len(ref)),
        "onnx_detections": int(len(cand)),
        "matched_iou_ge_0.9_same_class": len(matched),
    }
    if used != conf:
        summary["threshold_note"] = (
            f"No detections at conf={conf}; dropped to conf={used} so the comparison "
            "exercises the full decode and NMS path rather than comparing two empty "
            "box lists."
        )
    if matched:
        idx_a = np.array([m[0] for m in matched])
        idx_b = np.array([m[1] for m in matched])
        summary["max_box_delta_px"] = float(np.abs(ref[idx_a, :4] - cand[idx_b, :4]).max())
        summary["max_confidence_delta"] = float(np.abs(ref[idx_a, 4] - cand[idx_b, 4]).max())
        summary["min_matched_iou"] = float(ious[idx_a, idx_b].min())
    if len(ref):
        summary["top_class_pytorch"] = CLASS_NAMES[int(ref[0, 5])]
        summary["top_confidence_pytorch"] = round(float(ref[0, 4]), 4)
    if len(cand):
        summary["top_class_onnx"] = CLASS_NAMES[int(cand[0, 5])]
        summary["top_confidence_onnx"] = round(float(cand[0, 4]), 4)

    if not len(ref) and not len(cand):
        summary["verdict"] = "inconclusive: no detections from either backend at any threshold"
    elif len(ref) == len(cand) == len(matched):
        summary["verdict"] = "identical detections"
    else:
        summary["verdict"] = "detections differ"
    return summary


def benchmark_backends(
    staged: Path, onnx_path: Path, tensor: np.ndarray, n: int, thread_grid: Sequence[int]
) -> dict[str, Any]:
    """Forward-pass latency of each backend on CPU, each at its own best thread count.

    CPU on both sides deliberately: the point is what the graph costs once it
    leaves PyTorch, and ONNX Runtime has no MPS backend to compare against.

    Two things silently rig this comparison if they are left alone.

    * **Threads.** Importing `ultralytics` sets ``OMP_NUM_THREADS=1`` in the process
      environment, so PyTorch comes up single-threaded while ONNX Runtime defaults
      to every core. Measured on this 10-core M5: 1865 ms torch against 819 ms ORT,
      i.e. "ONNX is 2.3x faster", which is a fact about the harness and not about
      the model. Pinning both to the *same* count is not the fix either -- this is a
      heterogeneous CPU, and forcing ten threads costs ORT 3x (61 ms at six threads,
      204 ms at ten) because it spills onto the efficiency cores. So each backend is
      swept over `thread_grid` and quoted at its own best; the whole table is kept
      in the report so the choice is auditable.
    * **Fusion.** The exporter fuses Conv+BN before it traces, and `YOLO.predict`
      fuses on its first call, so an unfused `model` reference would hand ONNX a
      graph optimisation the PyTorch side was denied.
    """
    import onnxruntime as ort
    from ultralytics import YOLO

    torch_model = YOLO(str(staged)).model.float().eval().fuse()
    torch_input = torch.from_numpy(tensor)
    previous_threads = torch.get_num_threads()

    by_threads: dict[str, dict[str, float]] = {}
    try:
        for threads in thread_grid:
            torch.set_num_threads(threads)

            def torch_call() -> None:
                with torch.inference_mode():
                    torch_model(torch_input)

            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            session = ort.InferenceSession(
                str(onnx_path), options, providers=["CPUExecutionProvider"]
            )
            input_name = session.get_inputs()[0].name

            def ort_call() -> None:
                session.run(None, {input_name: tensor})

            by_threads[str(threads)] = {
                "pytorch_cpu_ms": timed(torch_call, n)["median_ms"],
                "onnxruntime_cpu_ms": timed(ort_call, n)["median_ms"],
            }
    finally:
        torch.set_num_threads(previous_threads)

    best_torch = min(by_threads.items(), key=lambda kv: kv[1]["pytorch_cpu_ms"])
    best_ort = min(by_threads.items(), key=lambda kv: kv[1]["onnxruntime_cpu_ms"])
    return {
        "iterations": n,
        "thread_grid": list(thread_grid),
        "measures": "forward pass only",
        "excludes": [
            "image decode",
            "letterbox / normalise",
            "host-to-device copy",
            "box decode and NMS",
            "Detection record building",
        ],
        "median_ms_by_threads": by_threads,
        "pytorch_fused": True,
        "pytorch_cpu": {
            "median_ms": best_torch[1]["pytorch_cpu_ms"],
            "best_threads": int(best_torch[0]),
        },
        "onnxruntime_cpu": {
            "median_ms": best_ort[1]["onnxruntime_cpu_ms"],
            "best_threads": int(best_ort[0]),
        },
        "speedup_onnx_vs_pytorch": round(
            best_torch[1]["pytorch_cpu_ms"] / max(best_ort[1]["onnxruntime_cpu_ms"], 1e-9), 2
        ),
        "note": (
            "This is the FORWARD PASS ONLY on one pre-letterboxed tensor: no image "
            "decode, no letterbox, no box decode, no NMS, no record building. It is "
            "not a frame rate and must not be quoted as one -- the same checkpoint "
            "through the full shipping path (src/benchmark.py, which times "
            "DefectDetector.predict end to end) is roughly 2-3x slower per frame on "
            "the same CPU. What this number is for is comparing two backends on "
            "identical work. Each backend is quoted at its own best thread count "
            "from thread_grid, both running a Conv+BN-fused graph; equalising "
            "neither would make the ratio an artefact of the harness rather than a "
            "property of the exported model. Absolute milliseconds depend on what "
            "else the host is running; the ratio is the result."
        ),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--weights", type=Path, default=None, help="default: resolve_weights()")
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                    help="static input size baked into the exported graph "
                         "(default: inference.DEFAULT_IMGSZ). Exporting at a size "
                         "the model is not accurate at produces a verified export "
                         "of a detector that does not detect.")
    ap.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    ap.add_argument(
        "--dynamic", action="store_true",
        help="dynamic batch/spatial axes; static shapes optimise better on edge runtimes",
    )
    ap.add_argument("--no-simplify", action="store_true", help="skip onnxslim graph cleanup")
    ap.add_argument("--no-coreml", action="store_true")
    ap.add_argument("--image", type=Path, default=None, help="verification frame")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--iters", type=int, default=20, help="latency iterations per backend")
    ap.add_argument(
        "--threads", type=int, nargs="+", default=None,
        help="intra-op thread counts to sweep; each backend is quoted at its own best "
             "(default: 1, half the cores, all the cores)",
    )
    ap.add_argument("--out-dir", type=Path, default=EXPORT_DIR)
    ap.add_argument("--report", type=Path, default=REPORTS_DIR / "export_summary.json")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    weights = resolve_weights(args.weights)
    image_path = pick_test_image(args.image)
    staged = stage_checkpoint(weights, args.out_dir)
    staged_sha = sha256(staged)

    print(f"[source]  {weights}  ({path_size_mb(weights)} MB)")
    print(f"[staged]  {staged}  sha256:{staged_sha[:16]}")
    print(f"[verify]  {image_path}")
    print(f"[config]  imgsz={args.imgsz} opset={args.opset} dynamic={args.dynamic}")

    artefacts: list[ArtefactReport] = []

    onnx_report = export_format(
        staged,
        "onnx",
        args.imgsz,
        args.opset,
        {"dynamic": args.dynamic, "simplify": not args.no_simplify},
    )
    artefacts.append(onnx_report)
    if not onnx_report.ok:
        print(f"[onnx]    FAILED: {onnx_report.error}", file=sys.stderr)
        return 1
    print(f"[onnx]    {onnx_report.path}  {onnx_report.size_mb} MB  "
          f"({onnx_report.seconds}s)")

    verification: dict[str, Any] = {}
    onnx_path = Path(onnx_report.path)
    tensor = letterbox_tensor(image_path, args.imgsz)

    verification["graph"] = check_onnx_graph(onnx_path)
    print(f"[onnx]    checker pass, {verification['graph']['nodes']} nodes, "
          f"opset {verification['graph']['opset']}")

    verification["raw_tensor"] = compare_raw_outputs(staged, onnx_path, tensor)
    raw = verification["raw_tensor"]
    if raw["match"]:
        print(
            f"[verify]  raw head tensor {raw['output_shape']}: "
            f"max|d|={raw['max_abs_diff']:.3e} mean|d|={raw['mean_abs_diff']:.3e} "
            f"cos={raw['cosine_similarity']:.9f}  "
            f"allclose(1e-3)={raw['allclose_atol_1e-3']}"
        )
    else:
        print(f"[verify]  raw tensor comparison IMPOSSIBLE: {raw['reason']}", file=sys.stderr)
        return 1

    verification["detections"] = compare_detections(
        staged, onnx_path, image_path, args.imgsz, args.conf, args.iou
    )
    det = verification["detections"]
    print(
        f"[verify]  detections @conf={det['conf_used']}: "
        f"pytorch={det['pytorch_detections']} onnx={det['onnx_detections']} "
        f"matched={det['matched_iou_ge_0.9_same_class']} -> {det['verdict']}"
    )
    if "max_box_delta_px" in det:
        print(
            f"[verify]  worst matched box delta {det['max_box_delta_px']:.4f} px, "
            f"worst confidence delta {det['max_confidence_delta']:.3e}, "
            f"min matched IoU {det['min_matched_iou']:.6f}"
        )

    cores = os.cpu_count() or 1
    thread_grid = sorted({max(1, int(t)) for t in (args.threads or (1, cores // 2, cores))})
    verification["latency"] = benchmark_backends(
        staged, onnx_path, tensor, args.iters, thread_grid
    )
    lat = verification["latency"]
    print(
        f"[latency] pytorch-cpu {lat['pytorch_cpu']['median_ms']} ms "
        f"@{lat['pytorch_cpu']['best_threads']} threads  "
        f"onnxruntime-cpu {lat['onnxruntime_cpu']['median_ms']} ms "
        f"@{lat['onnxruntime_cpu']['best_threads']} threads  "
        f"({lat['speedup_onnx_vs_pytorch']}x, both fused, best of {lat['thread_grid']})"
    )

    if not args.no_coreml:
        coreml_report = export_format(staged, "coreml", args.imgsz, None, {"nms": False})
        artefacts.append(coreml_report)
        if coreml_report.ok:
            print(f"[coreml]  {coreml_report.path}  {coreml_report.size_mb} MB  "
                  f"({coreml_report.seconds}s)")
        else:
            # Non-fatal by design: CoreML is a convenience target on this Mac.
            print(f"[coreml]  skipped -- export failed: {coreml_report.error}")

    params = int(sum(p.numel() for p in _load_module(staged).parameters()))
    sizes = {
        "checkpoint_pt_mb": path_size_mb(weights),
        "inference_weights_fp32_mb": round(params * 4 / 1e6, 2),
        **{f"{a.format}_mb": a.size_mb for a in artefacts if a.ok},
    }
    print(
        "[size]    "
        + "  ".join(f"{k}={v}" for k, v in sizes.items())
        + "   (the .pt also carries EMA and optimizer state; the ONNX graph does not)"
    )

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_checkpoint": str(weights),
        "staged_checkpoint": str(staged),
        "staged_checkpoint_sha256": staged_sha,
        "verification_image": str(image_path),
        "config": {
            "imgsz": args.imgsz,
            "opset": args.opset,
            "dynamic": args.dynamic,
            "simplify": not args.no_simplify,
            "conf": args.conf,
            "iou": args.iou,
        },
        "parameters": params,
        "sizes_mb": sizes,
        "artefacts": [asdict(a) for a in artefacts],
        "verification": verification,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(f"[write]   {args.report}")
    return 0


def _load_module(staged: Path):
    from ultralytics import YOLO

    return YOLO(str(staged)).model


if __name__ == "__main__":
    raise SystemExit(main())
