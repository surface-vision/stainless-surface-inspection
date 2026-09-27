"""Package a trained joint checkpoint for the live demo.

    .venv/bin/python scripts/build_demo_model.py --weights models/yolov8n_joint/weights/best.pt

1. Picks the network input size on the VALIDATION split (combined joint val mAP50), never on test.
2. Exports a static ONNX graph at that size to site/model/detector.onnx.
3. Fits an isotonic confidence calibrator on validation detections only (a detection is
   "right" if it overlaps a same-class ground-truth box at IoU >= 0.5, greedy by score),
   and reports expected calibration error on the untouched test split.
4. Writes site/js/calibration.js (same interface the page already imports) and
   reports/demo_model.json with every number the page quotes.

Everything runs on CPU so it does not compete with training on the GPU.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/joint_xdsafe"
SITE = ROOT / "site"
EPS = 1e-6
CACHE_CONF = 0.05
IOU_NMS = 0.45


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def labelled_detections(model, split, imgsz):
    """(raw score, 1 if the box is a true positive else 0) for every detection >= CACHE_CONF."""
    img_dir = DATA / split / "images"
    lbl_dir = DATA / split / "labels"
    scores, hits = [], []
    paths = sorted(img_dir.glob("*.jpg"))
    for r in model.predict([str(p) for p in paths], imgsz=imgsz, conf=CACHE_CONF, iou=IOU_NMS,
                           device="cpu", verbose=False, stream=True):
        h, w = r.orig_shape
        gt = []
        lf = lbl_dir / (Path(r.path).stem + ".txt")
        if lf.exists():
            for line in lf.read_text().split("\n"):
                if line.strip():
                    c, cx, cy, bw, bh = map(float, line.split()[:5])
                    gt.append((int(c), [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]))
        used = set()
        order = np.argsort(-r.boxes.conf.numpy())
        for k in order:
            c = int(r.boxes.cls[k]); s = float(r.boxes.conf[k]); b = r.boxes.xyxy[k].tolist()
            best, bj = 0.0, -1
            for j, (gc, gb) in enumerate(gt):
                if j in used or gc != c:
                    continue
                v = iou(b, gb)
                if v > best:
                    best, bj = v, j
            hit = best >= 0.5
            if hit:
                used.add(bj)
            scores.append(s); hits.append(1 if hit else 0)
    return np.array(scores), np.array(hits)


def ece(p, y, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[256, 320])
    ap.add_argument("--tag", default="joint")
    a = ap.parse_args()
    w = Path(a.weights).resolve()
    model = YOLO(str(w))
    names = [model.names[i] for i in range(len(model.names))]

    # 1. input size chosen on validation
    val_scores = {}
    for sz in a.sizes:
        m = model.val(data=str(DATA / "data.yaml"), split="val", imgsz=sz, batch=16, device="cpu",
                      plots=False, verbose=False)
        val_scores[sz] = float(m.box.map50)
    imgsz = max(val_scores, key=val_scores.get)
    print("val mAP50 by input size:", val_scores, "-> chosen", imgsz)

    # 2. ONNX export
    out = Path(model.export(format="onnx", imgsz=imgsz, opset=17, simplify=True, dynamic=False, device="cpu"))
    dst = SITE / "model/detector.onnx"
    shutil.copy(out, dst)

    # 3. calibration: fit on val, evaluate on test
    sv, yv = labelled_detections(model, "val", imgsz)
    st, yt = labelled_detections(model, "test", imgsz)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(sv, yv)
    kx, ky = iso.X_thresholds_.tolist(), iso.y_thresholds_.tolist()
    apply = lambda s: np.interp(np.clip(s, kx[0], kx[-1]), kx, ky) * (1 - EPS) + EPS * np.clip(s, 0, 1)
    rep = {
        "weights": str(w.relative_to(ROOT)), "tag": a.tag, "class_names": names, "imgsz": imgsz,
        "val_map50_by_imgsz": val_scores, "onnx_mb": round(dst.stat().st_size / 1e6, 1),
        "calibration": {"method": "isotonic", "fit_split": "val", "eval_split": "test",
                        "n_fit": int(len(sv)), "n_eval": int(len(st)),
                        "ece_before": ece(st, yt), "ece_after": ece(apply(st), yt),
                        "brier_before": float(np.mean((st - yt) ** 2)),
                        "brier_after": float(np.mean((apply(st) - yt) ** 2)),
                        "knots_x": kx, "knots_y": ky},
    }
    examples = {f"{v:.2f}": float(apply(np.array([v]))[0]) for v in (0.15, 0.25, 0.40, 0.60, 0.70, 0.80, 0.90)}
    rep["calibration"]["examples"] = examples
    json.dump(rep, open(ROOT / "reports/demo_model.json", "w"), indent=1)

    # 4. calibration.js with the interface app.js already imports
    c = rep["calibration"]
    js = f"""/**
 * Confidence calibration for the demo detector: an isotonic map fitted on the validation
 * split only, applied to the raw network score so that a stated confidence means what it
 * says. Generated by scripts/build_demo_model.py; do not edit by hand.
 */
const KNOTS_X = {json.dumps(kx)};
const KNOTS_Y = {json.dumps(ky)};
const EPS = {EPS};

export const CALIBRATION = {{
  method: 'isotonic',
  nKnots: KNOTS_X.length,
  eceBefore: {c['ece_before']},
  eceAfter: {c['ece_after']},
  brierBefore: {c['brier_before']},
  brierAfter: {c['brier_after']},
  fitSplit: 'val',
  evalSplit: 'test',
}};

export function calibrate(raw) {{
  const lo = KNOTS_X[0];
  const hi = KNOTS_X[KNOTS_X.length - 1];
  const s = Math.min(Math.max(raw, lo), hi);
  let base;
  if (s <= lo) base = KNOTS_Y[0];
  else if (s >= hi) base = KNOTS_Y[KNOTS_Y.length - 1];
  else {{
    let i = KNOTS_X.length - 1;
    while (i > 0 && KNOTS_X[i - 1] >= s) i--;
    const xa = KNOTS_X[i - 1], xb = KNOTS_X[i], ya = KNOTS_Y[i - 1], yb = KNOTS_Y[i];
    base = xb === xa ? yb : ya + ((yb - ya) * (s - xa)) / (xb - xa);
  }}
  return base * (1 - EPS) + EPS * Math.min(Math.max(raw, 0), 1);
}}

export const CALIBRATION_EXAMPLES = {json.dumps(examples, indent=2)};
"""
    (SITE / "js/calibration.js").write_text(js)
    print(json.dumps({k: v for k, v in rep.items() if k != "calibration"}, indent=1))
    print("calibration ECE test:", round(c["ece_before"], 4), "->", round(c["ece_after"], 4),
          f"(fit on {c['n_fit']} val detections)")


if __name__ == "__main__":
    main()
