"""Held-out scorecard for a demo candidate, on the untouched test split of data/joint_xdsafe.

    .venv/bin/python scripts/eval_demo_model.py --weights models/yolov8n_joint/weights/best.pt --imgsz 320 --tag joint

Every candidate is scored on the same images with the same rules, so the numbers compare:
  neu_map50        NEU-DET test (180 lab images), box accuracy
  mill_map50       Severstal test crops with defects, box accuracy
  right_type       NEU-DET test images whose strongest box names the right defect type
  clean_flagged    clean mill-strip test crops (300) with any box at the demo setting (0.15)
  defect_flagged   defective mill-strip test crops (288) with any box at 0.15
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data/joint_xdsafe"
CONF = 0.15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    m = YOLO(a.weights)
    res = {"tag": a.tag, "weights": str(Path(a.weights).resolve().relative_to(ROOT)), "imgsz": a.imgsz, "conf": CONF}

    # the eval yamls point their "val" key at the untouched test images
    for key, yaml in (("neu_map50", "eval_neu_test.yaml"), ("mill_map50", "eval_sev_test.yaml")):
        if len(m.names) != 10:
            res[key] = None  # the old 6-class model cannot be scored on the 10-class files
            continue
        v = m.val(data=str(D / yaml), split="val", imgsz=a.imgsz, batch=16, device="cpu", plots=False, verbose=False)
        res[key] = float(v.box.map50)

    right = total = 0
    neu = sorted((D / "test/images").glob("*.jpg"))
    neu = [p for p in neu if not p.name.startswith("sev_")]
    for r in m.predict([str(p) for p in neu], imgsz=a.imgsz, conf=0.05, iou=0.45, device="cpu", verbose=False, stream=True):
        total += 1
        true_cls = Path(r.path).stem.rsplit("_", 1)[0]
        if len(r.boxes):
            right += m.names[int(r.boxes.cls[int(r.boxes.conf.argmax())])] == true_cls
    res["right_type"] = [right, total]

    clean = defect = clean_hit = defect_hit = 0
    sev = sorted((D / "test/images").glob("sev_*.jpg"))
    for r in m.predict([str(p) for p in sev], imgsz=a.imgsz, conf=CONF, iou=0.45, device="cpu", verbose=False, stream=True):
        lf = D / "test/labels" / (Path(r.path).stem + ".txt")
        labelled = lf.exists() and lf.read_text().strip()
        if labelled:
            defect += 1; defect_hit += len(r.boxes) > 0
        else:
            clean += 1; clean_hit += len(r.boxes) > 0
    res["clean_flagged"] = [clean_hit, clean]
    res["defect_flagged"] = [defect_hit, defect]
    json.dump(res, open(ROOT / f"reports/demo_eval_{a.tag}.json", "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
