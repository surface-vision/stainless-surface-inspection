"""Multi-seed retrain of the demo model (joint NEU-DET + Severstal + clean-strip negatives).

Same recipe as models/yolov8n_joint (warm start from the NEU-DET model with the head widened
to 10 classes, holdout-safe data, 320 px, cosine LR), repeated over independent seeds so the
shipped model is chosen on validation and its test score comes with a seed spread.

    .venv/bin/python scripts/train_demo_seeds.py --seed 1 --epochs 120
"""
import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
WARM = ROOT / "models/yolov8n_joint/warmstart_10cls.pt"
DATA = ROOT / "data/joint_xdsafe/data.yaml"

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--epochs", type=int, default=120)
ap.add_argument("--workers", type=int, default=3)
a = ap.parse_args()

name = f"demo_seed{a.seed}"
t0 = time.time()
model = YOLO(str(WARM))
model.train(data=str(DATA), epochs=a.epochs, patience=40, imgsz=320, batch=32, seed=a.seed,
            deterministic=False, cos_lr=True, close_mosaic=15, cache="ram", workers=a.workers,
            device="mps", project=str(ROOT / "models"), name=name, exist_ok=True, plots=False, verbose=False)
json.dump({"seed": a.seed, "epochs": a.epochs, "wall_s": round(time.time() - t0, 1), "warm_start": str(WARM),
           "data": str(DATA), "best": str(ROOT / "models" / name / "weights/best.pt")},
          open(ROOT / "models" / name / "run.json", "w"), indent=1)
print("done", name)
