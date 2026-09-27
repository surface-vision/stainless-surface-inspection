"""Fine-tune a YOLO detector on the NEU-DET steel surface defect split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "neu-det" / "data.yaml"


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--name", default=None)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    device = pick_device(args.device)
    name = args.name or f"{Path(args.model).stem}_e{args.epochs}"
    print(f"[train] model={args.model} device={device} epochs={args.epochs}")

    model = YOLO(args.model)
    model.train(
        data=str(DATA),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        project=str(ROOT / "models"),
        name=name,
        exist_ok=True,
        patience=args.patience,
        seed=1337,
        deterministic=False,  # MPS deterministic kernels are ~5x slower; seed alone is adequate
        pretrained=True,
        optimizer="auto",
        cos_lr=True,
        # Steel strip texture: flips and mild scale/translate are label-preserving,
        # colour jitter is not informative on near-greyscale mill imagery.
        fliplr=0.5,
        flipud=0.5,
        degrees=10.0,
        scale=0.4,
        translate=0.1,
        shear=2.0,
        mosaic=1.0,
        close_mosaic=15,
        mixup=0.1,
        hsv_h=0.0,
        hsv_s=0.3,
        hsv_v=0.4,
        plots=True,
        val=True,
    )

    # Held-out test set: the split the model never saw during selection.
    metrics = model.val(data=str(DATA), split="test", device=device, plots=True,
                        project=str(ROOT / "models"), name=f"{name}_test", exist_ok=True)

    summary = {
        "model": args.model,
        "name": name,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "device": device,
        "test": {
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "per_class_mAP50": {
                metrics.names[c]: float(metrics.box.ap50[i])
                for i, c in enumerate(metrics.ap_class_index)
            },
        },
    }
    out = ROOT / "reports" / f"train_summary_{name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["test"], indent=2))
    print(f"[train] wrote {out}")


if __name__ == "__main__":
    main()
