"""Run the live demo model on your own stainless photos and produce slide-ready results.

    .venv/bin/python Round2/stainless_test/run_stainless_test.py            # uses Round2/stainless_test/photos/
    .venv/bin/python Round2/stainless_test/run_stainless_test.py --photos some/other/folder --out some/out

Name each photo  <finish>_<label>_<nn>.jpg   e.g.  2B_scratch_01.jpg, BA_clean_03.jpg, No4_pit_02.jpg
  finish: 2B, 2D, BA, No4, HL or other      label: clean, or the defect you can see (scratch, pit, stain, ...)
Anything not labelled "clean" counts as defective.

What it does, the same way the live demo does:
  - converts to greyscale (the model was trained on greyscale line-scan images)
  - cuts a large photo into overlapping square windows so small defects keep their detail
  - runs the model at 320 px, confidence 0.15, and merges the windows with one NMS
  - reports calibrated confidence using the same calibration the site uses
Writes: out/results.json, out/summary.json, out/annotated/*.jpg, out/stainless_sheet.png
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont, ImageOps
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WEIGHTS = ROOT / "models/yolov8n_joint/weights/best.pt"
CALIB = ROOT / "reports/demo_model.json"
IMGSZ, CONF, IOU = 320, 0.15, 0.45
DISPLAY = {"crazing": "Crazing", "inclusion": "Inclusion", "patches": "Patches", "pitted_surface": "Pitted surface",
           "rolled-in_scale": "Rolled-in scale", "scratches": "Scratches"}
BOX = (255, 138, 0)
FINISHES = ("2B", "2D", "BA", "No4", "HL")


def display(name: str) -> str:
    return DISPLAY.get(name, "Unclassified defect")


def calibrator():
    c = json.load(open(CALIB))["calibration"]
    kx, ky = np.array(c["knots_x"]), np.array(c["knots_y"])
    return lambda s: float(np.interp(np.clip(s, kx[0], kx[-1]), kx, ky) * (1 - 1e-6) + 1e-6 * min(max(s, 0), 1))


def windows(w: int, h: int, tile: int, overlap: float = 0.2):
    """Square windows of side min(tile, w, h) covering the frame with the given overlap."""
    side = min(tile, w, h)
    step = max(1, int(side * (1 - overlap)))
    xs = list(range(0, max(w - side, 0) + 1, step)) or [0]
    ys = list(range(0, max(h - side, 0) + 1, step)) or [0]
    if xs[-1] != w - side:
        xs.append(w - side)
    if ys[-1] != h - side:
        ys.append(h - side)
    return [(x, y, side) for y in ys for x in xs]


def parse_name(p: Path):
    parts = p.stem.split("_")
    finish = parts[0] if parts and parts[0] in FINISHES else "other"
    label = parts[1].lower() if len(parts) > 1 else "unknown"
    return finish, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", default=str(HERE / "photos"))
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--tile", type=int, default=800, help="window side in photo pixels")
    a = ap.parse_args()
    photos = sorted(p for p in Path(a.photos).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".heic", ".webp"))
    if not photos:
        raise SystemExit(f"no photos in {a.photos} - see README.md for what to shoot and how to name files")
    out = Path(a.out); (out / "annotated").mkdir(parents=True, exist_ok=True)
    model = YOLO(str(WEIGHTS)); cal = calibrator()
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 28)
    results = []
    heic_dir = out / "_converted"
    for p in photos:
        finish, label = parse_name(p)
        src = p
        if p.suffix.lower() == ".heic":  # iPhone photos: convert with macOS sips, PIL cannot read HEIC
            heic_dir.mkdir(exist_ok=True)
            src = heic_dir / (p.stem + ".jpg")
            subprocess.run(["sips", "-s", "format", "jpeg", str(p), "--out", str(src)], check=True, capture_output=True)
        img = ImageOps.exif_transpose(Image.open(src)).convert("L").convert("RGB")
        w, h = img.size
        boxes, scores, classes = [], [], []
        for x, y, side in windows(w, h, a.tile):
            r = model.predict(img.crop((x, y, x + side, y + side)), imgsz=IMGSZ, conf=CONF, iou=IOU, device="cpu", verbose=False)[0]
            for b, s, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                boxes.append([b[0] + x, b[1] + y, b[2] + x, b[3] + y]); scores.append(s); classes.append(int(c))
        dets = []
        if boxes:
            bt, st, ct = torch.tensor(boxes), torch.tensor(scores), torch.tensor(classes)
            keep = torchvision.ops.batched_nms(bt, st, ct, IOU).tolist()
            dets = [{"type": display(model.names[classes[i]]), "raw": round(scores[i], 3),
                     "confidence": round(cal(scores[i]), 3), "box": [round(v) for v in boxes[i]]} for i in keep]
        dets.sort(key=lambda d: -d["confidence"])
        results.append({"file": p.name, "finish": finish, "label": label, "defective": label != "clean",
                        "flagged": bool(dets), "detections": dets})
        # annotated copy
        im = img.copy(); dr = ImageDraw.Draw(im); lw = max(3, w // 300)
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            dr.rectangle([x1, y1, x2, y2], outline=BOX, width=lw)
            t = f"{d['type']} {int(d['confidence'] * 100)}%"
            tw = dr.textlength(t, font=font)
            dr.rectangle([x1, max(0, y1 - 38), x1 + tw + 14, max(0, y1 - 38) + 36], fill=BOX)
            dr.text((x1 + 7, max(0, y1 - 38) + 3), t, fill=(20, 20, 20), font=font)
        im.thumbnail((1200, 1200)); im.save(out / "annotated" / (p.stem + ".jpg"), quality=88)

    # summary
    clean = [r for r in results if not r["defective"]]; bad = [r for r in results if r["defective"]]
    by_finish = defaultdict(lambda: {"photos": 0, "clean": 0, "clean_flagged": 0, "defective": 0, "defective_flagged": 0})
    for r in results:
        f = by_finish[r["finish"]]; f["photos"] += 1
        if r["defective"]:
            f["defective"] += 1; f["defective_flagged"] += r["flagged"]
        else:
            f["clean"] += 1; f["clean_flagged"] += r["flagged"]
    summary = {"photos": len(results), "finishes": sorted(by_finish),
               "clean": len(clean), "clean_flagged": sum(r["flagged"] for r in clean),
               "defective": len(bad), "defective_flagged": sum(r["flagged"] for r in bad),
               "by_finish": by_finish, "types_reported": Counter(d["type"] for r in results for d in r["detections"]),
               "settings": {"model": str(WEIGHTS.relative_to(ROOT)), "imgsz": IMGSZ, "conf": CONF, "tile_px": a.tile, "greyscale": True}}
    json.dump(results, open(out / "results.json", "w"), indent=1)
    json.dump(summary, open(out / "summary.json", "w"), indent=1)

    # contact sheet for the deck: up to 8 photos, defective first, each captioned
    pick = (bad + clean)[:8]; cell = 360; cols = 4; rows = (len(pick) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * (cell + 44)), "white"); sd = ImageDraw.Draw(sheet)
    cap_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 20)
    for i, r in enumerate(pick):
        im = Image.open(out / "annotated" / (Path(r["file"]).stem + ".jpg")); im = ImageOps.fit(im, (cell - 8, cell - 8))
        x, y = (i % cols) * cell, (i // cols) * (cell + 44)
        sheet.paste(im, (x + 4, y + 4))
        verdict = "FLAGGED" if r["flagged"] else "no alarm"
        sd.text((x + 8, y + cell + 8), f"{r['finish']} · {r['label']} · {verdict}", fill=(20, 58, 90), font=cap_font)
    sheet.save(out / "stainless_sheet.png")
    print(json.dumps({k: summary[k] for k in ("photos", "finishes", "clean", "clean_flagged", "defective", "defective_flagged")}))
    print("types reported:", dict(summary["types_reported"]))


if __name__ == "__main__":
    main()
