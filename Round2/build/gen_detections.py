"""Render the defect gallery: our joint model's real detections on held-out NEU-DET test images.

Picks, per defect class, the test frame where every box is the correct class and mean
confidence is highest, then draws it deck-ready. Run from the repo root:
    .venv/bin/python Round2/build/gen_detections.py
"""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "img"
NAME = {"crazing": "Crazing", "inclusion": "Inclusion", "patches": "Patches", "pitted_surface": "Pitted surface",
        "rolled-in_scale": "Rolled-in scale", "scratches": "Scratches"}
BOX = (255, 138, 0)
S = 480

m = YOLO(str(ROOT / "models/yolov8n_joint/weights/best.pt"))
best = {}
imgs = sorted((ROOT / "data/neu-det/test/images").glob("*.jpg"))
for r in m.predict([str(p) for p in imgs], imgsz=256, conf=0.25, iou=0.45, verbose=False, stream=True):
    p = Path(r.path)
    true_cls = p.stem.rsplit("_", 1)[0]
    boxes = [(m.names[int(c)], float(s), [float(v) for v in b]) for c, s, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy)]
    correct = [b for b in boxes if b[0] == true_cls]
    if not correct or len(correct) != len(boxes) or len(correct) > 4:
        continue
    score = sum(s for _, s, _ in correct) / len(correct)
    if score > best.get(true_cls, (0,))[0]:
        best[true_cls] = (score, p.name, correct)
json.dump({k: {"file": v[1], "boxes": v[2]} for k, v in best.items()}, open(OUT / "detections.json", "w"), indent=1)

font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 26)
for cls, (_, fname, boxes) in best.items():
    im = Image.open(ROOT / "data/neu-det/test/images" / fname).convert("RGB")
    k = S / im.width
    im = im.resize((S, S), Image.LANCZOS)
    dr = ImageDraw.Draw(im)
    for _, conf, (x1, y1, x2, y2) in boxes:
        x1, y1, x2, y2 = max(x1 * k, 3), max(y1 * k, 3), min(x2 * k, S - 4), min(y2 * k, S - 4)
        dr.rectangle([x1 - 1, y1 - 1, x2 + 1, y2 + 1], outline=(255, 255, 255), width=2)
        dr.rectangle([x1 + 1, y1 + 1, x2 - 1, y2 - 1], outline=BOX, width=5)
        lab = f"{NAME[cls]}  {conf:.2f}"
        tw = dr.textlength(lab, font=font)
        ly = y1 - 38 if y1 > 42 else y1 + 6
        lx = min(max(x1, 4), S - tw - 20)
        dr.rectangle([lx, ly, lx + tw + 16, ly + 34], fill=BOX)
        dr.text((lx + 8, ly + 3), lab, fill=(20, 20, 20), font=font)
    im.save(OUT / f"det_{cls}.png")
print({k: (v[1], round(v[0], 2)) for k, v in best.items()})
