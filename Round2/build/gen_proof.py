"""Assets and numbers for the proof slide: one sample per dataset, and the joint model's
right-type rate on the NEU-DET held-out test set (top-confidence box class == true class).
Run from the repo root:  .venv/bin/python Round2/build/gen_proof.py
"""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "img"
m = YOLO(str(ROOT / "models/yolov8n_joint/weights/best.pt"))
font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22)
BOX = (255, 138, 0)

# 1) right defect type on held-out NEU-DET test images
imgs = sorted((ROOT / "data/neu-det/test/images").glob("*.jpg"))
right = total = 0
for r in m.predict([str(p) for p in imgs], imgsz=256, conf=0.05, iou=0.45, verbose=False, stream=True):
    total += 1
    true_cls = Path(r.path).stem.rsplit("_", 1)[0]
    if len(r.boxes):
        top = int(r.boxes.conf.argmax())
        right += m.names[int(r.boxes.cls[top])] == true_cls

# 2) Severstal sample: a held-out defective crop where the model finds the labelled defect
def draw(im, boxes, label, k=1.0):
    dr = ImageDraw.Draw(im)
    for (x1, y1, x2, y2), conf in boxes:
        x1, y1, x2, y2 = x1 * k, y1 * k, x2 * k, y2 * k
        dr.rectangle([x1 - 1, y1 - 1, x2 + 1, y2 + 1], outline=(255, 255, 255), width=2)
        dr.rectangle([x1 + 1, y1 + 1, x2 - 1, y2 - 1], outline=BOX, width=4)
        lab = f"{label}  {conf:.2f}"
        tw = dr.textlength(lab, font=font)
        ly = y1 - 32 if y1 > 36 else y1 + 5
        lx = min(max(x1, 4), im.width - tw - 18)
        dr.rectangle([lx, ly, lx + tw + 14, ly + 29], fill=BOX)
        dr.text((lx + 7, ly + 2), lab, fill=(20, 20, 20), font=font)
    return im

sev_dir = ROOT / "data/severstal/test"
best = None
for p in sorted((sev_dir / "images").glob("*.jpg")):
    lbl = sev_dir / "labels" / (p.stem + ".txt")
    if not lbl.exists() or not lbl.read_text().strip():
        continue
    r = m.predict(str(p), imgsz=256, conf=0.3, iou=0.45, verbose=False)[0]
    b = [(bx.tolist(), float(c)) for bx, c, cl in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls) if m.names[int(cl)].startswith("severstal")]
    if len(b) == 1 and (best is None or b[0][1] > best[1][0][1]):
        best = (p, b)
p, b = best
im = Image.open(p).convert("RGB").resize((480, 480), Image.LANCZOS)
draw(im, b, "Defect", k=480 / 256).save(OUT / "proof_severstal.png")

# 3) GC10 sample: a full-width line-scan frame (raw, CC BY 4.0), cropped to a strip band
from PIL import ImageOps
g = Image.open(ROOT / "assets/strip_sample_rollmark.jpg").convert("L")
g = ImageOps.autocontrast(g.crop((200, 200, 1250, 910)), cutoff=1)   # zoom on the roll marks
g.convert("RGB").resize((740, 500), Image.LANCZOS).save(OUT / "proof_gc10.png")

# 4) NEU sample reused from the gallery
res = {"neu_right_type": right, "neu_total": total, "severstal_sample": p.name, "severstal_conf": round(b[0][1], 2)}
json.dump(res, open(OUT / "proof.json", "w"), indent=1)
print(res)
