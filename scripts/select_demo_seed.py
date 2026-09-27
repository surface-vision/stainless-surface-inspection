"""After the seed runs: choose the demo model on VALIDATION only, then score every seed on test.

    .venv/bin/python scripts/select_demo_seed.py
Writes reports/demo_seed_selection.json. Test scores are reported for every seed (the spread),
but the choice is made before looking at them.
"""
import json
import subprocess
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/joint_xdsafe/data.yaml"
cands = {"joint_seed1337": ROOT / "models/yolov8n_joint/weights/best.pt"}
for s in (1, 2, 3):
    w = ROOT / f"models/demo_seed{s}/weights/best.pt"
    if w.exists():
        cands[f"seed{s}"] = w

val = {}
for tag, w in cands.items():
    m = YOLO(str(w))
    val[tag] = {sz: float(m.val(data=str(DATA), split="val", imgsz=sz, batch=16, device="cpu",
                                 plots=False, verbose=False).box.map50) for sz in (256, 320)}
best_tag, best_sz = max(((t, sz) for t in val for sz in val[t]), key=lambda k: val[k[0]][k[1]])

test = {}
for tag, w in cands.items():
    sz = max(val[tag], key=val[tag].get)
    subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/eval_demo_model.py"),
                    "--weights", str(w), "--imgsz", str(sz), "--tag", tag], check=True, capture_output=True)
    test[tag] = json.load(open(ROOT / f"reports/demo_eval_{tag}.json"))

out = {"chosen_on_val": {"tag": best_tag, "imgsz": best_sz, "weights": str(cands[best_tag].relative_to(ROOT))},
       "val_map50": val, "test": test}
json.dump(out, open(ROOT / "reports/demo_seed_selection.json", "w"), indent=1)
print(json.dumps(out["chosen_on_val"]))
for t, r in test.items():
    print(t, "val", {k: round(v, 4) for k, v in val[t].items()},
          "| test neu", round(r["neu_map50"], 4), "mill", round(r["mill_map50"], 4),
          "right", r["right_type"], "clean", r["clean_flagged"], "defect", r["defect_flagged"])
