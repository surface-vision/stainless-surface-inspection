"""Joint NEU-DET + Severstal training: the fix for the cross-domain false-alarm gap.

WHY THIS MODULE EXISTS
----------------------
The shipped detector (``models/yolov8n_neudet/weights/best.pt``) was fitted on
NEU-DET, in which *every* image contains a defect. It therefore never had to
learn what defect-free steel looks like, and `src/cross_domain.py` measures the
consequence on real Severstal strip: 93.7% of genuinely defect-free frames raise
an alarm at the shipped conf 0.15, a verified-clean coil is put on HOLD by the
project's own disposition rules, and clean/defective separation is ROC AUC 0.608
[0.576, 0.640] against 0.968 in domain.

THE TRAP THIS MODULE AVOIDS
---------------------------
The auditor already measured the obvious repair: warm-start on NEU-DET plus
Severstal *negatives only*. Clean-frame false alarms collapse (94% -> 5.8%) at
no in-domain cost, and it is worthless -- Severstal defect recall collapses with
it (0.057) and AUC does not move. Importing negatives from a domain that
contributes no positives teaches a *domain* classifier: "this texture is not
NEU-DET, so it is clean". The only way the model can learn a *defect* classifier
on Severstal is to see Severstal defects and Severstal clean steel in the same
optimisation. Hence joint training on `data/joint`, which carries both.

THE HOLD-OUT RULE, AND WHY THIS MODULE FILTERS THE DATASET IT WAS GIVEN
----------------------------------------------------------------------
`src/cross_domain.py` splits every Severstal source frame by a deterministic
hash (``is_holdout(stem, 0.5)``) and scores only the holdout half, reserving the
other half as legal training material. `src/prepare_severstal.py` does not know
about that rule: 1133 of the 2182 source frames behind ``data/joint/train`` --
including 109 clean and 24 masked-defective frames that the harness actually
scores -- fall on the scored side. Training on them would inflate every
cross-domain number by construction.

So this module never trains on ``data/joint`` directly. It derives
``data/joint_xdsafe`` -- symlinks only, no pixels copied -- by dropping from
*train and val* every Severstal crop whose source frame satisfies
``is_holdout(stem, 0.5)``. The rule is the harness's own, applied without
reference to which frames the auditor happened to download, so the record stays
honest even if someone later re-runs `cross_domain.py` against a larger pull of
the same dataset. It costs 51% of the Severstal training crops (4794 -> 2329)
and that cost is reported rather than hidden.

The joint *test* split is left whole: it is never trained on under either rule.

WARM START: THE HEAD IS TRANSPLANTED, NOT DISCARDED
---------------------------------------------------
The shipped checkpoint has a 6-class detection head; the joint dataset needs 10.
Ultralytics' default transfer (`intersect_dicts`) drops every tensor whose shape
changed, which throws away the three learned 1x1 classification convolutions and
re-randomises all ten classes. That is a needless in-domain loss: only the four
new output channels are actually unknown. `build_warmstart_checkpoint` builds the
10-class model, runs the standard transfer for backbone and neck, then copies the
learned ``cv3[i][-1]`` weight and bias into output channels 0..5 and leaves 6..9
at Ultralytics' own ``bias_init`` prior. Box regression (``cv2``) is
class-independent and transfers unchanged.

USAGE
-----
    python -m src.train_joint verify
    python -m src.train_joint build
    python -m src.train_joint warmstart
    python -m src.train_joint train --epochs 60
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch
import yaml

try:  # works as `python src/train_joint.py` and as `from src import train_joint`
    from .cross_domain import HOLDOUT_SEED, is_holdout
except ImportError:  # pragma: no cover - direct-script fallback
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.cross_domain import HOLDOUT_SEED, is_holdout

ROOT = Path(__file__).resolve().parent.parent
JOINT_DIR = ROOT / "data" / "joint"
NEU_DIR = ROOT / "data" / "neu-det"
SEVERSTAL_CACHE = ROOT / "data" / ".severstal_cache"
SAFE_DIR = ROOT / "data" / "joint_xdsafe"
BASE_WEIGHTS = ROOT / "models" / "yolov8n_neudet" / "weights" / "best.pt"
RUN_DIR = ROOT / "models" / "yolov8n_joint"
WARMSTART_PT = RUN_DIR / "warmstart_10cls.pt"
REPORTS = ROOT / "reports"

SPLITS: tuple[str, ...] = ("train", "val", "test")
NEU_NAMES: tuple[str, ...] = (
    "crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches",
)
SEVERSTAL_NAMES: tuple[str, ...] = ("severstal_1", "severstal_2", "severstal_3", "severstal_4")
JOINT_NAMES: tuple[str, ...] = NEU_NAMES + SEVERSTAL_NAMES

# `sev_<source frame id>_x<left edge>.jpg` -- the tiler's naming, and the only
# link back from a crop to the frame the holdout rule is defined on.
SEV_STEM = re.compile(r"^sev_([0-9a-fA-F]+)_x(\d+)$")

HOLDOUT_FRAC = 0.5

# Half of the last decimal place YOLO label files are written to.
_LABEL_EPS = 1e-6

# The 200x200 crops the deployed system inspects come out of these tiles, so the
# training geometry is the deployment geometry at 1.28x, exactly as for NEU-DET.
TILE_PX = 256


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def source_frame_id(stem: str) -> str | None:
    """Severstal source frame id behind a crop stem, or None for a NEU-DET image."""
    match = SEV_STEM.match(stem)
    return match.group(1) if match else None


def crop_offset(stem: str) -> int | None:
    match = SEV_STEM.match(stem)
    return int(match.group(2)) if match else None


def is_severstal(stem: str) -> bool:
    return SEV_STEM.match(stem) is not None


def read_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    """Parse a YOLO label file; a missing or blank file is a background image."""
    if not path.is_file():
        return []
    rows: list[tuple[int, float, float, float, float]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        rows.append((int(parts[0]), *(float(v) for v in parts[1:5])))
    return rows


def images_in(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------


@dataclass
class Check:
    """One named assertion about the dataset, with the evidence behind it."""

    name: str
    ok: bool
    detail: str

    def line(self) -> str:
        return f"[{'PASS' if self.ok else 'FAIL'}] {self.name}: {self.detail}"


def verify_dataset(
    joint_dir: Path = JOINT_DIR,
    neu_dir: Path = NEU_DIR,
    cache_dir: Path = SEVERSTAL_CACHE,
    samples_json: Path | None = None,
    *,
    fidelity_sample: int = 120,
    seed: int = 1337,
) -> tuple[list[Check], dict[str, Any]]:
    """Independently re-derive every claim `data/joint/manifest.json` makes.

    Nothing here trusts the manifest: counts are recounted from the files, the
    NEU-DET half is compared byte-for-byte against `data/neu-det`, background
    labels are re-read, splits are re-hashed, and a random sample of Severstal
    crops is checked to be a byte-exact sub-rectangle of its cached source frame.
    """
    checks: list[Check] = []
    facts: dict[str, Any] = {}
    manifest = json.loads((joint_dir / "manifest.json").read_text())
    cfg = yaml.safe_load((joint_dir / "data.yaml").read_text())

    # --- 1. class vocabulary and index order -------------------------------
    names = [cfg["names"][i] for i in range(cfg["nc"])]
    ok = names == list(JOINT_NAMES) and cfg["nc"] == 10
    checks.append(Check(
        "class vocabulary",
        ok,
        f"nc={cfg['nc']} names={names}",
    ))
    neu_yaml = yaml.safe_load((neu_dir / "data.yaml").read_text())
    neu_names = [neu_yaml["names"][i] for i in range(neu_yaml["nc"])]
    checks.append(Check(
        "NEU-DET indices 0-5 unchanged",
        names[:6] == neu_names,
        f"joint[0:6]={names[:6]} == neu-det={neu_names}",
    ))

    # --- 2. recount every split --------------------------------------------
    per_split: dict[str, dict[str, Any]] = {}
    stems_by_split: dict[str, set[str]] = {}
    hashes_by_split: dict[str, dict[str, str]] = {}
    frames_by_split: dict[str, set[str]] = {}
    for split in SPLITS:
        imgs = images_in(joint_dir / split / "images")
        stems_by_split[split] = {p.stem for p in imgs}
        instances: Counter[int] = Counter()
        background = 0
        labelled = 0
        bad_geometry: list[str] = []
        bad_class: list[str] = []
        neu_with_sev_class: list[str] = []
        sev_with_neu_class: list[str] = []
        for img in imgs:
            rows = read_label(joint_dir / split / "labels" / f"{img.stem}.txt")
            if not rows:
                background += 1
                continue
            labelled += 1
            sev = is_severstal(img.stem)
            for cls, cx, cy, bw, bh in rows:
                instances[cls] += 1
                if not (0 <= cls < 10):
                    bad_class.append(img.stem)
                elif sev and cls < 6:
                    sev_with_neu_class.append(img.stem)
                elif not sev and cls >= 6:
                    neu_with_sev_class.append(img.stem)
                # Labels are written to 6 decimal places, so a box that ends
                # exactly on the tile edge can round to 1.0000005. The tolerance
                # is one unit in the last written place -- 2.6e-4 px on a 256 px tile
                # -- not a licence for genuinely out-of-frame boxes.
                if not (0.0 < bw <= 1.0 + _LABEL_EPS and 0.0 < bh <= 1.0 + _LABEL_EPS):
                    bad_geometry.append(img.stem)
                elif not (-_LABEL_EPS <= cx - bw / 2 and cx + bw / 2 <= 1.0 + _LABEL_EPS):
                    bad_geometry.append(img.stem)
                elif not (-_LABEL_EPS <= cy - bh / 2 and cy + bh / 2 <= 1.0 + _LABEL_EPS):
                    bad_geometry.append(img.stem)
        frames_by_split[split] = {
            fid for fid in (source_frame_id(s) for s in stems_by_split[split]) if fid
        }
        per_split[split] = {
            "images": len(imgs),
            "labelled": labelled,
            "background": background,
            "instances": dict(sorted(instances.items())),
            "total_instances": sum(instances.values()),
            "neu_images": sum(1 for s in stems_by_split[split] if not is_severstal(s)),
            "sev_images": sum(1 for s in stems_by_split[split] if is_severstal(s)),
            "source_frames": len(frames_by_split[split]),
            "bad_geometry": sorted(set(bad_geometry))[:5],
            "bad_class": sorted(set(bad_class))[:5],
            "class_domain_violations": sorted(set(sev_with_neu_class + neu_with_sev_class))[:5],
        }
        claimed = manifest["joint"]["counts"][split]
        agree = (
            per_split[split]["images"] == claimed["images"]
            and per_split[split]["background"] == claimed["background_images"]
            and per_split[split]["labelled"] == claimed["labelled_images"]
            and per_split[split]["total_instances"] == claimed["total_instances"]
            and {str(k): v for k, v in per_split[split]["instances"].items()} == claimed["instances"]
        )
        checks.append(Check(
            f"counts recounted, {split}",
            agree,
            f"{per_split[split]['images']} images "
            f"({per_split[split]['labelled']} labelled / {per_split[split]['background']} background), "
            f"{per_split[split]['total_instances']} instances -- manifest agrees: {agree}",
        ))
    facts["per_split"] = per_split

    checks.append(Check(
        "label geometry in range",
        all(not per_split[s]["bad_geometry"] for s in SPLITS),
        "every box has 0 < w,h <= 1 and lies inside the tile"
        if all(not per_split[s]["bad_geometry"] for s in SPLITS)
        else str({s: per_split[s]["bad_geometry"] for s in SPLITS}),
    ))
    checks.append(Check(
        "class ids stay in their own domain",
        all(not per_split[s]["class_domain_violations"] for s in SPLITS)
        and all(not per_split[s]["bad_class"] for s in SPLITS),
        "NEU-DET images carry only classes 0-5, Severstal crops only 6-9",
    ))

    # --- 3. NEU-DET half is byte-identical to data/neu-det -----------------
    neu_mismatch: list[str] = []
    neu_missing: list[str] = []
    neu_counts: dict[str, int] = {}
    for split in SPLITS:
        src_imgs = {p.stem: p for p in images_in(neu_dir / split / "images")}
        joint_imgs = {
            s: joint_dir / split / "images" / f"{s}.jpg"
            for s in stems_by_split[split] if not is_severstal(s)
        }
        neu_counts[split] = len(joint_imgs)
        if set(src_imgs) != set(joint_imgs):
            neu_missing.append(
                f"{split}: {len(set(joint_imgs) ^ set(src_imgs))} stems differ"
            )
        for stem, path in joint_imgs.items():
            src = src_imgs.get(stem)
            if src is None or sha256_file(src) != sha256_file(path):
                neu_mismatch.append(f"{split}/{stem}")
            src_lab = neu_dir / split / "labels" / f"{stem}.txt"
            if read_label(src_lab) != read_label(joint_dir / split / "labels" / f"{stem}.txt"):
                neu_mismatch.append(f"{split}/{stem}:label")
    checks.append(Check(
        "NEU-DET images/labels byte-identical and in the same split",
        not neu_mismatch and not neu_missing,
        f"{neu_counts} images compared; "
        f"{len(neu_mismatch)} mismatches, {len(neu_missing)} split differences"
        + (f" e.g. {neu_mismatch[:3]}" if neu_mismatch else ""),
    ))
    facts["neu_images_per_split"] = neu_counts

    # --- 4. content-hash disjointness ---------------------------------------
    for split in SPLITS:
        hashes_by_split[split] = {
            p.stem: sha256_file(p) for p in images_in(joint_dir / split / "images")
        }
    collisions: dict[str, int] = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = set(hashes_by_split[a].values()) & set(hashes_by_split[b].values())
        collisions[f"{a}^{b}"] = len(shared)
    dupes = {s: len(hashes_by_split[s]) - len(set(hashes_by_split[s].values())) for s in SPLITS}
    checks.append(Check(
        "splits are content-hash disjoint",
        all(v == 0 for v in collisions.values()),
        f"shared image hashes {collisions}; intra-split duplicate images {dupes}",
    ))
    facts["hash_collisions"] = collisions
    facts["intra_split_duplicate_images"] = dupes

    frame_collisions = {
        f"{a}^{b}": len(frames_by_split[a] & frames_by_split[b])
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    checks.append(Check(
        "no Severstal source frame straddles a split",
        all(v == 0 for v in frame_collisions.values()),
        f"shared source frames {frame_collisions}",
    ))
    facts["source_frame_collisions"] = frame_collisions

    # --- 5. background crops are genuinely empty ----------------------------
    clean_flags: dict[str, bool] = {}
    if samples_json and Path(samples_json).is_file():
        payload = json.loads(Path(samples_json).read_text())["samples"]
        for sample in payload:
            image_id = str(sample.get("image_id", ""))
            if image_id:
                clean_flags[Path(image_id).stem] = str(sample.get("has_defect")) == "True"
    facts["has_defect_lookup"] = len(clean_flags)
    if clean_flags:
        wrong_bg: list[str] = []
        wrong_fg: list[str] = []
        unknown = 0
        for split in SPLITS:
            for stem in stems_by_split[split]:
                fid = source_frame_id(stem)
                if fid is None:
                    continue
                flag = clean_flags.get(fid)
                if flag is None:
                    unknown += 1
                    continue
                labelled = bool(read_label(joint_dir / split / "labels" / f"{stem}.txt"))
                if not labelled and flag:
                    wrong_bg.append(f"{split}/{stem}")
                if labelled and not flag:
                    wrong_fg.append(f"{split}/{stem}")
        checks.append(Check(
            "background crops come only from has_defect=False source frames",
            not wrong_bg and not wrong_fg,
            f"{len(wrong_bg)} background crops from defective frames, "
            f"{len(wrong_fg)} labelled crops from clean frames, {unknown} frames not in samples.json",
        ))
    else:
        checks.append(Check(
            "background crops come only from has_defect=False source frames",
            False,
            "samples.json not supplied - cannot verify the source flag "
            "(pass --samples-json to check it)",
        ))

    # --- 6. crops are byte-exact sub-rectangles of the cached source --------
    rng = np.random.default_rng(seed)
    sev_all = [
        (split, stem)
        for split in SPLITS
        for stem in sorted(stems_by_split[split])
        if is_severstal(stem)
    ]
    take = min(fidelity_sample, len(sev_all))
    picks = [sev_all[i] for i in rng.choice(len(sev_all), size=take, replace=False)]
    worst = 0
    checked = 0
    missing_source = 0
    for split, stem in picks:
        fid, x0 = source_frame_id(stem), crop_offset(stem)
        src = cache_dir / f"{fid}.jpg"
        if not src.is_file():
            missing_source += 1
            continue
        frame = cv2.imread(str(src), cv2.IMREAD_COLOR)
        crop = cv2.imread(str(joint_dir / split / "images" / f"{stem}.jpg"), cv2.IMREAD_COLOR)
        if frame is None or crop is None:
            missing_source += 1
            continue
        ref = frame[:, x0 : x0 + TILE_PX]
        worst = max(worst, int(np.abs(ref.astype(np.int32) - crop.astype(np.int32)).max()))
        checked += 1
    checks.append(Check(
        "crops are the source frame's own pixels",
        checked > 0 and worst <= 12,
        f"{checked} random crops re-cut from data/.severstal_cache; "
        f"worst absolute channel delta {worst} (JPEG re-encode only, no resampling); "
        f"{missing_source} sources unavailable",
    ))
    facts["fidelity_worst_abs_delta"] = worst
    facts["fidelity_crops_checked"] = checked

    # --- 7. the cross-domain holdout rule ------------------------------------
    holdout_stats: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        held = {f for f in frames_by_split[split] if is_holdout(f, HOLDOUT_FRAC)}
        crops_held = sum(
            1 for s in stems_by_split[split]
            if (fid := source_frame_id(s)) and is_holdout(fid, HOLDOUT_FRAC)
        )
        holdout_stats[split] = {
            "source_frames": len(frames_by_split[split]),
            "source_frames_on_scored_side": len(held),
            "crops_on_scored_side": crops_held,
        }
    facts["cross_domain_holdout"] = holdout_stats
    leaks = holdout_stats["train"]["crops_on_scored_side"] + holdout_stats["val"]["crops_on_scored_side"]
    checks.append(Check(
        "data/joint respects the cross_domain holdout rule",
        leaks == 0,
        f"{leaks} train+val crops come from frames cross_domain.py scores "
        f"(seed {HOLDOUT_SEED!r}, frac {HOLDOUT_FRAC}) -- "
        "this is why `build` derives data/joint_xdsafe instead of training on data/joint",
    ))

    return checks, facts


# ---------------------------------------------------------------------------
# holdout-safe dataset
# ---------------------------------------------------------------------------


def build_holdout_safe(
    joint_dir: Path = JOINT_DIR,
    out_dir: Path = SAFE_DIR,
    *,
    holdout_frac: float = HOLDOUT_FRAC,
    filter_splits: Sequence[str] = ("train", "val"),
) -> dict[str, Any]:
    """Derive a training view of `data/joint` that the harness may never score.

    Symlinks only: no pixels are copied and `data/joint` is never written to.
    Every Severstal crop whose source frame falls on the harness's scored side is
    dropped from `filter_splits`; `test` is copied whole because nothing trains
    on it. Two extra single-split views (`eval_neu_test`, `eval_sev_test`) let a
    10-class checkpoint be scored on the 180 NEU-DET test frames alone -- the
    only way to put its in-domain mAP50 next to the shipped 0.7524.
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    stats: dict[str, Any] = {"splits": {}, "holdout_frac": holdout_frac, "holdout_seed": HOLDOUT_SEED}

    def link_set(dest: Path, pairs: Sequence[tuple[Path, Path | None]]) -> None:
        (dest / "images").mkdir(parents=True, exist_ok=True)
        (dest / "labels").mkdir(parents=True, exist_ok=True)
        for img, lab in pairs:
            (dest / "images" / img.name).symlink_to(img)
            target = dest / "labels" / f"{img.stem}.txt"
            if lab is not None and lab.is_file():
                target.symlink_to(lab)
            else:  # background image: an explicit empty label file, not an absent one
                target.write_text("")

    for split in SPLITS:
        imgs = images_in(joint_dir / split / "images")
        kept: list[tuple[Path, Path | None]] = []
        dropped = 0
        dropped_frames: set[str] = set()
        for img in imgs:
            fid = source_frame_id(img.stem)
            if split in filter_splits and fid is not None and is_holdout(fid, holdout_frac):
                dropped += 1
                dropped_frames.add(fid)
                continue
            kept.append((img, joint_dir / split / "labels" / f"{img.stem}.txt"))
        link_set(out_dir / split, kept)
        n_neu = sum(1 for img, _ in kept if not is_severstal(img.stem))
        n_pos = sum(1 for img, lab in kept if lab is not None and lab.is_file() and lab.stat().st_size > 0)
        stats["splits"][split] = {
            "filtered": split in filter_splits,
            "images_in": len(imgs),
            "images_kept": len(kept),
            "images_dropped": dropped,
            "source_frames_dropped": len(dropped_frames),
            "neu_kept": n_neu,
            "severstal_kept": len(kept) - n_neu,
            "labelled_kept": n_pos,
            "background_kept": len(kept) - n_pos,
        }

    # Per-domain evaluation views of the untouched test split.
    for view, predicate in (
        ("eval_neu_test", lambda s: not is_severstal(s)),
        ("eval_sev_test", is_severstal),
    ):
        pairs = [
            (img, joint_dir / "test" / "labels" / f"{img.stem}.txt")
            for img in images_in(joint_dir / "test" / "images")
            if predicate(img.stem)
        ]
        link_set(out_dir / view, pairs)
        stats["splits"][view] = {"filtered": False, "images_kept": len(pairs)}

    names_block = {i: n for i, n in enumerate(JOINT_NAMES)}
    (out_dir / "data.yaml").write_text(
        "# Holdout-safe view of data/joint, written by src/train_joint.py.\n"
        "# train/val exclude every Severstal crop whose source frame is scored by\n"
        "# src/cross_domain.py (is_holdout(stem, 0.5)); test is the untouched split.\n"
        + yaml.safe_dump(
            {
                "path": str(out_dir),
                "train": "train/images",
                "val": "val/images",
                "test": "test/images",
                "nc": 10,
                "names": names_block,
            },
            sort_keys=False,
        )
    )
    for view in ("eval_neu_test", "eval_sev_test"):
        (out_dir / f"{view}.yaml").write_text(
            f"# Single-split view for scoring a 10-class checkpoint on {view}.\n"
            + yaml.safe_dump(
                {
                    "path": str(out_dir),
                    "train": f"{view}/images",
                    "val": f"{view}/images",
                    "nc": 10,
                    "names": names_block,
                },
                sort_keys=False,
            )
        )
    (out_dir / "manifest.json").write_text(json.dumps(stats, indent=2))
    return stats


# ---------------------------------------------------------------------------
# warm start
# ---------------------------------------------------------------------------


def build_warmstart_checkpoint(
    base: Path = BASE_WEIGHTS,
    out: Path = WARMSTART_PT,
    names: Sequence[str] = JOINT_NAMES,
) -> dict[str, Any]:
    """Widen the shipped 6-class checkpoint to a 10-class one, keeping the head.

    Returns the transfer statistics so the report can state exactly how much of
    the shipped model survived, instead of asserting "warm-started".
    """
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.torch_utils import intersect_dicts

    ckpt = torch.load(str(base), map_location="cpu", weights_only=False)
    old = (ckpt.get("ema") or ckpt["model"]).float()
    old.eval()
    old_nc = int(old.nc)
    new_nc = len(names)
    if new_nc < old_nc:
        raise ValueError(f"Cannot narrow a head: {old_nc} -> {new_nc}")

    cfg = dict(old.yaml)
    cfg["nc"] = new_nc
    new = DetectionModel(cfg, ch=cfg.get("ch", 3), nc=new_nc, verbose=False)

    old_state = old.state_dict()
    csd = intersect_dicts(old_state, new.state_dict())
    new.load_state_dict(csd, strict=False)

    # The three classification 1x1 convolutions are the only shape-changed
    # tensors worth rescuing: channels 0..old_nc-1 mean exactly what they meant
    # before, and the rest keep Ultralytics' own bias_init prior for new_nc.
    head_old, head_new = old.model[-1], new.model[-1]
    transplanted = []
    with torch.no_grad():
        for i, (seq_old, seq_new) in enumerate(zip(head_old.cv3, head_new.cv3)):
            conv_old, conv_new = seq_old[-1], seq_new[-1]
            conv_new.weight[:old_nc].copy_(conv_old.weight)
            conv_new.bias[:old_nc].copy_(conv_old.bias)
            transplanted.append(f"cv3.{i}.{len(seq_new) - 1}")

    new.nc = new_nc
    new.names = {i: n for i, n in enumerate(names)}
    new.args = getattr(old, "args", None)
    new.yaml = cfg

    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(ckpt)
    payload.update(
        {
            "model": new.half(),
            "ema": None,
            "updates": None,
            "optimizer": None,
            "scaler": None,
            "epoch": -1,
            "best_fitness": None,
            "train_args": dict(ckpt.get("train_args") or {}),
        }
    )
    torch.save(payload, str(out))

    total_new = len(new.state_dict())
    stats = {
        "base": str(base),
        "base_sha256": sha256_file(base),
        "out": str(out),
        "out_sha256": sha256_file(out),
        "old_nc": old_nc,
        "new_nc": new_nc,
        "tensors_transferred_by_shape": len(csd),
        "tensors_in_new_model": total_new,
        "transfer_fraction": round(len(csd) / total_new, 4),
        "head_layers_transplanted": transplanted,
        "class_channels_transplanted": list(range(old_nc)),
        "class_channels_initialised": list(range(old_nc, new_nc)),
        "names": list(names),
    }
    return stats


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


# Identical to src/train_detector.py: steel strip texture makes flips and mild
# geometry label-preserving and colour jitter uninformative. Held fixed so the
# only differences between this run and the shipped one are the data and the
# warm start.
AUGMENTATION: dict[str, Any] = {
    "fliplr": 0.5,
    "flipud": 0.5,
    "degrees": 10.0,
    "scale": 0.4,
    "translate": 0.1,
    "shear": 2.0,
    "mosaic": 1.0,
    "close_mosaic": 15,
    "mixup": 0.1,
    "hsv_h": 0.0,
    "hsv_s": 0.3,
    "hsv_v": 0.4,
}


def train_joint(
    *,
    data: Path,
    weights: Path,
    epochs: int,
    imgsz: int = 320,
    batch: int = 32,
    device: str = "auto",
    workers: int = 8,
    patience: int = 20,
    name: str = "yolov8n_joint",
    project: Path = ROOT / "models",
    seed: int = 1337,
    plots: bool = True,
) -> dict[str, Any]:
    """Run the joint fine-tune and return the run directory plus wall clock."""
    from ultralytics import YOLO

    dev = pick_device(device)
    started = time.perf_counter()
    model = YOLO(str(weights))
    model.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=dev,
        workers=workers,
        project=str(project),
        name=name,
        exist_ok=True,
        patience=patience,
        seed=seed,
        deterministic=False,  # MPS deterministic kernels are ~5x slower; seed alone is adequate
        pretrained=True,
        optimizer="auto",
        cos_lr=True,
        plots=plots,
        val=True,
        **AUGMENTATION,
    )
    elapsed = time.perf_counter() - started
    return {
        "run_dir": str(project / name),
        "weights": str(project / name / "weights" / "best.pt"),
        "elapsed_s": elapsed,
        "epochs_requested": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": dev,
    }




# ---------------------------------------------------------------------------
# runtime registration of the four Severstal classes
# ---------------------------------------------------------------------------

# `src/inference.py` hard-codes six class names and a six-entry `DEFECT_INFO`
# severity table. `score_detection` indexes that table with the predicted class
# name and raises rather than guessing, so a 10-class checkpoint cannot run
# through the shipped inference path at all:
#
#     KeyError: 'class_9'   (src/inference.py:211, via predict_batch)
#
# That is a genuine ship blocker and it is reported as one. `src/inference.py`
# is not this task's file to edit, so the four classes are registered *in the
# running process* instead, by mutating the module's own dictionaries in place.
# Nothing is written to disk; the shipped default is untouched; every consumer
# (report.py, cross_domain.py) sees the entries because they hold references to
# the same objects.
#
# THE TIER IS A PLACEHOLDER AND IS TREATED AS ONE. The Severstal competition
# never published semantic names for its four defect ids, so nobody knows what
# they are, and inventing a quality judgement for them would be fabrication. The
# default is the middle tier; `coil --tier low|critical` re-runs the disposition
# at both extremes so the reader can see whether any verdict depends on the
# guess. Severity plays no part at all in the cross-domain rates, which are
# computed from confidences.
SEVERSTAL_TIER_DEFAULT = "medium"

_SEVERSTAL_COLORS: dict[str, tuple[int, int, int]] = {
    "severstal_1": (0, 160, 200),
    "severstal_2": (200, 120, 60),
    "severstal_3": (140, 200, 60),
    "severstal_4": (200, 60, 160),
}


def _loaded_inference_modules() -> list[Any]:
    """Every module object that is `src/inference.py`, and there can be two.

    `evaluate.py`, `false_alarm.py` and `calibrate.py` each carry a
    ``try: from .inference import ... except ImportError: from inference import ...``
    fallback for direct script execution, and one of those except-branches also
    puts `src/` on `sys.path`. The result is that a single process can end up
    holding both ``src.inference`` and a second, independent ``inference`` module
    object built from the same file -- with two independent copies of
    ``CLASS_NAMES`` and ``DEFECT_INFO``. Patching only one of them produces
    exactly the failure this shim exists to prevent, several minutes into a run,
    so both are patched and neither is imported into existence here.
    """
    target = (ROOT / "src" / "inference.py").resolve()
    mods: list[Any] = []
    for name in ("src.inference", "inference"):
        mod = sys.modules.get(name)
        if mod is None:
            continue
        path = getattr(mod, "__file__", None)
        if path and Path(path).resolve() == target and mod not in mods:
            mods.append(mod)
    return mods


def register_severstal_classes(tier: str = SEVERSTAL_TIER_DEFAULT) -> dict[str, Any]:
    """Teach the running process the four Severstal classes. Process-local."""
    from . import inference  # noqa: F401  - guarantees at least one is loaded

    if tier not in inference.SEVERITY_BANDS:
        raise ValueError(f"tier must be one of {inference.SEVERITY_BANDS}, got {tier!r}")

    for module in _loaded_inference_modules():
        _register_into(module, tier)
    return {
        "tier": tier,
        "class_names": list(inference.CLASS_NAMES),
        "registered": list(SEVERSTAL_NAMES),
        "modules_patched": [m.__name__ for m in _loaded_inference_modules()],
    }


def _register_into(inference: Any, tier: str) -> None:
    for i, name in enumerate(SEVERSTAL_NAMES, start=1):
        inference.DEFECT_INFO[name] = {
            "cause": (
                "Severstal hot-rolled carbon strip surface defect class "
                f"{i}. The dataset publishes no semantic name for this class, so "
                "no root cause is asserted here."
            ),
            "severity": tier,
            "action": (
                "Placeholder. The severity tier and the disposition action for "
                "this class must be set by the quality department before this "
                "checkpoint is served."
            ),
        }
        inference.CLASS_COLORS.setdefault(name, _SEVERSTAL_COLORS[name])

    # Making the list exactly 10 long also flips DefectDetector onto the trained
    # head's own names, so detections read "severstal_3" instead of "class_8".
    inference.CLASS_NAMES[:] = list(JOINT_NAMES)

# ---------------------------------------------------------------------------
# axis (a) and (b): detection metrics per domain
# ---------------------------------------------------------------------------


def _box_metrics(result: Any) -> dict[str, Any]:
    """Pull the headline and per-class numbers out of an Ultralytics result."""
    return {
        "mAP50": float(result.box.map50),
        "mAP50_95": float(result.box.map),
        "precision": float(result.box.mp),
        "recall": float(result.box.mr),
        "per_class": {
            result.names[c]: {
                "AP50": float(result.box.ap50[i]),
                "AP50_95": float(result.box.ap[i]),
                "P": float(result.box.p[i]),
                "R": float(result.box.r[i]),
            }
            for i, c in enumerate(result.ap_class_index)
        },
    }


def evaluate_detection(
    weights: Path,
    *,
    safe_dir: Path = SAFE_DIR,
    neu_yaml: Path = NEU_DIR / "data.yaml",
    imgszs: Sequence[int] = (256, 320),
    device: str = "auto",
    scratch: Path | None = None,
) -> dict[str, Any]:
    """Score a checkpoint on the NEU-DET test frames and the Severstal test crops.

    A 10-class checkpoint cannot be validated against the 6-class
    `data/neu-det/data.yaml`, so the NEU-DET half of the joint test split is
    presented through `eval_neu_test.yaml` -- the same 180 images and the same
    labels, declared with the 10-class vocabulary. Ultralytics averages AP only
    over classes that appear in the ground truth, so classes 6-9 (absent from
    NEU-DET) do not enter the mean and the number is directly comparable with the
    shipped 0.7524. A 6-class checkpoint is scored through the canonical NEU-DET
    yaml instead, and `warmstart` reproduces the shipped figure through both, to
    eight significant figures, which is the check that the two paths agree.
    """
    from ultralytics import YOLO

    weights = Path(weights)
    scratch = Path(scratch) if scratch else (RUN_DIR / "val_runs")
    dev = pick_device(device)
    model = YOLO(str(weights))
    nc = int(getattr(model.model, "nc", 0) or len(getattr(model, "names", {})))

    out: dict[str, Any] = {
        "weights": str(weights),
        "weights_sha256": sha256_file(weights),
        "nc": nc,
        "device": dev,
        "neu_det": {},
        "severstal": {},
    }
    neu_data = str(neu_yaml) if nc == 6 else str(safe_dir / "eval_neu_test.yaml")
    neu_split = "test" if nc == 6 else "val"
    out["neu_det_yaml"] = neu_data

    for imgsz in imgszs:
        res = model.val(
            data=neu_data, split=neu_split, imgsz=imgsz, device=dev, plots=False,
            project=str(scratch), name=f"neu_{imgsz}", exist_ok=True, verbose=False,
        )
        out["neu_det"][str(imgsz)] = _box_metrics(res)

    if nc >= 10:
        for imgsz in imgszs:
            res = model.val(
                data=str(safe_dir / "eval_sev_test.yaml"), split="val", imgsz=imgsz,
                device=dev, plots=False, project=str(scratch),
                name=f"sev_{imgsz}", exist_ok=True, verbose=False,
            )
            metrics = _box_metrics(res)
            sev_only = {k: v for k, v in metrics["per_class"].items() if k.startswith("severstal_")}
            metrics["severstal_mAP50"] = (
                float(np.mean([v["AP50"] for v in sev_only.values()])) if sev_only else None
            )
            metrics["severstal_mAP50_95"] = (
                float(np.mean([v["AP50_95"] for v in sev_only.values()])) if sev_only else None
            )
            metrics["severstal_recall"] = (
                float(np.mean([v["R"] for v in sev_only.values()])) if sev_only else None
            )
            out["severstal"][str(imgsz)] = metrics
    else:
        out["severstal"] = {
            "note": "a 6-class checkpoint cannot emit a Severstal class, so every "
                    "Severstal AP is 0 by construction; the meaningful cross-model "
                    "comparison is the class-agnostic recall in reports/cross_domain.json"
        }
    return out


# ---------------------------------------------------------------------------
# axis: operating threshold, picked on VALIDATION
# ---------------------------------------------------------------------------


@dataclass
class ValFrame:
    """One validation image reduced to what the sweep needs."""

    path: Path
    domain: str  # "neu" | "severstal"
    labelled: bool
    boxes: np.ndarray  # (n,4) xyxy in pixels
    classes: np.ndarray  # (n,)


def load_val_frames(split_dir: Path) -> list[ValFrame]:
    """Read a joint split into pixel-space records, 10 classes and all."""
    frames: list[ValFrame] = []
    for img in images_in(split_dir / "images"):
        image = cv2.imread(str(img), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Unreadable validation image: {img}")
        height, width = image.shape[:2]
        rows = read_label(split_dir / "labels" / f"{img.stem}.txt")
        boxes = np.zeros((len(rows), 4), dtype=np.float64)
        classes = np.zeros((len(rows),), dtype=np.int64)
        for i, (cls, cx, cy, bw, bh) in enumerate(rows):
            classes[i] = cls
            boxes[i] = [
                (cx - bw / 2) * width, (cy - bh / 2) * height,
                (cx + bw / 2) * width, (cy + bh / 2) * height,
            ]
        frames.append(ValFrame(
            path=img,
            domain="severstal" if is_severstal(img.stem) else "neu",
            labelled=bool(rows),
            boxes=boxes,
            classes=classes,
        ))
    return frames


def cache_val_predictions(
    frames: Sequence[ValFrame],
    *,
    weights: Path,
    device: str,
    imgsz: int,
    cache_conf: float,
    iou: float,
    batch_size: int = 32,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Score every validation frame once at `cache_conf` and keep all boxes.

    Filtering that cache at `conf >= t` reproduces exactly what a detector built
    at `t` would return: NMS is greedy in descending confidence, so a box above t
    can only ever have been suppressed by another box above t. The whole sweep is
    therefore one pass over the data, not one pass per threshold.
    """
    from .inference import DefectDetector

    # The shipped severity table has no entry for the Severstal classes, so the
    # inference path raises on them. Only class ids, confidences and boxes are
    # used below, so the tier is irrelevant here -- the call is what makes the
    # forward pass survive at all.
    register_severstal_classes()
    detector = DefectDetector(
        weights=weights, device=device, conf=cache_conf, iou=iou, imgsz=imgsz
    )
    cached: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for start in range(0, len(frames), batch_size):
        group = frames[start : start + batch_size]
        images = [cv2.imread(str(f.path), cv2.IMREAD_COLOR) for f in group]
        for result in detector.predict_batch(images, batch_size=batch_size):
            if result.detections:
                boxes = np.asarray([d.bbox_xyxy for d in result.detections], dtype=np.float64)
                conf = np.asarray([d.confidence for d in result.detections], dtype=np.float64)
                cls = np.asarray([d.class_id for d in result.detections], dtype=np.int64)
            else:
                boxes = np.zeros((0, 4)); conf = np.zeros((0,)); cls = np.zeros((0,), dtype=np.int64)
            cached.append((boxes, conf, cls))
    return cached


def sweep_validation(
    frames: Sequence[ValFrame],
    cached: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    thresholds: Sequence[float],
    *,
    iou_match: float = 0.5,
) -> list[dict[str, Any]]:
    """D(t) on labelled frames and F(t) on genuinely defect-free frames.

    `D(t)` is `src/evaluate.py`'s own definition -- a frame counts as detected
    when at least one prediction claims a ground-truth box of the *right class*
    at IoU >= 0.5, not merely when a box lands somewhere on it. `F(t)` is the
    fraction of certified defect-free Severstal crops carrying at least one box
    of any class: a real measurement on real clean steel, so unlike
    `false_alarm.py` it needs no Poisson area extrapolation to reach frame level.
    """
    from .evaluate import match_image

    rows: list[dict[str, Any]] = []
    labelled = [(f, c) for f, c in zip(frames, cached) if f.labelled]
    clean = [(f, c) for f, c in zip(frames, cached) if not f.labelled]
    by_domain = {
        d: [(f, c) for f, c in labelled if f.domain == d] for d in ("neu", "severstal")
    }

    for t in thresholds:
        def detected(pairs: Sequence[tuple[ValFrame, tuple[np.ndarray, np.ndarray, np.ndarray]]]) -> int:
            hits = 0
            for frame, (boxes, conf, cls) in pairs:
                keep = conf >= t
                match = match_image(
                    frame.boxes, frame.classes, boxes[keep], conf[keep], cls[keep], iou_match
                )
                hits += int(int(match.pred_is_tp.sum()) > 0)
            return hits

        def flagged(pairs: Sequence[tuple[ValFrame, tuple[np.ndarray, np.ndarray, np.ndarray]]]) -> int:
            return sum(1 for _, (_, conf, _) in pairs if int((conf >= t).sum()) > 0)

        flagged_clean = flagged(clean)
        boxes_clean = sum(int((c >= t).sum()) for _, (b, c, _) in clean)
        rows.append({
            "threshold": float(t),
            "n_labelled": len(labelled),
            "n_clean": len(clean),
            # strict: a class-correct box on the defect, evaluate.py's definition
            "defect_detection_rate": detected(labelled) / max(len(labelled), 1),
            "defect_detection_rate_neu": (
                detected(by_domain["neu"]) / len(by_domain["neu"]) if by_domain["neu"] else None
            ),
            "defect_detection_rate_severstal": (
                detected(by_domain["severstal"]) / len(by_domain["severstal"])
                if by_domain["severstal"] else None
            ),
            # lenient: the frame got flagged at all, so an operator looks at it and
            # the miss cost is not incurred even if the box is in the wrong place
            "defect_flag_rate": flagged(labelled) / max(len(labelled), 1),
            "defect_flag_rate_neu": (
                flagged(by_domain["neu"]) / len(by_domain["neu"]) if by_domain["neu"] else None
            ),
            "defect_flag_rate_severstal": (
                flagged(by_domain["severstal"]) / len(by_domain["severstal"])
                if by_domain["severstal"] else None
            ),
            "clean_frame_false_alarm_rate": flagged_clean / max(len(clean), 1),
            "boxes_per_clean_frame": boxes_clean / max(len(clean), 1),
        })
    return rows


def pick_operating_point(
    rows: Sequence[dict[str, Any]],
    *,
    cost_model: dict[str, Any] | None = None,
    detection_key: str = "defect_detection_rate",
) -> dict[str, Any]:
    """Cheapest validation threshold that still clears the detection floor.

    The cost function, the miss:false-alarm ratio, the prevalence and the
    detection floor are `false_alarm.COST_MODEL` unchanged; only F(t) is
    different, and it is different because it is now measured on genuinely
    defect-free steel instead of extrapolated from mined patches.
    """
    from .false_alarm import COST_MODEL, CostPoint, choose_threshold, effective_lambda

    cfg = dict(cost_model or COST_MODEL)
    ratio = float(cfg["miss_cost_ratio"])
    prevalence = float(cfg["defect_frame_prevalence"])
    floor = float(cfg["min_detection_rate"])

    def curve_for(r: float) -> list[CostPoint]:
        out: list[CostPoint] = []
        for row in rows:
            detection = float(row[detection_key])
            miss = prevalence * r * (1.0 - detection)
            fa = (1.0 - prevalence) * row["clean_frame_false_alarm_rate"]
            out.append(CostPoint(
                threshold=row["threshold"],
                detection_rate=detection,
                clean_frame_fa=row["clean_frame_false_alarm_rate"],
                clean_patch_fa=row["clean_frame_false_alarm_rate"],
                legacy_fa=float("nan"),
                cost_frame=miss + fa,
                cost_patch=miss + fa,
            ))
        return out

    curve = curve_for(ratio)
    chosen, floor_met, at_edge = choose_threshold(curve, floor, basis="frame")
    unconstrained = min(curve, key=lambda p: (p.cost_frame, -p.threshold))

    sensitivity = []
    for r in cfg.get("sensitivity_ratios", (3.0, 10.0, 12.0, 30.0)):
        c = curve_for(float(r))
        best, met, _ = choose_threshold(c, floor, basis="frame")
        free = min(c, key=lambda p: (p.cost_frame, -p.threshold))
        sensitivity.append({
            "miss_cost_ratio": float(r),
            "effective_lambda": round(effective_lambda(float(r), prevalence), 5),
            "constrained_threshold": round(best.threshold, 3),
            "constrained_detection_rate": round(best.detection_rate, 4),
            "constrained_clean_fa": round(best.clean_frame_fa, 4),
            "unconstrained_threshold": round(free.threshold, 3),
            "floor_reachable": met,
        })

    return {
        "cost_model": cfg,
        "detection_key": detection_key,
        "effective_lambda": round(effective_lambda(ratio, prevalence), 5),
        "chosen": chosen.to_dict(),
        "detection_floor_met": floor_met,
        "chosen_at_sweep_edge": at_edge,
        "unconstrained": unconstrained.to_dict(),
        "sensitivity": sensitivity,
    }


def choose_threshold_on_validation(
    weights: Path,
    *,
    split_dir: Path = SAFE_DIR / "val",
    device: str = "auto",
    imgsz: int = 256,
    iou: float = 0.45,
    cache_conf: float = 0.01,
    thresholds: Sequence[float] | None = None,
) -> dict[str, Any]:
    """The whole validation threshold study, start to finish."""
    from .false_alarm import SWEEP_THRESHOLDS

    frames = load_val_frames(Path(split_dir))
    cached = cache_val_predictions(
        frames, weights=Path(weights), device=pick_device(device),
        imgsz=imgsz, cache_conf=cache_conf, iou=iou,
    )
    rows = sweep_validation(frames, cached, thresholds or SWEEP_THRESHOLDS)
    picked = pick_operating_point(rows)
    # The same rule applied to each population separately. Pooling two domains of
    # very different difficulty under one detection floor can push the pick onto
    # the sweep edge without that meaning anything about either domain, so the
    # per-domain picks are computed rather than argued about.
    variants = {
        key: pick_operating_point(rows, detection_key=key)
        for key in (
            "defect_detection_rate",
            "defect_detection_rate_neu",
            "defect_detection_rate_severstal",
            "defect_flag_rate",
            "defect_flag_rate_neu",
            "defect_flag_rate_severstal",
        )
    }
    return {
        "weights": str(weights),
        "weights_sha256": sha256_file(Path(weights)),
        "split_dir": str(split_dir),
        "imgsz": imgsz,
        "iou": iou,
        "cache_conf": cache_conf,
        "population": {
            "frames": len(frames),
            "labelled": sum(1 for f in frames if f.labelled),
            "clean_severstal": sum(1 for f in frames if not f.labelled),
            "neu": sum(1 for f in frames if f.domain == "neu"),
            "severstal": sum(1 for f in frames if f.domain == "severstal"),
        },
        "sweep": rows,
        "operating_point": picked,
        "operating_points_by_population": variants,
    }

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_verify = sub.add_parser("verify", help="re-derive every claim data/joint makes")
    p_verify.add_argument("--joint-dir", default=str(JOINT_DIR))
    p_verify.add_argument("--neu-dir", default=str(NEU_DIR))
    p_verify.add_argument("--cache-dir", default=str(SEVERSTAL_CACHE))
    p_verify.add_argument("--samples-json", default=None)
    p_verify.add_argument("--fidelity-sample", type=int, default=120)
    p_verify.add_argument("--json", default=None)

    p_build = sub.add_parser("build", help="derive data/joint_xdsafe (symlinks)")
    p_build.add_argument("--joint-dir", default=str(JOINT_DIR))
    p_build.add_argument("--out", default=str(SAFE_DIR))

    p_warm = sub.add_parser("warmstart", help="widen the shipped head 6 -> 10 classes")
    p_warm.add_argument("--base", default=str(BASE_WEIGHTS))
    p_warm.add_argument("--out", default=str(WARMSTART_PT))

    p_train = sub.add_parser("train", help="joint fine-tune")
    p_train.add_argument("--data", default=str(SAFE_DIR / "data.yaml"))
    p_train.add_argument("--weights", default=str(WARMSTART_PT))
    p_train.add_argument("--epochs", type=int, required=True)
    p_train.add_argument("--imgsz", type=int, default=320)
    p_train.add_argument("--batch", type=int, default=32)
    p_train.add_argument("--device", default="auto")
    p_train.add_argument("--workers", type=int, default=8)
    p_train.add_argument("--patience", type=int, default=20)
    p_train.add_argument("--name", default="yolov8n_joint")
    p_train.add_argument("--json", default=None)

    p_eval = sub.add_parser("eval", help="NEU-DET and Severstal held-out test metrics")
    p_eval.add_argument("--weights", required=True)
    p_eval.add_argument("--imgsz", type=int, nargs="+", default=[256, 320])
    p_eval.add_argument("--device", default="auto")
    p_eval.add_argument("--safe-dir", default=str(SAFE_DIR))
    p_eval.add_argument("--json", default=None)

    p_thr = sub.add_parser("threshold", help="re-pick the operating point on VALIDATION")
    p_thr.add_argument("--weights", required=True)
    p_thr.add_argument("--split-dir", default=str(SAFE_DIR / "val"))
    p_thr.add_argument("--imgsz", type=int, default=256)
    p_thr.add_argument("--iou", type=float, default=0.45)
    p_thr.add_argument("--device", default="auto")
    p_thr.add_argument("--json", default=None)

    p_xd = sub.add_parser(
        "crossdomain",
        help="register the Severstal classes, then run src/cross_domain.py unchanged",
    )
    p_xd.add_argument("--tier", default=SEVERSTAL_TIER_DEFAULT)
    p_xd.add_argument("rest", nargs=argparse.REMAINDER,
                      help="arguments forwarded verbatim to src.cross_domain")

    p_coil = sub.add_parser(
        "coil", help="run the shipped disposition chain over a directory of frames"
    )
    p_coil.add_argument("--tier", default=SEVERSTAL_TIER_DEFAULT)
    p_coil.add_argument("rest", nargs=argparse.REMAINDER,
                        help="arguments forwarded verbatim to src.report")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "verify":
        checks, facts = verify_dataset(
            Path(args.joint_dir),
            Path(args.neu_dir),
            Path(args.cache_dir),
            Path(args.samples_json) if args.samples_json else None,
            fidelity_sample=args.fidelity_sample,
        )
        for check in checks:
            print(check.line())
        print(json.dumps(facts, indent=2, default=str))
        if args.json:
            Path(args.json).write_text(json.dumps(
                {"checks": [c.__dict__ for c in checks], "facts": facts}, indent=2, default=str
            ))
        # The holdout check is expected to fail on data/joint as delivered; it is
        # informational, and `build` is the response to it.
        hard = [c for c in checks if not c.ok and "holdout" not in c.name]
        return 1 if hard else 0

    if args.command == "build":
        stats = build_holdout_safe(Path(args.joint_dir), Path(args.out))
        print(json.dumps(stats, indent=2))
        return 0

    if args.command == "warmstart":
        stats = build_warmstart_checkpoint(Path(args.base), Path(args.out))
        print(json.dumps(stats, indent=2))
        return 0

    if args.command == "eval":
        result = evaluate_detection(
            Path(args.weights), safe_dir=Path(args.safe_dir),
            imgszs=tuple(args.imgsz), device=args.device,
        )
        print(json.dumps(result, indent=2))
        if args.json:
            Path(args.json).write_text(json.dumps(result, indent=2))
        return 0

    if args.command == "threshold":
        result = choose_threshold_on_validation(
            Path(args.weights), split_dir=Path(args.split_dir),
            device=args.device, imgsz=args.imgsz, iou=args.iou,
        )
        print(json.dumps(result, indent=2))
        if args.json:
            Path(args.json).write_text(json.dumps(result, indent=2))
        return 0

    if args.command == "crossdomain":
        from . import cross_domain  # imported first: it is what forks the module

        print(json.dumps(register_severstal_classes(args.tier)))
        forwarded = [a for a in args.rest if a != "--"]
        return cross_domain.main(forwarded)

    if args.command == "coil":
        from . import report as report_module

        print(json.dumps(register_severstal_classes(args.tier)))
        forwarded = [a for a in args.rest if a != "--"]
        return report_module._cli(forwarded)

    if args.command == "train":
        result = train_joint(
            data=Path(args.data),
            weights=Path(args.weights),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            patience=args.patience,
            name=args.name,
        )
        print(json.dumps(result, indent=2))
        if args.json:
            Path(args.json).write_text(json.dumps(result, indent=2))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
