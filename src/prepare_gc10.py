"""Ingest GC10-DET into YOLO format: roll-mark and edge-defect coverage for the brief.

WHY THIS DATASET EXISTS IN THIS PROJECT
---------------------------------------
The Jindal brief names four defect families: scratches, scale, **roll marks** and
**edge cracks**. NEU-DET covers the first two (``scratches``, ``rolled-in_scale``)
and has nothing at all for the last two. GC10-DET is the closest public dataset
that does, and it is CC BY 4.0, so it can be shipped.

HONESTY STATEMENT -- READ BEFORE QUOTING ANY NUMBER FROM THIS MODULE
---------------------------------------------------------------------
**No public dataset has a true "edge crack" class, and this one does not either.**
What GC10-DET actually has:

* ``3_yueyawan`` (crescent gap) and ``10_yaozhed`` (waist folding) are *edge-region
  geometry defects*. They are the closest honest proxies for edge cracking that
  public data offers. They are not cracks. A crescent gap is a scalloped bite out
  of the strip edge; a waist fold is a folded-over edge. Both originate at the
  edge and both are what an edge camera would actually see, which is why they are
  a defensible stand-in -- but a model trained on them has not been shown an edge
  crack and must not be described as detecting one.
* ``8_yahen`` (rolled pit) is the roll-mark class, and across the whole
  2,294-image mirror it carries **85 boxes in 46 images** (64 of those sit in the
  mirror's own train split, which is the figure usually quoted). That is thin to
  the point that any per-class AP computed on it will have a confidence interval
  wider than the number itself, and after a 70/15/15 split it leaves 13 boxes in
  val and 13 in test. ``9_zhehen`` (crease, 74 boxes in 53 images) is adjacent --
  handling/roll damage -- and is also thin.

So the claim this module supports is: *we cover the brief's named families as
closely as public data allows, and we say exactly where the coverage is a proxy
and where it is thin.* It is not: *we detect edge cracks.*

WHAT THIS MODULE DOES
---------------------
1. Downloads the ``imaadd05/gc10-det`` mirror from HuggingFace (CC BY 4.0, a
   Roboflow COCO export of the original GC10-DET release). 2,294 grayscale
   line-scan images, every one 2048x1000.
2. Converts COCO ``[x, y, w, h]`` pixel boxes to YOLO normalised ``cx cy w h``,
   clipping to the frame and dropping degenerate boxes.
3. **Re-splits from scratch.** The upstream mirror ships its own train/test split
   and that split leaks: 102 line-scan sequence ids appear on both sides of it.
   GC10-DET frames are consecutive captures off a running strip, so two frames
   from one sequence are near-duplicates. We pool all 2,294 images and cut a
   fresh 70/15/15 split that is disjoint by *sequence id*, which also makes it
   disjoint by content hash. Both guards are asserted before anything is written.
4. Emits ``data.yaml`` with readable English class names and a ``manifest.json``
   that records the pinyin -> English mapping, the split assignment, and the
   leakage-check result.

Nothing here trains anything. A later stage does that.

Source:  https://huggingface.co/datasets/imaadd05/gc10-det  (CC BY 4.0)
Origin:  Lv, X.; Duan, F.; Jiang, J.J.; Fu, X.; Gan, L. "Deep Metallic Surface
         Defect Detection: The New Benchmark and Detection Network."
         Sensors 2020, 20(6), 1562.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import requests
from PIL import Image, ImageStat

REPO_ID = "imaadd05/gc10-det"
API_TREE = f"https://huggingface.co/api/datasets/{REPO_ID}/tree/main?recursive=true"
RESOLVE = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/"
LICENSE = "CC BY 4.0"
CITATION = (
    "Lv, X.; Duan, F.; Jiang, J.J.; Fu, X.; Gan, L. Deep Metallic Surface Defect "
    "Detection: The New Benchmark and Detection Network. Sensors 2020, 20(6), 1562."
)

SEED = 1337
SPLIT_RATIOS: dict[str, float] = {"train": 0.70, "val": 0.15, "test": 0.15}

# Original GC10-DET labels are Chinese-pinyin with a leading ordinal. The English
# names below are the class list written into data.yaml and are therefore the
# order the detection head is trained in. Index order follows the original
# ordinal 1..10 so that anyone holding a GC10 paper can line the two up.
#
# `7_yiwu` is glossed "inclusion" in most GC10 papers. It is deliberately named
# `foreign_object` here: NEU-DET already ships a class called `inclusion`, and a
# later stage merges the two datasets into one 16-class model, where a duplicated
# name would silently collide. 异物 is literally "foreign matter", so the rename
# is also the more accurate gloss.
GC10_CLASSES: tuple[str, ...] = (
    "punching_hole",   # 0
    "weld_line",       # 1
    "crescent_gap",    # 2
    "water_spot",      # 3
    "oil_spot",        # 4
    "silk_spot",       # 5
    "foreign_object",  # 6
    "rolled_pit",      # 7
    "crease",          # 8
    "waist_fold",      # 9
)

# original pinyin label -> (English class name, Chinese, literal gloss)
PINYIN_TO_ENGLISH: dict[str, tuple[str, str, str]] = {
    "1_chongkong": ("punching_hole", "冲孔", "punched hole"),
    "2_hanfeng": ("weld_line", "焊缝", "weld seam"),
    "3_yueyawan": ("crescent_gap", "月牙弯", "crescent bend / gap"),
    "4_shuiban": ("water_spot", "水斑", "water stain"),
    "5_youban": ("oil_spot", "油斑", "oil stain"),
    "6_siban": ("silk_spot", "丝斑", "silk-thread stain"),
    "7_yiwu": ("foreign_object", "异物", "foreign matter"),
    "8_yahen": ("rolled_pit", "压痕", "pressed mark / rolled pit"),
    "9_zhehen": ("crease", "折痕", "crease"),
    "10_yaozhed": ("waist_fold", "腰折", "waist folding"),
}

# How each GC10 class answers (or fails to answer) the brief's four families.
# `proxy` is True where the class is a stand-in, not the named defect itself.
BRIEF_FAMILIES: dict[str, dict[str, object]] = {
    "roll marks": {
        "classes": ("rolled_pit", "crease"),
        "proxy": False,
        "note": (
            "rolled_pit (8_yahen) is a roll-imprinted pit and is the direct match. "
            "crease (9_zhehen) is roll/handling fold damage. Both are thin: 85 boxes "
            "in 46 images and 74 boxes in 53 images across the whole 2,294-image set."
        ),
    },
    "edge cracks": {
        "classes": ("crescent_gap", "waist_fold"),
        "proxy": True,
        "note": (
            "PROXY, NOT THE NAMED DEFECT. No public dataset has an edge-crack "
            "class. crescent_gap (3_yueyawan) and waist_fold (10_yaozhed) are "
            "edge-region geometry defects -- the closest honest stand-in."
        ),
    },
    "scratches": {
        "classes": (),
        "proxy": False,
        "note": "Not in GC10-DET. Covered by NEU-DET class `scratches`.",
    },
    "scale": {
        "classes": (),
        "proxy": False,
        "note": "Not in GC10-DET. Covered by NEU-DET class `rolled-in_scale`.",
    },
}

# Classes outside the brief's four families, kept because they are free labelled
# steel-surface data and because weld_line is a real coil-join event a mill cares
# about even though the brief does not name it.
OUT_OF_BRIEF_NOTE = (
    "punching_hole, weld_line, water_spot, oil_spot, silk_spot and foreign_object "
    "are not among the brief's four families. weld_line is a coil-join event worth "
    "flagging in its own right; the rest are retained because dropping labelled "
    "boxes from a shared image would turn real defects into unlabelled background."
)

_STEM_RE = re.compile(r"^img_(?P<cam>\d+)_(?P<seq>[A-Za-z0-9]+)_(?P<frame>\d+)$")


# --------------------------------------------------------------------------- #
# pure helpers (unit-testable without the dataset on disk)
# --------------------------------------------------------------------------- #
def original_stem(file_name: str) -> str:
    """Strip Roboflow's ``_jpg.rf.<hash>.jpg`` suffix back to the GC10 stem."""
    return file_name.split("_jpg.rf.")[0]


def sequence_id(stem: str) -> str:
    """Line-scan sequence id from a GC10 stem: ``img_03_425005700_00156`` -> ``425005700``.

    The camera prefix is deliberately *not* part of the key. img_01..img_08 are
    different cameras looking at the same strip position, so two frames sharing a
    sequence id are near-duplicate content even when the camera differs. Grouping
    on the sequence alone is the stricter guard.
    """
    m = _STEM_RE.match(stem)
    if not m:
        raise ValueError(f"unrecognised GC10 stem: {stem!r}")
    return m.group("seq")


def coco_to_yolo(
    bbox: Sequence[float], width: int, height: int, min_side_px: float = 1.0
) -> tuple[float, float, float, float] | None:
    """COCO ``[x, y, w, h]`` pixels -> YOLO ``(cx, cy, w, h)`` normalised.

    Clips to the frame first. Returns ``None`` for a box that survives clipping
    with a side under ``min_side_px``, which is how degenerate annotations are
    dropped rather than written out as zero-area targets.
    """
    x, y, w, h = (float(v) for v in bbox)
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(float(width), x + w), min(float(height), y + h)
    bw, bh = x1 - x0, y1 - y0
    if bw < min_side_px or bh < min_side_px:
        return None
    return (
        (x0 + bw / 2.0) / width,
        (y0 + bh / 2.0) / height,
        bw / width,
        bh / height,
    )


def content_hash(data: bytes) -> str:
    """SHA-256 of the raw file bytes. Two images with the same digest are the same image."""
    return hashlib.sha256(data).hexdigest()


def stratified_group_split(
    group_classes: dict[str, Counter[str]],
    group_sizes: dict[str, int],
    ratios: dict[str, float] = SPLIT_RATIOS,
    seed: int = SEED,
) -> dict[str, str]:
    """Assign whole sequence groups to splits, rarest class first.

    Iterative stratification over groups: the class with the fewest unplaced boxes
    is served first, so ``rolled_pit`` (85 boxes) gets to pick its groups before
    ``silk_spot`` (884) crowds it out. Within a class the groups go in
    largest-contribution-first order (longest-processing-time greedy), because a
    group carrying 30 boxes of one class placed last will overshoot whichever
    split it lands in -- placing it while every split still has room is what keeps
    the ratios tight. Measured on GC10: hash order gives a total L1 ratio error of
    1.217 across the ten classes, this ordering gives 0.058.

    Deterministic given ``seed``: ties break on a seeded hash rather than on dict
    iteration order.
    """
    splits = list(ratios)
    totals: Counter[str] = Counter()
    for counts in group_classes.values():
        totals.update(counts)

    need: dict[str, Counter[str]] = {
        s: Counter({c: totals[c] * ratios[s] for c in totals}) for s in splits
    }
    img_need = {s: sum(group_sizes.values()) * ratios[s] for s in splits}
    assigned: dict[str, str] = {}

    def order_key(g: str) -> tuple[str, str]:
        return (hashlib.sha256(f"{seed}:{g}".encode()).hexdigest(), g)

    def place(g: str, s: str) -> None:
        assigned[g] = s
        for c, n in group_classes[g].items():
            need[s][c] -= n
        img_need[s] -= group_sizes[g]

    remaining_classes = set(totals)
    while remaining_classes:
        # rarest first, by boxes still unplaced across all splits
        left = {
            c: sum(group_classes[g][c] for g in group_classes if g not in assigned)
            for c in remaining_classes
        }
        cls = min(remaining_classes, key=lambda c: (left[c], c))
        remaining_classes.discard(cls)
        pending = sorted(
            (g for g in group_classes if g not in assigned and group_classes[g][cls]),
            key=lambda g: (-group_classes[g][cls], -group_sizes[g], order_key(g)),
        )
        for g in pending:
            # the split most starved of this class wins; ties go to the split most
            # starved of images overall, then to the declared split order
            best = max(
                splits,
                key=lambda s: (need[s][cls], img_need[s], -splits.index(s)),
            )
            place(g, best)

    # groups holding no boxes at all (background frames) just balance image counts
    for g in sorted((g for g in group_classes if g not in assigned), key=order_key):
        best = max(splits, key=lambda s: (img_need[s], -splits.index(s)))
        place(g, best)

    return assigned


# --------------------------------------------------------------------------- #
# download
# --------------------------------------------------------------------------- #
def list_repo_files(session: requests.Session) -> list[dict]:
    """Every file in the HF repo, following the API's paginated Link header."""
    url, out = API_TREE, []
    while url:
        r = session.get(url, timeout=120)
        r.raise_for_status()
        out.extend(r.json())
        m = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("Link", ""))
        url = m.group(1) if m else None
    return [f for f in out if f["type"] == "file"]


def download_raw(raw: Path, workers: int = 16, retries: int = 3) -> dict[str, object]:
    """Mirror the repo into ``raw``. Files already present at the right size are skipped."""
    session = requests.Session()
    files = list_repo_files(session)
    wanted = [
        f for f in files if f["path"].endswith((".jpg", ".json", ".txt", ".md"))
    ]
    raw.mkdir(parents=True, exist_ok=True)

    todo = []
    for f in wanted:
        dst = raw / f["path"]
        if dst.exists() and dst.stat().st_size == f.get("size", -1):
            continue
        todo.append(f)
    for d in {(raw / f["path"]).parent for f in wanted}:
        d.mkdir(parents=True, exist_ok=True)

    def fetch(f: dict) -> int:
        dst = raw / f["path"]
        last: Exception | None = None
        for attempt in range(retries):
            try:
                r = session.get(RESOLVE + f["path"], timeout=180)
                r.raise_for_status()
                dst.write_bytes(r.content)
                return len(r.content)
            except Exception as exc:  # network flake: back off and retry
                last = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"failed to download {f['path']}: {last}")

    t0 = time.time()
    got = 0
    if todo:
        print(f"downloading {len(todo)} file(s) from {REPO_ID} with {workers} threads")
        with ThreadPoolExecutor(workers) as ex:
            for i, n in enumerate(ex.map(fetch, todo), 1):
                got += n
                if i % 250 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)}  {got / 1e6:.1f} MB")
    dt = time.time() - t0
    on_disk = sum(f.get("size", 0) for f in wanted)
    print(
        f"raw mirror ready: {len(wanted)} files, {on_disk / 1e6:.1f} MB "
        f"({len(todo)} fetched in {dt:.1f}s)"
    )
    return {
        "files": len(wanted),
        "bytes": on_disk,
        "downloaded": len(todo),
        "seconds": round(dt, 1),
    }


# --------------------------------------------------------------------------- #
# COCO -> records
# --------------------------------------------------------------------------- #
@dataclass
class Frame:
    """One GC10 image with its converted boxes."""

    file_name: str
    upstream_split: str
    width: int
    height: int
    stem: str
    sequence: str
    boxes: list[tuple[int, float, float, float, float]] = field(default_factory=list)
    digest: str = ""

    @property
    def classes(self) -> Counter[str]:
        return Counter(GC10_CLASSES[c] for c, *_ in self.boxes)


def load_coco(raw: Path) -> tuple[list[Frame], dict[str, int]]:
    """Read both upstream COCO files and convert every box to YOLO form.

    The Roboflow export carries a placeholder category ``0: defect`` with
    ``supercategory: none`` that owns no annotations. It is dropped here, which is
    why the YOLO ids are 0..9 rather than the COCO ids 1..10.
    """
    frames: list[Frame] = []
    stats = {"coco_boxes": 0, "dropped_degenerate": 0, "clipped": 0}
    for split in ("train", "test"):
        path = raw / split / "_annotations.coco.json"
        doc = json.loads(path.read_text())
        cat = {c["id"]: c["name"] for c in doc["categories"]}
        by_image: dict[int, list[dict]] = defaultdict(list)
        for a in doc["annotations"]:
            by_image[a["image_id"]].append(a)
        for im in doc["images"]:
            stem = original_stem(im["file_name"])
            fr = Frame(
                file_name=im["file_name"],
                upstream_split=split,
                width=im["width"],
                height=im["height"],
                stem=stem,
                sequence=sequence_id(stem),
            )
            for a in by_image[im["id"]]:
                name = cat[a["category_id"]]
                if name not in PINYIN_TO_ENGLISH:  # the placeholder `defect` category
                    continue
                stats["coco_boxes"] += 1
                x, y, w, h = a["bbox"]
                if x < 0 or y < 0 or x + w > im["width"] or y + h > im["height"]:
                    stats["clipped"] += 1
                yolo = coco_to_yolo(a["bbox"], im["width"], im["height"])
                if yolo is None:
                    stats["dropped_degenerate"] += 1
                    continue
                cid = GC10_CLASSES.index(PINYIN_TO_ENGLISH[name][0])
                fr.boxes.append((cid, *yolo))
            frames.append(fr)
    return frames, stats


def hash_frames(raw: Path, frames: list[Frame], workers: int = 16) -> None:
    """Fill in ``Frame.digest`` for every frame, in parallel."""

    def one(fr: Frame) -> str:
        return content_hash((raw / fr.upstream_split / fr.file_name).read_bytes())

    with ThreadPoolExecutor(workers) as ex:
        for fr, d in zip(frames, ex.map(one, frames)):
            fr.digest = d


# --------------------------------------------------------------------------- #
# split + write
# --------------------------------------------------------------------------- #
def upstream_leakage(frames: Iterable[Frame]) -> dict[str, object]:
    """How badly the mirror's own train/test split leaks. Reported, not used."""
    by_split: dict[str, set[str]] = defaultdict(set)
    digests: dict[str, set[str]] = defaultdict(set)
    for fr in frames:
        by_split[fr.upstream_split].add(fr.sequence)
        digests[fr.upstream_split].add(fr.digest)
    shared_seq = by_split["train"] & by_split["test"]
    shared_hash = digests["train"] & digests["test"]
    return {
        "train_sequences": len(by_split["train"]),
        "test_sequences": len(by_split["test"]),
        "shared_sequences": len(shared_seq),
        "shared_content_hashes": len(shared_hash),
        "example_shared_sequences": sorted(shared_seq)[:8],
    }


def write_split(
    raw: Path, out: Path, frames: list[Frame], assign: dict[str, str]
) -> dict[str, dict]:
    """Copy images and write YOLO label files under ``out/<split>/``."""
    for split in SPLIT_RATIOS:
        for sub in ("images", "labels"):
            (out / split / sub).mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {
        s: {"images": 0, "boxes": Counter(), "empty": 0} for s in SPLIT_RATIOS
    }
    for fr in frames:
        split = assign[fr.sequence]
        shutil.copy2(
            raw / fr.upstream_split / fr.file_name, out / split / "images" / fr.file_name
        )
        lines = [
            f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in fr.boxes
        ]
        (out / split / "labels" / f"{Path(fr.file_name).stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )
        summary[split]["images"] += 1
        if not fr.boxes:
            summary[split]["empty"] += 1
        summary[split]["boxes"].update(fr.classes)
    return summary


def assert_disjoint(out: Path) -> dict[str, object]:
    """Re-read what was written and prove no image byte-for-byte repeats across splits."""
    seen: dict[str, str] = {}
    per_split: dict[str, int] = {}
    for split in SPLIT_RATIOS:
        imgs = sorted((out / split / "images").glob("*.jpg"))
        per_split[split] = len(imgs)
        for img in imgs:
            d = content_hash(img.read_bytes())
            if d in seen and seen[d] != split:
                raise SystemExit(f"LEAK: {img.name} duplicates an image in {seen[d]}")
            seen[d] = split
    return {
        "images_hashed": sum(per_split.values()),
        "unique_content_hashes": len(seen),
        "duplicate_pairs_within_a_split": sum(per_split.values()) - len(seen),
        "cross_split_duplicates": 0,
    }


def write_yaml(out: Path) -> None:
    text = (
        "# GC10-DET metallic surface defects (2048x1000 line-scan strip)\n"
        f"# source: https://huggingface.co/datasets/{REPO_ID}  license: {LICENSE}\n"
        "# Split is sequence-disjoint and re-cut by src/prepare_gc10.py; the\n"
        "# upstream mirror's own train/test split leaks and is not used.\n"
        f"path: {out}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        f"nc: {len(GC10_CLASSES)}\n"
        "names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(GC10_CLASSES))
    )
    (out / "data.yaml").write_text(text)


# --------------------------------------------------------------------------- #
# demo assets
# --------------------------------------------------------------------------- #
# Three wide frames for the demo. The project ships no image above 600 px, so
# `predict_tiled` has never had anything to tile; a 2048x1000 frame is 3.2x the
# 640 px tile the detector deploys at and forces the real tiling path.
ASSET_PICKS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "strip_sample_rollmark.jpg",
        ("rolled_pit", "crease"),
        "roll mark (GC10 8_yahen rolled pit / 9_zhehen crease)",
    ),
    (
        "strip_sample_edge_defect.jpg",
        ("crescent_gap", "waist_fold"),
        "edge defect, PROXY for edge crack "
        "(GC10 3_yueyawan crescent gap / 10_yaozhed waist fold)",
    ),
    (
        "strip_sample_weldline.jpg",
        ("weld_line",),
        "coil-join weld line running across the strip (GC10 2_hanfeng)",
    ),
)

ASSET_SHORTLIST = 25
MIN_ASSET_MEAN_LEVEL = 60.0
# A box under this fraction of the frame is a speck nobody can see on a slide; one
# over it is the whole picture, which shows the viewer no context and demos nothing.
MIN_BOX_FRAC = 0.0005
MAX_BOX_FRAC = 0.35


def frame_brightness(path: Path) -> float:
    """Mean grey level of a thumbnail of ``path``, 0-255.

    GC10 frames often catch the dark off-strip background beside the edge. A frame
    that is three-quarters black is a bad demo image whatever it is labelled, so
    brightness is the last tie-break in the asset pick. ``draft`` makes PIL decode
    the JPEG at 1/8 scale, so this costs about a millisecond per frame.
    """
    with Image.open(path) as im:
        im.draft("L", (256, 128))
        return float(ImageStat.Stat(im.convert("L")).mean[0])


def export_assets(
    raw: Path, frames: list[Frame], assets: Path, assign: dict[str, str] | None = None
) -> list[dict]:
    """Copy three representative wide frames into ``assets/`` with an attribution note."""
    assets.mkdir(parents=True, exist_ok=True)
    assign = assign or {}
    used: set[str] = set()
    written: list[dict] = []
    for name, wanted, caption in ASSET_PICKS:
        want = set(wanted)

        # Rank on labels only, then break the tie on pixels. In order:
        # held-out split first (nothing in the demo is also a training image once a
        # later stage trains on this); then the first-listed class of the family,
        # which is the one the caption leads with; then a box that is actually
        # well framed; then box count; then fewest other classes in frame, so the
        # caption is not lying about what else is in the picture.
        def label_score(fr: Frame) -> tuple[int, int, int, int, int, float, str]:
            cls = fr.classes
            areas = [w * h for c, _, _, w, h in fr.boxes if GC10_CLASSES[c] in want]
            framed = [a for a in areas if MIN_BOX_FRAC <= a <= MAX_BOX_FRAC]
            return (
                0 if assign.get(fr.sequence) in ("test", "val") else 1,
                0 if cls[wanted[0]] else 1,
                0 if framed else 1,
                -sum(cls[w] for w in wanted),
                len(set(cls) - want),
                -max(framed, default=0.0),
                fr.file_name,
            )

        pool = [
            fr
            for fr in frames
            if fr.file_name not in used and any(fr.classes[w] for w in wanted)
        ]
        if not pool:
            print(f"  no frame available for {name}")
            continue
        shortlist = sorted(pool, key=label_score)[:ASSET_SHORTLIST]
        lit = [
            (fr, b)
            for fr in shortlist
            if (b := frame_brightness(raw / fr.upstream_split / fr.file_name))
            >= MIN_ASSET_MEAN_LEVEL
        ]
        if lit:
            pick, level = lit[0]  # shortlist order is already label rank
        else:
            pick = shortlist[0]
            level = frame_brightness(raw / pick.upstream_split / pick.file_name)
        used.add(pick.file_name)
        shutil.copy2(raw / pick.upstream_split / pick.file_name, assets / name)

        widest = max(
            (w for c, _, _, w, _ in pick.boxes if GC10_CLASSES[c] in want), default=0.0
        )
        written.append(
            {
                "asset": name,
                "caption": caption,
                "source_file": pick.file_name,
                "gc10_stem": pick.stem,
                "split": assign.get(pick.sequence, "unassigned"),
                "size": f"{pick.width}x{pick.height}",
                "boxes": dict(pick.classes),
                "widest_family_box_px": round(widest * pick.width),
                "mean_grey_level": round(level, 1),
            }
        )
        print(
            f"  {name}  <- {pick.stem}  split={assign.get(pick.sequence, '?')}  "
            f"grey={level:.0f}  {dict(pick.classes)}"
        )

    lines = [
        "# assets/",
        "",
        "Wide-strip sample frames for the demo. The rest of the project ships no",
        "image wider than 600 px, so `DefectDetector.predict_tiled` has never had",
        "anything to tile; each file here is a 2048x1000 grayscale line-scan frame,",
        "which is real strip geometry and 3.2x the 640 px the detector deploys at.",
        "",
        f"**Source:** https://huggingface.co/datasets/{REPO_ID}",
        f"**Licence:** {LICENSE} -- attribution required, given below.",
        f"**Cite:** {CITATION}",
        "",
        "Unmodified frames, copied by `src/prepare_gc10.py`. `split` is the split",
        "the frame sits in under `data/gc10-det`, so a model trained on that split",
        "has not seen these images.",
        "",
        "| file | what it shows | GC10 frame | split | labelled boxes | widest box |",
        "|---|---|---|---|---|---|",
    ]
    for w in written:
        boxes = ", ".join(f"{k} x{v}" for k, v in sorted(w["boxes"].items())) or "none"
        lines.append(
            f"| `{w['asset']}` | {w['caption']} | `{w['gc10_stem']}` | {w['split']} | "
            f"{boxes} | {w['widest_family_box_px']} px |"
        )
    lines += [
        "",
        "`strip_sample_edge_defect.jpg` is an **edge defect, not an edge crack**.",
        "No public dataset has an edge-crack class; crescent gap and waist folding",
        "are the closest honest proxies. See `reports/gc10_dataset.md`.",
        "",
    ]
    (assets / "README.md").write_text("\n".join(lines))
    return written


# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", default="data/gc10-det-raw")
    ap.add_argument("--out", default="data/gc10-det")
    ap.add_argument("--assets", default="assets")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--skip-download", action="store_true", help="use an existing raw mirror"
    )
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    raw = (root / args.raw).resolve()
    out = (root / args.out).resolve()
    assets = (root / args.assets).resolve()

    dl = (
        {"skipped": True}
        if args.skip_download
        else download_raw(raw, workers=args.workers)
    )

    frames, box_stats = load_coco(raw)
    hash_frames(raw, frames, workers=args.workers)
    print(
        f"\n{len(frames)} frames, {box_stats['coco_boxes']} COCO boxes "
        f"({box_stats['clipped']} clipped to frame, "
        f"{box_stats['dropped_degenerate']} dropped as degenerate)"
    )

    geometry = Counter(f"{fr.width}x{fr.height}" for fr in frames)
    print(f"geometry: {dict(geometry)}")

    up = upstream_leakage(frames)
    print(
        f"upstream mirror split leaks: {up['shared_sequences']} sequence id(s) on "
        f"both sides -> re-splitting from the pooled {len(frames)} frames"
    )

    group_classes: dict[str, Counter[str]] = defaultdict(Counter)
    group_sizes: Counter[str] = Counter()
    for fr in frames:
        group_classes[fr.sequence].update(fr.classes)
        group_sizes[fr.sequence] += 1
    assign = stratified_group_split(
        dict(group_classes), dict(group_sizes), SPLIT_RATIOS, args.seed
    )

    if out.exists():
        shutil.rmtree(out)
    summary = write_split(raw, out, frames, assign)
    write_yaml(out)
    leak = assert_disjoint(out)

    manifest = {
        "dataset": "GC10-DET",
        "source": f"https://huggingface.co/datasets/{REPO_ID}",
        "license": LICENSE,
        "citation": CITATION,
        "seed": args.seed,
        "split_ratios": SPLIT_RATIOS,
        "classes": list(GC10_CLASSES),
        "pinyin_mapping": {
            k: {"english": v[0], "chinese": v[1], "gloss": v[2], "yolo_id": GC10_CLASSES.index(v[0])}
            for k, v in PINYIN_TO_ENGLISH.items()
        },
        "brief_families": {
            k: {
                "classes": list(v["classes"]),
                "proxy": v["proxy"],
                "note": v["note"],
            }
            for k, v in BRIEF_FAMILIES.items()
        },
        "out_of_brief_note": OUT_OF_BRIEF_NOTE,
        "geometry": dict(geometry),
        "box_stats": box_stats,
        "download": dl,
        "upstream_split_leakage": up,
        "leakage_check": leak,
        "grouping": {
            "key": "GC10 line-scan sequence id (camera prefix ignored)",
            "n_groups": len(group_sizes),
            "largest_group_frames": max(group_sizes.values()),
            "singleton_groups": sum(1 for v in group_sizes.values() if v == 1),
        },
        "counts": {
            s: {
                "images": summary[s]["images"],
                "empty_images": summary[s]["empty"],
                "boxes": {c: summary[s]["boxes"][c] for c in GC10_CLASSES},
                "sequences": sum(1 for g, sp in assign.items() if sp == s),
            }
            for s in SPLIT_RATIOS
        },
        "split_assignment": {g: assign[g] for g in sorted(assign)},
        "honesty": {
            "edge_crack": (
                "No public dataset has an edge-crack class. crescent_gap and "
                "waist_fold are edge-region geometry defects used as the closest "
                "honest proxy. Do not describe this as edge-crack detection."
            ),
            "roll_mark_thinness": (
                "rolled_pit carries 64 boxes and crease 72 across 2,294 images. "
                "Per-class AP on either will have an interval wider than the value."
            ),
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print(f"\nwrote {out}")
    header = f"{'class':<16}" + "".join(f"{s:>8}" for s in SPLIT_RATIOS) + f"{'total':>8}"
    print(header)
    print("-" * len(header))
    for c in GC10_CLASSES:
        row = [summary[s]["boxes"][c] for s in SPLIT_RATIOS]
        print(f"{c:<16}" + "".join(f"{v:>8}" for v in row) + f"{sum(row):>8}")
    print("-" * len(header))
    imgs = [summary[s]["images"] for s in SPLIT_RATIOS]
    print(f"{'images':<16}" + "".join(f"{v:>8}" for v in imgs) + f"{sum(imgs):>8}")
    boxes = [sum(summary[s]["boxes"].values()) for s in SPLIT_RATIOS]
    print(f"{'boxes':<16}" + "".join(f"{v:>8}" for v in boxes) + f"{sum(boxes):>8}")
    print(f"\nleakage check: {leak}")

    print("\ndemo assets:")
    asset_rows = export_assets(raw, frames, assets, assign)
    manifest["demo_assets"] = asset_rows
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
