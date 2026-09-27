"""Build a stratified train/val/test split of NEU-DET in YOLO format.

The upstream mirror ships 1620 train / 180 test images (270 / 30 per class).
Model selection on the test set would leak, so we carve a 180-image validation
set out of train and keep the original 180 test images fully held out.

Result: 1440 train / 180 val / 180 test, 240 / 30 / 30 per class.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The class order written into data.yaml is the order the head is trained in and
# the order every downstream module decodes with, so it is imported rather than
# restated here: a second copy is a second thing that can silently disagree.
from inference import CLASS_NAMES as CLASSES  # noqa: E402  (path bootstrap must run first)

SEED = 1337
VAL_PER_CLASS = 30


def class_of(stem: str) -> str:
    """`rolled-in_scale_42` -> `rolled-in_scale`. Longest matching prefix wins."""
    matches = [c for c in CLASSES if stem.startswith(c + "_")]
    if not matches:
        raise ValueError(f"cannot infer class from {stem!r}")
    return max(matches, key=len)


def read_boxes(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cid, *coords = parts
        boxes.append((int(cid), *(float(v) for v in coords)))
    return boxes


def copy_pair(img: Path, dst_root: Path, split: str) -> None:
    lbl = img.parent.parent / "labels" / f"{img.stem}.txt"
    shutil.copy2(img, dst_root / split / "images" / img.name)
    if lbl.exists():
        shutil.copy2(lbl, dst_root / split / "labels" / lbl.name)
    else:  # background image: YOLO treats a missing label as "no objects"
        (dst_root / split / "labels" / f"{img.stem}.txt").write_text("")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/NEU-DET-raw")
    ap.add_argument("--out", default="data/neu-det")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    raw = (root / args.raw).resolve()
    out = (root / args.out).resolve()

    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val", "test"):
        for sub in ("images", "labels"):
            (out / split / sub).mkdir(parents=True, exist_ok=True)

    train_imgs = sorted((raw / "train" / "images").glob("*.jpg"))
    test_imgs = sorted((raw / "test" / "images").glob("*.jpg"))

    by_class: dict[str, list[Path]] = defaultdict(list)
    for img in train_imgs:
        by_class[class_of(img.stem)].append(img)

    rng = random.Random(SEED)
    manifest: dict[str, dict[str, list[str]]] = {"train": {}, "val": {}, "test": {}}

    for cls in CLASSES:
        pool = sorted(by_class[cls], key=lambda p: p.name)
        rng.shuffle(pool)
        val, train = pool[:VAL_PER_CLASS], pool[VAL_PER_CLASS:]
        for img in val:
            copy_pair(img, out, "val")
        for img in train:
            copy_pair(img, out, "train")
        manifest["val"][cls] = sorted(p.name for p in val)
        manifest["train"][cls] = sorted(p.name for p in train)

    test_by_class: dict[str, list[str]] = defaultdict(list)
    for img in test_imgs:
        copy_pair(img, out, "test")
        test_by_class[class_of(img.stem)].append(img.name)
    manifest["test"] = {c: sorted(v) for c, v in test_by_class.items()}

    # Leakage guard: the same image must never appear in two splits.
    seen: dict[str, str] = {}
    for split in ("train", "val", "test"):
        for img in (out / split / "images").glob("*.jpg"):
            digest = hashlib.md5(img.read_bytes()).hexdigest()
            if digest in seen and seen[digest] != split:
                raise SystemExit(
                    f"leak: {img.name} duplicates an image in {seen[digest]}"
                )
            seen[digest] = split

    yaml_text = (
        f"# NEU-DET hot-rolled steel strip surface defects\n"
        f"path: {out}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"test: test/images\n"
        f"nc: {len(CLASSES)}\n"
        f"names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES))
    )
    (out / "data.yaml").write_text(yaml_text)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"wrote {out}")
    for split in ("train", "val", "test"):
        imgs = list((out / split / "images").glob("*.jpg"))
        boxes = Counter()
        empty = 0
        for img in imgs:
            b = read_boxes(out / split / "labels" / f"{img.stem}.txt")
            if not b:
                empty += 1
            for cid, *_ in b:
                boxes[CLASSES[cid]] += 1
        total = sum(boxes.values())
        print(f"\n{split}: {len(imgs)} images, {total} boxes, {empty} unlabelled")
        for c in CLASSES:
            print(f"    {c:<18} {boxes[c]:>5}")


if __name__ == "__main__":
    main()
