"""Tests for src/prepare_severstal.py and the datasets it writes.

Unit tests run anywhere with no network. The integration tests are skipped
unless ``data/joint`` and ``data/severstal`` have actually been built, and they
re-derive the on-disk labels from scratch rather than trusting the manifest.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.prepare_severstal import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MIN_BOX_SIDE,
    MIN_FRAG_PIXELS,
    NEU_CLASS_NAMES,
    NEU_OFFSET,
    SEVERSTAL_CLASS_NAMES,
    SPLITS,
    STRIDE,
    TILE,
    Box,
    assign_splits,
    crop_boxes,
    decode_mask,
    mask_components,
    order_positive_frames,
    tile_starts,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SEV_ROOT = REPO_ROOT / "data" / "severstal"
JOINT_ROOT = REPO_ROOT / "data" / "joint"
NEU_ROOT = REPO_ROOT / "data" / "neu-det"

built = pytest.mark.skipif(
    not (JOINT_ROOT / "data.yaml").is_file() or not (SEV_ROOT / "data.yaml").is_file(),
    reason="datasets not built; run python -m src.prepare_severstal",
)


# --------------------------------------------------------------------------
# tiling geometry
# --------------------------------------------------------------------------
def test_tile_starts_cover_the_strip_exactly() -> None:
    starts = tile_starts()
    assert starts == [0, 224, 448, 672, 896, 1120, 1344]
    assert starts[-1] + TILE == FRAME_WIDTH
    assert all(b - a == STRIDE for a, b in zip(starts, starts[1:]))
    covered = np.zeros(FRAME_WIDTH, dtype=bool)
    for x0 in starts:
        covered[x0 : x0 + TILE] = True
    assert covered.all()


def test_tile_starts_appends_a_flush_right_tile_when_stride_does_not_divide() -> None:
    starts = tile_starts(width=1000, tile=256, stride=224)
    assert starts[-1] + 256 == 1000
    assert len(set(starts)) == len(starts)


# --------------------------------------------------------------------------
# box derivation
# --------------------------------------------------------------------------
def _mask_with(cls: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    m = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
    m[y0:y1, x0:x1] = cls
    return m


def test_box_is_exact_for_a_component_inside_one_tile() -> None:
    comps = mask_components(_mask_with(3, 40, 90, 500, 560))
    boxes = crop_boxes(comps, 448)
    assert boxes is not None and len(boxes) == 1
    b = boxes[0]
    assert b.cls == 3
    # local x span 52..111 inclusive, y span 40..89 inclusive
    assert b.w == pytest.approx(60 / TILE)
    assert b.h == pytest.approx(50 / TILE)
    assert b.cx == pytest.approx((52 + 111 + 1) / 2 / TILE)
    assert b.cy == pytest.approx((40 + 89 + 1) / 2 / TILE)


def test_empty_crop_returns_empty_list_not_none() -> None:
    comps = mask_components(_mask_with(1, 10, 60, 20, 80))
    assert crop_boxes(comps, 1344) == []


def test_a_defect_wider_than_the_tile_is_kept_in_every_tile_it_spans() -> None:
    comps = mask_components(_mask_with(3, 0, FRAME_HEIGHT, 0, 900))
    for x0 in (0, 224, 448):
        boxes = crop_boxes(comps, x0)
        assert boxes is not None and len(boxes) == 1
        assert boxes[0].w == pytest.approx(1.0)
        assert boxes[0].h == pytest.approx(1.0)


def test_a_sliver_smaller_than_the_threshold_drops_the_whole_crop() -> None:
    # 4 px of a component pokes into the tile starting at 448 -> below MIN_BOX_SIDE.
    comps = mask_components(_mask_with(2, 50, 200, 300, 452))
    assert crop_boxes(comps, 448) is None
    # the neighbouring tile sees it whole and keeps it
    boxes = crop_boxes(comps, 224)
    assert boxes is not None and len(boxes) == 1


def test_thin_fragment_below_the_pixel_floor_drops_the_crop() -> None:
    # 20 px wide, 20 px tall but only a 3-px-thick diagonal-ish sliver of area
    m = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
    m[0:20, 0:2] = 4
    comps = mask_components(m)
    assert 2 < MIN_BOX_SIDE
    assert crop_boxes(comps, 0) is None


def test_thresholds_are_the_documented_ones() -> None:
    assert MIN_BOX_SIDE == 8
    assert MIN_FRAG_PIXELS == 64


def test_box_line_uses_the_right_index_space() -> None:
    b = Box(cls=1, cx=0.5, cy=0.5, w=0.25, h=0.25)
    assert b.line(0).split()[0] == "0"
    assert b.line(NEU_OFFSET).split()[0] == "6"
    assert Box(cls=4, cx=0.5, cy=0.5, w=0.1, h=0.1).line(NEU_OFFSET).split()[0] == "9"


def test_decode_mask_of_a_defect_free_sample_is_all_zero() -> None:
    m = decode_mask({"image_id": "x.jpg", "ground_truth": None})
    assert m.shape == (FRAME_HEIGHT, FRAME_WIDTH)
    assert m.dtype == np.uint8
    assert int(m.sum()) == 0


# --------------------------------------------------------------------------
# selection and splitting
# --------------------------------------------------------------------------
def test_assign_splits_is_disjoint_complete_and_deterministic() -> None:
    ids = [f"f{i:04d}.jpg" for i in range(1000)]
    a = assign_splits(ids, random.Random(7))
    b = assign_splits(ids, random.Random(7))
    assert a == b
    assert set(a) == set(ids)
    counts = {s: sum(1 for v in a.values() if v == s) for s in SPLITS}
    assert counts == {"train": 800, "val": 100, "test": 100}
    groups = {s: {k for k, v in a.items() if v == s} for s in SPLITS}
    assert not groups["train"] & groups["val"]
    assert not groups["train"] & groups["test"]
    assert not groups["val"] & groups["test"]


def test_order_positive_frames_front_loads_rare_classes() -> None:
    frames = [{"image_id": f"a{i}.jpg", "defect_classes": [3]} for i in range(100)]
    frames += [{"image_id": f"b{i}.jpg", "defect_classes": [2]} for i in range(5)]
    ordered = order_positive_frames(frames, random.Random(1))
    assert len(ordered) == len(frames)
    assert {f["image_id"] for f in ordered} == {f["image_id"] for f in frames}
    head = ordered[:10]
    assert sum(1 for f in head if f["defect_classes"] == [2]) == 5


# --------------------------------------------------------------------------
# integration: what actually landed on disk
# --------------------------------------------------------------------------
@built
def test_joint_yaml_lists_ten_classes_with_neu_indices_unchanged() -> None:
    d = yaml.safe_load((JOINT_ROOT / "data.yaml").read_text())
    assert d["nc"] == 10
    names = [d["names"][i] for i in range(10)]
    assert names[:6] == list(NEU_CLASS_NAMES)
    assert names[6:] == list(SEVERSTAL_CLASS_NAMES)
    neu = yaml.safe_load((NEU_ROOT / "data.yaml").read_text())
    assert [neu["names"][i] for i in range(6)] == names[:6]


@built
def test_severstal_yaml_lists_four_classes() -> None:
    d = yaml.safe_load((SEV_ROOT / "data.yaml").read_text())
    assert d["nc"] == 4
    assert [d["names"][i] for i in range(4)] == list(SEVERSTAL_CLASS_NAMES)


@built
def test_every_neu_image_survives_into_joint_in_its_original_split() -> None:
    for split in SPLITS:
        src = {p.name for p in (NEU_ROOT / split / "images").glob("*.jpg")}
        dst = {p.name for p in (JOINT_ROOT / split / "images").glob("*.jpg")}
        assert src <= dst, f"{split}: {len(src - dst)} NEU images missing from joint"
        for name in sorted(src)[:25]:
            a = (NEU_ROOT / split / "labels" / f"{Path(name).stem}.txt").read_text()
            b = (JOINT_ROOT / split / "labels" / f"{Path(name).stem}.txt").read_text()
            assert a == b, f"{name}: NEU label was rewritten"


@built
def test_class_indices_stay_inside_their_namespace() -> None:
    for split in SPLITS:
        for lbl in (SEV_ROOT / split / "labels").glob("sev_*.txt"):
            for line in lbl.read_text().splitlines():
                assert 0 <= int(line.split()[0]) <= 3
        for lbl in (JOINT_ROOT / split / "labels").glob("sev_*.txt"):
            for line in lbl.read_text().splitlines():
                assert 6 <= int(line.split()[0]) <= 9
        for lbl in (JOINT_ROOT / split / "labels").glob("*.txt"):
            if lbl.name.startswith("sev_"):
                continue
            for line in lbl.read_text().splitlines():
                assert 0 <= int(line.split()[0]) <= 5


@built
def test_yolo_boxes_are_normalised_and_in_range() -> None:
    n = 0
    for split in SPLITS:
        for lbl in (JOINT_ROOT / split / "labels").glob("*.txt"):
            for line in lbl.read_text().splitlines():
                _, cx, cy, w, h = line.split()
                cx, cy, w, h = float(cx), float(cy), float(w), float(h)
                assert 0.0 < w <= 1.0 and 0.0 < h <= 1.0
                assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0
                assert cx - w / 2 >= -1e-6 and cx + w / 2 <= 1 + 1e-6
                assert cy - h / 2 >= -1e-6 and cy + h / 2 <= 1 + 1e-6
                n += 1
    assert n > 5000


@built
def test_severstal_crops_are_square_tiles() -> None:
    import cv2

    for split in SPLITS:
        paths = sorted((SEV_ROOT / split / "images").glob("sev_*.jpg"))
        assert paths
        for p in paths[::97]:
            assert cv2.imread(str(p)).shape == (TILE, TILE, 3)


@built
def test_no_source_frame_straddles_a_split() -> None:
    def sources(root: Path, split: str) -> set[str]:
        return {p.stem.split("_")[1] for p in (root / split / "images").glob("sev_*.jpg")}

    for root in (SEV_ROOT, JOINT_ROOT):
        tr, va, te = (sources(root, s) for s in SPLITS)
        assert not tr & va and not tr & te and not va & te


@built
def test_split_content_hashes_are_disjoint() -> None:
    import hashlib

    import cv2

    def hashes(root: Path, split: str) -> set[str]:
        out = set()
        for p in sorted((root / split / "images").glob("*.jpg")):
            a = cv2.imread(str(p))
            out.add(hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
        return out

    for root in (SEV_ROOT, JOINT_ROOT):
        tr, va, te = (hashes(root, s) for s in SPLITS)
        assert not tr & va, f"{root.name}: train/val share image content"
        assert not tr & te, f"{root.name}: train/test share image content"
        assert not va & te, f"{root.name}: val/test share image content"


@built
def test_background_images_exist_in_every_split_and_are_truly_empty() -> None:
    for root in (SEV_ROOT, JOINT_ROOT):
        for split in SPLITS:
            empties = [
                p
                for p in (root / split / "labels").glob("sev_*.txt")
                if not p.read_text().strip()
            ]
            assert len(empties) >= 100, f"{root.name}/{split}: only {len(empties)} backgrounds"
            for p in empties:
                assert (root / split / "images" / f"{p.stem}.jpg").is_file()


@built
def test_manifest_records_a_passing_leakage_check() -> None:
    m = json.loads((JOINT_ROOT / "manifest.json").read_text())
    assert m["severstal"]["leakage"]["ok"] is True
    assert m["joint"]["leakage"]["ok"] is True
    assert m["neu_det_leakage"]["ok"] is True
    assert m["stats"]["negatives_pixel_verified"] == m["selection"]["negative_crops"]
    assert m["stats"]["failures"] == []
    total = sum(m["joint"]["counts"][s]["images"] for s in SPLITS)
    assert 6000 <= total <= 9000, f"joint dataset is {total} images, outside the training budget"


@built
def test_manifest_records_a_passing_roundtrip_verification() -> None:
    m = json.loads((JOINT_ROOT / "manifest.json").read_text())
    v = m["verification"]
    assert v["ok"] is True
    assert v["errors"] == []
    assert v["labelled_crops"] + v["background_crops"] == (
        m["selection"]["positive_crops"] + m["selection"]["negative_crops"]
    )
    assert v["boxes"] == sum(
        m["severstal"]["counts"][s]["total_instances"] for s in SPLITS
    )
