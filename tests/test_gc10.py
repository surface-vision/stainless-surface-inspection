"""Tests for the GC10-DET ingest that gives the project roll-mark and edge coverage.

Split in two halves. The first needs nothing on disk: it pins the class contract
(including the fact that no GC10 name collides with a NEU-DET name, because a
later stage merges the two into one 16-class head), the COCO->YOLO geometry, and
the group-split algorithm. The second half checks what `src/prepare_gc10.py`
actually wrote and skips cleanly when the dataset has not been built.

The leakage test is the one that matters: GC10 frames are consecutive line-scan
captures off a running strip, so a sequence id on both sides of a split is a
near-duplicate pair even though the bytes differ. Both guards are asserted --
content hash *and* sequence id.

Run with:
    .venv/bin/python -m pytest tests/test_gc10.py -v
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ.setdefault("YOLO_AUTOINSTALL", "False")

import prepare_gc10 as gc10  # noqa: E402
from inference import CLASS_NAMES as NEUDET_CLASSES  # noqa: E402

DATA = PROJECT_ROOT / "data" / "gc10-det"
ASSETS = PROJECT_ROOT / "assets"
SPLITS = ("train", "val", "test")

requires_dataset = pytest.mark.skipif(
    not (DATA / "manifest.json").is_file(),
    reason="run `.venv/bin/python src/prepare_gc10.py` to build data/gc10-det",
)


# --------------------------------------------------------------------------- #
# class contract
# --------------------------------------------------------------------------- #
def test_ten_classes_named_once_each() -> None:
    assert len(gc10.GC10_CLASSES) == 10
    assert len(set(gc10.GC10_CLASSES)) == 10


def test_class_names_are_readable_english_identifiers() -> None:
    for name in gc10.GC10_CLASSES:
        assert re.fullmatch(r"[a-z][a-z_]*[a-z]", name), name
        assert not re.search(r"\d", name), f"{name} still carries a GC10 ordinal"


def test_pinyin_mapping_covers_every_original_label_exactly_once() -> None:
    ordinals = sorted(int(k.split("_")[0]) for k in gc10.PINYIN_TO_ENGLISH)
    assert ordinals == list(range(1, 11))
    english = [v[0] for v in gc10.PINYIN_TO_ENGLISH.values()]
    assert sorted(english) == sorted(gc10.GC10_CLASSES)


def test_pinyin_mapping_records_the_chinese_and_a_gloss() -> None:
    for pinyin, (english, chinese, gloss) in gc10.PINYIN_TO_ENGLISH.items():
        assert english in gc10.GC10_CLASSES
        assert chinese and chinese != english, pinyin
        assert gloss, pinyin


def test_no_gc10_class_collides_with_a_neu_det_class() -> None:
    """A later stage merges both datasets into one 16-class head.

    A shared name there would silently fuse two different defects, which is why
    GC10's `7_yiwu` is `foreign_object` and not `inclusion`.
    """
    assert set(gc10.GC10_CLASSES).isdisjoint(NEUDET_CLASSES)
    assert len(set(gc10.GC10_CLASSES) | set(NEUDET_CLASSES)) == 16


# --------------------------------------------------------------------------- #
# the brief's four families, and the honesty around them
# --------------------------------------------------------------------------- #
def test_all_four_brief_families_are_accounted_for() -> None:
    assert set(gc10.BRIEF_FAMILIES) == {
        "scratches",
        "scale",
        "roll marks",
        "edge cracks",
    }


def test_every_family_class_is_a_real_gc10_class() -> None:
    for family, spec in gc10.BRIEF_FAMILIES.items():
        for name in spec["classes"]:
            assert name in gc10.GC10_CLASSES, f"{family} -> {name}"


def test_edge_cracks_are_declared_a_proxy_and_roll_marks_are_not() -> None:
    """The one claim that must never quietly become 'we detect edge cracks'."""
    edge = gc10.BRIEF_FAMILIES["edge cracks"]
    assert edge["proxy"] is True
    assert "PROXY" in str(edge["note"])
    assert set(edge["classes"]) == {"crescent_gap", "waist_fold"}
    assert gc10.BRIEF_FAMILIES["roll marks"]["proxy"] is False
    assert set(gc10.BRIEF_FAMILIES["roll marks"]["classes"]) == {"rolled_pit", "crease"}


def test_scratches_and_scale_are_left_to_neu_det() -> None:
    for family in ("scratches", "scale"):
        assert gc10.BRIEF_FAMILIES[family]["classes"] == ()
        assert "NEU-DET" in str(gc10.BRIEF_FAMILIES[family]["note"])


def test_the_module_docstring_states_the_proxy_limitation() -> None:
    doc = gc10.__doc__ or ""
    assert 'No public dataset has a true "edge crack" class' in doc
    assert "85 boxes in 46 images" in doc
    assert "13 boxes in\n  val and 13 in test" in doc


# --------------------------------------------------------------------------- #
# filename parsing
# --------------------------------------------------------------------------- #
def test_original_stem_strips_the_roboflow_suffix() -> None:
    assert (
        gc10.original_stem("img_02_425507000_00931_jpg.rf.2ce782b0b157.jpg")
        == "img_02_425507000_00931"
    )


@pytest.mark.parametrize(
    "stem,expected",
    [
        ("img_01_425005700_00156", "425005700"),
        ("img_08_425005700_09999", "425005700"),
        ("img_03_SIS001540_00771", "SIS001540"),
    ],
)
def test_sequence_id_ignores_the_camera_prefix(stem: str, expected: str) -> None:
    """img_01 and img_08 are two cameras on the same strip position, not two strips."""
    assert gc10.sequence_id(stem) == expected


def test_sequence_id_refuses_an_unrecognised_stem() -> None:
    with pytest.raises(ValueError):
        gc10.sequence_id("not_a_gc10_name")


# --------------------------------------------------------------------------- #
# COCO -> YOLO
# --------------------------------------------------------------------------- #
def test_coco_to_yolo_centres_and_normalises() -> None:
    assert gc10.coco_to_yolo([0, 0, 1024, 500], 2048, 1000) == (0.25, 0.25, 0.5, 0.5)
    assert gc10.coco_to_yolo([1024, 500, 1024, 500], 2048, 1000) == (
        0.75,
        0.75,
        0.5,
        0.5,
    )


def test_coco_to_yolo_clips_a_box_that_runs_off_the_frame() -> None:
    cx, cy, w, h = gc10.coco_to_yolo([-50, -50, 100, 100], 100, 100)
    assert (cx, cy, w, h) == (0.25, 0.25, 0.5, 0.5)
    cx, cy, w, h = gc10.coco_to_yolo([50, 50, 500, 500], 100, 100)
    assert (cx, cy, w, h) == (0.75, 0.75, 0.5, 0.5)


@pytest.mark.parametrize(
    "bbox", [[10, 10, 0, 50], [10, 10, 50, 0], [200, 10, 50, 50], [10, 10, 0.5, 0.5]]
)
def test_coco_to_yolo_drops_a_degenerate_box(bbox: list[float]) -> None:
    assert gc10.coco_to_yolo(bbox, 100, 100) is None


def test_coco_to_yolo_never_emits_a_coordinate_outside_the_unit_square() -> None:
    for bbox in ([-500, -500, 3000, 3000], [0, 0, 2048, 1000], [2040, 995, 20, 20]):
        out = gc10.coco_to_yolo(bbox, 2048, 1000)
        assert out is not None
        cx, cy, w, h = out
        assert 0.0 <= cx - w / 2 and cx + w / 2 <= 1.0 + 1e-9
        assert 0.0 <= cy - h / 2 and cy + h / 2 <= 1.0 + 1e-9


def test_content_hash_separates_different_bytes() -> None:
    assert gc10.content_hash(b"a") == gc10.content_hash(b"a")
    assert gc10.content_hash(b"a") != gc10.content_hash(b"b")
    assert gc10.content_hash(b"a") == hashlib.sha256(b"a").hexdigest()


# --------------------------------------------------------------------------- #
# the group split
# --------------------------------------------------------------------------- #
@pytest.fixture()
def synthetic_groups() -> tuple[dict[str, Counter], dict[str, int]]:
    """40 groups over 3 classes, one of them deliberately rare (10 groups)."""
    classes: dict[str, Counter] = {}
    sizes: dict[str, int] = {}
    for i in range(40):
        c: Counter = Counter()
        c["common"] = 1 + i % 7
        if i % 3 == 0:
            c["middling"] = 2
        if i % 4 == 0:  # the thin class, 10 groups
            c["rare"] = 1
        classes[f"g{i:03d}"] = c
        sizes[f"g{i:03d}"] = 1 + i % 5
    return classes, sizes


def test_split_assigns_every_group_exactly_once(synthetic_groups) -> None:
    classes, sizes = synthetic_groups
    assign = gc10.stratified_group_split(classes, sizes)
    assert set(assign) == set(classes)
    assert set(assign.values()) <= set(gc10.SPLIT_RATIOS)


def test_split_is_deterministic(synthetic_groups) -> None:
    classes, sizes = synthetic_groups
    a = gc10.stratified_group_split(classes, sizes)
    b = gc10.stratified_group_split(dict(reversed(list(classes.items()))), sizes)
    assert a == b


def test_a_different_seed_gives_a_different_split(synthetic_groups) -> None:
    classes, sizes = synthetic_groups
    a = gc10.stratified_group_split(classes, sizes, seed=1)
    b = gc10.stratified_group_split(classes, sizes, seed=2)
    assert a != b


def test_split_keeps_the_rare_class_in_every_split(synthetic_groups) -> None:
    classes, sizes = synthetic_groups
    assign = gc10.stratified_group_split(classes, sizes)
    per_split: dict[str, Counter] = defaultdict(Counter)
    for g, s in assign.items():
        per_split[s].update(classes[g])
    for split in gc10.SPLIT_RATIOS:
        assert per_split[split]["rare"] > 0, f"{split} has no rare-class boxes"


def test_split_holds_the_declared_ratios_to_within_five_points(
    synthetic_groups,
) -> None:
    classes, sizes = synthetic_groups
    assign = gc10.stratified_group_split(classes, sizes)
    per_split: dict[str, Counter] = defaultdict(Counter)
    for g, s in assign.items():
        per_split[s].update(classes[g])
    totals: Counter = Counter()
    for c in classes.values():
        totals.update(c)
    for cls in ("common", "middling"):  # `rare` is 10 boxes; no ratio survives that
        for split, want in gc10.SPLIT_RATIOS.items():
            got = per_split[split][cls] / totals[cls]
            assert abs(got - want) < 0.05, f"{cls} {split}: {got:.3f} vs {want}"


def test_a_group_with_no_boxes_still_lands_somewhere() -> None:
    classes = {"a": Counter({"x": 4}), "b": Counter({"x": 4}), "empty": Counter()}
    sizes = {"a": 1, "b": 1, "empty": 1}
    assign = gc10.stratified_group_split(classes, sizes)
    assert assign["empty"] in gc10.SPLIT_RATIOS


# --------------------------------------------------------------------------- #
# what was actually written
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def manifest() -> dict:
    if not (DATA / "manifest.json").is_file():
        pytest.skip("dataset not built")
    return json.loads((DATA / "manifest.json").read_text())


@requires_dataset
def test_data_yaml_agrees_with_the_class_contract() -> None:
    text = (DATA / "data.yaml").read_text()
    assert f"nc: {len(gc10.GC10_CLASSES)}" in text
    for i, name in enumerate(gc10.GC10_CLASSES):
        assert f"  {i}: {name}\n" in text


@requires_dataset
@pytest.mark.parametrize("split", SPLITS)
def test_every_image_has_a_label_file(split: str) -> None:
    images = sorted((DATA / split / "images").glob("*.jpg"))
    assert images, f"{split} is empty"
    for img in images:
        assert (DATA / split / "labels" / f"{img.stem}.txt").is_file(), img.name


@requires_dataset
@pytest.mark.parametrize("split", SPLITS)
def test_labels_are_well_formed_yolo_inside_the_class_contract(split: str) -> None:
    for lbl in (DATA / split / "labels").glob("*.txt"):
        for line in lbl.read_text().splitlines():
            parts = line.split()
            assert len(parts) == 5, f"{lbl.name}: {line!r}"
            cid = int(parts[0])
            assert 0 <= cid < len(gc10.GC10_CLASSES), f"{lbl.name}: class {cid}"
            cx, cy, w, h = (float(v) for v in parts[1:])
            assert 0.0 < w <= 1.0 and 0.0 < h <= 1.0, f"{lbl.name}: {line!r}"
            assert -1e-6 <= cx - w / 2 and cx + w / 2 <= 1.0 + 1e-6, lbl.name
            assert -1e-6 <= cy - h / 2 and cy + h / 2 <= 1.0 + 1e-6, lbl.name


@requires_dataset
def test_no_image_content_appears_in_two_splits() -> None:
    seen: dict[str, str] = {}
    for split in SPLITS:
        for img in (DATA / split / "images").glob("*.jpg"):
            digest = gc10.content_hash(img.read_bytes())
            assert seen.get(digest, split) == split, f"{img.name} leaks from {seen[digest]}"
            seen[digest] = split


@requires_dataset
def test_no_line_scan_sequence_appears_in_two_splits() -> None:
    """The guard that content hashing cannot give you: near-duplicate frames."""
    owner: dict[str, str] = {}
    for split in SPLITS:
        for img in (DATA / split / "images").glob("*.jpg"):
            seq = gc10.sequence_id(gc10.original_stem(img.name))
            assert owner.get(seq, split) == split, f"sequence {seq} leaks from {owner[seq]}"
            owner[seq] = split


@requires_dataset
def test_manifest_counts_match_the_files_on_disk(manifest: dict) -> None:
    for split in SPLITS:
        on_disk = len(list((DATA / split / "images").glob("*.jpg")))
        assert on_disk == manifest["counts"][split]["images"], split
        boxes = Counter()
        for lbl in (DATA / split / "labels").glob("*.txt"):
            for line in lbl.read_text().splitlines():
                boxes[gc10.GC10_CLASSES[int(line.split()[0])]] += 1
        assert dict(boxes) == {
            k: v for k, v in manifest["counts"][split]["boxes"].items() if v
        }, split


@requires_dataset
def test_the_split_is_a_partition_of_the_whole_mirror(manifest: dict) -> None:
    total = sum(manifest["counts"][s]["images"] for s in SPLITS)
    assert total == sum(manifest["geometry"].values())


@requires_dataset
def test_every_frame_is_wide_strip_geometry(manifest: dict) -> None:
    assert set(manifest["geometry"]) == {"2048x1000"}


@requires_dataset
def test_the_manifest_records_the_upstream_split_leak(manifest: dict) -> None:
    """Why we re-split instead of using the mirror's own train/test."""
    assert manifest["upstream_split_leakage"]["shared_sequences"] > 0
    assert manifest["leakage_check"]["cross_split_duplicates"] == 0


@requires_dataset
def test_the_manifest_carries_the_pinyin_mapping_and_the_honesty_note(
    manifest: dict,
) -> None:
    assert set(manifest["pinyin_mapping"]) == set(gc10.PINYIN_TO_ENGLISH)
    for pinyin, row in manifest["pinyin_mapping"].items():
        assert gc10.GC10_CLASSES[row["yolo_id"]] == row["english"]
        assert row["chinese"] == gc10.PINYIN_TO_ENGLISH[pinyin][1]
    assert "No public dataset has an edge-crack class" in manifest["honesty"]["edge_crack"]
    assert manifest["license"] == "CC BY 4.0"


# --------------------------------------------------------------------------- #
# demo assets
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", [p[0] for p in gc10.ASSET_PICKS])
def test_each_named_demo_asset_exists_and_is_a_wide_strip(name: str) -> None:
    path = ASSETS / name
    if not path.is_file():
        pytest.skip("assets not exported yet")
    from PIL import Image

    with Image.open(path) as im:
        width, height = im.size
    assert width >= 2048, f"{name} is {width} px wide; tiling needs a wide frame"
    assert width > height


def test_the_assets_note_carries_the_licence_and_the_proxy_caveat() -> None:
    path = ASSETS / "README.md"
    if not path.is_file():
        pytest.skip("assets not exported yet")
    text = path.read_text()
    assert "CC BY 4.0" in text
    assert "Sensors 2020" in text  # attribution is a licence condition, not a nicety
    assert "not an edge crack" in text


@pytest.mark.parametrize("name", [p[0] for p in gc10.ASSET_PICKS])
def test_the_wide_assets_actually_exercise_the_tiling_path(name: str) -> None:
    """The whole reason these files are in the repo.

    `predict_tiled` falls through to `predict` when the frame fits one tile, and
    every image the project shipped before these did. Mirrors the stride the
    method computes at its default 0.2 overlap.
    """
    from inference import DEFAULT_IMGSZ, _tile_origins

    path = ASSETS / name
    if not path.is_file():
        pytest.skip("assets not exported yet")
    from PIL import Image

    with Image.open(path) as im:
        width, height = im.size
    stride = max(1, round(DEFAULT_IMGSZ * 0.8))
    tiles = len(_tile_origins(width, DEFAULT_IMGSZ, stride)) * len(
        _tile_origins(height, DEFAULT_IMGSZ, stride)
    )
    assert tiles > 1, "a single tile covers the frame; tiling is not exercised"
