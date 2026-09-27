"""Tests for src/train_joint.py -- the joint NEU-DET + Severstal training path.

The integration tests read the real `data/joint` and `data/joint_xdsafe` trees
and skip themselves when those are absent, so the file is still runnable on a
checkout that has not built the datasets.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import train_joint as tj  # noqa: E402
from src.cross_domain import is_holdout  # noqa: E402

JOINT = ROOT / "data" / "joint"
SAFE = ROOT / "data" / "joint_xdsafe"
BASE_WEIGHTS = ROOT / "models" / "yolov8n_neudet" / "weights" / "best.pt"

needs_joint = pytest.mark.skipif(not JOINT.is_dir(), reason="data/joint not built")
needs_safe = pytest.mark.skipif(not SAFE.is_dir(), reason="data/joint_xdsafe not built")
needs_weights = pytest.mark.skipif(not BASE_WEIGHTS.is_file(), reason="shipped weights missing")


@pytest.fixture(autouse=True)
def _restore_inference_globals():
    """`register_severstal_classes` mutates module globals on purpose.

    It is process-local by design, which makes it test pollution unless every
    loaded copy of `src/inference.py` is snapshotted and put back -- and there can
    be two copies (see `_loaded_inference_modules`). Restoring only the one this
    file imported leaves the other dirty and breaks `tests/test_smoke.py`, which
    asserts the six-class contract.
    """
    modules = tj._loaded_inference_modules()
    saved = [
        (m, list(m.CLASS_NAMES), dict(m.DEFECT_INFO), dict(m.CLASS_COLORS))
        for m in modules
    ]
    try:
        yield
    finally:
        for module, names, info, colors in saved:
            module.CLASS_NAMES[:] = names
            module.DEFECT_INFO.clear()
            module.DEFECT_INFO.update(info)
            module.CLASS_COLORS.clear()
            module.CLASS_COLORS.update(colors)



# ---------------------------------------------------------------- name parsing


@pytest.mark.parametrize(
    "stem,expected",
    [
        ("sev_001d3d093_x0224", "001d3d093"),
        ("sev_fff0295e1_x0000", "fff0295e1"),
        ("crazing_1", None),
        ("rolled-in_scale_120", None),
        ("sev_notahexid_x0000", None),
    ],
)
def test_source_frame_id(stem: str, expected: str | None) -> None:
    assert tj.source_frame_id(stem) == expected
    assert tj.is_severstal(stem) is (expected is not None)


def test_crop_offset() -> None:
    assert tj.crop_offset("sev_001d3d093_x1344") == 1344
    assert tj.crop_offset("sev_001d3d093_x0000") == 0
    assert tj.crop_offset("crazing_1") is None


# ------------------------------------------------------------------- io helpers


def test_read_label_variants(tmp_path: Path) -> None:
    missing = tmp_path / "absent.txt"
    assert tj.read_label(missing) == []

    blank = tmp_path / "blank.txt"
    blank.write_text("\n  \n")
    assert tj.read_label(blank) == []

    real = tmp_path / "real.txt"
    real.write_text("3 0.5 0.25 0.1 0.2\n9 0.1 0.1 0.05 0.05\n")
    rows = tj.read_label(real)
    assert rows[0] == (3, 0.5, 0.25, 0.1, 0.2)
    assert rows[1][0] == 9


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    import hashlib

    path = tmp_path / "blob.bin"
    payload = b"steel" * 5000
    path.write_bytes(payload)
    assert tj.sha256_file(path) == hashlib.sha256(payload).hexdigest()


# ------------------------------------------------------ holdout-safe derivation


def _mini_joint(root: Path) -> Path:
    """A three-split joint dataset with hand-picked holdout / non-holdout ids."""
    import cv2

    held = [s for s in ("aaa000001", "aaa000002", "aaa000003", "aaa000004", "aaa000005",
                        "aaa000006", "aaa000007", "aaa000008") if is_holdout(s, 0.5)]
    kept = [s for s in ("bbb000001", "bbb000002", "bbb000003", "bbb000004", "bbb000005",
                        "bbb000006", "bbb000007", "bbb000008") if not is_holdout(s, 0.5)]
    assert held and kept, "test fixture needs at least one frame on each side"

    for split in ("train", "val", "test"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        img = np.full((256, 256, 3), 120, dtype=np.uint8)
        # one NEU-style image, one holdout Severstal crop, one non-holdout one
        cv2.imwrite(str(root / split / "images" / "crazing_1.jpg"), img)
        (root / split / "labels" / "crazing_1.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        cv2.imwrite(str(root / split / "images" / f"sev_{held[0]}_x0000.jpg"), img)
        (root / split / "labels" / f"sev_{held[0]}_x0000.txt").write_text("8 0.5 0.5 0.2 0.2\n")
        cv2.imwrite(str(root / split / "images" / f"sev_{kept[0]}_x0224.jpg"), img)
        (root / split / "labels" / f"sev_{kept[0]}_x0224.txt").write_text("")
    return root


def test_build_holdout_safe_filters_train_and_val_only(tmp_path: Path) -> None:
    joint = _mini_joint(tmp_path / "joint")
    out = tmp_path / "safe"
    stats = tj.build_holdout_safe(joint, out)

    assert stats["splits"]["train"]["images_dropped"] == 1
    assert stats["splits"]["val"]["images_dropped"] == 1
    assert stats["splits"]["test"]["images_dropped"] == 0
    assert stats["splits"]["test"]["filtered"] is False
    assert stats["holdout_seed"] == tj.HOLDOUT_SEED

    for split in ("train", "val"):
        stems = {p.stem for p in tj.images_in(out / split / "images")}
        for stem in stems:
            fid = tj.source_frame_id(stem)
            assert fid is None or not is_holdout(fid, 0.5)
    assert len(tj.images_in(out / "test" / "images")) == 3


def test_build_holdout_safe_writes_symlinks_and_empty_labels(tmp_path: Path) -> None:
    joint = _mini_joint(tmp_path / "joint")
    out = tmp_path / "safe"
    tj.build_holdout_safe(joint, out)

    for img in tj.images_in(out / "train" / "images"):
        assert img.is_symlink()
        assert img.resolve().is_file()
        label = out / "train" / "labels" / f"{img.stem}.txt"
        assert label.is_file(), "every image must have a label file, empty or not"
    backgrounds = [
        p for p in (out / "train" / "labels").iterdir() if p.read_text().strip() == ""
    ]
    assert backgrounds, "the background crop must survive as an explicit empty label"


def test_build_holdout_safe_yaml_and_eval_views(tmp_path: Path) -> None:
    joint = _mini_joint(tmp_path / "joint")
    out = tmp_path / "safe"
    tj.build_holdout_safe(joint, out)

    cfg = yaml.safe_load((out / "data.yaml").read_text())
    assert cfg["nc"] == 10
    assert [cfg["names"][i] for i in range(10)] == list(tj.JOINT_NAMES)
    assert cfg["names"][0] == "crazing" and cfg["names"][6] == "severstal_1"

    for view in ("eval_neu_test", "eval_sev_test"):
        view_cfg = yaml.safe_load((out / f"{view}.yaml").read_text())
        assert view_cfg["nc"] == 10
        assert view_cfg["val"] == f"{view}/images"
    assert len(tj.images_in(out / "eval_neu_test" / "images")) == 1
    assert len(tj.images_in(out / "eval_sev_test" / "images")) == 2


def test_build_holdout_safe_is_idempotent(tmp_path: Path) -> None:
    joint = _mini_joint(tmp_path / "joint")
    out = tmp_path / "safe"
    first = tj.build_holdout_safe(joint, out)
    second = tj.build_holdout_safe(joint, out)
    assert first == second


# ------------------------------------------------------------------- warm start


@needs_weights
def test_warmstart_widens_head_and_keeps_every_learned_tensor(tmp_path: Path) -> None:
    import torch

    out = tmp_path / "warm.pt"
    stats = tj.build_warmstart_checkpoint(tj.BASE_WEIGHTS, out)
    assert stats["old_nc"] == 6 and stats["new_nc"] == 10
    assert stats["class_channels_transplanted"] == [0, 1, 2, 3, 4, 5]
    assert stats["class_channels_initialised"] == [6, 7, 8, 9]
    assert stats["transfer_fraction"] > 0.98

    old = torch.load(str(tj.BASE_WEIGHTS), map_location="cpu", weights_only=False)
    new = torch.load(str(out), map_location="cpu", weights_only=False)
    old_model = (old.get("ema") or old["model"]).float()
    new_model = new["model"].float()
    assert new_model.nc == 10
    assert [new_model.names[i] for i in range(10)] == list(tj.JOINT_NAMES)

    # Every classification convolution keeps the shipped weights in channels 0-5.
    for seq_old, seq_new in zip(old_model.model[-1].cv3, new_model.model[-1].cv3):
        assert torch.allclose(seq_new[-1].weight[:6].float(), seq_old[-1].weight.float(), atol=1e-3)
        assert torch.allclose(seq_new[-1].bias[:6].float(), seq_old[-1].bias.float(), atol=1e-3)
        assert seq_new[-1].weight.shape[0] == 10

    # And nothing else moved.
    old_state = old_model.state_dict()
    new_state = new_model.state_dict()
    same_shape = [k for k in old_state if k in new_state and old_state[k].shape == new_state[k].shape]
    assert len(same_shape) >= 340
    for key in same_shape:
        assert torch.allclose(old_state[key].float(), new_state[key].float(), atol=1e-3), key


@needs_weights
def test_warmstart_reproduces_shipped_detections(tmp_path: Path) -> None:
    """The transplant is only real if the widened model predicts the same boxes."""
    import cv2
    from ultralytics import YOLO

    sample = sorted((ROOT / "data" / "neu-det" / "test" / "images").glob("*.jpg"))
    if not sample:
        pytest.skip("NEU-DET test split not present")
    image = cv2.imread(str(sample[0]), cv2.IMREAD_COLOR)

    out = tmp_path / "warm.pt"
    tj.build_warmstart_checkpoint(tj.BASE_WEIGHTS, out)
    old = YOLO(str(tj.BASE_WEIGHTS)).predict(image, imgsz=256, conf=0.25, device="cpu", verbose=False)
    new = YOLO(str(out)).predict(image, imgsz=256, conf=0.25, device="cpu", verbose=False)

    a = old[0].boxes.data.numpy()
    b = new[0].boxes.data.numpy()
    b = b[b[:, 5] < 6]  # the four new classes did not exist in the shipped head
    assert a.shape == b.shape
    np.testing.assert_allclose(np.sort(a[:, 4]), np.sort(b[:, 4]), atol=2e-3)


# -------------------------------------------------------------- validation sweep


def _frame(labelled: bool, domain: str = "severstal") -> tj.ValFrame:
    boxes = np.array([[10.0, 10.0, 50.0, 50.0]]) if labelled else np.zeros((0, 4))
    classes = np.array([8]) if labelled else np.zeros((0,), dtype=np.int64)
    return tj.ValFrame(Path("x.jpg"), domain, labelled, boxes, classes)


def test_sweep_validation_detection_and_false_alarm() -> None:
    frames = [_frame(True), _frame(True), _frame(False), _frame(False)]
    hit = (np.array([[10.0, 10.0, 50.0, 50.0]]), np.array([0.6]), np.array([8]))
    miss = (np.array([[200.0, 200.0, 240.0, 240.0]]), np.array([0.6]), np.array([8]))
    fp = (np.array([[10.0, 10.0, 50.0, 50.0]]), np.array([0.3]), np.array([2]))
    none = (np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=np.int64))

    rows = tj.sweep_validation(frames, [hit, miss, fp, none], [0.1, 0.5, 0.9])
    by_t = {r["threshold"]: r for r in rows}
    # 0.1: one labelled frame localised out of two; one clean frame carries a box
    assert by_t[0.1]["defect_detection_rate"] == pytest.approx(0.5)
    assert by_t[0.1]["clean_frame_false_alarm_rate"] == pytest.approx(0.5)
    assert by_t[0.1]["boxes_per_clean_frame"] == pytest.approx(0.5)
    # 0.5: the 0.3-confidence false positive is gone
    assert by_t[0.5]["clean_frame_false_alarm_rate"] == pytest.approx(0.0)
    assert by_t[0.5]["defect_detection_rate"] == pytest.approx(0.5)
    # 0.9: nothing survives
    assert by_t[0.9]["defect_detection_rate"] == pytest.approx(0.0)


def test_sweep_validation_wrong_class_is_not_a_detection() -> None:
    frames = [_frame(True)]
    wrong_class = (np.array([[10.0, 10.0, 50.0, 50.0]]), np.array([0.9]), np.array([1]))
    rows = tj.sweep_validation(frames, [wrong_class], [0.1])
    assert rows[0]["defect_detection_rate"] == 0.0


def test_sweep_validation_reports_per_domain() -> None:
    frames = [_frame(True, "neu"), _frame(True, "severstal")]
    hit = (np.array([[10.0, 10.0, 50.0, 50.0]]), np.array([0.9]), np.array([8]))
    none = (np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=np.int64))
    rows = tj.sweep_validation(frames, [hit, none], [0.1])
    assert rows[0]["defect_detection_rate_neu"] == pytest.approx(1.0)
    assert rows[0]["defect_detection_rate_severstal"] == pytest.approx(0.0)


def test_sweep_validation_thresholds_are_monotone() -> None:
    frames = [_frame(True) for _ in range(3)] + [_frame(False) for _ in range(3)]
    cached = [
        (np.array([[10.0, 10.0, 50.0, 50.0]]), np.array([c]), np.array([8]))
        for c in (0.2, 0.5, 0.8)
    ] + [
        (np.array([[1.0, 1.0, 9.0, 9.0]]), np.array([c]), np.array([3]))
        for c in (0.2, 0.5, 0.8)
    ]
    rows = tj.sweep_validation(frames, cached, [0.1, 0.3, 0.6, 0.9])
    d = [r["defect_detection_rate"] for r in rows]
    f = [r["clean_frame_false_alarm_rate"] for r in rows]
    assert d == sorted(d, reverse=True)
    assert f == sorted(f, reverse=True)


# ------------------------------------------------------------- operating point


def _rows(pairs: list[tuple[float, float, float]]) -> list[dict[str, float]]:
    return [
        {
            "threshold": t,
            "defect_detection_rate": d,
            "clean_frame_false_alarm_rate": f,
            "boxes_per_clean_frame": 0.0,
            "defect_detection_rate_neu": d,
            "defect_detection_rate_severstal": d,
            "n_labelled": 100,
            "n_clean": 100,
        }
        for t, d, f in pairs
    ]


def test_pick_operating_point_respects_the_detection_floor() -> None:
    rows = _rows([(0.1, 0.99, 0.60), (0.5, 0.95, 0.10), (0.9, 0.40, 0.00)])
    picked = tj.pick_operating_point(rows)
    assert picked["detection_floor_met"] is True
    assert picked["chosen"]["threshold"] == 0.5
    # Ignoring the floor, the cheapest point may be somewhere else entirely.
    assert picked["unconstrained"]["threshold"] in (0.5, 0.9)


def test_pick_operating_point_reports_an_unreachable_floor() -> None:
    rows = _rows([(0.1, 0.50, 0.60), (0.5, 0.30, 0.10), (0.9, 0.10, 0.00)])
    picked = tj.pick_operating_point(rows)
    assert picked["detection_floor_met"] is False


def test_pick_operating_point_uses_the_shipped_cost_model() -> None:
    from src.false_alarm import COST_MODEL

    rows = _rows([(0.1, 0.99, 0.60), (0.5, 0.95, 0.10)])
    picked = tj.pick_operating_point(rows)
    assert picked["cost_model"]["miss_cost_ratio"] == COST_MODEL["miss_cost_ratio"]
    assert picked["cost_model"]["defect_frame_prevalence"] == COST_MODEL["defect_frame_prevalence"]
    assert picked["cost_model"]["min_detection_rate"] == COST_MODEL["min_detection_rate"]
    expected = (
        COST_MODEL["defect_frame_prevalence"] * COST_MODEL["miss_cost_ratio"] * (1 - 0.95)
        + (1 - COST_MODEL["defect_frame_prevalence"]) * 0.10
    )
    assert picked["chosen"]["expected_cost_frame_basis"] == pytest.approx(expected, abs=5e-5)


def test_pick_operating_point_sensitivity_covers_every_ratio() -> None:
    from src.false_alarm import COST_MODEL

    rows = _rows([(0.1, 0.99, 0.60), (0.3, 0.97, 0.30), (0.5, 0.95, 0.10), (0.9, 0.40, 0.00)])
    picked = tj.pick_operating_point(rows)
    ratios = [row["miss_cost_ratio"] for row in picked["sensitivity"]]
    assert ratios == [float(r) for r in COST_MODEL["sensitivity_ratios"]]
    # A higher price on a miss can never recommend a stricter threshold.
    chosen = [row["constrained_threshold"] for row in picked["sensitivity"]]
    assert chosen == sorted(chosen, reverse=True)


# ------------------------------------------------------------- label loading


def test_load_val_frames_converts_boxes_to_pixels(tmp_path: Path) -> None:
    import cv2

    split = tmp_path / "val"
    (split / "images").mkdir(parents=True)
    (split / "labels").mkdir(parents=True)
    cv2.imwrite(str(split / "images" / "sev_aaa111222_x0000.jpg"),
                np.full((256, 256, 3), 90, dtype=np.uint8))
    (split / "labels" / "sev_aaa111222_x0000.txt").write_text("7 0.5 0.5 0.25 0.5\n")
    cv2.imwrite(str(split / "images" / "crazing_9.jpg"),
                np.full((200, 200, 3), 90, dtype=np.uint8))
    (split / "labels" / "crazing_9.txt").write_text("")

    frames = {f.path.stem: f for f in tj.load_val_frames(split)}
    sev = frames["sev_aaa111222_x0000"]
    assert sev.domain == "severstal" and sev.labelled
    np.testing.assert_allclose(sev.boxes[0], [96.0, 64.0, 160.0, 192.0])
    assert sev.classes[0] == 7
    neu = frames["crazing_9"]
    assert neu.domain == "neu" and not neu.labelled and neu.boxes.shape == (0, 4)


# ------------------------------------------------------------------ integration


@needs_joint
def test_real_joint_dataset_passes_every_structural_check() -> None:
    checks, facts = tj.verify_dataset(fidelity_sample=12)
    named = {c.name: c for c in checks}
    for name in (
        "class vocabulary",
        "NEU-DET indices 0-5 unchanged",
        "counts recounted, train",
        "counts recounted, val",
        "counts recounted, test",
        "label geometry in range",
        "class ids stay in their own domain",
        "NEU-DET images/labels byte-identical and in the same split",
        "splits are content-hash disjoint",
        "no Severstal source frame straddles a split",
        "crops are the source frame's own pixels",
    ):
        assert named[name].ok, named[name].line()
    assert facts["neu_images_per_split"] == {"train": 1440, "val": 180, "test": 180}
    assert facts["per_split"]["train"]["images"] == 6234


@needs_joint
def test_real_joint_dataset_leaks_into_the_cross_domain_holdout() -> None:
    """The reason `build` exists. If this ever starts failing, drop the filter."""
    checks, facts = tj.verify_dataset(fidelity_sample=1)
    holdout = facts["cross_domain_holdout"]
    assert holdout["train"]["crops_on_scored_side"] > 0
    named = {c.name: c for c in checks}
    assert named["data/joint respects the cross_domain holdout rule"].ok is False


@needs_safe
def test_holdout_safe_tree_trains_on_nothing_the_harness_scores() -> None:
    for split in ("train", "val"):
        for image in tj.images_in(SAFE / split / "images"):
            fid = tj.source_frame_id(image.stem)
            if fid is not None:
                assert not is_holdout(fid, 0.5), f"{image.name} is scored by cross_domain.py"


@needs_safe
def test_holdout_safe_tree_keeps_every_neu_det_image() -> None:
    manifest = json.loads((SAFE / "manifest.json").read_text())
    assert manifest["splits"]["train"]["neu_kept"] == 1440
    assert manifest["splits"]["val"]["neu_kept"] == 180
    assert manifest["splits"]["test"]["images_kept"] == 768
    assert len(tj.images_in(SAFE / "eval_neu_test" / "images")) == 180
    assert len(tj.images_in(SAFE / "eval_sev_test" / "images")) == 588


@needs_safe
def test_holdout_safe_eval_views_partition_the_test_split() -> None:
    neu = {p.stem for p in tj.images_in(SAFE / "eval_neu_test" / "images")}
    sev = {p.stem for p in tj.images_in(SAFE / "eval_sev_test" / "images")}
    whole = {p.stem for p in tj.images_in(SAFE / "test" / "images")}
    assert neu.isdisjoint(sev)
    assert neu | sev == whole


def test_augmentation_recipe_matches_train_detector() -> None:
    """The joint run must differ from the shipped run in data, not in recipe."""
    source = (ROOT / "src" / "train_detector.py").read_text()
    for key, value in tj.AUGMENTATION.items():
        needle = f"{key}={value}," if not isinstance(value, bool) else f"{key}={value},"
        assert needle in source, f"{key}={value} is not the shipped setting"


# ------------------------------------------- runtime registration of new classes


def test_register_severstal_classes_patches_every_inference_module() -> None:
    import importlib

    inference = importlib.import_module("src.inference")
    result = tj.register_severstal_classes("medium")
    assert result["tier"] == "medium"
    assert "src.inference" in result["modules_patched"]
    assert len(inference.CLASS_NAMES) == 10
    assert inference.CLASS_NAMES[6:] == list(tj.SEVERSTAL_NAMES)
    for name in tj.SEVERSTAL_NAMES:
        assert inference.DEFECT_INFO[name]["severity"] == "medium"
        # Scoring must now work rather than raising KeyError.
        assert inference.score_detection(name, 0.9, 0.1) > 0.0
    # Every loaded copy, not just the one this file imported.
    for module in tj._loaded_inference_modules():
        assert len(module.CLASS_NAMES) == 10
        assert set(tj.SEVERSTAL_NAMES) <= set(module.DEFECT_INFO)


def test_register_severstal_classes_rejects_an_unknown_tier() -> None:
    with pytest.raises(ValueError):
        tj.register_severstal_classes("catastrophic")


def test_register_severstal_classes_is_idempotent() -> None:
    import importlib

    inference = importlib.import_module("src.inference")
    tj.register_severstal_classes("low")
    tj.register_severstal_classes("low")
    assert len(inference.CLASS_NAMES) == 10
    assert inference.CLASS_NAMES.count("severstal_1") == 1


def test_registration_does_not_leak_out_of_a_test() -> None:
    """The six-class contract the rest of the suite asserts must survive this file.

    `src/inference.py` now carries the four Severstal classes natively, so their
    presence in DEFECT_INFO is the shipped state rather than leaked test state.
    What must not leak is the shim's *mutation*: CLASS_NAMES back to six, and the
    knowledge-base entries back to the shipped wording rather than the shim's
    placeholder. That distinction is the whole point of the restore fixture, so
    it is asserted on the text rather than on mere membership.
    """
    import importlib

    inference = importlib.import_module("src.inference")
    assert len(inference.CLASS_NAMES) == 6
    assert set(tj.SEVERSTAL_NAMES) <= set(inference.DEFECT_INFO)
    for name in tj.SEVERSTAL_NAMES:
        info = inference.DEFECT_INFO[name]
        assert info["severity"] == inference.SEVERSTAL_TIER
        assert "Placeholder." not in info["action"], (
            f"{name} still carries the shim's placeholder text; the restore "
            "fixture did not put the shipped entry back"
        )


def test_the_shim_is_now_redundant_with_the_shipped_inference_module() -> None:
    """`register_severstal_classes` predates 10-class support in inference.py.

    It exists because `src/inference.py` was six-class-only and was not that
    task's file to edit. It is now, and the four classes are registered at
    import, so the shim should be deleted by whoever owns `src/train_joint.py`.
    Until then it must at least agree with the module it patches, or a joint
    evaluation run and the shipped runtime would band the same detection
    differently.
    """
    import importlib

    inference = importlib.import_module("src.inference")
    assert tj.SEVERSTAL_TIER_DEFAULT == inference.SEVERSTAL_TIER
    assert list(tj.SEVERSTAL_NAMES) == list(inference.SEVERSTAL_CLASS_NAMES)
    assert list(tj.JOINT_NAMES) == inference.JOINT_CLASS_NAMES
    # And its colours must not fight the shipped palette.
    for name, colour in tj._SEVERSTAL_COLORS.items():
        assert name in inference.CLASS_COLORS


def test_loaded_inference_modules_only_returns_the_project_module() -> None:
    mods = tj._loaded_inference_modules()
    assert mods, "src.inference must be importable"
    target = (ROOT / "src" / "inference.py").resolve()
    for module in mods:
        assert Path(module.__file__).resolve() == target
    assert len({id(m) for m in mods}) == len(mods)


# ------------------------------------------------ flag rate and per-population picks


def test_sweep_reports_flag_rate_alongside_localised_detection() -> None:
    frames = [_frame(True), _frame(False)]
    wrong_place = (np.array([[200.0, 200.0, 240.0, 240.0]]), np.array([0.9]), np.array([8]))
    none = (np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=np.int64))
    rows = tj.sweep_validation(frames, [wrong_place, none], [0.1])
    # The box is on the frame but not on the defect: flagged, not detected.
    assert rows[0]["defect_flag_rate"] == pytest.approx(1.0)
    assert rows[0]["defect_detection_rate"] == pytest.approx(0.0)


def test_pick_operating_point_honours_the_detection_key() -> None:
    rows = _rows([(0.1, 0.50, 0.60), (0.5, 0.30, 0.10)])
    for row in rows:
        row["defect_flag_rate"] = 0.99 if row["threshold"] == 0.5 else 1.0
    strict = tj.pick_operating_point(rows)
    lenient = tj.pick_operating_point(rows, detection_key="defect_flag_rate")
    assert strict["detection_floor_met"] is False
    assert lenient["detection_floor_met"] is True
    assert lenient["detection_key"] == "defect_flag_rate"
    assert lenient["chosen"]["threshold"] == 0.5
