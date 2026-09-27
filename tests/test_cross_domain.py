"""Tests for the cross-domain generalisation harness (`src/cross_domain.py`).

Everything here runs without the checkpoint and without the Severstal pull: the
statistics, the tiling, the labelling rule and the artifact plumbing are all
exercised on synthetic inputs, because a measurement harness whose arithmetic is
only ever checked by the measurement it produces is not checked at all.

The one test that touches `reports/cross_domain.json` does not assert any
measured value -- it asserts the record is internally consistent (rates match
their own counts, intervals bracket their point estimates, the sweep is monotone).
A test that pinned the false alarm rate would fail the moment the fix worked.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cross_domain as xd  # noqa: E402


# ---------------------------------------------------------------------------
# tiling
# ---------------------------------------------------------------------------


def test_a_1600px_strip_cuts_into_eight_abutting_200px_tiles() -> None:
    origins = xd.crop_origins(1600, 200)
    assert origins == [0, 200, 400, 600, 800, 1000, 1200, 1400]


def test_a_256px_height_gives_one_centred_row() -> None:
    assert xd.crop_origins(256, 200) == [28]


@pytest.mark.parametrize("extent", [200, 256, 320, 512, 1000, 1600, 2048])
@pytest.mark.parametrize("size", [100, 128, 200])
def test_no_tile_ever_runs_off_the_frame(extent: int, size: int) -> None:
    for origin in xd.crop_origins(extent, size):
        assert origin >= 0
        assert origin + size <= max(extent, size)


def test_a_frame_narrower_than_one_tile_still_yields_one_origin() -> None:
    assert xd.crop_origins(120, 200) == [0]


# ---------------------------------------------------------------------------
# the held-out split
# ---------------------------------------------------------------------------


def test_holdout_assignment_is_deterministic() -> None:
    assert [xd.is_holdout(f"frame{i}", 0.5) for i in range(50)] == [
        xd.is_holdout(f"frame{i}", 0.5) for i in range(50)
    ]


def test_holdout_fraction_is_approximately_honoured() -> None:
    stems = [f"{i:06x}" for i in range(4000)]
    share = sum(xd.is_holdout(s, 0.5) for s in stems) / len(stems)
    assert 0.47 < share < 0.53


def test_holdout_endpoints_take_everything_or_nothing() -> None:
    stems = [f"frame{i}" for i in range(200)]
    assert all(xd.is_holdout(s, 1.0) for s in stems)
    assert not any(xd.is_holdout(s, 0.0) for s in stems)


def test_holdout_is_nested_so_a_smaller_fraction_never_adds_frames() -> None:
    # Threshold on a fixed hash, so shrinking the fraction can only remove frames.
    # A training run that used --holdout-frac 0.5 is therefore still disjoint from
    # an evaluation at 0.3, which is what makes the split composable.
    stems = [f"frame{i}" for i in range(500)]
    big = {s for s in stems if xd.is_holdout(s, 0.5)}
    small = {s for s in stems if xd.is_holdout(s, 0.3)}
    assert small <= big


# ---------------------------------------------------------------------------
# rank statistics
# ---------------------------------------------------------------------------


def test_ranks_average_over_ties() -> None:
    ranks = xd._average_ranks(np.array([1.0, 2.0, 2.0, 5.0]))
    assert list(ranks) == [1.0, 2.5, 2.5, 4.0]


def test_auc_is_one_for_perfect_separation() -> None:
    assert xd.roc_auc(np.array([0.9, 0.8]), np.array([0.1, 0.2])) == pytest.approx(1.0)


def test_auc_is_zero_when_the_ordering_is_inverted() -> None:
    assert xd.roc_auc(np.array([0.1, 0.2]), np.array([0.9, 0.8])) == pytest.approx(0.0)


def test_auc_is_a_half_when_every_score_ties() -> None:
    # This is the case that matters: every crop with no box scores exactly 0.0,
    # and at the shipped threshold that is a large share of the population.
    assert xd.roc_auc(np.zeros(40), np.zeros(60)) == pytest.approx(0.5)


def test_auc_counts_a_tie_as_half_a_win() -> None:
    # one positive, two negatives: beats one, ties the other -> (1 + 0.5) / 2
    assert xd.roc_auc(np.array([0.5]), np.array([0.1, 0.5])) == pytest.approx(0.75)


def test_auc_is_undefined_rather_than_zero_when_an_arm_is_empty() -> None:
    assert np.isnan(xd.roc_auc(np.array([]), np.array([0.1])))
    assert np.isnan(xd.roc_auc(np.array([0.1]), np.array([])))


def test_auc_is_invariant_to_input_order() -> None:
    rng = np.random.default_rng(7)
    pos, neg = rng.random(90), rng.random(140)
    baseline = xd.roc_auc(pos, neg)
    assert xd.roc_auc(pos[::-1], neg[::-1]) == pytest.approx(baseline)


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------


def test_cluster_bootstrap_interval_brackets_the_auc() -> None:
    rng = np.random.default_rng(11)
    pos = [rng.normal(0.6, 0.1, 8) for _ in range(40)]
    neg = [rng.normal(0.4, 0.1, 8) for _ in range(40)]
    point = xd.roc_auc(np.concatenate(pos), np.concatenate(neg))
    lo, hi = xd.cluster_bootstrap_auc(pos, neg, n_boot=400, seed=3)
    assert lo <= point <= hi


def test_clustering_widens_the_auc_interval_when_frames_are_the_real_unit() -> None:
    # Same number of crops both ways. In the clustered design every crop of a
    # frame carries that frame's score, so the evidence is 20 frames, not 200
    # crops -- and the interval has to say so.
    rng = np.random.default_rng(5)
    frame_scores_pos = rng.normal(0.6, 0.15, 20)
    frame_scores_neg = rng.normal(0.4, 0.15, 20)
    clustered_pos = [np.full(10, v) for v in frame_scores_pos]
    clustered_neg = [np.full(10, v) for v in frame_scores_neg]
    independent_pos = [np.array([v]) for v in np.repeat(frame_scores_pos, 10)]
    independent_neg = [np.array([v]) for v in np.repeat(frame_scores_neg, 10)]

    c_lo, c_hi = xd.cluster_bootstrap_auc(clustered_pos, clustered_neg, n_boot=600, seed=1)
    i_lo, i_hi = xd.cluster_bootstrap_auc(independent_pos, independent_neg, n_boot=600, seed=1)
    assert (c_hi - c_lo) > (i_hi - i_lo)


def test_bootstrap_is_reproducible_from_its_seed() -> None:
    pos = [np.array([0.7, 0.6]) for _ in range(12)]
    neg = [np.array([0.3, 0.4]) for _ in range(12)]
    assert xd.cluster_bootstrap_auc(pos, neg, n_boot=200, seed=42) == xd.cluster_bootstrap_auc(
        pos, neg, n_boot=200, seed=42
    )


# ---------------------------------------------------------------------------
# crop labelling
# ---------------------------------------------------------------------------


def _write_frame(path: Path, width: int = 400, height: int = 200) -> None:
    import cv2

    cv2.imwrite(str(path), np.full((height, width, 3), 120, dtype=np.uint8))


def test_clean_frames_produce_only_clean_crops(tmp_path: Path) -> None:
    frame = tmp_path / "clean0.jpg"
    _write_frame(frame)
    crops = xd.build_crops([frame], crop_px=200)
    assert len(crops) == 2
    assert {c.label for c in crops} == {0}


def test_a_tile_is_defective_only_above_the_pixel_floor(tmp_path: Path) -> None:
    frame = tmp_path / "defect0.jpg"
    _write_frame(frame)
    mask = np.zeros((200, 400), dtype=np.uint8)
    mask[0:10, 0:10] = 3  # 100 px in the left tile
    mask[0:2, 300:304] = 1  # 8 px in the right tile
    mask_path = tmp_path / "defect0.jpg.mask.npy"
    np.save(mask_path, mask)

    crops = xd.build_crops([frame], {frame: mask_path}, crop_px=200, min_defect_px=64)
    by_x = {c.x0: c for c in crops}
    assert by_x[0].label == 1 and by_x[0].defect_px == 100
    # 8 labelled pixels is below the floor, so the tile is unlabelled, never clean:
    # nobody verified this region, and calling it a negative would be a relabelling.
    assert by_x[200].label == -1 and by_x[200].defect_px == 8


def test_an_unmasked_region_of_a_defective_frame_is_never_counted_clean(tmp_path: Path) -> None:
    frame = tmp_path / "defect1.jpg"
    _write_frame(frame)
    mask = np.zeros((200, 400), dtype=np.uint8)
    mask[0:40, 0:40] = 2
    mask_path = tmp_path / "defect1.jpg.mask.npy"
    np.save(mask_path, mask)
    labels = {c.x0: c.label for c in xd.build_crops([frame], {frame: mask_path}, crop_px=200)}
    assert labels == {0: 1, 200: -1}


def test_a_mask_of_the_wrong_geometry_is_an_error_not_a_silent_crop(tmp_path: Path) -> None:
    frame = tmp_path / "defect2.jpg"
    _write_frame(frame)
    mask_path = tmp_path / "defect2.jpg.mask.npy"
    np.save(mask_path, np.zeros((100, 100), dtype=np.uint8))
    with pytest.raises(RuntimeError, match="same geometry"):
        xd.build_crops([frame], {frame: mask_path}, crop_px=200)


@pytest.mark.parametrize(
    "layout", ["sidecar_full", "sidecar_stem", "masks_dir", "masks_dir_png"]
)
def test_masks_are_found_across_the_ingest_layouts(tmp_path: Path, layout: str) -> None:
    import cv2

    frame = tmp_path / "f.jpg"
    _write_frame(frame)
    blank = np.zeros((200, 400), dtype=np.uint8)
    if layout == "sidecar_full":
        np.save(tmp_path / "f.jpg.mask.npy", blank)
    elif layout == "sidecar_stem":
        np.save(tmp_path / "f.mask.npy", blank)
    elif layout == "masks_dir":
        (tmp_path / "masks").mkdir()
        np.save(tmp_path / "masks" / "f.npy", blank)
    else:
        (tmp_path / "masks").mkdir()
        cv2.imwrite(str(tmp_path / "masks" / "f.png"), blank)
    found = xd.find_mask(frame)
    assert found is not None
    assert xd.load_mask(found).shape == (200, 400)


# ---------------------------------------------------------------------------
# dataset discovery
# ---------------------------------------------------------------------------


def _tiny_dataset(root: Path, n_clean: int = 12, n_defect: int = 8) -> None:
    (root / "clean").mkdir(parents=True)
    (root / "defect").mkdir(parents=True)
    for i in range(n_clean):
        _write_frame(root / "clean" / f"c{i:03d}.jpg")
    for i in range(n_defect):
        frame = root / "defect" / f"d{i:03d}.jpg"
        _write_frame(frame)
        mask = np.zeros((200, 400), dtype=np.uint8)
        mask[0:40, 0:40] = 1
        np.save(root / "defect" / f"d{i:03d}.jpg.mask.npy", mask)


def test_discovery_splits_clean_frames_into_scored_and_withheld(tmp_path: Path) -> None:
    _tiny_dataset(tmp_path)
    split = xd.discover_dataset(tmp_path, holdout_frac=0.5)
    assert len(split.clean) + len(split.clean_train) == 12
    assert not {p.stem for p in split.clean} & {p.stem for p in split.clean_train}


def test_discovery_refuses_a_frame_that_is_both_clean_and_defective(tmp_path: Path) -> None:
    _tiny_dataset(tmp_path)
    _write_frame(tmp_path / "clean" / "d000.jpg")
    with pytest.raises(RuntimeError, match="both the clean and defective"):
        xd.discover_dataset(tmp_path, holdout_frac=1.0)


def test_an_excluded_frame_is_never_scored(tmp_path: Path) -> None:
    _tiny_dataset(tmp_path)
    split = xd.discover_dataset(tmp_path, holdout_frac=1.0, exclude_stems=["c000", "c001"])
    stems = {p.stem for p in split.clean}
    assert "c000" not in stems and "c001" not in stems
    assert split.excluded == ["c000", "c001"]


def test_a_defective_frame_with_no_mask_is_dropped_rather_than_guessed(tmp_path: Path) -> None:
    _tiny_dataset(tmp_path)
    _write_frame(tmp_path / "defect" / "d999.jpg")  # no mask alongside it
    split = xd.discover_dataset(tmp_path, holdout_frac=1.0)
    assert "d999" not in {p.stem for p in split.defect}
    assert all(p in split.masks for p in split.defect)


def test_discovery_fails_loudly_on_a_root_with_no_clean_frames(tmp_path: Path) -> None:
    (tmp_path / "defect").mkdir()
    with pytest.raises(RuntimeError, match="No defect-free frames"):
        xd.discover_dataset(tmp_path)


def test_the_manifest_hash_identifies_the_scored_set(tmp_path: Path) -> None:
    a = [Path("x/one.jpg"), Path("y/two.jpg")]
    assert xd.manifest_hash(a) == xd.manifest_hash(list(reversed(a)))
    assert xd.manifest_hash(a) != xd.manifest_hash(a + [Path("three.jpg")])


# ---------------------------------------------------------------------------
# the threshold sweep
# ---------------------------------------------------------------------------


def _scored(frame: str, label: int, confs: list[float], classes: list[int]) -> xd.ScoredCrop:
    return xd.ScoredCrop(
        crop=xd.Crop(frame=Path(frame), x0=0, y0=0, size=200, label=label),
        confidences=np.asarray(confs, dtype=np.float64),
        classes=np.asarray(classes, dtype=np.int64),
    )


def test_sweep_reports_crop_and_frame_false_alarms_on_the_right_denominators() -> None:
    clean = [
        _scored("a.jpg", 0, [0.30], [0]),   # alarms at 0.2, silent at 0.5
        _scored("a.jpg", 0, [], []),
        _scored("b.jpg", 0, [], []),
        _scored("b.jpg", 0, [], []),
    ]
    defect = [_scored("d.jpg", 1, [0.60], [1])]
    points = xd.sweep_thresholds(clean, defect, [0.2, 0.5], ["crazing", "inclusion"], n_boot=100)
    low, high = points

    assert low["clean_crops"] == 4 and low["clean_crops_flagged"] == 1
    assert low["clean_crop_fa"] == pytest.approx(0.25)
    # One of two frames alarms, even though only one of its two tiles did.
    assert low["clean_frames"] == 2 and low["clean_frame_fa"] == pytest.approx(0.5)
    assert high["clean_crop_fa"] == pytest.approx(0.0)
    assert high["clean_frame_fa"] == pytest.approx(0.0)


def test_sweep_counts_boxes_per_crop_and_names_the_false_positives() -> None:
    clean = [
        _scored("a.jpg", 0, [0.9, 0.8, 0.1], [1, 1, 0]),
        _scored("b.jpg", 0, [0.7], [2]),
    ]
    point = xd.sweep_thresholds(
        clean, [_scored("d.jpg", 1, [0.9], [0])], [0.5],
        ["crazing", "inclusion", "patches"], n_boot=100,
    )[0]
    assert point["boxes_total"] == 3
    assert point["boxes_per_clean_crop"] == pytest.approx(1.5)
    assert point["false_positive_classes"] == {"inclusion": 2, "patches": 1}
    assert point["false_positive_class_fracs"]["inclusion"] == pytest.approx(2 / 3)


def test_recall_uses_only_labelled_defective_tiles() -> None:
    defect = [
        _scored("d.jpg", 1, [0.9], [0]),    # hit
        _scored("d.jpg", 1, [], []),        # miss
        _scored("d.jpg", -1, [0.9], [0]),   # unlabelled region: neither hit nor miss
    ]
    point = xd.sweep_thresholds(
        [_scored("c.jpg", 0, [], [])], defect, [0.5], ["crazing"], n_boot=100
    )[0]
    assert point["defect_crops"] == 2
    assert point["defect_crop_recall"] == pytest.approx(0.5)
    assert point["unlabelled_region_crops"] == 1
    assert point["unlabelled_region_flag_rate"] == pytest.approx(1.0)
    # The frame carried at least one hit, so it counts as recalled at frame level.
    assert point["defect_frame_recall"] == pytest.approx(1.0)


def test_every_interval_brackets_its_own_point_estimate() -> None:
    rng = np.random.default_rng(19)

    def crop(name: str, label: int) -> xd.ScoredCrop:
        confs = rng.random(int(rng.integers(0, 3)))
        return xd.ScoredCrop(
            crop=xd.Crop(frame=Path(name), x0=0, y0=0, size=200, label=label),
            confidences=confs,
            classes=np.zeros(confs.size, dtype=np.int64),
        )

    clean = [crop(f"c{i // 8}.jpg", 0) for i in range(64)]
    defect = [crop(f"d{i // 8}.jpg", 1) for i in range(40)]
    for point in xd.sweep_thresholds(clean, defect, [0.15, 0.5], ["crazing"], n_boot=300):
        for value, ci in (
            (point["clean_crop_fa"], point["clean_crop_fa_ci"]),
            (point["clean_frame_fa"], point["clean_frame_fa_ci"]),
            (point["defect_crop_recall"], point["defect_crop_recall_ci"]),
        ):
            assert ci[0] <= value <= ci[1]


def test_false_alarms_and_recall_both_fall_as_the_threshold_rises() -> None:
    rng = np.random.default_rng(23)
    clean = [
        xd.ScoredCrop(
            crop=xd.Crop(frame=Path(f"c{i // 8}.jpg"), x0=0, y0=0, size=200, label=0),
            confidences=rng.random(3) * 0.9,
            classes=np.zeros(3, dtype=np.int64),
        )
        for i in range(80)
    ]
    defect = [
        xd.ScoredCrop(
            crop=xd.Crop(frame=Path(f"d{i // 8}.jpg"), x0=0, y0=0, size=200, label=1),
            confidences=rng.random(3) * 0.9,
            classes=np.zeros(3, dtype=np.int64),
        )
        for i in range(48)
    ]
    points = xd.sweep_thresholds(clean, defect, [0.1, 0.3, 0.5, 0.7], ["crazing"], n_boot=100)
    for key in ("clean_crop_fa", "clean_frame_fa", "defect_crop_recall", "boxes_per_clean_crop"):
        values = [p[key] for p in points]
        assert values == sorted(values, reverse=True)


def test_separation_reports_the_same_auc_the_rank_function_does() -> None:
    clean = [_scored("c.jpg", 0, [0.1], [0]), _scored("c.jpg", 0, [], [])]
    defect = [_scored("d.jpg", 1, [0.9], [0]), _scored("d.jpg", -1, [0.99], [0])]
    result = xd.separation(clean, defect, n_boot=200)
    assert result["n_positive_crops"] == 1  # the unlabelled tile is not a positive
    assert result["n_negative_crops"] == 2
    assert result["auc"] == pytest.approx(xd.roc_auc(np.array([0.9]), np.array([0.1, 0.0])))


# ---------------------------------------------------------------------------
# CLI and artifacts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["0", "1", "1.5", "-0.2", ""])
def test_the_sweep_parser_rejects_thresholds_outside_the_open_unit_interval(bad: str) -> None:
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        xd._parse_sweep(bad)


def test_the_sweep_parser_accepts_a_comma_list() -> None:
    assert xd._parse_sweep("0.15, 0.4,0.6") == (0.15, 0.4, 0.6)


def test_the_cli_exposes_the_arguments_the_workflow_depends_on() -> None:
    args = xd.build_parser().parse_args([])
    for name in ("weights", "imgsz", "conf_sweep", "tag"):
        assert hasattr(args, name)
    assert args.tag == "baseline"
    assert args.imgsz == xd.DEFAULT_IMGSZ


def test_a_new_tag_is_appended_and_an_existing_tag_is_replaced(tmp_path: Path) -> None:
    path = tmp_path / "cross_domain.json"
    first = {"tag": "baseline", "generated_at": "2026-09-09T10:00:00+05:30", "sweep": []}
    second = {"tag": "with_negatives", "generated_at": "2026-09-09T11:00:00+05:30", "sweep": []}
    xd.write_json(path, xd.merge_record(path, first))
    xd.write_json(path, xd.merge_record(path, second))
    tags = [r["tag"] for r in json.loads(path.read_text())["runs"]]
    assert tags == ["baseline", "with_negatives"]

    redone = {"tag": "baseline", "generated_at": "2026-09-09T12:00:00+05:30", "sweep": []}
    xd.write_json(path, xd.merge_record(path, redone))
    runs = json.loads(path.read_text())["runs"]
    assert [r["tag"] for r in runs] == ["with_negatives", "baseline"]
    assert len(runs) == 2


def _fake_run(tag: str) -> dict:
    return {
        "tag": tag,
        "generated_at": "2026-09-09T10:00:00+05:30",
        "model": {"weights": "w.pt", "weights_sha256": "0" * 64, "imgsz": 256, "iou": 0.45,
                  "device": "mps"},
        "protocol": {"crop_px": 200, "magnification": 1.28, "deploy_conf": 0.15,
                     "holdout_frac": 0.5, "holdout_seed": "seed"},
        "dataset": {"root": "/data", "clean_frames_scored": 2, "clean_crops": 4,
                    "defect_frames_scored": 1, "defect_crops": 2,
                    "clean_frames_withheld": 1, "defect_frames_withheld": 1,
                    "clean_manifest_sha256": "a" * 64, "unlabelled_region_crops": 1},
        "sweep": [
            {"conf": 0.15, "clean_crop_fa": 0.5, "clean_crop_fa_ci": [0.3, 0.7],
             "clean_crop_design_effect": 1.4, "clean_crops": 4, "clean_crops_flagged": 2,
             "clean_frame_fa": 1.0, "clean_frame_fa_ci": [0.6, 1.0], "clean_frames": 2,
             "clean_frames_flagged": 2, "boxes_per_clean_crop": 2.0, "boxes_total": 8,
             "false_positive_classes": {"inclusion": 8},
             "false_positive_class_fracs": {"inclusion": 1.0},
             "defect_crop_recall": 0.5, "defect_crop_recall_ci": [0.2, 0.8],
             "defect_crop_design_effect": 1.1, "defect_crops": 2, "defect_crops_hit": 1,
             "defect_frame_recall": 1.0, "defect_frames": 1,
             "unlabelled_region_flag_rate": 0.5, "unlabelled_region_crops": 1},
        ],
        "separation_cross_domain": {"auc": 0.61, "auc_ci": [0.58, 0.64],
                                    "n_positive_crops": 2, "n_negative_crops": 4},
        "separation_in_domain": {"auc": 0.96, "auc_ci": [0.94, 0.98], "split": "test",
                                 "n_positive_crops": 197, "n_negative_crops": 771,
                                 "magnification": 1.28, "crop_sizes_px": [60, 80],
                                 "clearance_px": 6, "auc_scale_stratified": 0.95,
                                 "per_scale": [{"crop_px": 60, "n_positive": 40,
                                                "n_negative": 448, "auc": 0.95}]},
        "clean_coil_disposition": {
            "source_frames": 2,
            "tiles": {"coil_id": "C", "disposition": "HOLD", "reasons": ["Defect rate high."],
                      "total_frames": 4, "defect_rate": 0.5, "total_detections": 8,
                      "p95_severity": 80.0},
            "whole_frames": {"coil_id": "C-W", "disposition": "HOLD", "reasons": ["x"],
                             "total_frames": 2, "defect_rate": 0.5, "total_detections": 2,
                             "p95_severity": 70.0},
        },
    }


def test_the_markdown_states_both_false_alarm_units_and_the_verdict(tmp_path: Path) -> None:
    payload = {"runs": [_fake_run("baseline")], "updated_at": "2026-09-09T10:00:00+05:30"}
    text = xd.write_markdown(tmp_path / "cross_domain.md", payload).read_text()
    assert "clean-crop FA" in text and "clean-FRAME FA" in text
    assert "HOLD" in text
    assert "inclusion" in text
    assert "0.61" in text and "0.96" in text


def test_the_chart_draws_one_curve_per_tag(tmp_path: Path) -> None:
    payload = {"runs": [_fake_run("baseline"), _fake_run("with_negatives")]}
    out = xd.plot_fa_vs_recall(payload, tmp_path / "curve.png")
    assert out.is_file() and out.stat().st_size > 5_000


def test_non_finite_values_are_written_as_null_not_bare_nan(tmp_path: Path) -> None:
    path = xd.write_json(tmp_path / "x.json", {"runs": [], "a": float("nan")})
    assert json.loads(path.read_text())["a"] is None


# ---------------------------------------------------------------------------
# the shipped artifact
# ---------------------------------------------------------------------------

ARTIFACT = PROJECT_ROOT / "reports" / "cross_domain.json"


@pytest.mark.skipif(not ARTIFACT.is_file(), reason="run src/cross_domain.py first")
def test_the_published_record_is_internally_consistent() -> None:
    """No measured value is pinned here -- only that the record cannot lie to itself."""
    payload = json.loads(ARTIFACT.read_text())
    assert payload["schema"] == "cross_domain/1"
    assert payload["runs"], "no runs recorded"
    for run in payload["runs"]:
        data = run["dataset"]
        assert data["clean_frames_scored"] > 0
        assert data["clean_crops"] >= data["clean_frames_scored"]
        for point in run["sweep"]:
            assert point["clean_crop_fa"] == pytest.approx(
                point["clean_crops_flagged"] / point["clean_crops"]
            )
            assert point["clean_frame_fa"] == pytest.approx(
                point["clean_frames_flagged"] / point["clean_frames"]
            )
            assert point["defect_crop_recall"] == pytest.approx(
                point["defect_crops_hit"] / point["defect_crops"]
            )
            for value, ci in (
                (point["clean_crop_fa"], point["clean_crop_fa_ci"]),
                (point["clean_frame_fa"], point["clean_frame_fa_ci"]),
                (point["defect_crop_recall"], point["defect_crop_recall_ci"]),
            ):
                assert ci[0] <= value <= ci[1]
            # The design effect is a variance ratio, so it must be positive and
            # finite. It is usually above 1 (alarms clump inside frames), but it
            # can dip just below when alarms are so rare that the per-frame counts
            # are under-dispersed relative to binomial -- which is exactly what a
            # checkpoint that has stopped firing looks like.
            deff = point["clean_crop_design_effect"]
            assert 0.0 < deff < 100.0
        for key in ("clean_crop_fa", "clean_frame_fa", "defect_crop_recall"):
            values = [p[key] for p in run["sweep"]]
            assert values == sorted(values, reverse=True)
        cross = run["separation_cross_domain"]
        assert cross["auc_ci"][0] <= cross["auc"] <= cross["auc_ci"][1]


@pytest.mark.skipif(not ARTIFACT.is_file(), reason="run src/cross_domain.py first")
def test_the_published_record_scored_a_held_out_set_disjoint_from_training() -> None:
    payload = json.loads(ARTIFACT.read_text())
    for run in payload["runs"]:
        proto, data = run["protocol"], run["dataset"]
        assert 0.0 < proto["holdout_frac"] <= 1.0
        assert data["clean_manifest_sha256"] and len(data["clean_manifest_sha256"]) == 64
        if proto["holdout_frac"] < 1.0:
            assert data["clean_frames_withheld"] > 0


def test_the_coil_tile_directory_refuses_to_delete_files_it_did_not_write(
    tmp_path: Path,
) -> None:
    (tmp_path / "precious.jpg").write_bytes(b"not ours")
    with pytest.raises(RuntimeError, match="not empty and was not written"):
        xd.clean_coil_disposition(
            [], weights=None, device="cpu", conf=0.15, iou=0.45, imgsz=256,
            coil_dir=tmp_path, write_html=False,
        )
    assert (tmp_path / "precious.jpg").is_file()


# ---------------------------------------------------------------------------
# the assumption the whole sweep rests on
# ---------------------------------------------------------------------------

_WEIGHTS = PROJECT_ROOT / "models" / "yolov8n_neudet" / "weights" / "best.pt"


@pytest.mark.skipif(
    not (_WEIGHTS.is_file() and xd.DEFAULT_DATA_ROOT.is_dir()),
    reason="needs the shipped checkpoint and a cross-domain data root",
)
def test_filtering_the_box_cache_equals_running_the_detector_at_that_threshold() -> None:
    """The sweep is exact, not an approximation -- checked against live inference.

    `score_crops` caches every box above CACHE_CONF once and `sweep_thresholds`
    then filters that cache. The argument is that greedy NMS runs in descending
    confidence, so a box above t can only have been suppressed by another box
    above t. Every number in the report inherits that argument, so it is checked
    against detectors actually constructed at each threshold rather than assumed.
    """
    from false_alarm import CACHE_CONF, NMS_IOU  # noqa: PLC0415 - optional at import time
    from inference import DefectDetector, resolve_device, resolve_weights  # noqa: PLC0415

    split = xd.discover_dataset(xd.DEFAULT_DATA_ROOT, limit_clean=4, limit_defect=0)
    crops = xd.build_crops(split.clean)
    assert crops
    weights, device = resolve_weights(_WEIGHTS), resolve_device("auto")

    cached = xd.score_crops(
        crops,
        DefectDetector(weights=weights, device=device, conf=CACHE_CONF, iou=NMS_IOU, imgsz=256),
        verbose=False,
    )
    for threshold in (0.25, 0.50):
        live = xd.score_crops(
            crops,
            DefectDetector(
                weights=weights, device=device, conf=threshold, iou=NMS_IOU, imgsz=256
            ),
            verbose=False,
        )
        from_cache = [int((s.confidences >= threshold).sum()) for s in cached]
        from_live = [len(s.confidences) for s in live]
        assert from_cache == from_live
