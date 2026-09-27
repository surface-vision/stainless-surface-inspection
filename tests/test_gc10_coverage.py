"""Tests for the GC10-DET detector run and the coverage report it produces.

These guard three things the report would be worthless without: that the
checkpoint really carries GC10's ten classes in the prepared order, that every
number in `reports/gc10_coverage.md` traces back to `gc10_eval.json` and to the
label files on disk, and that no per-class AP is ever quoted without the box
count it was computed on -- which is the whole point of the exercise for classes
holding 11 test boxes.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "gc10-det"
RUN = ROOT / "models" / "yolov8n_gc10_640"
EVAL_JSON = RUN / "gc10_eval.json"
REPORT = ROOT / "reports" / "gc10_coverage.md"
TRAIN_SUMMARY = RUN / "train_summary.json"

CLASSES = ["punching_hole", "weld_line", "crescent_gap", "water_spot", "oil_spot",
           "silk_spot", "foreign_object", "rolled_pit", "crease", "waist_fold"]
BRIEF_CLASSES = ["rolled_pit", "crease", "crescent_gap", "waist_fold"]

# The shipped NEU-DET detector must be byte-identical after this work: GC10
# training writes to its own run directory and must never touch the deployed one.
SHIPPED_SHA256 = "6661c7a09037137aa5a2fac73cb9588d4906118f4880b4e2902b39aea9f5f985"

needs_eval = pytest.mark.skipif(not EVAL_JSON.is_file(), reason="GC10 evaluation not run yet")
needs_report = pytest.mark.skipif(not REPORT.is_file(), reason="GC10 coverage report not written yet")


@pytest.fixture(scope="module")
def evaluation() -> dict:
    return json.loads(EVAL_JSON.read_text())


@pytest.fixture(scope="module")
def report_text() -> str:
    return REPORT.read_text()


def label_box_counts(split: str) -> Counter:
    counts: Counter = Counter()
    for lp in sorted((DATA / split / "labels").glob("*.txt")):
        for line in lp.read_text().splitlines():
            if line.strip():
                counts[CLASSES[int(line.split()[0])]] += 1
    return counts


# --- the shipped model is untouched ---------------------------------------

def test_shipped_neudet_checkpoint_is_unchanged() -> None:
    import hashlib

    weights = ROOT / "models" / "yolov8n_neudet" / "weights" / "best.pt"
    assert weights.is_file()
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    assert digest == SHIPPED_SHA256, "GC10 work must not modify the deployed detector"


def test_gc10_run_is_a_separate_directory() -> None:
    assert "gc10" in RUN.name
    assert RUN.resolve() != (ROOT / "models" / "yolov8n_neudet").resolve()


# --- the checkpoint carries the prepared class list ------------------------

@pytest.mark.skipif(not (RUN / "weights" / "best.pt").is_file(), reason="no GC10 checkpoint yet")
def test_checkpoint_class_names_match_the_prepared_dataset() -> None:
    import torch

    ckpt = torch.load(RUN / "weights" / "best.pt", map_location="cpu", weights_only=False)
    names = ckpt["model"].names
    assert [names[i] for i in range(len(names))] == CLASSES


def test_data_yaml_class_order_matches_this_test() -> None:
    text = (DATA / "data.yaml").read_text()
    found = re.findall(r"^\s*(\d+):\s*(\S+)\s*$", text, flags=re.MULTILINE)
    assert [name for _, name in found] == CLASSES
    assert [int(i) for i, _ in found] == list(range(10))


# --- split integrity, re-checked independently of the ingest ---------------

@pytest.mark.parametrize("split,images", [("train", 1641), ("val", 320), ("test", 333)])
def test_split_sizes(split: str, images: int) -> None:
    assert len(list((DATA / split / "images").glob("*.jpg"))) == images
    assert len(list((DATA / split / "labels").glob("*.txt"))) == images


def test_the_four_brief_classes_are_present_in_every_split() -> None:
    for split in ("train", "val", "test"):
        counts = label_box_counts(split)
        for name in BRIEF_CLASSES:
            assert counts[name] > 0, f"{name} missing from {split}"


def test_test_split_brief_class_counts_are_what_the_report_claims() -> None:
    counts = label_box_counts("test")
    assert counts["rolled_pit"] == 13
    assert counts["crease"] == 11
    assert counts["crescent_gap"] == 40
    assert counts["waist_fold"] == 21


# --- the evaluation JSON is internally consistent --------------------------

@needs_eval
def test_evaluation_covers_val_and_test(evaluation: dict) -> None:
    assert set(evaluation["splits"]) == {"val", "test"}
    assert evaluation["classes"] == CLASSES


@needs_eval
def test_reported_instance_counts_match_the_label_files(evaluation: dict) -> None:
    for split in ("val", "test"):
        gt = evaluation["splits"][split]["ground_truth"]["boxes_per_class"]
        assert gt == dict(label_box_counts(split))


@needs_eval
def test_every_class_has_an_ap_and_an_instance_count(evaluation: dict) -> None:
    test = evaluation["splits"]["test"]
    for name in CLASSES:
        assert name in test["per_class"], f"{name} absent from per-class results"
        assert test["ground_truth"]["boxes_per_class"].get(name, 0) > 0
        assert 0.0 <= test["per_class"][name]["AP50"] <= 1.0


@needs_eval
def test_bootstrap_reproduces_the_validator_number(evaluation: dict) -> None:
    fidelity = evaluation["splits"]["test"]["bootstrap_fidelity"]
    assert fidelity["reproduces"] is True
    assert fidelity["abs_difference"] < 1e-6
    assert fidelity["images_captured"] == 333


@needs_eval
def test_bootstrap_intervals_bracket_the_point_estimate(evaluation: dict) -> None:
    """A CI that excludes the number it is an interval for would be a bug."""
    test = evaluation["splits"]["test"]
    boot = test["bootstrap"]["per_class_AP50"]
    assert test["bootstrap"]["resamples"] >= 1000
    for name in CLASSES:
        ci = boot[name]
        if not ci.get("n_resamples"):
            continue
        assert ci["ci95_lo"] <= ci["ci95_hi"]
        assert ci["ci95_lo"] <= ci["mean"] <= ci["ci95_hi"]
        assert 0.0 <= ci["ci95_lo"] and ci["ci95_hi"] <= 1.0


@needs_eval
def test_rare_classes_have_wider_intervals_than_common_ones(evaluation: dict) -> None:
    """The report's central claim: 11-box AP is noisier than 129-box AP.

    If this ever fails, the report's caveats are overstated and should be redrawn.
    """
    boot = evaluation["splits"]["test"]["bootstrap"]["per_class_AP50"]
    width = {n: boot[n]["ci95_hi"] - boot[n]["ci95_lo"] for n in CLASSES
             if boot[n].get("n_resamples")}
    assert width["crease"] > width["silk_spot"]
    assert width["rolled_pit"] > width["silk_spot"]


@needs_eval
def test_training_summary_records_the_imgsz_used_for_evaluation(evaluation: dict) -> None:
    summary = json.loads(TRAIN_SUMMARY.read_text())
    assert summary["imgsz"] == evaluation["imgsz"] == 640
    assert summary["seed"] == 1337
    assert "gc10" in summary["dataset"].lower()


# --- the report cannot quote a number that is not in the JSON --------------

@needs_report
@needs_eval
def test_report_headline_map_matches_the_evaluation(report_text: str, evaluation: dict) -> None:
    map50 = evaluation["splits"]["test"]["mAP50"]
    assert f"{map50:.4f}" in report_text or f"{map50:.3f}" in report_text


@needs_report
def test_report_never_quotes_a_brief_class_ap_without_its_box_count(report_text: str) -> None:
    """Every table row naming a brief class must carry its instance count."""
    counts = label_box_counts("test")
    for line in report_text.splitlines():
        if not line.startswith("|"):
            continue
        for name in BRIEF_CLASSES:
            if name in line and re.search(r"0\.\d{2,}", line):
                assert str(counts[name]) in line, (
                    f"row quotes an AP for {name} without its {counts[name]} test boxes: {line}"
                )


@needs_report
def test_report_states_that_edge_crack_is_a_proxy(report_text: str) -> None:
    low = report_text.lower()
    assert "proxy" in low
    assert "edge crack" in low
    # the claim that must never be made
    assert "detects edge cracks" not in low


@needs_report
def test_report_names_what_real_data_would_be_needed(report_text: str) -> None:
    low = report_text.lower()
    assert "what would be needed" in low or "what it would take" in low
