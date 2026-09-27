"""The 10-class joint checkpoint, end to end through the shipped inference path.

`models/yolov8n_joint` scores better than the shipped checkpoint on the held-out
NEU-DET test split (mAP50 0.7642 vs 0.7524 at imgsz 256) and turns the
cross-domain false-alarm problem around (clean-frame false alarms 93.7% ->
32.5%). It could not be served anyway, because `src/inference.py` was written
for a six-class head: the detector truncated its own class list to six so
indices 6-9 came back as `class_6` .. `class_9`, and the first frame that fired
on one died with

    KeyError: 'class_8'   (score_detection, via _finalise, via predict)

This file is the guard on that being fixed properly rather than papered over.
It covers the whole 10-class path -- names, colours, knowledge base, predict,
predict_batch, predict_tiled, annotate, the severity score and the coil
disposition -- and it skips itself cleanly on a checkout without the checkpoint
or the joint dataset.

Everything that needs the GPU shares one detector at imgsz 320 and touches a
handful of frames, so the suite stays polite to a training job on the same
accelerator.

Run with:
    .venv/bin/python -m pytest tests/test_joint_head.py -v
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _extra in (PROJECT_ROOT / "src", PROJECT_ROOT / "demo"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

os.environ.setdefault("YOLO_AUTOINSTALL", "False")

import inference  # noqa: E402
import report  # noqa: E402
from inference import (  # noqa: E402
    CLASS_COLORS,
    CLASS_NAMES,
    DEFECT_INFO,
    JOINT_CLASS_NAMES,
    SEVERITY_BANDS,
    InferenceResult,
    _head_class_names,
    _severity_bucket,
    aggregate_severity,
    resolve_weights,
    score_detection,
)

JOINT_WEIGHTS = PROJECT_ROOT / "models" / "yolov8n_joint" / "weights" / "best.pt"
JOINT_DATA = PROJECT_ROOT / "data" / "joint_xdsafe"

TEST_IMGSZ = 320
TEST_CONF = 0.25
TEST_IOU = 0.45

needs_weights = pytest.mark.skipif(
    not JOINT_WEIGHTS.is_file(), reason="models/yolov8n_joint has not been trained here"
)
needs_data = pytest.mark.skipif(
    not JOINT_DATA.is_dir(), reason="data/joint_xdsafe has not been built here"
)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def joint_detector() -> Iterator[inference.DefectDetector]:
    if not JOINT_WEIGHTS.is_file():
        pytest.skip("models/yolov8n_joint has not been trained here")
    model = inference.DefectDetector(
        weights=JOINT_WEIGHTS,
        device="auto",
        conf=TEST_CONF,
        iou=TEST_IOU,
        imgsz=TEST_IMGSZ,
    )
    model.warmup(1)
    yield model


@pytest.fixture(scope="session")
def severstal_frames() -> list[Path]:
    """Held-out crops whose labels carry a Severstal class, i.e. index 6-9.

    Chosen from the labels rather than from the filename so the fixture keeps
    working if the crop naming changes.
    """
    labels = JOINT_DATA / "test" / "labels"
    images = JOINT_DATA / "test" / "images"
    if not labels.is_dir():
        pytest.skip(f"no joint test split at {labels}")
    picked: list[Path] = []
    for label_path in sorted(labels.glob("*.txt")):
        classes = {
            int(line.split()[0])
            for line in label_path.read_text().splitlines()
            if line.strip()
        }
        if not classes & {6, 7, 8, 9}:
            continue
        image = images / f"{label_path.stem}.jpg"
        if image.is_file():
            picked.append(image)
        if len(picked) == 8:
            break
    if not picked:
        pytest.skip("no held-out frame carries a Severstal class")
    return picked


@pytest.fixture(scope="session")
def severstal_hit(
    joint_detector: inference.DefectDetector, severstal_frames: list[Path]
) -> tuple[Path, InferenceResult]:
    """A frame the joint head actually calls `severstal_*`, scored once."""
    for path in severstal_frames:
        result = joint_detector.predict(path)
        if any(d.class_name.startswith("severstal_") for d in result.detections):
            return path, result
    pytest.skip("the joint checkpoint emitted no Severstal class on these frames")


# --------------------------------------------------------------------------- #
# 1. the class contract of a head that is not the shipped one
# --------------------------------------------------------------------------- #


def test_head_class_names_accepts_a_complete_mapping() -> None:
    assert _head_class_names({0: "a", 1: "b"}) == ["a", "b"]
    assert _head_class_names({"0": "a", "1": "b"}) == ["a", "b"]


@pytest.mark.parametrize(
    "names",
    [
        None,
        {},
        {0: "a", 2: "c"},           # sparse: index 1 would be a hole
        {1: "a", 2: "b"},           # does not start at 0
        {0: "a", 1: ""},            # empty name
        {0: "a", 1: None},          # not a string
        {0: "a", "x": "b"},         # unusable key
    ],
)
def test_head_class_names_refuses_anything_it_cannot_index(names) -> None:
    """A partial mapping must fall back, not produce a list with holes in it."""
    assert _head_class_names(names) is None


@needs_data
def test_the_joint_class_contract_matches_the_dataset_it_was_trained_on() -> None:
    cfg = yaml.safe_load((JOINT_DATA / "data.yaml").read_text())
    assert int(cfg["nc"]) == len(JOINT_CLASS_NAMES) == 10
    assert [cfg["names"][i] for i in range(10)] == JOINT_CLASS_NAMES
    # NEU-DET indices are unchanged by construction, which is what lets the two
    # checkpoints' test numbers be compared at all.
    assert JOINT_CLASS_NAMES[:6] == CLASS_NAMES


@needs_weights
def test_the_detector_reads_ten_names_off_the_checkpoint(
    joint_detector: inference.DefectDetector,
) -> None:
    """The regression: this used to silently truncate to the six NEU-DET names."""
    assert joint_detector.class_names == JOINT_CLASS_NAMES
    assert len(joint_detector.class_names) == 10


@needs_weights
def test_the_shipped_checkpoint_still_reads_six(
    joint_detector: inference.DefectDetector,
) -> None:
    """Widening the rule must not have changed the six-class head's behaviour."""
    shipped = inference.DefectDetector(
        weights=resolve_weights(), device="cpu", imgsz=TEST_IMGSZ
    )
    assert shipped.class_names == CLASS_NAMES


def test_every_joint_class_is_colourable_and_explainable() -> None:
    for name in JOINT_CLASS_NAMES:
        assert name in CLASS_COLORS, f"{name} would be drawn white on grey steel"
        info = DEFECT_INFO[name]
        assert info["severity"] in SEVERITY_BANDS
        assert info["cause"].strip() and info["action"].strip()


# --------------------------------------------------------------------------- #
# 2. scoring: the KeyError that made this checkpoint unservable
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", list(inference.SEVERSTAL_CLASS_NAMES))
def test_scoring_a_severstal_class_no_longer_raises(name: str) -> None:
    score = score_detection(name, 0.9, 0.1)
    assert 0.0 < score <= 100.0
    assert _severity_bucket(score) in SEVERITY_BANDS


def test_an_unknown_class_is_scored_visibly_rather_than_crashing_the_line() -> None:
    """A head this module was not built for must degrade, not stop production.

    Scoring it zero would be worse than crashing: a real box would sit in the
    report behind a clean-looking frame score.
    """
    score = score_detection("class_47", 0.9, 0.1)
    assert score == pytest.approx(
        score_detection("patches", 0.9, 0.1)  # patches is also the middle tier
    )
    assert score > 0.0


# --------------------------------------------------------------------------- #
# 3. the full inference path on the 10-class head
# --------------------------------------------------------------------------- #


def _assert_joint_contract(result: InferenceResult) -> None:
    width, height = result.image_size
    assert width > 0 and height > 0
    assert result.defect_count == len(result.detections)
    assert (result.verdict == "DEFECT") == bool(result.detections)
    assert 0.0 <= result.severity_score <= 100.0
    assert result.severity_score == pytest.approx(
        aggregate_severity([d.severity_score for d in result.detections])
    )
    for det in result.detections:
        assert 0 <= det.class_id < len(JOINT_CLASS_NAMES)
        assert det.class_name == JOINT_CLASS_NAMES[det.class_id]
        # The two failure modes of the old code, asserted directly.
        assert not det.class_name.startswith("class_"), "class id was not resolved"
        assert det.class_name in DEFECT_INFO and det.class_name in CLASS_COLORS
        assert det.severity == _severity_bucket(det.severity_score)
        x1, y1, x2, y2 = det.bbox_xyxy
        assert 0.0 <= x1 <= x2 <= width and 0.0 <= y1 <= y2 <= height


@needs_weights
@needs_data
def test_predict_scores_a_severstal_frame(
    severstal_hit: tuple[Path, InferenceResult]
) -> None:
    _, result = severstal_hit
    _assert_joint_contract(result)
    assert result.severity_score > 0.0, "a Severstal box must move the frame score"


@needs_weights
@needs_data
def test_predict_batch_agrees_with_predict_on_the_joint_head(
    joint_detector: inference.DefectDetector, severstal_frames: list[Path]
) -> None:
    frames = severstal_frames[:4]
    singles = [joint_detector.predict(p) for p in frames]
    batched = joint_detector.predict_batch(frames, batch_size=len(frames))
    assert len(batched) == len(singles)
    for one, many in zip(singles, batched):
        _assert_joint_contract(many)
        assert one.defect_count == many.defect_count
        assert [d.class_name for d in one.detections] == [
            d.class_name for d in many.detections
        ]
        assert one.severity_score == pytest.approx(many.severity_score)


@needs_weights
@needs_data
def test_predict_tiled_works_on_the_joint_head(
    joint_detector: inference.DefectDetector, severstal_frames: list[Path]
) -> None:
    strip = np.concatenate(
        [inference._to_rgb(p) for p in severstal_frames[:6]], axis=1
    )
    height, width = strip.shape[:2]
    result = joint_detector.predict_tiled(strip, tile=joint_detector.imgsz, overlap=0.2)
    assert result.image_size == (width, height)
    _assert_joint_contract(result)


@needs_weights
@needs_data
def test_annotate_draws_every_joint_class_in_its_own_colour(
    joint_detector: inference.DefectDetector,
    severstal_hit: tuple[Path, InferenceResult],
) -> None:
    path, result = severstal_hit
    canvas = inference._to_rgb(path)
    annotated = joint_detector.annotate(canvas, result)

    assert annotated.shape == canvas.shape and annotated.dtype == np.uint8
    assert not np.array_equal(annotated, canvas), "nothing was drawn"
    # The caller's array is never written through.
    assert np.array_equal(canvas, inference._to_rgb(path))
    # White is the fallback for a class with no colour; none of ours should use it.
    for det in result.detections:
        assert CLASS_COLORS.get(det.class_name, (255, 255, 255)) != (255, 255, 255)


@needs_weights
@needs_data
def test_a_joint_result_survives_a_json_round_trip(
    severstal_hit: tuple[Path, InferenceResult]
) -> None:
    _, result = severstal_hit
    payload = json.loads(json.dumps(result.to_dict()))
    assert len(payload["detections"]) == len(result.detections)
    for encoded, det in zip(payload["detections"], result.detections):
        assert encoded["class_name"] == det.class_name
        assert encoded["severity"] == det.severity
        assert encoded["severity_score"] == pytest.approx(det.severity_score, abs=5e-3)


# --------------------------------------------------------------------------- #
# 4. what the layer above does with an unnamed defect
# --------------------------------------------------------------------------- #


@needs_weights
@needs_data
def test_the_coil_report_dispositions_severstal_frames_sensibly(
    joint_detector: inference.DefectDetector, severstal_frames: list[Path]
) -> None:
    """A frame whose only defect is an unnamed class must still be reportable.

    The disposition itself is not pinned -- it depends on how many of these
    frames the checkpoint fires on -- but it must be one of the three real
    verdicts, with a reason, and it must not have been reached by the
    zero-tolerance class rule, which no Severstal class belongs to.
    """
    records = [
        report.FrameRecord.from_result(i, str(path), joint_detector.predict(path))
        for i, path in enumerate(severstal_frames)
    ]
    stats = report._aggregate(records)
    disposition, reasons = report.DispositionRules().evaluate(stats)

    assert disposition in ("ACCEPT", "DOWNGRADE", "HOLD")
    assert reasons
    assert not any("zero-tolerance" in reason for reason in reasons)
    assert set(stats.band_counts) >= set(SEVERITY_BANDS)
    for record in records:
        assert record.severity_band in set(SEVERITY_BANDS) | {"clean"}


def test_the_placeholder_tier_cannot_hold_a_coil_by_itself() -> None:
    """One unnamed defect must not stop the line while nobody knows what it is.

    `DispositionRules` holds unconditionally on a single frame in the critical
    band. A `SEVERSTAL_TIER` detection maxes out below that cutoff, so an
    unnamed defect can cost a coil its prime grade but cannot stop production on
    its own. If someone raises SEVERSTAL_TIER to "critical", this fails and they
    have to say so out loud.
    """
    worst = max(
        score_detection(name, 1.0, 1.0)
        for name in inference.SEVERSTAL_CLASS_NAMES
    )
    assert _severity_bucket(worst) != "critical"

    rules = report.DispositionRules()
    assert not set(inference.SEVERSTAL_CLASS_NAMES) & set(rules.hold_classes)


# --------------------------------------------------------------------------- #
# 5. which checkpoint the runtime serves
# --------------------------------------------------------------------------- #


@needs_weights
def test_the_default_checkpoint_is_still_the_shipped_one() -> None:
    """The joint model is recommended but NOT yet the default -- deliberately.

    `resolve_weights` carries the full argument and the four things that have to
    happen first; the short version is that `demo/app.py` still opens on
    `yolov8n_neudet`, and `false_alarm.py` and `export_model.py` still index
    `CLASS_NAMES` with a predicted class id. Flipping the default without those
    would make the console and the report generator serve different models.

    This test is the tripwire on flipping it silently, not a judgement that the
    shipped checkpoint is better.
    """
    assert resolve_weights().name == "best.pt"
    assert resolve_weights().parent.parent.name == "yolov8n_neudet"
    assert inference._PREFERENCE_ORDER[0].startswith("yolov8n_neudet/")


@needs_weights
def test_the_joint_checkpoint_is_reachable_without_a_path_literal() -> None:
    """"Not the default" must not mean "hard to use"."""
    assert inference.JOINT_WEIGHTS.is_file()
    assert resolve_weights(inference.JOINT_WEIGHTS) == JOINT_WEIGHTS.resolve()
    detector = inference.load_detector(
        weights=inference.JOINT_WEIGHTS, device="cpu", imgsz=TEST_IMGSZ
    )
    assert len(detector.class_names) == 10
