"""End-to-end smoke tests for the Jindal Stainless surface-defect system.

This is the suite that has to pass before the repository is considered coherent.
It exercises the pieces that everything else is built on, in the order they are
depended upon:

* **dataset integrity** - the split really is 1440/180/180, stratified 240/30/30
  per class, with no image leaking between splits by name or by content, and
  every label file well formed and inside the class contract;
* **the inference contract** - `InferenceResult` field types, value ranges,
  internal consistency, JSON round trip, and boxes that never leave the frame;
* **tiled inference** - fall-through when the frame fits one tile, global NMS
  across tile seams, and coordinates lifted correctly back into full-image space;
* **annotation** - shape, dtype and non-mutation of the caller's array;
* **severity scoring** - monotone in confidence, in area, in defect count and in
  defect seriousness, and bounded on 0-100;
* **robustness** - greyscale, RGBA, float, 1x1 and multi-megapixel frames all
  score instead of crashing;
* **the layers above inference** - the console's decode path, the coil report's
  disposition rules and the evaluator's matching, so a change to the contract
  breaks here rather than in front of an operator.

Anything that needs the GPU shares one session-scoped detector at imgsz 320 and
touches a handful of frames: this suite is expected to run while a training job
is using the same accelerator.

Run with:
    .venv/bin/python -m pytest tests/ -v
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _extra in (PROJECT_ROOT / "src", PROJECT_ROOT / "demo"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

# Ultralytics pip-installs its way out of a version conflict on import unless
# told not to. A test run must never mutate the environment it is testing.
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

import evaluate  # noqa: E402
import false_alarm  # noqa: E402
import inference  # noqa: E402
import report  # noqa: E402
from inference import (  # noqa: E402
    CLASS_COLORS,
    CLASS_NAMES,
    DEFECT_INFO,
    JOINT_CLASS_NAMES,
    SEVERITY_BANDS,
    Detection,
    InferenceResult,
    _severity_bucket,
    _tile_origins,
    aggregate_severity,
    resolve_weights,
    score_detection,
)

DATA_ROOT = PROJECT_ROOT / "data" / "neu-det"
SPLITS = ("train", "val", "test")
EXPECTED_IMAGES = {"train": 1440, "val": 180, "test": 180}
EXPECTED_PER_CLASS = {"train": 240, "val": 30, "test": 30}

# Small enough that the suite stays polite to a training job on the same GPU.
TEST_IMGSZ = 320
TEST_CONF = 0.25
TEST_IOU = 0.45


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


def _split_images(split: str) -> list[Path]:
    return sorted((DATA_ROOT / split / "images").glob("*.jpg"))


@pytest.fixture(scope="session")
def detector() -> Iterator[inference.DefectDetector]:
    """The one detector every GPU test shares, at a deliberately small input."""
    try:
        weights = resolve_weights()
    except FileNotFoundError as exc:
        pytest.skip(f"no trained checkpoint available: {exc}")
    model = inference.DefectDetector(
        weights=weights, device="auto", conf=TEST_CONF, iou=TEST_IOU, imgsz=TEST_IMGSZ
    )
    model.warmup(1)
    yield model


@pytest.fixture(scope="session")
def sample_frames() -> list[Path]:
    """One held-out frame from each of the six defect families."""
    paths = _split_images("test")
    if not paths:
        pytest.skip(f"no test images under {DATA_ROOT / 'test' / 'images'}")
    by_family: dict[str, Path] = {}
    for path in paths:
        by_family.setdefault(path.stem.rsplit("_", 1)[0], path)
    return [by_family[name] for name in CLASS_NAMES if name in by_family]


@pytest.fixture(scope="session")
def defective_result(
    detector: inference.DefectDetector, sample_frames: list[Path]
) -> tuple[Path, InferenceResult]:
    """A real frame the detector actually fires on, scored once and reused."""
    for path in sample_frames:
        result = detector.predict(path)
        if result.detections:
            return path, result
    pytest.skip("the current checkpoint detects nothing on any sample frame")


# --------------------------------------------------------------------------- #
# 1. dataset integrity
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("split", SPLITS)
def test_split_has_the_expected_image_count(split: str) -> None:
    assert len(_split_images(split)) == EXPECTED_IMAGES[split]


@pytest.mark.parametrize("split", SPLITS)
def test_split_is_stratified_across_all_six_classes(split: str) -> None:
    counts = Counter(p.stem.rsplit("_", 1)[0] for p in _split_images(split))
    assert set(counts) == set(CLASS_NAMES)
    assert set(counts.values()) == {EXPECTED_PER_CLASS[split]}


def test_no_image_filename_appears_in_two_splits() -> None:
    seen: dict[str, str] = {}
    for split in SPLITS:
        for path in _split_images(split):
            previous = seen.setdefault(path.name, split)
            assert previous == split, f"{path.name} is in both {previous} and {split}"


def test_no_image_content_appears_in_two_splits() -> None:
    """The stronger leakage guard: identical pixels under two different names."""
    seen: dict[str, tuple[str, str]] = {}
    for split in SPLITS:
        for path in _split_images(split):
            digest = hashlib.md5(path.read_bytes()).hexdigest()
            previous = seen.setdefault(digest, (split, path.name))
            assert previous[0] == split, (
                f"{path.name} ({split}) is byte-identical to {previous[1]} ({previous[0]})"
            )


@pytest.mark.parametrize("split", SPLITS)
def test_every_image_has_a_label_file(split: str) -> None:
    labels_dir = DATA_ROOT / split / "labels"
    missing = [p.name for p in _split_images(split) if not (labels_dir / f"{p.stem}.txt").is_file()]
    assert missing == []


@pytest.mark.parametrize("split", SPLITS)
def test_labels_are_well_formed_yolo(split: str) -> None:
    """Class ids inside the contract, coordinates normalised, boxes non-degenerate."""
    labels_dir = DATA_ROOT / split / "labels"
    for label_path in sorted(labels_dir.glob("*.txt")):
        for line_no, line in enumerate(label_path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            parts = line.split()
            where = f"{label_path.name}:{line_no}"
            assert len(parts) == 5, f"{where}: expected 5 fields, got {len(parts)}"
            class_id = int(parts[0])
            cx, cy, bw, bh = (float(v) for v in parts[1:])
            assert 0 <= class_id < len(CLASS_NAMES), f"{where}: class id {class_id}"
            assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0, f"{where}: centre off-frame"
            assert 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0, f"{where}: degenerate box"


def test_data_yaml_agrees_with_the_class_contract() -> None:
    """data.yaml is what the head was trained against; index order must match."""
    yaml_path = DATA_ROOT / "data.yaml"
    assert yaml_path.is_file(), f"missing {yaml_path}"
    names: dict[int, str] = {}
    declared_nc = None
    for line in yaml_path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("nc:"):
            declared_nc = int(stripped.split(":", 1)[1])
        elif stripped[:1].isdigit() and ":" in stripped:
            index, name = stripped.split(":", 1)
            names[int(index)] = name.strip()
    assert declared_nc == len(CLASS_NAMES)
    assert [names[i] for i in range(len(CLASS_NAMES))] == CLASS_NAMES


def test_every_severity_band_has_a_colour_in_every_presentation_layer() -> None:
    """Bands are defined once in inference.py; both renderers must cover them all."""
    import app

    assert set(app.SEVERITY_COLORS) == set(SEVERITY_BANDS)
    assert set(report._BAND_COLORS) == set(SEVERITY_BANDS) | {"clean"}


def test_every_class_has_colour_and_knowledge_base_entries() -> None:
    """A class the UI cannot colour or explain is a half-integrated class.

    The contract is the *joint* ten, not the NEU-DET six: `models/yolov8n_joint`
    is on disk and loadable, and a class it can emit with no colour and no
    knowledge-base entry is exactly the half-integration that made that
    checkpoint unservable.
    """
    assert list(DEFECT_INFO) == JOINT_CLASS_NAMES
    assert set(CLASS_COLORS) == set(JOINT_CLASS_NAMES)
    for name, info in DEFECT_INFO.items():
        assert set(info) == {"cause", "severity", "action"}, name
        assert info["severity"] in SEVERITY_BANDS, name
        assert info["cause"].strip() and info["action"].strip(), name


def test_the_six_class_neu_det_contract_is_still_six_long() -> None:
    """CLASS_NAMES is the NEU-DET contract and widening it breaks other things.

    `data/neu-det/data.yaml`, the dataset tests above, `false_alarm.py`'s
    "chance is 1/len(CLASS_NAMES)" and the console's atlas all index it.
    """
    assert len(CLASS_NAMES) == 6
    assert JOINT_CLASS_NAMES[:6] == CLASS_NAMES
    assert JOINT_CLASS_NAMES[6:] == list(inference.SEVERSTAL_CLASS_NAMES)
    # Built from a copy, so a consumer mutating CLASS_NAMES cannot reach in.
    assert JOINT_CLASS_NAMES is not CLASS_NAMES


def test_class_colours_stay_distinguishable_on_screen() -> None:
    """Ten boxes on one strip need ten telling-apart colours.

    The floor is the tightest pair the six shipped classes already ship with, so
    this asserts the four added classes did not make the palette worse.
    """
    import itertools
    import math

    pairs = {
        frozenset((a, b)): math.dist(CLASS_COLORS[a], CLASS_COLORS[b])
        for a, b in itertools.combinations(JOINT_CLASS_NAMES, 2)
    }
    shipped = min(
        d for k, d in pairs.items() if all(n in CLASS_NAMES for n in k)
    )
    added = min(
        d for k, d in pairs.items() if any(n not in CLASS_NAMES for n in k)
    )
    assert added >= shipped, (
        f"closest pair involving an added class is {added:.1f} apart, worse than "
        f"the tightest shipped pair at {shipped:.1f}"
    )


def test_the_severstal_tier_is_a_declared_placeholder_not_a_diagnosis() -> None:
    """The four Severstal classes have no published meaning; say so, do not invent.

    The Kaggle release publishes a mask value 1-4 and nothing else -- no defect
    name, no metallurgy, no grading rule -- which is why the classes are named
    after the mask value in the first place (`reports/severstal_dataset.md`).
    This pins the two ways that could quietly rot: someone writing a confident
    reheat-furnace root cause for `severstal_3`, and someone promoting the
    placeholder tier to one that stops a line.
    """
    assert inference.SEVERSTAL_TIER in SEVERITY_BANDS
    for name in inference.SEVERSTAL_CLASS_NAMES:
        info = DEFECT_INFO[name]
        assert info["severity"] == inference.SEVERSTAL_TIER
        # The entry must admit what is unknown rather than assert metallurgy.
        assert "no root cause is asserted" in info["cause"].lower()
        assert "publishes only the numeric class id" in info["cause"].lower()
        assert "quality department" in info["action"].lower()

    # A placeholder tier must not be able to stop a line on its own: the worst a
    # single such detection can score is below the "critical" cutoff, which is
    # the band `report.DispositionRules` turns into an unconditional HOLD.
    worst = max(
        score_detection(name, 1.0, 1.0) for name in inference.SEVERSTAL_CLASS_NAMES
    )
    assert _severity_bucket(worst) != "critical", worst
    assert not set(inference.SEVERSTAL_CLASS_NAMES) & set(
        report.DispositionRules().hold_classes
    )


# --------------------------------------------------------------------------- #
# 2. severity scoring
# --------------------------------------------------------------------------- #


def test_severity_bucket_boundaries_are_half_open() -> None:
    assert _severity_bucket(0.0) == "low"
    assert _severity_bucket(29.999) == "low"
    assert _severity_bucket(30.0) == "medium"
    assert _severity_bucket(54.999) == "medium"
    assert _severity_bucket(55.0) == "high"
    assert _severity_bucket(79.999) == "high"
    assert _severity_bucket(80.0) == "critical"
    assert _severity_bucket(1000.0) == "critical"


@pytest.mark.parametrize("name", CLASS_NAMES)
def test_score_is_monotone_in_confidence(name: str) -> None:
    scores = [score_detection(name, c, 0.05) for c in np.linspace(0.0, 1.0, 21)]
    assert all(b >= a for a, b in zip(scores, scores[1:]))
    assert scores[-1] > scores[0]


@pytest.mark.parametrize("name", CLASS_NAMES)
def test_score_is_monotone_in_area(name: str) -> None:
    scores = [score_detection(name, 0.8, a) for a in np.linspace(0.0, 1.0, 21)]
    assert all(b >= a for a, b in zip(scores, scores[1:]))
    assert scores[-1] > scores[0]


def test_score_ranks_classes_by_their_base_tier() -> None:
    """At equal confidence and area, a critical class must outscore a medium one."""
    ranked = sorted(
        CLASS_NAMES, key=lambda n: SEVERITY_BANDS.index(DEFECT_INFO[n]["severity"])
    )
    scores = [score_detection(n, 0.7, 0.1) for n in ranked]
    assert all(b >= a for a, b in zip(scores, scores[1:]))


@pytest.mark.parametrize("name", CLASS_NAMES)
def test_score_stays_on_the_0_100_scale(name: str) -> None:
    for conf in (-1.0, 0.0, 0.5, 1.0, 2.0):
        for area in (-1.0, 0.0, 0.25, 1.0, 5.0):
            assert 0.0 <= score_detection(name, conf, area) <= 100.0


def test_aggregate_of_nothing_is_zero() -> None:
    assert aggregate_severity([]) == 0.0


def test_aggregate_of_one_detection_is_its_own_score() -> None:
    for score in (0.0, 12.5, 47.3, 99.9):
        assert aggregate_severity([score]) == pytest.approx(score, abs=1e-9)


def test_aggregate_is_monotone_in_defect_count() -> None:
    """Adding a defect can never make a coil look better."""
    running = [40.0]
    previous = aggregate_severity(running)
    for _ in range(8):
        running.append(40.0)
        current = aggregate_severity(running)
        assert current >= previous
        previous = current


def test_aggregate_is_monotone_in_defect_severity() -> None:
    base = [30.0, 20.0, 10.0]
    worse = [60.0, 20.0, 10.0]
    assert aggregate_severity(worse) > aggregate_severity(base)


def test_aggregate_is_bounded_at_100() -> None:
    assert aggregate_severity([100.0] * 50) <= 100.0
    assert aggregate_severity([99.0] * 200) <= 100.0


def test_detection_severity_band_matches_its_own_score() -> None:
    """The band on a Detection must be the band its score buckets into."""
    for name in CLASS_NAMES:
        for conf in (0.26, 0.6, 0.99):
            for area in (0.001, 0.05, 0.6):
                score = score_detection(name, conf, area)
                assert _severity_bucket(score) in SEVERITY_BANDS


# --------------------------------------------------------------------------- #
# 3. tiling geometry
# --------------------------------------------------------------------------- #


def test_a_frame_that_fits_one_tile_yields_one_origin() -> None:
    assert _tile_origins(320, 320, 256) == [0]
    assert _tile_origins(200, 320, 256) == [0]


@pytest.mark.parametrize("extent", [321, 500, 1000, 2049, 4000])
@pytest.mark.parametrize("overlap", [0.0, 0.2, 0.5])
def test_tile_origins_cover_the_whole_extent(extent: int, overlap: float) -> None:
    tile = 320
    stride = max(1, int(round(tile * (1.0 - overlap))))
    origins = _tile_origins(extent, tile, stride)

    assert origins[0] == 0
    assert origins[-1] == extent - tile, "last window must sit flush with the far edge"
    assert origins == sorted(origins)
    covered = np.zeros(extent, dtype=bool)
    for x0 in origins:
        assert 0 <= x0 <= extent - tile
        covered[x0 : x0 + tile] = True
    assert covered.all(), "sliding window left a gap in coverage"


# --------------------------------------------------------------------------- #
# 4. the inference contract
# --------------------------------------------------------------------------- #


def _assert_result_contract(
    result: InferenceResult, class_names: Sequence[str] = CLASS_NAMES
) -> None:
    """Every invariant `InferenceResult` promises its consumers.

    `class_names` is the class list of the head that produced the result -- the
    NEU-DET six by default, the joint ten when a 10-class checkpoint was served.
    """
    width, height = result.image_size
    assert isinstance(width, int) and isinstance(height, int)
    assert width > 0 and height > 0

    assert result.verdict in ("PASS", "DEFECT")
    assert result.defect_count == len(result.detections)
    assert (result.verdict == "DEFECT") == bool(result.detections)
    assert 0.0 <= result.max_confidence <= 1.0
    assert 0.0 <= result.severity_score <= 100.0
    assert result.conf_threshold >= 0.0 and result.iou_threshold >= 0.0
    assert result.model_name

    for value in (
        result.preprocess_ms,
        result.inference_ms,
        result.postprocess_ms,
        result.total_ms,
    ):
        assert isinstance(value, float) and np.isfinite(value) and value >= 0.0

    if result.detections:
        assert result.severity_score == pytest.approx(
            aggregate_severity([d.severity_score for d in result.detections])
        )
        dominant = max(result.detections, key=lambda d: d.confidence)
        assert result.dominant_class == dominant.class_name
        assert result.max_confidence == pytest.approx(dominant.confidence)
        # Detections are handed to the UI already ranked.
        confidences = [d.confidence for d in result.detections]
        assert confidences == sorted(confidences, reverse=True)
    else:
        assert result.dominant_class is None
        assert result.max_confidence == 0.0
        assert result.severity_score == 0.0

    frame_area = float(width * height)
    for det in result.detections:
        assert isinstance(det, Detection)
        assert isinstance(det.class_id, int) and 0 <= det.class_id < len(class_names)
        assert det.class_name == class_names[det.class_id]
        assert 0.0 <= det.confidence <= 1.0
        assert det.severity in SEVERITY_BANDS
        # The band on the record is the band of the score on the record, and the
        # frame aggregates exactly those scores.
        assert det.severity == _severity_bucket(det.severity_score)
        assert 0.0 <= det.severity_score <= 100.0

        x1, y1, x2, y2 = det.bbox_xyxy
        assert 0.0 <= x1 <= x2 <= width, f"box x out of frame: {det.bbox_xyxy}"
        assert 0.0 <= y1 <= y2 <= height, f"box y out of frame: {det.bbox_xyxy}"
        assert det.area_px == pytest.approx((x2 - x1) * (y2 - y1), abs=1e-6)
        assert det.area_frac == pytest.approx(det.area_px / frame_area, abs=1e-9)
        assert 0.0 <= det.area_frac <= 1.0


def test_single_frame_results_satisfy_the_contract(
    detector: inference.DefectDetector, sample_frames: list[Path]
) -> None:
    for result in detector.predict_batch(sample_frames, batch_size=len(sample_frames)):
        _assert_result_contract(result)


def test_result_survives_a_json_round_trip(
    defective_result: tuple[Path, InferenceResult]
) -> None:
    _, result = defective_result
    payload = json.loads(json.dumps(result.to_dict()))  # must be plain JSON, no NaN

    assert payload["verdict"] == result.verdict
    assert payload["defect_count"] == result.defect_count
    assert payload["dominant_class"] == result.dominant_class
    assert payload["image_size"] == [result.image_size[0], result.image_size[1]]
    assert set(payload["timing_ms"]) == {"preprocess", "inference", "postprocess", "total"}
    assert len(payload["detections"]) == len(result.detections)

    for encoded, det in zip(payload["detections"], result.detections):
        assert encoded["class_id"] == det.class_id
        assert encoded["class_name"] == det.class_name
        assert encoded["severity"] == det.severity
        assert encoded["confidence"] == pytest.approx(det.confidence, abs=5e-5)
        assert len(encoded["bbox_xyxy"]) == 4
        for got, want in zip(encoded["bbox_xyxy"], det.bbox_xyxy):
            assert got == pytest.approx(want, abs=5e-3)


def test_batch_scoring_agrees_with_single_frame_scoring(
    detector: inference.DefectDetector, sample_frames: list[Path]
) -> None:
    """Throughput path and latency path must not disagree about the boxes."""
    frames = sample_frames[:3]
    singles = [detector.predict(p) for p in frames]
    batched = detector.predict_batch(frames, batch_size=len(frames))

    assert len(singles) == len(batched)
    for one, many in zip(singles, batched):
        assert one.image_size == many.image_size
        assert one.verdict == many.verdict
        assert one.defect_count == many.defect_count
        for a, b in zip(one.detections, many.detections):
            assert a.class_id == b.class_id
            assert a.confidence == pytest.approx(b.confidence, abs=1e-3)
            for va, vb in zip(a.bbox_xyxy, b.bbox_xyxy):
                assert va == pytest.approx(vb, abs=0.01)


def test_boxes_are_clamped_into_the_frame(detector: inference.DefectDetector) -> None:
    """Letterbox rounding lets the head place a box slightly off-frame.

    Real NEU-DET crops rarely trigger it, so the clamp is exercised directly on the
    record builder: a box hanging off every edge must come back inside the frame,
    with its area recomputed from the clamped corners rather than the raw ones.
    """
    raw = np.array(
        [
            [-30.0, -12.0, 260.0, 220.0, 0.77, 1],  # over all four edges
            [-50.0, -50.0, -10.0, -10.0, 0.55, 0],  # entirely outside
        ],
        dtype=np.float64,
    )
    detections = detector._build_detections(raw, width=200, height=200)

    assert len(detections) == 2
    for det in detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        assert 0.0 <= x1 <= x2 <= 200.0
        assert 0.0 <= y1 <= y2 <= 200.0
        assert det.area_px == pytest.approx((x2 - x1) * (y2 - y1))
        assert 0.0 <= det.area_frac <= 1.0

    inside, outside = detections
    assert inside.bbox_xyxy == (0.0, 0.0, 200.0, 200.0)
    assert inside.area_frac == pytest.approx(1.0)
    # Nothing of it survives inside the frame: a degenerate box at the edge, not
    # an inverted one.
    assert outside.bbox_xyxy == (0.0, 0.0, 0.0, 0.0)
    assert outside.area_px == 0.0


def test_raising_the_confidence_threshold_only_removes_boxes(
    detector: inference.DefectDetector, defective_result: tuple[Path, InferenceResult]
) -> None:
    path, low = defective_result
    previous = detector.conf
    try:
        detector.conf = 0.9
        high = detector.predict(path)
    finally:
        detector.conf = previous
    assert high.defect_count <= low.defect_count
    assert all(d.confidence >= 0.9 for d in high.detections)


# --------------------------------------------------------------------------- #
# 5. tiled inference
# --------------------------------------------------------------------------- #


def test_tiling_falls_through_when_the_frame_fits_one_tile(
    detector: inference.DefectDetector, defective_result: tuple[Path, InferenceResult]
) -> None:
    path, single = defective_result
    tiled = detector.predict_tiled(path, tile=TEST_IMGSZ, overlap=0.2)

    assert tiled.image_size == single.image_size
    assert tiled.defect_count == single.defect_count
    for a, b in zip(tiled.detections, single.detections):
        assert a.class_id == b.class_id
        for va, vb in zip(a.bbox_xyxy, b.bbox_xyxy):
            assert va == pytest.approx(vb, abs=0.01)
    # The decode is charged to the fall-through result, not silently dropped.
    assert tiled.total_ms >= single.total_ms * 0.0
    assert tiled.preprocess_ms > 0.0


@pytest.fixture(scope="session")
def strip(defective_result: tuple[Path, InferenceResult]) -> np.ndarray:
    """A synthetic full-width strip: one defective frame repeated five times.

    Repeating a real frame means the ground truth is known by construction - the
    same defect appears at five known x offsets - which is what makes the
    seam-merging and coordinate-lifting assertions below meaningful.
    """
    import cv2

    path, _ = defective_result
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(np.concatenate([rgb] * 5, axis=1))


def test_tiled_inference_lifts_boxes_into_full_image_coordinates(
    detector: inference.DefectDetector, strip: np.ndarray
) -> None:
    height, width = strip.shape[:2]
    result = detector.predict_tiled(strip, tile=TEST_IMGSZ, overlap=0.2)

    assert result.image_size == (width, height)
    _assert_result_contract(result)
    assert result.detections, "tiled inference found nothing on a strip of real defects"


def test_tiled_inference_finds_defects_beyond_the_first_tile(
    detector: inference.DefectDetector, strip: np.ndarray
) -> None:
    """A tiled pass must actually walk the strip, not score its left edge."""
    width = strip.shape[1]
    result = detector.predict_tiled(strip, tile=TEST_IMGSZ, overlap=0.2)
    centres = [0.5 * (d.bbox_xyxy[0] + d.bbox_xyxy[2]) for d in result.detections]
    assert max(centres) > width * 0.6, "no detection in the right-hand half of the strip"


def test_global_nms_deduplicates_defects_across_tile_seams(
    detector: inference.DefectDetector, strip: np.ndarray
) -> None:
    """A defect straddling a seam must survive once, not once per tile."""
    result = detector.predict_tiled(strip, tile=TEST_IMGSZ, overlap=0.4)
    boxes = np.array([d.bbox_xyxy for d in result.detections], dtype=np.float64)
    classes = np.array([d.class_id for d in result.detections], dtype=np.int64)
    if len(boxes) < 2:
        pytest.skip("fewer than two detections; nothing to deduplicate")

    ious = evaluate.iou_matrix(boxes, boxes)
    same_class = classes[:, None] == classes[None, :]
    np.fill_diagonal(ious, 0.0)
    worst = float(np.max(np.where(same_class, ious, 0.0)))
    assert worst <= detector.iou + 1e-6, (
        f"two surviving boxes of the same class overlap at IoU {worst:.3f}, "
        f"above the NMS threshold {detector.iou}"
    )


def test_tiling_overlap_is_clamped_instead_of_exploding(
    detector: inference.DefectDetector, strip: np.ndarray
) -> None:
    """A bad UI slider must degrade, not stall the line with a tile avalanche."""
    result = detector.predict_tiled(strip, tile=TEST_IMGSZ, overlap=5.0)
    _assert_result_contract(result)


# --------------------------------------------------------------------------- #
# 5b. tiled inference is LOSSLESS -- the claim the deck makes about it
# --------------------------------------------------------------------------- #
#
# "Tiling preserves full optical resolution, unlike downscaling a whole frame"
# is a load-bearing claim: it is why a 2048 px strip is not simply squashed into
# the network. These tests hold the pipeline to it byte for byte, because the
# claim is one careless `tile=` default away from being false.


@pytest.fixture
def resize_spy(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    """Record every `cv2.resize` any layer makes, ours or the framework's.

    Ultralytics' letterbox does `import cv2` at module scope and calls
    `cv2.resize`, so patching the attribute on the shared module object catches
    it too. That is the only place a resample can hide in this path.
    """
    import cv2

    calls: list[tuple] = []
    real = cv2.resize

    def spy(src, dsize, *args, **kwargs):
        calls.append((tuple(np.shape(src)[:2]), tuple(dsize)))
        return real(src, dsize, *args, **kwargs)

    monkeypatch.setattr(cv2, "resize", spy)
    return calls


@pytest.fixture
def crop_spy(
    detector: inference.DefectDetector, monkeypatch: pytest.MonkeyPatch
) -> list[np.ndarray]:
    """Copy of every array handed to the network, in tile order."""
    seen: list[np.ndarray] = []
    real = detector._forward

    def spy(bgr_batch: list[np.ndarray]):
        seen.extend(np.array(b, copy=True) for b in bgr_batch)
        return real(bgr_batch)

    monkeypatch.setattr(detector, "_forward", spy)
    return seen


def _tile_plan(
    width: int, height: int, tile: int, overlap: float
) -> tuple[list[tuple[int, int]], int, int]:
    """The origins `predict_tiled` will use, recomputed from its own helper."""
    tile_w, tile_h = min(tile, width), min(tile, height)
    xs = _tile_origins(width, tile_w, max(1, int(round(tile_w * (1.0 - overlap)))))
    ys = _tile_origins(height, tile_h, max(1, int(round(tile_h * (1.0 - overlap)))))
    return [(x0, y0) for y0 in ys for x0 in xs], tile_w, tile_h


def test_tiling_at_the_default_tile_size_never_resamples(
    detector: inference.DefectDetector,
    strip: np.ndarray,
    resize_spy: list[tuple],
) -> None:
    """Zero interpolation calls in the whole tiled pass. Not "few". Zero.

    At tile == imgsz the crop is already the network input size, so the
    framework's letterbox computes r == 1.0 and zero padding and short-circuits
    its resize entirely. Every pixel the network sees is a source pixel.
    """
    detector.predict_tiled(strip, tile=detector.imgsz, overlap=0.2)
    assert resize_spy == [], f"tiled path resampled: {resize_spy}"


def test_every_tile_is_a_byte_exact_sub_rectangle_of_the_source(
    detector: inference.DefectDetector,
    strip: np.ndarray,
    crop_spy: list[np.ndarray],
) -> None:
    """Not "close to" the source crop. Equal to it, under np.array_equal."""
    height, width = strip.shape[:2]
    detector.predict_tiled(strip, tile=detector.imgsz, overlap=0.2)

    plan, tile_w, tile_h = _tile_plan(width, height, detector.imgsz, 0.2)
    assert len(crop_spy) == len(plan), "tile count does not match the window plan"
    for (x0, y0), crop_bgr in zip(plan, crop_spy):
        # predict_tiled hands the framework BGR; the fixture strip is RGB.
        want = strip[y0 : y0 + tile_h, x0 : x0 + tile_w][:, :, ::-1]
        assert crop_bgr.shape == want.shape, (x0, y0, crop_bgr.shape, want.shape)
        assert np.array_equal(crop_bgr, want), f"tile at ({x0}, {y0}) is not the source"


def test_the_window_plan_leaves_no_pixel_unseen(
    detector: inference.DefectDetector, strip: np.ndarray
) -> None:
    """Lossless is worth nothing if the windows skip a stripe of the coil."""
    height, width = strip.shape[:2]
    plan, tile_w, tile_h = _tile_plan(width, height, detector.imgsz, 0.2)
    cover = np.zeros((height, width), dtype=np.int32)
    for x0, y0 in plan:
        cover[y0 : y0 + tile_h, x0 : x0 + tile_w] += 1
    assert int(cover.min()) >= 1, f"{int((cover == 0).sum())} source pixels never scored"


def test_whole_frame_prediction_does_resample_and_by_how_much(
    detector: inference.DefectDetector,
    strip: np.ndarray,
    resize_spy: list[tuple],
) -> None:
    """The control. `predict()` on the same strip is one big interpolation.

    This is the comparison the deck rests on, so it is asserted rather than
    asserted-about: one resize, and the network sees a small fraction of the
    frame's pixels.
    """
    height, width = strip.shape[:2]
    detector.predict(strip)

    assert len(resize_spy) == 1, resize_spy
    (src_h, src_w), (dst_w, dst_h) = resize_spy[0]
    assert (src_h, src_w) == (height, width)
    assert dst_w < src_w and dst_h < src_h
    assert (dst_w * dst_h) / (src_w * src_h) < 0.5


@pytest.mark.parametrize("tile", [128, 512])
def test_a_non_default_tile_size_is_resampling_and_is_documented_as_such(
    detector: inference.DefectDetector,
    strip: np.ndarray,
    resize_spy: list[tuple],
    tile: int,
) -> None:
    """tile != imgsz breaks the contract: below it upscales, above it downscales.

    Kept as a test rather than a comment because the failure is invisible -- the
    call still works, still returns boxes, and quietly interpolates every tile.
    """
    detector.predict_tiled(strip, tile=tile, overlap=0.0)
    assert resize_spy, f"tile={tile} with imgsz={detector.imgsz} should resample"
    for (src_h, src_w), (dst_w, dst_h) in resize_spy:
        assert (src_h, src_w) != (dst_h, dst_w), "recorded a resize that was a no-op"
        grew = dst_w * dst_h > src_w * src_h
        assert grew is (tile < detector.imgsz), (
            f"tile={tile}, imgsz={detector.imgsz}: ({src_h}, {src_w}) -> "
            f"({dst_h}, {dst_w}) is the wrong direction"
        )


def test_boxes_lift_into_full_image_coordinates_exactly(
    detector: inference.DefectDetector, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The offset arithmetic is exact, not approximately exact.

    The network is stubbed with a fixed box in tile coordinates so the expected
    full-image answer is known in closed form: every tile must contribute
    box + (x0, y0), to the last bit, before NMS gets a say.
    """
    tile = detector.imgsz
    height, width = tile, tile * 3
    frame = np.full((height, width, 3), 120, dtype=np.uint8)
    box = np.array([[10.0, 20.0, 60.0, 90.0, 0.9, 1]], dtype=np.float64)

    monkeypatch.setattr(
        detector, "_forward",
        lambda batch: ([box.copy() for _ in batch],
                       {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}),
    )
    # overlap 0 so the three windows are disjoint and nothing is NMS'd away.
    result = detector.predict_tiled(frame, tile=tile, overlap=0.0)

    plan, _, _ = _tile_plan(width, height, tile, 0.0)
    expected = sorted((x0 + 10.0, y0 + 20.0, x0 + 60.0, y0 + 90.0) for x0, y0 in plan)
    assert sorted(d.bbox_xyxy for d in result.detections) == expected


def test_tiled_boxes_are_exactly_the_per_tile_boxes_shifted(
    detector: inference.DefectDetector, strip: np.ndarray
) -> None:
    """Same check against the real network: tiling adds an offset and nothing else.

    Every surviving tiled box must be one of the boxes the same detector produces
    on that crop scored alone, translated by the crop origin. NMS may drop boxes
    across seams, so the relation asserted is containment, not equality.
    """
    height, width = strip.shape[:2]
    tile = detector.imgsz
    plan, tile_w, tile_h = _tile_plan(width, height, tile, 0.0)

    per_tile: set[tuple[int, tuple[float, ...]]] = set()
    for x0, y0 in plan:
        crop = np.ascontiguousarray(strip[y0 : y0 + tile_h, x0 : x0 + tile_w])
        for det in detector.predict(crop).detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            per_tile.add(
                (det.class_id, (round(x1 + x0, 3), round(y1 + y0, 3),
                                round(x2 + x0, 3), round(y2 + y0, 3)))
            )

    tiled = detector.predict_tiled(strip, tile=tile, overlap=0.0)
    if not tiled.detections:
        pytest.skip("the current checkpoint fires on nothing in this strip")
    for det in tiled.detections:
        key = (det.class_id, tuple(round(v, 3) for v in det.bbox_xyxy))
        assert key in per_tile, f"tiled box {key} is not a shifted per-tile box"


# --------------------------------------------------------------------------- #
# 6. annotation
# --------------------------------------------------------------------------- #


def test_annotation_shape_dtype_and_input_immutability(
    detector: inference.DefectDetector, defective_result: tuple[Path, InferenceResult]
) -> None:
    path, result = defective_result
    source = inference._to_rgb(path)
    original = source.copy()

    annotated = detector.annotate(source, result)

    assert annotated.shape == source.shape
    assert annotated.dtype == np.uint8
    assert annotated.ndim == 3 and annotated.shape[2] == 3
    assert annotated is not source
    assert np.array_equal(source, original), "annotate() mutated the caller's array"
    assert not np.array_equal(annotated, original), "annotate() drew nothing"


def test_annotation_of_a_clean_frame_is_a_faithful_copy(
    detector: inference.DefectDetector
) -> None:
    blank = np.full((64, 96, 3), 130, dtype=np.uint8)
    annotated = detector.annotate(blank, InferenceResult(image_size=(96, 64)))
    assert annotated.shape == blank.shape
    assert annotated.dtype == np.uint8
    assert np.array_equal(annotated, blank)


@pytest.mark.parametrize("size", [(48, 48), (200, 200), (1400, 900)])
def test_annotation_scales_to_any_frame_size(
    detector: inference.DefectDetector, size: tuple[int, int]
) -> None:
    height, width = size
    canvas = np.full((height, width, 3), 150, dtype=np.uint8)
    detection = Detection(
        class_id=1,
        class_name="inclusion",
        confidence=0.83,
        bbox_xyxy=(2.0, 2.0, width - 3.0, height - 3.0),
        area_px=float((width - 5) * (height - 5)),
        area_frac=0.5,
        severity="critical",
    )
    result = InferenceResult(detections=[detection], image_size=(width, height))
    annotated = detector.annotate(canvas, result)
    assert annotated.shape == (height, width, 3)
    assert annotated.dtype == np.uint8
    assert not np.array_equal(annotated, canvas)


# --------------------------------------------------------------------------- #
# 7. robustness of the input path
# --------------------------------------------------------------------------- #


def test_greyscale_two_dimensional_array_is_accepted(
    detector: inference.DefectDetector
) -> None:
    grey = np.random.default_rng(0).integers(80, 170, (200, 200), dtype=np.uint8)
    result = detector.predict(grey)
    assert result.image_size == (200, 200)
    _assert_result_contract(result)


def test_rgba_and_float_arrays_are_normalised(detector: inference.DefectDetector) -> None:
    rgba = np.dstack(
        [np.full((120, 160, 3), 120, np.uint8), np.full((120, 160, 1), 255, np.uint8)]
    )
    assert detector.predict(rgba).image_size == (160, 120)

    floats = np.full((120, 160, 3), 300.7, dtype=np.float32)  # out of range on purpose
    assert detector.predict(floats).image_size == (160, 120)


def test_one_by_one_image_scores_instead_of_crashing(
    detector: inference.DefectDetector
) -> None:
    result = detector.predict(np.full((1, 1, 3), 128, dtype=np.uint8))
    assert result.image_size == (1, 1)
    assert result.verdict == "PASS"
    _assert_result_contract(result)


def test_multi_megapixel_frame_is_handled(detector: inference.DefectDetector) -> None:
    """A mill-resolution capture must go through the single-pass path intact."""
    big = np.random.default_rng(1).integers(90, 160, (1600, 2400, 3), dtype=np.uint8)
    result = detector.predict(big)
    assert result.image_size == (2400, 1600)
    _assert_result_contract(result)


def test_unsupported_input_types_fail_loudly(detector: inference.DefectDetector) -> None:
    with pytest.raises(TypeError):
        detector.predict(42)
    with pytest.raises(ValueError):
        detector.predict(np.zeros((8, 8, 5), dtype=np.uint8))
    with pytest.raises(FileNotFoundError):
        detector.predict(PROJECT_ROOT / "definitely" / "not" / "a" / "frame.jpg")


def test_empty_batch_returns_an_empty_list(detector: inference.DefectDetector) -> None:
    assert detector.predict_batch([]) == []


# --------------------------------------------------------------------------- #
# 8. the console layer (headless)
# --------------------------------------------------------------------------- #


def _png_bytes(array: np.ndarray, mode: str | None = None) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(array, mode=mode).save(buffer, format="PNG")
    return buffer.getvalue()


def test_console_decodes_greyscale_to_rgb() -> None:
    import app

    grey = np.random.default_rng(2).integers(0, 255, (64, 96), dtype=np.uint8)
    loaded = app.decode_image(_png_bytes(grey, mode="L"), "grey.png")

    assert loaded.array.shape == (64, 96, 3)
    assert loaded.array.dtype == np.uint8
    assert loaded.size == (96, 64)
    assert loaded.source_mode == "L"
    assert "Converted from L" in loaded.note


def test_console_rejects_unusable_frames() -> None:
    import app

    with pytest.raises(app.ImageLoadError):
        app.decode_image(b"", "empty.png")
    with pytest.raises(app.ImageLoadError):
        app.decode_image(b"this is not an image", "junk.png")
    with pytest.raises(app.ImageLoadError):
        # Long edge past MAX_ASPECT_RATIO x the short edge: the short edge would
        # letterbox down to zero pixels and take the whole batch with it.
        sliver = np.full((2, 2 * app.MAX_ASPECT_RATIO + 20, 3), 128, dtype=np.uint8)
        app.decode_image(_png_bytes(sliver), "sliver.png")


def test_console_display_rescaling_keeps_frames_readable() -> None:
    import app

    small = app.display_array(np.zeros((200, 200, 3), dtype=np.uint8))
    assert max(small.shape[:2]) >= app.MIN_DISPLAY_EDGE
    assert small.dtype == np.uint8

    large = app.display_array(np.zeros((4000, 2000, 3), dtype=np.uint8))
    assert max(large.shape[:2]) <= app.MAX_DISPLAY_EDGE

    untouched = np.zeros((900, 700, 3), dtype=np.uint8)
    assert app.display_array(untouched) is untouched


def test_console_tables_keep_their_schema_when_empty() -> None:
    import app

    assert list(app.frame_table([], []).columns) == list(app.FRAME_COLUMNS)
    assert app.detection_table([], []).empty
    assert list(app.class_distribution([])["defect"]) == CLASS_NAMES
    assert app.latency_stats([])["fps_mean"] == 0.0
    assert app.running_defect_rate([]) == []


def test_console_defaults_to_the_evaluated_operating_point(tmp_path: Path) -> None:
    """src/evaluate.py writes the threshold; the console has to actually read it."""
    import app

    written = tmp_path / "operating_point.json"
    written.write_text(
        json.dumps(
            {
                "conf_threshold": 0.35,
                "iou_threshold": 0.5,
                "tuned_on_split": "val",
                "expected_at_threshold": {
                    "defect_detection_rate": 0.91,
                    "false_alarm_rate": 0.22,
                },
            }
        )
    )
    point = app.load_operating_point(written)
    assert point.conf == pytest.approx(0.35)
    assert point.iou == pytest.approx(0.5)
    assert point.tuned_on == "val"
    assert point.detection_rate == pytest.approx(0.91)
    assert "operating_point.json" in point.summary

    # Off-grid and out-of-range thresholds are snapped onto the slider, never
    # handed to st.slider as an illegal default.
    low, high, step = app.CONF_SLIDER
    for raw in (0.0, 0.031, 0.5, 0.99, 7.0):
        written.write_text(json.dumps({"conf_threshold": raw}))
        snapped = app.load_operating_point(written).conf
        assert low <= snapped <= high
        assert round(snapped / step) == pytest.approx(snapped / step, abs=1e-9)


@pytest.mark.parametrize(
    "payload", [None, "not json at all", '{"no_threshold_here": 1}', '{"conf_threshold": "x"}']
)
def test_console_falls_back_when_no_operating_point_is_readable(
    tmp_path: Path, payload: str | None
) -> None:
    """A missing or malformed report must not stop the console from starting."""
    import app

    target = tmp_path / "operating_point.json"
    if payload is not None:
        target.write_text(payload)
    point = app.load_operating_point(target)
    assert point.conf == app.FALLBACK_CONF
    assert point.iou == app.FALLBACK_IOU
    assert point.source is None
    assert "make eval" in point.summary


def test_the_shipped_operating_point_is_loadable_if_present() -> None:
    """When an evaluation has been run, its threshold must survive the round trip."""
    import app

    if not app.OPERATING_POINT_PATH.is_file():
        pytest.skip("no evaluation has been run yet")
    recorded = json.loads(app.OPERATING_POINT_PATH.read_text())
    point = app.load_operating_point()
    assert point.source == app.OPERATING_POINT_PATH
    assert point.conf == pytest.approx(_snap_conf(recorded["conf_threshold"]))


def _snap_conf(value: float) -> float:
    import app

    low, high, step = app.CONF_SLIDER
    return min(max(round(float(value) / step) * step, low), high)


def test_the_console_renders_without_raising() -> None:
    """Run the real Streamlit script end to end and assert it comes up clean.

    Every other console test exercises the pure layer above the `# UI` divider.
    This one executes the whole script through Streamlit's own harness, which is
    the only check that catches a widget the API no longer accepts, a layout call
    that raises on an empty state, or a detector that will not load in the app's
    own default configuration. It is the slowest test here (it loads a checkpoint
    at the console's own input size) and it is the one worth the seconds.
    """
    from streamlit.testing.v1 import AppTest

    if not (PROJECT_ROOT / "models").is_dir():
        pytest.skip("no models/ directory; the console would show its no-checkpoint screen")

    app_test = AppTest.from_file(str(PROJECT_ROOT / "demo" / "app.py"), default_timeout=240).run()

    assert not app_test.exception, [e.value for e in app_test.exception]
    assert not app_test.error, [e.value for e in app_test.error]
    assert len(app_test.tabs) == 4, "single frame / batch / simulation / atlas"

    sliders = {s.label: s.value for s in app_test.sidebar.slider}
    assert sliders["Confidence"] == pytest.approx(load_operating_point_conf())
    assert "Checkpoint" in [s.label for s in app_test.sidebar.selectbox]


def test_the_console_offers_both_trained_models_and_defaults_to_the_better_one() -> None:
    """The picker must list both runs, and open on the one that scores highest.

    At the 320 px training size yolov8n beats yolov8s on the held-out test split
    (mAP50 0.7286 vs 0.6598, bootstrap CI on the gap excludes zero); at each
    model's own best input size the two are tied and nano is 3.7x smaller. Either
    way nano is the one to open on, while still offering small, because the
    comparison is the point of having trained both. This guards the ordering in
    `preferred_checkpoint_index`, which is easy to "fix" back to the conventional
    bigger-is-better order by someone who has not read the evaluation.
    """
    import app

    checkpoints = app.discover_checkpoints()
    if not checkpoints:
        pytest.skip("no checkpoints on disk")

    runs = {c.run for c in checkpoints}
    for expected in ("yolov8n_neudet", "yolov8s_neudet"):
        if expected not in runs:
            pytest.skip(f"{expected} has not been trained on this working tree")

    default = checkpoints[app.preferred_checkpoint_index(checkpoints)]
    assert (default.run, default.filename) == ("yolov8n_neudet", "best.pt")

    # And the engine must agree with the console, or the report generator and the
    # UI would quietly serve two different models.
    assert inference.resolve_weights() == default.path.resolve()


def test_the_console_opens_at_the_input_size_the_study_chose() -> None:
    """The default network input must be the one selection actually picked.

    Input size is an accuracy control, not a quality-of-rendering control. This
    checkpoint was trained at 320 px and its test mAP50 falls 0.752 -> 0.729 ->
    0.623 -> 0.344 across 256 / 320 / 416 / 640 px, so a console that opens at
    640 px shows a detector finding less than half the defects it can find. It
    did open at 640 until this was caught. The default is pinned here against
    reports/model_study.json rather than to a literal, so the two cannot drift.
    """
    import app

    assert app.DEFAULT_IMGSZ in app.IMGSZ_OPTIONS

    study = PROJECT_ROOT / "reports" / "model_study.json"
    if not study.is_file():
        pytest.skip("model_study.json has not been generated on this working tree")
    selection = json.loads(study.read_text()).get("selection", {})
    chosen = selection.get("imgsz")
    if chosen is None:
        pytest.skip("model_study.json carries no selected input size")
    assert app.DEFAULT_IMGSZ == int(chosen), (
        f"console opens at {app.DEFAULT_IMGSZ} px but the study selected {chosen} px"
    )


def load_operating_point_conf() -> float:
    """The confidence the console should be defaulting to on this working tree."""
    import app

    return app.load_operating_point().conf


def test_console_bands_a_frame_exactly_as_the_engine_does(
    defective_result: tuple[Path, InferenceResult]
) -> None:
    import app

    _, result = defective_result
    row = app.frame_record("frame.jpg", result)
    assert row["severity_band"] == _severity_bucket(result.severity_score)
    assert row["verdict"] == result.verdict
    assert row["defects"] == result.defect_count


# --------------------------------------------------------------------------- #
# 9. the layers above inference
# --------------------------------------------------------------------------- #


def _stats(**overrides) -> report.CoilStats:
    base = dict(
        total_frames=100,
        defect_frames=1,
        clean_frames=99,
        defect_rate=0.01,
        total_detections=1,
        detections_per_frame=0.01,
        class_counts={name: 0 for name in CLASS_NAMES},
        class_frame_counts={name: 0 for name in CLASS_NAMES},
        band_counts={"clean": 99, "low": 1, "medium": 0, "high": 0, "critical": 0},
        mean_severity=2.0,
        max_severity=20.0,
        p95_severity=10.0,
        mean_latency_ms=12.0,
        throughput_fps=83.0,
    )
    base.update(overrides)
    return report.CoilStats(**base)


def test_a_clean_coil_is_accepted() -> None:
    disposition, reasons = report.DispositionRules().evaluate(_stats())
    assert disposition == "ACCEPT"
    assert reasons


def test_a_single_inclusion_frame_holds_the_coil() -> None:
    """Inclusions cannot be removed downstream, so one frame is enough."""
    counts = {name: 0 for name in CLASS_NAMES}
    counts["inclusion"] = 1
    disposition, reasons = report.DispositionRules().evaluate(
        _stats(class_frame_counts=counts)
    )
    assert disposition == "HOLD"
    assert any("inclusion" in reason for reason in reasons)


def test_a_critical_frame_holds_the_coil() -> None:
    bands = {"clean": 98, "low": 1, "medium": 0, "high": 0, "critical": 1}
    disposition, _ = report.DispositionRules().evaluate(_stats(band_counts=bands))
    assert disposition == "HOLD"


def test_a_high_severity_frame_downgrades_the_coil() -> None:
    bands = {"clean": 98, "low": 1, "medium": 0, "high": 1, "critical": 0}
    disposition, _ = report.DispositionRules().evaluate(_stats(band_counts=bands))
    assert disposition == "DOWNGRADE"


def test_report_bands_a_frame_with_the_engine_bands(
    defective_result: tuple[Path, InferenceResult]
) -> None:
    _, result = defective_result
    record = report.FrameRecord.from_result(0, "frame.jpg", result)
    assert record.severity_band == _severity_bucket(result.severity_score)
    assert record.severity_band in SEVERITY_BANDS
    assert json.loads(json.dumps(record.to_dict()))["defect_count"] == result.defect_count

    clean = report.FrameRecord.from_result(1, "clean.jpg", InferenceResult(image_size=(10, 10)))
    assert clean.severity_band == "clean"


def test_iou_matrix_matches_hand_computed_overlap() -> None:
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 15.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    ious = evaluate.iou_matrix(a, b)
    assert ious.shape == (1, 3)
    assert ious[0, 0] == pytest.approx(1.0)
    assert ious[0, 1] == pytest.approx(50.0 / 150.0)
    assert ious[0, 2] == pytest.approx(0.0)


def test_matching_is_class_aware_and_greedy() -> None:
    """A well-placed box with the wrong class is a false positive AND a miss."""
    gt_boxes = np.array([[0.0, 0.0, 10.0, 10.0]])
    gt_classes = np.array([0])

    right = evaluate.match_image(
        gt_boxes, gt_classes,
        np.array([[0.0, 0.0, 10.0, 10.0]]), np.array([0.9]), np.array([0]),
    )
    assert right.pred_is_tp.tolist() == [True]
    assert right.gt_matched.tolist() == [True]

    wrong_class = evaluate.match_image(
        gt_boxes, gt_classes,
        np.array([[0.0, 0.0, 10.0, 10.0]]), np.array([0.9]), np.array([3]),
    )
    assert wrong_class.pred_is_tp.tolist() == [False]
    assert wrong_class.gt_matched.tolist() == [False]

    # Two boxes on one label: the more confident one claims it, the other is an FP.
    duplicated = evaluate.match_image(
        gt_boxes, gt_classes,
        np.array([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 9.0, 9.0]]),
        np.array([0.4, 0.9]), np.array([0, 0]),
    )
    assert duplicated.pred_is_tp.tolist() == [False, True]
    assert duplicated.gt_matched.tolist() == [True]


def test_the_operating_point_respects_the_detection_floor() -> None:
    """Cost alone prefers a high threshold; the floor must veto it."""
    def point(threshold: float, detection_rate: float, false_alarm: float) -> evaluate.SweepPoint:
        return evaluate.SweepPoint(
            threshold=threshold, tp=1, fp=1, fn=1,
            precision=0.5, recall=detection_rate, f1=0.5,
            fp_per_image=false_alarm, false_alarm_rate=false_alarm,
            defect_detection_rate=detection_rate, class_accuracy=detection_rate,
            images_flagged=detection_rate,
        )

    sweep = [point(0.10, 0.95, 0.60), point(0.50, 0.93, 0.10), point(0.90, 0.40, 0.00)]
    chosen = evaluate.recommend_threshold(sweep, miss_cost_ratio=12.0, min_detection_rate=0.90)
    assert chosen.constraint_met
    assert chosen.threshold == 0.50
    assert chosen.point.defect_detection_rate >= 0.90
    assert chosen.sensitivity

    starved = [point(0.10, 0.55, 0.30), point(0.90, 0.20, 0.01)]
    fallback = evaluate.recommend_threshold(starved, min_detection_rate=0.90)
    assert not fallback.constraint_met
    assert "WARNING" in fallback.rationale


def test_evaluate_refuses_to_tune_and_report_on_the_same_split() -> None:
    with pytest.raises(ValueError, match="chosen on the same data"):
        evaluate.evaluate(split="val", tuning_split="val")


# ---------------------------------------------------------------------------
# false alarm proxy: the three statistics that a naive implementation gets wrong
# ---------------------------------------------------------------------------


def test_cluster_bootstrap_is_wider_than_wilson_when_patches_clump() -> None:
    """Correlated crops must not be reported with an independent-sample interval.

    Ten images, each contributing five crops that all agree with each other: the
    real sample size is ten, not fifty. The binomial interval does not know that,
    so the clustered interval has to be materially wider and the design effect
    has to be well above one.
    """
    groups = [[1] * 5 for _ in range(4)] + [[0] * 5 for _ in range(6)]
    lo, hi, deff = false_alarm.cluster_bootstrap_interval(groups, n_boot=4000, seed=7)
    w_lo, w_hi = false_alarm.wilson_interval(20, 50)
    assert deff > 3.0
    assert (hi - lo) > 1.5 * (w_hi - w_lo)
    assert lo < 0.4 < hi


def test_cluster_bootstrap_design_effect_is_one_without_clustering() -> None:
    """One crop per image is a simple random sample; the correction must vanish."""
    groups = [[1]] * 30 + [[0]] * 70
    _, _, deff = false_alarm.cluster_bootstrap_interval(groups, n_boot=2000, seed=7)
    assert deff == pytest.approx(1.0, abs=0.05)


def test_poisson_area_fit_finds_the_best_member_of_its_own_family() -> None:
    """The chi-square of a badly-estimated parameter is not evidence about a model.

    Data generated exactly from ``1 - exp(-lambda A)`` must be recovered, and no
    other intensity may fit the same cells better under the statistic the
    goodness-of-fit test then reports.
    """
    areas = [3600.0, 6400.0, 10000.0, 14400.0]
    true_lam = 6e-5
    totals = [4000] * 4
    hits = [int(round(n * (1.0 - np.exp(-true_lam * a)))) for a, n in zip(areas, totals)]
    lam, _, _ = false_alarm.poisson_area_fit(areas, hits, totals)
    assert lam == pytest.approx(true_lam, rel=0.05)

    fitted = false_alarm._chi2_binomial_fit(
        hits, totals, [1.0 - np.exp(-lam * a) for a in areas], 1
    )
    for other in (true_lam * 0.5, true_lam * 2.0):
        worse = false_alarm._chi2_binomial_fit(
            hits, totals, [1.0 - np.exp(-other * a) for a in areas], 1
        )
        assert worse["chi2"] > fitted["chi2"]


def test_a_flat_rate_across_areas_rejects_spatial_independence() -> None:
    """The core claim of the report, on synthetic data with a known answer."""
    areas = [3600.0, 6400.0, 10000.0, 14400.0]
    totals = [448, 212, 300, 300]
    hits = [int(round(0.24 * n)) for n in totals]
    lam, _, _ = false_alarm.poisson_area_fit(areas, hits, totals)
    poisson = false_alarm._chi2_binomial_fit(
        hits, totals, [1.0 - np.exp(-lam * a) for a in areas], 1
    )
    pooled = sum(hits) / sum(totals)
    constant = false_alarm._chi2_binomial_fit(hits, totals, [pooled] * 4, 1)
    assert constant["chi2"] < poisson["chi2"]
    assert poisson["p_value"] is not None and poisson["p_value"] < 0.01


def test_iid_image_prediction_is_averaged_per_image_not_at_the_mean() -> None:
    """1 - (1-p)^m is concave, so evaluating it at the mean m overstates it.

    The report used that overstatement as evidence of clustering, which inflated
    the claim. Guarding the direction here so it cannot come back.
    """
    p = 0.2374
    sizes = [1] * 20 + [17] * 20
    at_mean = 1.0 - (1.0 - p) ** (sum(sizes) / len(sizes))
    per_image = float(np.mean([1.0 - (1.0 - p) ** m for m in sizes]))
    assert at_mean > per_image + 0.05


def test_patch_imgsz_holds_the_deployment_magnification() -> None:
    """A crop must reach the network at the pixels-per-mm it was trained on."""
    for size in (60, 80, 100, 120):
        imgsz = false_alarm.patch_imgsz(size, 1.6)
        assert imgsz % 32 == 0
        assert imgsz / size == pytest.approx(1.6, abs=0.02)


def test_clean_patches_keep_a_hard_margin_from_every_label() -> None:
    """A crop that merely grazes a labelled defect is not a clean-steel sample."""
    record = evaluate.GroundTruth(
        image_path=Path("synthetic.jpg"),
        width=200,
        height=200,
        boxes=np.array([[80.0, 80.0, 120.0, 120.0]]),
        classes=np.array([0]),
    )
    margin = int(false_alarm.PATCH_MINING["clearance_px"])
    patches = false_alarm.mine_clean_patches([record], false_alarm.PATCH_MINING, 1.6)
    assert patches
    for patch in patches:
        grown = np.array(
            [
                patch.x0 - margin,
                patch.y0 - margin,
                patch.x0 + patch.size + margin,
                patch.y0 + patch.size + margin,
            ],
            dtype=np.float64,
        )
        assert float(false_alarm._intersection_area(grown, record.boxes).max()) == 0.0


def test_the_operating_point_file_does_not_claim_a_basis_it_did_not_use() -> None:
    """The demo caption is driven by this field; it must match the cost basis."""
    path = PROJECT_ROOT / "reports" / "operating_point.json"
    if not path.is_file():
        pytest.skip("no operating point has been written yet")
    payload = json.loads(path.read_text())
    basis = (payload.get("expected_at_threshold") or {}).get("cost_basis")
    if basis is None:
        pytest.skip("operating point predates the basis field")
    described = payload.get("false_alarm_basis", "")
    if basis == "patch":
        assert "area-extrapolated" not in described
    else:
        assert "area-extrapolated" in described


# --------------------------------------------------------------------------- #
# 12. photometric normalisation (src/domain_shift.py) and the
#     out-of-distribution gate (src/ood_guard.py)
#
# These two modules sit either side of the detector: the gate decides whether a
# frame is worth scoring at all, and the normaliser decides what the pixels look
# like when it is. Both are pure numpy/OpenCV, so everything below runs without
# the GPU except where a detector fixture is asked for.
# --------------------------------------------------------------------------- #

import cv2  # noqa: E402

import domain_shift  # noqa: E402
import ood_guard  # noqa: E402


def _synthetic_steel(seed: int = 0, size: int = 200) -> np.ndarray:
    """A grey, textured, in-focus frame -- the shape of thing the gate must pass."""
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.normal(128, 26, (size, size)), (0, 0), 1.4)
    return np.dstack([np.clip(base, 0, 255).astype(np.uint8)] * 3)


def _synthetic_logo() -> np.ndarray:
    """White field, black lettering, one flat bar: the audit's 'scratches 0.47'."""
    canvas = np.full((260, 520, 3), 255, np.uint8)
    cv2.putText(canvas, "JINDAL", (18, 140), cv2.FONT_HERSHEY_DUPLEX, 3.0, (30, 30, 30), 7)
    cv2.rectangle(canvas, (18, 170), (500, 196), (40, 40, 40), -1)
    return canvas


@pytest.fixture(scope="session")
def reference() -> domain_shift.ReferenceStats:
    """The cached NEU-DET training distribution, or skip -- never rebuild in a test."""
    if not domain_shift.REFERENCE_PATH.is_file():
        pytest.skip(f"no cached reference at {domain_shift.REFERENCE_PATH}")
    return domain_shift.load_reference(build_if_missing=False)


# --- domain_shift -----------------------------------------------------------


def test_domain_shift_restates_the_inference_decode_contract_exactly() -> None:
    """`domain_shift` re-implements `_to_rgb` to stay torch-free; it must not drift."""
    rng = np.random.default_rng(11)
    cases = [
        rng.integers(0, 256, (24, 31, 3), dtype=np.uint8),
        rng.integers(0, 256, (16, 16), dtype=np.uint8),
        rng.integers(0, 256, (12, 20, 4), dtype=np.uint8),
        (rng.random((9, 9, 3)) * 255.0),
    ]
    for case in cases:
        mine = domain_shift._to_rgb(case)
        theirs = inference._to_rgb(case)
        assert mine.shape == theirs.shape
        assert mine.dtype == theirs.dtype == np.uint8
        assert np.array_equal(mine, theirs)


def test_the_cached_reference_describes_the_training_split_and_nothing_else(
    reference: domain_shift.ReferenceStats,
) -> None:
    """The reference is part of the model: it may only have seen what the model saw."""
    assert reference.split == domain_shift.REFERENCE_SPLIT == "train"
    assert reference.histogram.shape == (domain_shift.LEVELS,)
    assert reference.frame_histogram.shape == (domain_shift.LEVELS,)
    assert reference.n_images == EXPECTED_IMAGES["train"]
    assert reference.n_pixels == pytest.approx(
        reference.histogram.sum(), rel=1e-9
    ), "pixel count must equal the histogram mass it was built from"
    for cdf in (reference.cdf, reference.frame_cdf):
        assert np.all(np.diff(cdf) >= -1e-12), "a CDF may not decrease"
        assert cdf[-1] == pytest.approx(1.0, abs=1e-9)


def test_the_reference_survives_a_json_round_trip(
    reference: domain_shift.ReferenceStats,
) -> None:
    restored = domain_shift.ReferenceStats.from_dict(json.loads(json.dumps(reference.to_dict())))
    assert restored.n_images == reference.n_images
    assert restored.mean == pytest.approx(reference.mean, abs=5e-4)
    assert restored.frame_std == pytest.approx(reference.frame_std, abs=5e-4)
    assert np.array_equal(restored.histogram, reference.histogram)


def test_a_malformed_reference_histogram_is_refused() -> None:
    """A silently-wrong reference would mis-map every frame; fail loudly instead."""
    for bad in (np.ones(255), -np.ones(domain_shift.LEVELS), np.zeros(domain_shift.LEVELS)):
        with pytest.raises(ValueError):
            domain_shift._validate_histogram(bad, "test histogram")


@pytest.mark.parametrize("target", domain_shift.TARGETS)
@pytest.mark.parametrize("strength", (0.0, 0.25, 0.5, 1.0))
def test_the_matching_lut_is_monotone_so_it_cannot_invert_a_defect(
    reference: domain_shift.ReferenceStats, target: str, strength: float
) -> None:
    """The whole safety argument for normalising: contrast may move, never flip."""
    rng = np.random.default_rng(3)
    counts = np.bincount(
        rng.integers(20, 210, 40_000, dtype=np.uint8), minlength=domain_shift.LEVELS
    ).astype(np.float64)
    lut = domain_shift.build_lut(counts, reference.cdf_for(target), strength=strength)
    assert lut.shape == (domain_shift.LEVELS,)
    assert lut.dtype == np.uint8
    assert np.all(np.diff(lut.astype(np.int16)) >= 0)


def test_normalisation_at_zero_strength_is_bit_identical_to_doing_nothing(
    reference: domain_shift.ReferenceStats,
) -> None:
    """`method='none'` and `strength=0` exist so callers can A/B without branching."""
    frame = _synthetic_steel(seed=5)
    for kwargs in ({"method": "none"}, {"method": "histogram", "strength": 0.0},
                   {"method": "affine", "strength": 0.0}):
        out = domain_shift.match_to_reference(frame, reference, **kwargs)
        assert np.array_equal(out, frame)
        assert out is not frame, "must not hand back the caller's buffer"


def test_full_strength_matching_moves_a_shifted_frame_towards_the_reference(
    reference: domain_shift.ReferenceStats,
) -> None:
    """The claim the module exists for, on a frame darkened the way Severstal is."""
    frame = np.clip(_synthetic_steel(seed=7).astype(np.int16) - 45, 0, 255).astype(np.uint8)
    before = domain_shift.histogram_distance(frame, reference)
    after = domain_shift.histogram_distance(
        domain_shift.match_to_reference(
            frame, reference, method="histogram", target="population", strength=1.0
        ),
        reference,
    )
    # A perfect landing is impossible: a 200x200 frame has 40k pixels to spread
    # over 256 levels, so the matched histogram is quantised, not equal. The
    # residual is a few grey levels; the shift it removed is fifty.
    assert after < before / 8.0, f"{before:.1f} -> {after:.1f} is not a match"
    assert after < 10.0


def test_normalisation_preserves_the_frames_shape_and_dtype(
    reference: domain_shift.ReferenceStats,
) -> None:
    frame = _synthetic_steel(seed=9, size=64)
    for method in domain_shift.METHODS:
        out = domain_shift.match_to_reference(frame, reference, method=method, strength=0.5)
        assert out.shape == frame.shape
        assert out.dtype == np.uint8


def test_an_unknown_method_or_target_is_a_loud_error(
    reference: domain_shift.ReferenceStats,
) -> None:
    frame = _synthetic_steel(seed=2, size=32)
    with pytest.raises(ValueError):
        domain_shift.match_to_reference(frame, reference, method="clahe")
    with pytest.raises(ValueError):
        domain_shift.match_to_reference(frame, reference, target="galaxy")
    with pytest.raises(ValueError):
        reference.cdf_for("galaxy")


def test_normalise_frame_reports_the_evidence_for_what_it_did(
    reference: domain_shift.ReferenceStats,
) -> None:
    """The console has to be able to justify changing an operator's pixels."""
    frame = np.clip(_synthetic_steel(seed=13).astype(np.int16) - 40, 0, 255).astype(np.uint8)
    out, record = domain_shift.normalise_frame(frame, reference, strength=1.0, target="population")
    assert out.shape == frame.shape
    assert record["method"] == "histogram"
    assert record["strength"] == pytest.approx(1.0)
    assert set(record["before"]) == set(record["after"])
    assert record["note"] and "mean" in record["note"]
    assert json.loads(json.dumps(record))["reference"] == reference.describe()
    assert record["after"]["hist_distance"] < record["before"]["hist_distance"]


def test_the_recommended_defaults_are_the_ones_the_measurement_chose() -> None:
    """`reports/ood_guard.md` justifies these two numbers; keep them in step."""
    assert domain_shift.RECOMMENDED_TARGET == "frame"
    assert domain_shift.RECOMMENDED_STRENGTH == 0.25
    assert domain_shift.RECOMMENDED_TARGET in domain_shift.TARGETS


# --- ood_guard --------------------------------------------------------------


def test_the_gate_passes_every_frame_of_the_headline_test_split(
    reference: domain_shift.ReferenceStats,
) -> None:
    """A gate that withholds held-out steel would silently invalidate mAP50 0.7524."""
    paths = _split_images("test")
    if not paths:
        pytest.skip("no test images on this working tree")
    rejected = [p.name for p in paths if not ood_guard.inspect(p, reference=reference).ok]
    assert rejected == [], f"the gate rejected genuine held-out steel: {rejected}"


def test_the_gate_passes_synthetic_steel_across_exposure_and_degradation(
    reference: domain_shift.ReferenceStats,
) -> None:
    """Mill reality: gradients, motion blur, hard JPEG, upscaling, a warm cast."""
    frame = _synthetic_steel(seed=17)
    gradient = np.tile(np.linspace(0.55, 1.35, frame.shape[1]), (frame.shape[0], 1))[:, :, None]
    variants = {
        "plain": frame,
        "underexposed": np.clip(frame * 0.5, 0, 255).astype(np.uint8),
        "overexposed": np.clip(frame * 1.7, 0, 255).astype(np.uint8),
        "lighting_gradient": np.clip(frame * gradient, 0, 255).astype(np.uint8),
        "motion_blur": cv2.filter2D(frame, -1, np.ones((1, 15), np.float32) / 15),
        "jpeg_q20": cv2.imdecode(
            cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 20])[1], cv2.IMREAD_COLOR
        ),
        "upscaled_4x": cv2.resize(frame, (800, 800)),
        "warm_cast": np.clip(
            frame.astype(np.float32) * np.array([1.06, 1.0, 0.94], np.float32), 0, 255
        ).astype(np.uint8),
    }
    failures = {
        name: [c.name for c in ood_guard.inspect(img, reference=reference).failed]
        for name, img in variants.items()
        if not ood_guard.inspect(img, reference=reference).ok
    }
    assert failures == {}, f"the gate rejected degraded but genuine steel: {failures}"


@pytest.mark.parametrize(
    ("case", "image", "expected_check"),
    (
        ("solid_white", np.full((400, 400, 3), 255, np.uint8), "tonal_range"),
        ("solid_black", np.zeros((400, 400, 3), np.uint8), "tonal_range"),
        ("solid_grey", np.full((400, 400, 3), 130, np.uint8), "focus"),
        ("logo", _synthetic_logo(), "tonal_range"),
        (
            "solid_red",
            np.dstack(
                [np.full((300, 300), 200, np.uint8)] + [np.zeros((300, 300), np.uint8)] * 2
            ),
            "colour",
        ),
        ("tiny", cv2.resize(_synthetic_steel(seed=1), (8, 8)), "resolution"),
        (
            "blurred",
            cv2.GaussianBlur(_synthetic_steel(seed=1), (0, 0), 6),
            "focus",
        ),
        (
            "noise",
            np.dstack(
                [np.random.default_rng(4).integers(0, 256, (400, 400), dtype=np.uint8)] * 3
            ),
            "noise",
        ),
    ),
)
def test_the_gate_withholds_the_verdict_on_things_that_are_not_strip(
    case: str, image: np.ndarray, expected_check: str, reference: domain_shift.ReferenceStats
) -> None:
    """Each audit failure, and the named check that has to be the one to catch it."""
    verdict = ood_guard.inspect(image, reference=reference)
    assert verdict.decision == ood_guard.REJECT, f"{case} was not rejected"
    assert not verdict.ok
    failed = {c.name for c in verdict.failed}
    assert expected_check in failed, f"{case} rejected by {failed}, not {expected_check}"


def test_a_rejection_explains_itself_in_terms_of_the_checks_that_failed(
    reference: domain_shift.ReferenceStats,
) -> None:
    """The reason string is what an operator reads; it may not be boilerplate."""
    verdict = ood_guard.inspect(_synthetic_logo(), reference=reference)
    assert verdict.failed
    assert verdict.headline
    for check in verdict.failed:
        assert check.explanation in verdict.reason
    for check in verdict.checks:
        if check.passed:
            assert check.explanation not in verdict.reason


def test_the_verdict_is_json_serialisable_and_internally_consistent(
    reference: domain_shift.ReferenceStats,
) -> None:
    for image in (_synthetic_steel(seed=21), _synthetic_logo()):
        verdict = ood_guard.inspect(image, reference=reference)
        payload = json.loads(json.dumps(verdict.to_dict()))
        assert payload["decision"] in (ood_guard.PASS, ood_guard.REVIEW, ood_guard.REJECT)
        assert payload["ok"] == (payload["decision"] != ood_guard.REJECT)
        assert payload["review"] == (payload["decision"] == ood_guard.REVIEW)
        assert bool(payload["failed"]) == (payload["decision"] == ood_guard.REJECT)
        assert len(payload["checks"]) == 6
        assert payload["reason"] and payload["headline"]
        assert ood_guard.is_steel_like(image) == verdict.ok


def test_the_gate_never_touches_the_callers_pixels(
    reference: domain_shift.ReferenceStats,
) -> None:
    """It runs before the model on the operator's own array; it must be read-only."""
    frame = _synthetic_steel(seed=23)
    original = frame.copy()
    ood_guard.inspect(frame, reference=reference)
    assert np.array_equal(frame, original)


def test_tonal_bits_counts_effective_grey_levels_and_ignores_clipped_pixels() -> None:
    """The statistic that separates a blown-out strip frame from a printed logo."""
    assert ood_guard.tonal_bits(np.full((32, 32), 255, np.uint8)) == 0.0
    assert ood_guard.tonal_bits(np.zeros((32, 32), np.uint8)) == 0.0
    # 64 equally-used unclipped levels is exactly 6 bits, whatever else is present.
    plane = np.tile(np.arange(64, 128, dtype=np.uint8), (64, 1))
    assert ood_guard.tonal_bits(plane) == pytest.approx(6.0, abs=1e-9)
    blown = plane.copy()
    blown[:8] = 255
    assert ood_guard.tonal_bits(blown) == pytest.approx(
        ood_guard.tonal_bits(plane[8:]), abs=1e-9
    ), "clipped rows must not change the answer"


def test_thresholds_are_data_a_mill_can_change_not_constants_baked_into_logic(
    reference: domain_shift.ReferenceStats,
) -> None:
    """A commissioning engineer must be able to loosen a gate without editing code."""
    blurred = cv2.GaussianBlur(_synthetic_steel(seed=1), (0, 0), 6)
    assert not ood_guard.inspect(blurred, reference=reference).ok
    loosened = ood_guard.GuardThresholds(min_focus=0.0, min_tonal_bits=0.0)
    assert ood_guard.inspect(blurred, thresholds=loosened, reference=reference).ok
    assert json.loads(json.dumps(loosened.to_dict()))["min_focus"] == 0.0


def test_calibration_reports_real_headroom_on_known_good_frames(
    reference: domain_shift.ReferenceStats,
) -> None:
    """The commissioning tool: nothing rejected, and every margin on the safe side."""
    paths = _split_images("test")[:40]
    if not paths:
        pytest.skip("no test images on this working tree")
    report_ = ood_guard.calibrate(paths, reference=reference)
    assert report_["n_images"] == len(paths)
    assert report_["rejected"] == 0
    assert set(report_["checks"]) == {
        "resolution", "aspect", "colour", "noise", "focus", "tonal_range"
    }
    for name, row in report_["checks"].items():
        assert row["n_failed"] == 0
        assert row["worst_margin"] > 1.0, f"{name} has no headroom on real steel"
    with pytest.raises(ValueError):
        ood_guard.calibrate([])


def test_suggested_thresholds_only_ever_loosen_the_gate(
    reference: domain_shift.ReferenceStats,
) -> None:
    """Rejecting real steel is the failure mode; calibration may not make it likelier."""
    paths = _split_images("test")[:24]
    if not paths:
        pytest.skip("no test images on this working tree")
    base = ood_guard.DEFAULT_THRESHOLDS
    tuned = ood_guard.suggest_thresholds(paths, safety=2.0, base=base, reference=reference)
    assert tuned.max_saturation >= base.max_saturation
    assert tuned.max_edge_density >= base.max_edge_density
    assert tuned.min_focus <= base.min_focus
    assert tuned.min_tonal_bits <= base.min_tonal_bits
    with pytest.raises(ValueError):
        ood_guard.suggest_thresholds(paths, safety=0.5, reference=reference)


def test_histogram_distance_is_reported_but_is_never_allowed_to_reject(
    reference: domain_shift.ReferenceStats,
) -> None:
    """It cannot separate: a blank white frame scores inside the range real steel occupies.

    This is the reason the gate uses texture rather than exposure, and the reason
    a large distance produces REVIEW rather than REJECT.
    """
    white = np.full((400, 400, 3), 255, np.uint8)
    assert domain_shift.histogram_distance(white, reference) > 100.0
    verdict = ood_guard.inspect(white, reference=reference)
    assert {c.name for c in verdict.checks}.isdisjoint({"hist_distance", "exposure"})
    # Far-from-training but otherwise sound steel is scored, with a caveat.
    dark = np.clip(_synthetic_steel(seed=29).astype(np.int16) - 100, 0, 255).astype(np.uint8)
    dark_verdict = ood_guard.inspect(dark, reference=reference)
    if dark_verdict.signals.hist_distance > ood_guard.DEFAULT_THRESHOLDS.review_hist_distance:
        assert dark_verdict.decision == ood_guard.REVIEW
        assert dark_verdict.ok and dark_verdict.review
        assert "domain_shift" in dark_verdict.reason


def test_a_focused_greyscale_texture_photograph_is_a_documented_miss(
    reference: domain_shift.ReferenceStats,
) -> None:
    """Characterisation test for the gate's stated limit, not an aspiration.

    The gate tests physics, not semantics: anything monochrome, in focus and
    textured is indistinguishable from rough strip on these six signals. If this
    test starts failing, the gate got stronger -- update `reports/ood_guard.md`
    rather than deleting the test.
    """
    grid = np.mgrid[0:400, 0:400]
    wood = (128 + 70 * np.sin(grid[1] / 6.0 + 4 * np.sin(grid[0] / 55.0))).astype(np.uint8)
    verdict = ood_guard.inspect(np.dstack([wood] * 3), reference=reference)
    assert verdict.ok, "documented miss became a catch; update reports/ood_guard.md"


def test_the_gate_is_cheap_enough_to_run_in_front_of_every_frame(
    reference: domain_shift.ReferenceStats,
) -> None:
    """It must never be the reason a console is slow to say 'no'."""
    import time

    frame = _synthetic_steel(seed=31)
    for _ in range(5):
        ood_guard.inspect(frame, reference=reference)
    start = time.perf_counter()
    for _ in range(50):
        ood_guard.inspect(frame, reference=reference)
    per_frame_ms = (time.perf_counter() - start) / 50 * 1000
    assert per_frame_ms < 20.0, f"the gate costs {per_frame_ms:.1f} ms on a 200x200 frame"
