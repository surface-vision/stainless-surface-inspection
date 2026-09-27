"""Tests for the operator console, `demo/app.py`.

Five audited failures are closed in that file and every one of them is a *wrong
answer on screen* rather than a crash, so the tests that matter here assert what
the console decides and what it says, not that it runs:

* the zero-click landing screen must not open on the model's weakest class
  (`ordered_families`, `family_confidence_note`);
* a wide strip capture must be tiled without anyone touching a control, and the
  screen must say why (`resolve_tiling`, `TilingDecision`);
* a frame that is not steel must get a withheld verdict, not a confident
  critical disposition (`src/ood_guard.py` wired ahead of the detector);
* the line-speed KPI must be labelled as one camera's field-of-view advance and
  must print the accelerator count beside it (`MillCapacity`);
* the operator-facing confidence must be the calibrated probability with the raw
  score kept beside it (`CalibrationInfo`, `detection_frame`).

Plus the physical-units readout, which is an assumption the operator sets and
must never be presentable as a measurement.

The pure layer is tested directly. Two tests drive the real script through
`streamlit.testing.v1.AppTest` because the withheld-verdict state and the
on-demand explanation only exist as rendered output.

Run with:
    .venv/bin/python -m pytest tests/test_demo_app.py -v
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _extra in (PROJECT_ROOT / "src", PROJECT_ROOT / "demo"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

os.environ.setdefault("YOLO_AUTOINSTALL", "False")

import app  # noqa: E402
import inference  # noqa: E402

APP_PATH = PROJECT_ROOT / "demo" / "app.py"
SAMPLES = PROJECT_ROOT / "data" / "neu-det" / "test" / "images"
BENCHMARK = PROJECT_ROOT / "reports" / "benchmark.json"


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def detector() -> inference.DefectDetector:
    weights = PROJECT_ROOT / "models" / "yolov8n_neudet" / "weights" / "best.pt"
    if not weights.is_file():
        pytest.skip("the production checkpoint is not on this working tree")
    return app.load_detector(weights=str(weights), device="auto")


@pytest.fixture(scope="module")
def steel_frame() -> np.ndarray:
    frames = sorted(SAMPLES.glob("patches_*.jpg"))
    if not frames:
        pytest.skip("no held-out sample frames on this working tree")
    return app.load_sample_image(frames[0]).array


def _logo_png() -> bytes:
    """A flat two-colour graphic - the "judge uploads our logo" case."""
    img = Image.new("RGB", (600, 240), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.ellipse((30, 40, 190, 200), fill=(214, 32, 39))
    draw.rectangle((220, 90, 560, 150), fill=(20, 60, 160))
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _white_png() -> bytes:
    import io

    buf = io.BytesIO()
    Image.new("RGB", (400, 400), (255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# GAP 10 - the landing screen must not open on the worst class
# --------------------------------------------------------------------------- #


def test_the_sample_picker_is_ordered_by_measured_confidence_not_alphabetically() -> None:
    """Alphabetical order put `crazing` on the zero-click screen.

    Crazing's median peak confidence is 0.289 with 93% of its frames under 0.50,
    so the console's automatic first impression was "peak confidence 0.22" beside
    "severity 79.8/100 high". Ordering by measured strength is the whole fix, and
    it must not be quietly reverted to `sorted()`.
    """
    families = app.ordered_families({name: [] for name in app.FAMILY_PEAK_CONFIDENCE})

    assert families[0] == "patches", "the console must open on a representative class"
    assert families[-1] == "crazing", "the weakest class belongs last, not first"
    assert families != sorted(families), "alphabetical order is the bug being fixed"

    medians = [app.FAMILY_PEAK_CONFIDENCE[f] for f in families]
    assert medians == sorted(medians, reverse=True), "strongest first, monotonically"


def test_the_landing_family_beats_the_all_class_median() -> None:
    """Whatever opens first has to be representative, not cherry-picked or weak."""
    first = app.ordered_families({name: [] for name in app.FAMILY_PEAK_CONFIDENCE})[0]
    assert app.FAMILY_PEAK_CONFIDENCE[first] > app.ALL_CLASS_PEAK_CONFIDENCE


def test_families_with_no_measurement_sort_last_rather_than_inheriting_a_rank() -> None:
    catalogue = {"crazing": [], "patches": [], "zzz_new_class": [], "aaa_new_class": []}
    families = app.ordered_families(catalogue)

    assert families[:2] == ["patches", "crazing"]
    assert families[2:] == ["aaa_new_class", "zzz_new_class"]


def test_every_family_prints_its_own_measured_confidence_under_the_picker() -> None:
    """The weak class stays reachable, but never without its number beside it."""
    for family in app.FAMILY_PEAK_CONFIDENCE:
        note = app.family_confidence_note(family)
        assert f"{app.FAMILY_PEAK_CONFIDENCE[family]:.2f}" in note
        assert f"{app.FAMILY_SHARE_UNDER_HALF[family]:.0%}" in note
        assert app.FAMILY_STATS_BASIS in note

    weakest = app.family_confidence_note("crazing")
    assert "weakest" in weakest, "the hard class must be named as hard, not hidden"

    unknown = app.family_confidence_note("not_a_real_family")
    assert "No held-out confidence measurement" in unknown


# --------------------------------------------------------------------------- #
# GAP 4 - wide strips must tile themselves
# --------------------------------------------------------------------------- #


def test_a_wide_strip_tiles_itself_with_nobody_touching_a_control() -> None:
    """The audited failure: a 3072x256 strip returned nothing with tiling off."""
    decision = app.resolve_tiling(app.TILING_AUTO, 3072, 256, 256)

    assert decision.tiled is True
    assert decision.trigger == "aspect"
    assert decision.automatic is True
    assert decision.badge == "TILED"
    assert "auto-tile trigger" in decision.reason


def test_a_large_frame_tiles_on_the_downscale_trigger_even_when_it_is_not_wide() -> None:
    """A real 2048x1000 line-scan frame is only 2.05:1, so aspect alone misses it.

    It is squashed 8x to reach a 256 px input, which is where the texture that
    separates crazing from pitting goes. Measured on the shipped sample: 0
    detections whole-frame, 20 tiled.
    """
    decision = app.resolve_tiling(app.TILING_AUTO, 2048, 1000, 256)

    assert decision.aspect < app.AUTO_TILE_ASPECT, "the aspect trigger must not fire"
    assert decision.tiled is True
    assert decision.trigger == "downscale"
    assert decision.downscale == pytest.approx(8.0)


def test_an_ordinary_neu_det_crop_is_left_alone() -> None:
    """The default landing screen must be untouched by either trigger."""
    decision = app.resolve_tiling(app.TILING_AUTO, 200, 200, 256)

    assert decision.tiled is False
    assert decision.trigger == "none"
    assert decision.badge == "SINGLE PASS"
    assert decision.automatic is False


@pytest.mark.parametrize(
    "mode,width,height,expected_tiled",
    [
        (app.TILING_ALWAYS, 200, 200, True),
        (app.TILING_ALWAYS, 3072, 256, True),
        (app.TILING_NEVER, 3072, 256, False),
        (app.TILING_NEVER, 200, 200, False),
    ],
)
def test_the_manual_override_still_wins_in_both_directions(
    mode: str, width: int, height: int, expected_tiled: bool
) -> None:
    decision = app.resolve_tiling(mode, width, height, 256)
    assert decision.tiled is expected_tiled
    assert decision.mode == mode


def test_overriding_tiling_off_on_a_frame_that_needs_it_says_so_on_screen() -> None:
    """Silence is what made this a bug. Turning the fix off must be visible."""
    decision = app.resolve_tiling(app.TILING_NEVER, 2048, 1000, 256)

    assert decision.tiled is False
    assert decision.trigger == "manual"
    assert "would otherwise have been tiled" in decision.reason


def test_the_console_ships_real_wide_strip_frames_so_tiling_can_be_demonstrated() -> None:
    """Nothing else in the project is wider than 600 px."""
    strips = app.wide_strip_samples()
    if not strips:
        pytest.skip("assets/ carries no strip samples on this working tree")

    for path in strips:
        loaded = app.load_sample_image(path)
        width, height = loaded.size
        assert max(width, height) >= 2000, f"{path.name} is not a strip capture"
        assert app.resolve_tiling(app.TILING_AUTO, width, height, 256).tiled, path.name


def test_tiling_actually_recovers_detections_a_single_pass_misses(
    detector: inference.DefectDetector,
) -> None:
    """The feature has to work, not merely engage.

    Measured on this working tree at conf 0.15 / IoU 0.45 / 256 px:
    `assets/strip_sample_edge_defect.jpg` gives 0 detections in a single pass and
    20 tiled. The assertion is the inequality rather than the literal, so it
    survives a retrain while still failing if auto-tiling stops helping.
    """
    strips = app.wide_strip_samples()
    if not strips:
        pytest.skip("assets/ carries no strip samples on this working tree")
    loaded = app.load_sample_image(strips[0])

    auto = app.RuntimeSettings(conf=0.15, iou=0.45, imgsz=256, tiling=app.TILING_AUTO)
    never = app.RuntimeSettings(conf=0.15, iou=0.45, imgsz=256, tiling=app.TILING_NEVER)

    tiled = app.run_frame(detector, loaded.array, auto)
    single = app.run_frame(detector, loaded.array, never)

    assert tiled.defect_count > single.defect_count
    assert tiled.verdict == "DEFECT"


# --------------------------------------------------------------------------- #
# GAP 2 - non-steel must not get a confident critical verdict
# --------------------------------------------------------------------------- #


def test_the_console_gates_frames_before_it_disposition_them() -> None:
    """The gate has to be wired in, not merely importable."""
    source = APP_PATH.read_text()
    assert "from ood_guard import" in source
    # ...and called ahead of the verdict in both scoring tabs.
    assert source.count("inspect_frame(") >= 2


@pytest.mark.parametrize("payload_name", ["logo", "white"])
def test_a_frame_that_is_not_steel_is_refused_by_the_gate(payload_name: str) -> None:
    """Measured on the shipped model: a logo scored `scratches 0.47`, a white
    frame `pitted_surface 0.739` at severity 86.3 critical."""
    payload = _logo_png() if payload_name == "logo" else _white_png()
    loaded = app.decode_image(payload, f"{payload_name}.png")

    verdict = app.inspect_frame(loaded.array, thresholds=app.GUARD_THRESHOLDS)

    assert verdict.ok is False, f"{payload_name} must not reach the detector"
    assert verdict.reason, "a withheld verdict without a reason is a silent suppression"
    assert any(not check.passed for check in verdict.checks)


def test_real_steel_passes_the_gate(steel_frame: np.ndarray) -> None:
    """A gate that rejects the product is worse than no gate."""
    verdict = app.inspect_frame(steel_frame, thresholds=app.GUARD_THRESHOLDS)
    assert verdict.ok is True


def test_a_real_strip_capture_passes_the_gate() -> None:
    strips = app.wide_strip_samples()
    if not strips:
        pytest.skip("assets/ carries no strip samples on this working tree")
    loaded = app.load_sample_image(strips[0])
    assert app.inspect_frame(loaded.array, thresholds=app.GUARD_THRESHOLDS).ok is True


def test_the_withheld_screen_shows_every_check_with_its_number() -> None:
    """"Rejected" without evidence is just a different opaque answer."""
    loaded = app.decode_image(_logo_png(), "logo.png")
    verdict = app.inspect_frame(loaded.array, thresholds=app.GUARD_THRESHOLDS)

    table = app.guard_check_frame(verdict)

    assert list(table.columns) == [
        "Check", "Measured", "Rule", "Units", "Result", "What it means",
    ]
    assert len(table) == len(verdict.checks)
    assert "FAIL" in set(table["Result"])
    assert all(str(text).strip() for text in table["What it means"])


# --------------------------------------------------------------------------- #
# GAP 3 - the line-speed KPI
# --------------------------------------------------------------------------- #


def test_the_console_never_calls_a_single_camera_figure_a_line_speed() -> None:
    """The audited KPI read "Max line speed 1267 m/min" from one 200x200 crop."""
    source = APP_PATH.read_text()

    assert "Max line speed" not in source
    assert "FOV advance" in source, "the tile must be labelled as what it is"
    assert "not a line speed" in source


def test_the_capacity_model_is_read_from_the_benchmark_not_restated() -> None:
    if not BENCHMARK.is_file():
        pytest.skip("reports/benchmark.json has not been generated on this tree")
    capacity = app.MillCapacity.load(256)
    assert capacity is not None

    payload = json.loads(BENCHMARK.read_text())
    geometry = payload["mill"]["geometry"]
    assert capacity.strip_width_m == pytest.approx(geometry["strip_width_m"])
    assert capacity.cameras_across_width == geometry["cameras_across_width"]
    assert capacity.mm_per_px == pytest.approx(geometry["optical_resolution_mm_per_px"])
    assert capacity.strip_advance_m == pytest.approx(geometry["strip_advance_per_frame_m"])


def test_the_accelerator_count_reproduces_deck_slide_2() -> None:
    """Slide 2 says 250 m/min needs 18 accelerators, quoted at the 320 px input.

    This is the number the demo used to contradict by ~30x, so it is pinned: the
    console recomputes it from `reports/benchmark.json` and prints it beside its
    own figure rather than quoting the slide.
    """
    if not BENCHMARK.is_file():
        pytest.skip("reports/benchmark.json has not been generated on this tree")
    deck = app.MillCapacity.load(app.DECK_REFERENCE_IMGSZ)
    assert deck is not None
    assert deck.imgsz == 320
    assert deck.accelerators_for(app.REFERENCE_LINE_SPEED_M_PER_MIN) == 18


def test_full_width_capacity_is_the_one_camera_rate_divided_by_the_camera_count() -> None:
    """A single camera cannot cover the strip, which is the whole finding."""
    if not BENCHMARK.is_file():
        pytest.skip("reports/benchmark.json has not been generated on this tree")
    capacity = app.MillCapacity.load(256)

    assert capacity.m_per_min_full_width == pytest.approx(
        capacity.m_per_min_one_camera / capacity.cameras_across_width
    )
    assert capacity.m_per_min_full_width < capacity.m_per_min_one_camera
    assert capacity.accelerators_for(0.0) == 0


def test_an_unmeasured_input_size_falls_back_to_the_nearest_measured_one() -> None:
    """The sidebar offers 512/960/1280; the benchmark swept 256/320/416/640."""
    if not BENCHMARK.is_file():
        pytest.skip("reports/benchmark.json has not been generated on this tree")
    swept = {int(k) for k in
             json.loads(BENCHMARK.read_text())["mill"]["deployment"]
             ["accelerator_seconds_per_frame_tiled"]}

    for requested in app.IMGSZ_OPTIONS:
        capacity = app.MillCapacity.load(requested)
        assert capacity is not None
        assert capacity.imgsz in swept


def test_a_missing_benchmark_degrades_instead_of_taking_the_tab_down(tmp_path: Path) -> None:
    assert app.MillCapacity.load(256, tmp_path / "nope.json") is None
    assert app.default_mm_per_px(tmp_path / "nope.json") == app.DEFAULT_MM_PER_PX


# --------------------------------------------------------------------------- #
# the sidebar's default threshold must come from val, never test
# --------------------------------------------------------------------------- #


def test_the_default_threshold_is_tuned_on_val_not_test() -> None:
    """The sidebar sliders default to this. If it ever says `tuned_on == "test"`,
    the console would be showing a judge a threshold picked on the split it is
    also being scored on -- the leak this module exists to rule out."""
    OPERATING_POINT = Path("reports") / "operating_point.json"
    if not OPERATING_POINT.is_file():
        pytest.skip("reports/operating_point.json has not been generated on this tree")
    point = app.load_operating_point(OPERATING_POINT)

    assert point.tuned_on == "val"
    assert point.source == OPERATING_POINT
    assert 0.0 <= point.conf <= 1.0
    assert "val" in point.summary
    assert "test" not in point.summary.lower().split("`")[0]


def test_a_missing_operating_point_degrades_to_the_built_in_defaults(tmp_path: Path) -> None:
    point = app.load_operating_point(tmp_path / "absent.json")

    assert point.conf == app.FALLBACK_CONF
    assert point.iou == app.FALLBACK_IOU
    assert point.source is None
    assert "make eval" in point.summary


def test_a_truncated_operating_point_degrades_instead_of_crashing(tmp_path: Path) -> None:
    """A partially-written file (a run caught mid-save) must not take the sidebar down."""
    broken = tmp_path / "operating_point.json"
    broken.write_text('{"conf_threshold": ')  # truncated JSON

    point = app.load_operating_point(broken)

    assert point.conf == app.FALLBACK_CONF
    assert point.iou == app.FALLBACK_IOU


# --------------------------------------------------------------------------- #
# GAP 5 - the number on screen must be a probability
# --------------------------------------------------------------------------- #


def test_the_operator_facing_number_is_the_calibrated_probability() -> None:
    """Measured ECE 0.142, under-confident in every bin.

    The calibrator is loaded from `reports/calibration.json`, so this asserts the
    console actually picks it up rather than silently showing raw scores under a
    label that says "calibrated".
    """
    info = app.load_calibration()
    if not info.fitted:
        pytest.skip("no calibrator on disk; run `make calibrate`")

    assert info.ece_after is not None and info.ece_before is not None
    assert info.ece_after < info.ece_before, "calibration must improve ECE"
    assert "P(this box is a true positive)" in info.summary
    # Under-confident: the fitted map lifts mid-range scores.
    assert info.apply(0.60) > 0.60


def test_calibration_is_monotone_so_it_cannot_reorder_anything() -> None:
    info = app.load_calibration()
    raw = np.linspace(0.05, 0.95, 19)
    mapped = [info.apply(float(r)) for r in raw]
    assert mapped == sorted(mapped)
    assert all(0.0 <= m <= 1.0 for m in mapped)


def test_a_missing_calibrator_degrades_to_the_raw_score_and_says_so(tmp_path: Path) -> None:
    """A console that labels raw scores "calibrated" is worse than an honest one."""
    info = app.load_calibration(tmp_path / "absent.json")

    assert info.fitted is False
    assert info.apply(0.42) == pytest.approx(0.42)
    assert "raw detector score" in info.summary


def test_both_numbers_reach_the_table_and_the_export(
    detector: inference.DefectDetector, steel_frame: np.ndarray
) -> None:
    """Neither number may be hidden: one is the probability, one is the evidence."""
    info = app.load_calibration()
    settings = app.RuntimeSettings(conf=0.15, iou=0.45, imgsz=256)
    result = app.run_frame(detector, steel_frame, settings)
    if not result.detections:
        pytest.skip("no detections on this frame at the shipped operating point")

    table = app.detection_frame(result, info.apply, settings.mm_per_px)
    assert "P(true)" in table.columns and "Raw score" in table.columns

    record = app.frame_record("f.jpg", result, info.apply)
    assert "calibrated_confidence" in record and "max_confidence" in record

    export = app.detection_table(["f.jpg"], [result], info.apply, settings.mm_per_px)
    assert "calibrated_confidence" in export.columns
    assert "confidence" in export.columns


def test_a_frame_with_no_box_shows_no_probability() -> None:
    """The calibrator maps score 0.0 to 0.09, which beside a PASS reads as
    "this clean frame is 9% defective". Both fields must go to a dash."""
    source = APP_PATH.read_text()
    assert "no box to score" in source
    assert 'calibrated_text = f"{calibrated:.2f}" if scored else "-"' in source


# --------------------------------------------------------------------------- #
# physical units - an assumption, never a measurement
# --------------------------------------------------------------------------- #


def test_the_optical_scale_default_comes_from_the_same_geometry_as_the_capacity_model() -> None:
    if not BENCHMARK.is_file():
        pytest.skip("reports/benchmark.json has not been generated on this tree")
    geometry = json.loads(BENCHMARK.read_text())["mill"]["geometry"]
    assert app.default_mm_per_px() == pytest.approx(
        geometry["optical_resolution_mm_per_px"]
    )


def test_millimetres_scale_with_the_assumption_the_operator_sets(
    detector: inference.DefectDetector, steel_frame: np.ndarray
) -> None:
    result = app.run_frame(detector, steel_frame, app.RuntimeSettings(conf=0.15, imgsz=256))
    if not result.detections:
        pytest.skip("no detections on this frame at the shipped operating point")

    coarse = app.detection_frame(result, app.identity_calibration, 0.4)
    fine = app.detection_frame(result, app.identity_calibration, 0.2)

    assert coarse["Length mm"].iloc[0] == pytest.approx(fine["Length mm"].iloc[0] * 2, rel=1e-3)
    assert coarse["Across strip mm"].iloc[0] == pytest.approx(
        fine["Across strip mm"].iloc[0] * 2, rel=1e-3
    )


def test_the_detection_export_carries_its_own_scale_assumption(
    detector: inference.DefectDetector, steel_frame: np.ndarray
) -> None:
    """A CSV of millimetres without the mm/px it was computed at is a trap."""
    result = app.run_frame(detector, steel_frame, app.RuntimeSettings(conf=0.15, imgsz=256))
    if not result.detections:
        pytest.skip("no detections on this frame at the shipped operating point")

    export = app.detection_table(["f.jpg"], [result], app.identity_calibration, 0.35)

    assert "mm_per_px" in export.columns
    assert set(export["mm_per_px"]) == {0.35}
    assert export["length_mm"].iloc[0] == pytest.approx(
        max(export["x2"].iloc[0] - export["x1"].iloc[0],
            export["y2"].iloc[0] - export["y1"].iloc[0]) * 0.35,
        rel=1e-2,
    )


def test_the_millimetre_assumption_is_labelled_as_an_assumption() -> None:
    source = APP_PATH.read_text()
    assert "ASSUMPTION" in source
    assert "not a measurement" in source


# --------------------------------------------------------------------------- #
# GAP 7 - explainability, on demand
# --------------------------------------------------------------------------- #


def test_explain_is_not_imported_at_console_start() -> None:
    """EigenCAM costs seconds on its first call, so it must stay off the import
    path and off every rerun. Importing `app` must not drag `explain` in."""
    source = APP_PATH.read_text()
    assert "from explain import" in source, "the console must reach explainability"
    # The import lives inside compute_explanation, not at module scope.
    module_scope = source.split("def compute_explanation")[0]
    assert "from explain import" not in module_scope


def test_an_explanation_belongs_to_one_frame_and_one_model(
    detector: inference.DefectDetector,
) -> None:
    """A cached CAM from another frame must never sit beside a fresh verdict."""
    settings = app.RuntimeSettings(imgsz=256)
    other = app.RuntimeSettings(imgsz=320)
    shape = (200, 200, 3)

    same = app.explanation_signature("a.jpg", detector, settings, shape)
    assert app.explanation_signature("a.jpg", detector, settings, shape) == same
    assert app.explanation_signature("b.jpg", detector, settings, shape) != same
    assert app.explanation_signature("a.jpg", detector, other, shape) != same


def test_the_explain_button_produces_an_overlay_of_the_right_shape(
    detector: inference.DefectDetector, steel_frame: np.ndarray
) -> None:
    explanation, message = app.compute_explanation(
        detector, app.RuntimeSettings(conf=0.15, imgsz=256), steel_frame
    )
    if explanation is None:
        pytest.skip(f"explainability unavailable here: {message}")

    overlay = explanation.overlay(steel_frame)
    assert overlay.shape == steel_frame.shape
    assert overlay.dtype == np.uint8
    assert explanation.method in {"eigencam", "occlusion"}
    assert explanation.stats


def test_a_broken_explainer_returns_a_message_instead_of_taking_the_page_down(
    detector: inference.DefectDetector,
) -> None:
    """A one-pixel frame is not explainable; the console must say so, not crash."""
    explanation, message = app.compute_explanation(
        detector, app.RuntimeSettings(imgsz=256), np.zeros((1, 1, 3), dtype=np.uint8)
    )
    assert explanation is None or message == ""
    if explanation is None:
        assert message, "a failure must come back with something to show the operator"


# --------------------------------------------------------------------------- #
# the saturated defect rate
# --------------------------------------------------------------------------- #


def test_a_hundred_percent_defect_rate_is_explained_not_left_hanging() -> None:
    """Every one of the 1,800 NEU-DET frames carries an annotated defect, so the
    split cannot produce a PASS. Verified on this tree:
    `find data/neu-det/*/labels -name '*.txt' -size -1c` returns nothing."""
    note = app.saturated_defect_rate_note(12, 12, from_samples=True)
    assert "Why 100%" in note
    assert "1,800" in note

    assert app.saturated_defect_rate_note(11, 12, from_samples=True) == ""
    assert app.saturated_defect_rate_note(0, 0, from_samples=True) == ""
    uploads = app.saturated_defect_rate_note(5, 5, from_samples=False)
    assert uploads and "NEU-DET" not in uploads


# --------------------------------------------------------------------------- #
# the rendered console
# --------------------------------------------------------------------------- #


def test_the_console_opens_on_a_representative_frame_with_a_calibrated_number() -> None:
    """Zero clicks. This is the screen a judge sees before touching anything."""
    from streamlit.testing.v1 import AppTest

    if not (PROJECT_ROOT / "models").is_dir():
        pytest.skip("no models/ directory")

    at = AppTest.from_file(str(APP_PATH), default_timeout=400).run()
    assert not at.exception, [e.value for e in at.exception]

    family = at.get_by_key("sample_family")
    assert family.value == "patches", "the landing screen must not open on crazing"

    verdict = [m.value for m in at.markdown if '<div class="jsw-verdict">' in m.value]
    assert verdict, "no verdict strip rendered"
    assert "Peak P(true positive)" in verdict[0]
    assert "Raw detector score" in verdict[0], "the raw score must stay visible"

    badge = [m.value for m in at.markdown if '<div class="jsw-badge">' in m.value]
    assert badge and "SINGLE PASS" in badge[0]

    covers = [m for m in at.metric if m.label == "Frame covers"]
    assert covers and "mm" in covers[0].value
    assert "assumed" in (covers[0].delta or "")


def test_a_logo_upload_gets_a_withheld_verdict_and_no_defect_call() -> None:
    """The worst audited failure, driven the way a judge produces it.

    Before the gate this rendered a defect verdict with operator guidance telling
    the operator to quarantine the coil.
    """
    from streamlit.testing.v1 import AppTest

    if not (PROJECT_ROOT / "models").is_dir():
        pytest.skip("no models/ directory")

    at = AppTest.from_file(str(APP_PATH), default_timeout=400).run()
    assert not at.exception, [e.value for e in at.exception]

    source = [r for r in at.radio if r.label == "Frame source"][0]
    source.set_value("Upload").run()
    uploader = [u for u in at.file_uploader if u.label == "Drop a strip frame"][0]
    uploader.set_value(("company_logo.png", _logo_png(), "image/png")).run()

    assert not at.exception, [e.value for e in at.exception]

    withheld = [m.value for m in at.markdown if '<div class="jsw-withheld">' in m.value]
    assert withheld, "the gate did not render a withheld verdict"
    assert "VERDICT" in withheld[0] and "WITHHELD" in withheld[0]

    verdict = [m.value for m in at.markdown if '<div class="jsw-verdict">' in m.value]
    assert not verdict, "a rejected frame must not also get a PASS/DEFECT flag"

    # The operator must see the evidence and be able to force the demonstration.
    assert at.dataframe, "the failing checks table did not render"
    assert any("Score it anyway" in c.label for c in at.checkbox)
