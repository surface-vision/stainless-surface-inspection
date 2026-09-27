"""Tests for `src/calibrate.py` -- the confidence calibrator.

The properties worth pinning here are not "does the ECE go down" (that is a
measurement, and it lives in `reports/calibration.md`). They are the invariants a
calibrator has to hold for the rest of the system to stay true:

* **strict monotonicity**, because average precision is a function of the ranking
  and a calibrator that creates ties silently makes mAP tie-break dependent;
* **split discipline**, because a calibrator fitted on the split it is scored on
  is a self-report;
* **graceful degradation**, because a console that cannot find the calibration
  file must show raw scores loudly, not calibrated-looking ones quietly.

Everything here runs on synthetic arrays or on the persisted artefact. Nothing in
this file needs the GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _extra in (PROJECT_ROOT / "src",):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import calibrate  # noqa: E402
from calibrate import (  # noqa: E402
    IdentityCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    TemperatureCalibrator,
    _average_precision,
    _grouped_folds,
    _weighted_pava,
    brier_score,
    calibrator_from_dict,
    expected_calibration_error,
    fit_isotonic,
    fit_platt,
    fit_temperature,
    reliability_bins,
    verify_rank_preservation,
)

GRID = np.linspace(0.0, 1.0, 501)
CALIBRATION_JSON = PROJECT_ROOT / "reports" / "calibration.json"


@pytest.fixture(scope="module")
def synthetic() -> tuple[np.ndarray, np.ndarray]:
    """Scores that are under-confident by construction, like the real detector.

    True probability is `sqrt(s)`, so every score below 1 is pessimistic and the
    bias runs the same direction at both ends -- the shape a single temperature
    provably cannot represent.
    """
    rng = np.random.default_rng(1337)
    scores = rng.uniform(0.05, 0.95, size=4000)
    labels = (rng.uniform(size=scores.size) < np.sqrt(scores)).astype(np.float64)
    return scores, labels


# --------------------------------------------------------------------------- #
# 1. every calibrator is a strictly increasing map on [0, 1]
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "calibrator",
    [
        IdentityCalibrator(),
        TemperatureCalibrator(temperature=0.5),
        TemperatureCalibrator(temperature=2.5),
        PlattCalibrator(slope=0.8, intercept=1.1),
        IsotonicCalibrator(knots_x=(0.0, 0.3, 0.3, 0.9, 1.0),
                           knots_y=(0.1, 0.1, 0.7, 0.7, 0.95)),
    ],
)
def test_every_calibrator_is_strictly_increasing(calibrator: calibrate.Calibrator) -> None:
    out = calibrator.transform(GRID)
    assert np.all(np.diff(out) > 0.0), "a tie in the calibrated score makes AP tie-break dependent"
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_transform_preserves_shape_and_scalars() -> None:
    cal = TemperatureCalibrator(temperature=0.7)
    assert cal.transform(np.zeros((3, 4))).shape == (3, 4)
    assert isinstance(cal(0.4), float)
    assert cal.transform(np.float64(0.4)).shape == ()


def test_a_probability_outside_the_unit_interval_is_clamped_not_nan() -> None:
    cal = TemperatureCalibrator(temperature=0.4)
    assert np.isfinite(cal.transform(np.array([0.0, 1.0]))).all()


# --------------------------------------------------------------------------- #
# 2. serialisation round trip
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "calibrator",
    [
        IdentityCalibrator(),
        TemperatureCalibrator(temperature=0.61),
        PlattCalibrator(slope=1.3, intercept=-0.2),
        IsotonicCalibrator(knots_x=(0.0, 0.5, 1.0), knots_y=(0.2, 0.6, 0.9)),
    ],
)
def test_a_calibrator_survives_a_json_round_trip(calibrator: calibrate.Calibrator) -> None:
    restored = calibrator_from_dict(json.loads(json.dumps(calibrator.to_dict())))
    assert type(restored) is type(calibrator)
    np.testing.assert_allclose(restored.transform(GRID), calibrator.transform(GRID), atol=0, rtol=0)


def test_an_unknown_method_is_an_error_not_a_silent_identity() -> None:
    with pytest.raises(ValueError, match="Unknown calibration method"):
        calibrator_from_dict({"method": "magic", "params": {}})


# --------------------------------------------------------------------------- #
# 3. fitting
# --------------------------------------------------------------------------- #


def test_temperature_fits_but_cannot_close_a_two_sided_bias(
    synthetic: tuple[np.ndarray, np.ndarray]
) -> None:
    scores, labels = synthetic
    temp = fit_temperature(scores, labels)
    iso = fit_isotonic(scores, labels)
    ece_raw = expected_calibration_error(scores, labels)
    ece_temp = expected_calibration_error(temp.transform(scores), labels)
    ece_iso = expected_calibration_error(iso.transform(scores), labels)
    assert temp.temperature > 0.0
    # This is the justification for shipping isotonic, asserted rather than argued:
    # one temperature leaves most of a two-sided bias on the table.
    assert ece_iso < ece_temp
    assert ece_iso < 0.25 * ece_raw


def test_platt_never_ships_a_negative_slope(synthetic: tuple[np.ndarray, np.ndarray]) -> None:
    scores, labels = synthetic
    assert fit_platt(scores, labels).slope > 0.0


def test_isotonic_smoothing_refuses_to_claim_certainty() -> None:
    """All-positive top plateau must not calibrate to a flat 1.000."""
    scores = np.linspace(0.1, 0.9, 60)
    labels = (scores > 0.5).astype(np.float64)  # every score above 0.5 is a true positive
    smoothed = fit_isotonic(scores, labels, smoothing=0.5)
    raw = fit_isotonic(scores, labels, smoothing=0.0)
    assert max(raw.knots_y) == pytest.approx(1.0)
    assert max(smoothed.knots_y) < 1.0
    # ... and it still has to be a monotone map.
    assert np.all(np.diff(smoothed.transform(GRID)) > 0.0)


def test_weighted_pava_returns_a_non_decreasing_sequence() -> None:
    values = np.array([0.9, 0.1, 0.5, 0.4, 0.8])
    weights = np.array([1.0, 3.0, 2.0, 5.0, 1.0])
    out = _weighted_pava(values, weights)
    assert out.shape == values.shape
    assert np.all(np.diff(out) >= -1e-12)
    # PAVA is a projection: it preserves the weighted mean.
    assert float(out @ weights) == pytest.approx(float(values @ weights))


def test_grouped_folds_never_split_one_image_across_folds() -> None:
    image_index = np.repeat(np.arange(37), 3)
    folds = _grouped_folds(image_index, k=5, seed=7)
    assert len(folds) == 5
    assert np.array_equal(np.sum(folds, axis=0), np.ones(image_index.size))
    for held in folds:
        images_in = set(image_index[held].tolist())
        images_out = set(image_index[~held].tolist())
        assert not (images_in & images_out), "a frame leaked across the fold boundary"


# --------------------------------------------------------------------------- #
# 4. metrics
# --------------------------------------------------------------------------- #


def test_ece_is_near_zero_for_a_perfectly_calibrated_source() -> None:
    rng = np.random.default_rng(11)
    scores = rng.uniform(0.02, 0.98, size=40000)
    labels = (rng.uniform(size=scores.size) < scores).astype(np.float64)
    assert expected_calibration_error(scores, labels) < 0.01


def test_ece_grows_with_a_deliberate_shift() -> None:
    rng = np.random.default_rng(12)
    scores = rng.uniform(0.05, 0.75, size=20000)
    labels = (rng.uniform(size=scores.size) < np.clip(scores + 0.2, 0, 1)).astype(np.float64)
    assert expected_calibration_error(scores, labels) > 0.15


def test_reliability_bins_partition_the_population() -> None:
    rng = np.random.default_rng(3)
    scores = rng.uniform(0.0, 1.0, size=1000)
    labels = (rng.uniform(size=1000) < 0.4).astype(np.float64)
    rows = reliability_bins(scores, labels)
    assert sum(r["n"] for r in rows) == 1000, "an empty top bin means a detection at exactly 1.0 was lost"


def test_brier_score_is_zero_for_a_perfect_oracle() -> None:
    labels = np.array([1.0, 0.0, 1.0, 0.0])
    assert brier_score(labels, labels) == 0.0


# --------------------------------------------------------------------------- #
# 5. the property the whole module rests on: mAP cannot move
# --------------------------------------------------------------------------- #


def test_average_precision_is_invariant_under_a_strictly_increasing_map() -> None:
    rng = np.random.default_rng(5)
    scores = rng.uniform(0.05, 0.95, size=300)
    is_tp = rng.uniform(size=300) < scores
    calibrator = IsotonicCalibrator(knots_x=(0.0, 0.2, 0.2, 0.8, 1.0),
                                    knots_y=(0.05, 0.05, 0.6, 0.6, 0.99))
    before = _average_precision(scores, is_tp, n_gt=200)
    after = _average_precision(calibrator.transform(scores), is_tp, n_gt=200)
    assert before == after, "AP moved under a rank-preserving rescoring"


def test_verify_rank_preservation_catches_a_calibrator_that_creates_ties() -> None:
    scores = np.array([0.10, 0.11, 0.12, 0.80, 0.81])
    good = IsotonicCalibrator(knots_x=(0.0, 0.5, 1.0), knots_y=(0.1, 0.4, 0.9))
    assert verify_rank_preservation(good, scores)["strictly_increasing_on_data"]

    flat = IsotonicCalibrator(knots_x=(0.0, 0.05, 0.9, 1.0), knots_y=(0.3, 0.3, 0.3, 0.3), eps=0.0)
    report = verify_rank_preservation(flat, scores)
    assert report["collapsed_pairs"] > 0
    assert not report["strictly_increasing_on_data"]


def test_run_refuses_to_fit_and_score_on_the_same_split() -> None:
    with pytest.raises(ValueError, match="one mistake this module exists to avoid"):
        calibrate.run(fit_split="test", eval_split="test")


# --------------------------------------------------------------------------- #
# 6. the front door the console imports
# --------------------------------------------------------------------------- #


def test_a_missing_calibration_file_degrades_loudly_to_the_identity(tmp_path: Path) -> None:
    with pytest.warns(RuntimeWarning, match="No calibration file"):
        calibrator = calibrate.load_calibrator(tmp_path / "absent.json")
    assert isinstance(calibrator, IdentityCalibrator)
    np.testing.assert_array_equal(calibrator.transform(GRID), GRID)


def test_a_corrupt_calibration_file_degrades_loudly_to_the_identity(tmp_path: Path) -> None:
    target = tmp_path / "calibration.json"
    target.write_text("{not json")
    with pytest.warns(RuntimeWarning, match="Could not read a calibrator"):
        assert isinstance(calibrate.load_calibrator(target), IdentityCalibrator)


def test_calibrate_confidence_is_a_probability_and_preserves_order() -> None:
    if not CALIBRATION_JSON.is_file():
        pytest.skip("no fitted calibrator; run `make calibrate`")
    raw = [0.05, 0.15, 0.25, 0.4, 0.6, 0.75, 0.9, 0.99]
    out = [calibrate.calibrate_confidence(v) for v in raw]
    assert all(isinstance(v, float) for v in out)
    assert all(0.0 <= v <= 1.0 for v in out)
    assert all(b > a for a, b in zip(out, out[1:])), "the console would re-order the detection table"


def test_the_vectorised_and_scalar_front_doors_agree() -> None:
    if not CALIBRATION_JSON.is_file():
        pytest.skip("no fitted calibrator; run `make calibrate`")
    raw = np.array([0.07, 0.31, 0.68, 0.94])
    np.testing.assert_allclose(
        calibrate.calibrate_confidences(raw),
        [calibrate.calibrate_confidence(float(v)) for v in raw],
    )


# --------------------------------------------------------------------------- #
# 7. the shipped artefact says what it should
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def shipped() -> dict:
    if not CALIBRATION_JSON.is_file():
        pytest.skip("no fitted calibrator; run `make calibrate`")
    return json.loads(CALIBRATION_JSON.read_text())


def test_the_shipped_calibrator_was_fitted_on_val_and_scored_on_test(shipped: dict) -> None:
    assert shipped["fit"]["split"] == "val"
    assert shipped["evaluation"]["split"] == "test"
    assert shipped["fit"]["split"] != shipped["evaluation"]["split"]
    assert len(shipped["fit"]["date"]) == 10, "the fit date is what makes a stale calibrator visible"


def test_the_shipped_calibrator_reduced_ece_without_touching_map(shipped: dict) -> None:
    ece = shipped["metrics"]["ece"]
    assert ece["after"] < ece["before"]
    assert shipped["map50"]["identical"] is True
    assert shipped["map50"]["before"] == shipped["map50"]["after"]
    assert shipped["rank_preservation"]["order_identical"] is True
    assert shipped["rank_preservation"]["collapsed_pairs"] == 0


def test_the_shipped_calibrator_reloads_and_reproduces_its_own_examples(shipped: dict) -> None:
    calibrator = calibrator_from_dict(shipped["calibrator"])
    for raw, expected in shipped["examples"].items():
        assert calibrator(float(raw)) == pytest.approx(expected, abs=1e-12)


def test_the_shipped_calibrator_never_prints_certainty(shipped: dict) -> None:
    calibrator = calibrator_from_dict(shipped["calibrator"])
    assert float(calibrator.transform(np.float64(1.0))) < 1.0
