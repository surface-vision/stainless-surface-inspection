"""Tests for the inference-resolution study and the report text it backs.

Three things go wrong with a resolution study, and all three have already
happened in this repository:

1. **A grid that starts above the peak.** `src/model_study.py` swept
   256/320/416/512/640 and reported 256 px as the best size. Extending the grid
   down to 128 px moves the *val* peak to 224 px, so the sweep was reporting a
   boundary, not a maximum. The test here asserts the shipped grid brackets its
   own peak on both splits -- if a future sweep peaks at an endpoint again, it
   fails.
2. **A prose claim that contradicts the arithmetic.** `reports/model_study.md`
   asserted that "256 px is the only size in the grid that does not [upsample]".
   256/200 = 1.28, so it upsamples too. The test greps for that sentence and for
   the arithmetic that refutes it.
3. **Numbers in Markdown that no longer match the JSON they came from.** Every
   mAP50 quoted in the report body is cross-checked against
   `reports/resolution_study.json`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS = PROJECT_ROOT / "reports"
STUDY_JSON = REPORTS / "resolution_study.json"
STUDY_MD = REPORTS / "resolution_study.md"
MODEL_STUDY_MD = REPORTS / "model_study.md"

# The grid the study is required to cover. 128/160/192 are at or below the
# 200x200 native size; 224 and 256 straddle it; 320 is the training size.
REQUIRED_SIZES = (128, 160, 192, 224, 256, 320, 416, 512, 640)


def _load(path: Path) -> Any:
    if not path.exists():
        pytest.skip(f"{path.name} not generated yet")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def study() -> dict[str, Any]:
    return _load(STUDY_JSON)


@pytest.fixture(scope="module")
def md_text() -> str:
    if not STUDY_MD.exists():
        pytest.skip("resolution_study.md not generated yet")
    return STUDY_MD.read_text()


def _cells(study: dict[str, Any], tag: str, split: str) -> list[dict[str, Any]]:
    return sorted(
        (r for r in study["runs"] if r["tag"] == tag and r["split"] == split),
        key=lambda r: r["imgsz"],
    )


# --------------------------------------------------------------------------
# the grid itself
# --------------------------------------------------------------------------


@pytest.mark.parametrize("split", ["val", "test"])
def test_shipped_checkpoint_is_swept_over_the_whole_grid(
    study: dict[str, Any], split: str
) -> None:
    sizes = [c["imgsz"] for c in _cells(study, "yolov8n", split)]
    assert sizes == list(REQUIRED_SIZES), f"{split}: swept {sizes}"


def test_grid_extends_below_the_200px_native_size(study: dict[str, Any]) -> None:
    """The whole point of the extension: sizes that do not upsample must be in it."""
    sizes = {c["imgsz"] for c in _cells(study, "yolov8n", "val")}
    assert {s for s in sizes if s <= 200} >= {128, 160, 192}


@pytest.mark.parametrize("split", ["val", "test"])
def test_the_peak_is_interior_to_the_grid(study: dict[str, Any], split: str) -> None:
    """A peak at an endpoint means the sweep found a boundary, not a maximum."""
    cells = _cells(study, "yolov8n", split)
    best = max(cells, key=lambda c: c["mAP50"])
    assert cells[0]["imgsz"] < best["imgsz"] < cells[-1]["imgsz"], (
        f"{split} peaks at the grid endpoint {best['imgsz']} px"
    )


def test_val_and_test_peaks_are_recorded_even_when_they_disagree(
    study: dict[str, Any],
) -> None:
    """The val peak (224) and the test peak (256) differ; both must be stated."""
    peaks = study["peaks"]["yolov8n"]
    assert set(peaks) >= {"val", "test"}
    for split in ("val", "test"):
        cells = _cells(study, "yolov8n", split)
        assert peaks[split]["imgsz"] == max(cells, key=lambda c: c["mAP50"])["imgsz"]


def test_shipped_operating_point_reproduces(study: dict[str, Any]) -> None:
    """yolov8n @256 on test is the headline number; it must not move."""
    cell = next(c for c in _cells(study, "yolov8n", "test") if c["imgsz"] == 256)
    assert cell["mAP50"] == pytest.approx(0.7524, abs=5e-4)
    assert cell["mAP50_95"] == pytest.approx(0.3967, abs=5e-4)


def test_mAP_collapses_above_the_training_size(study: dict[str, Any]) -> None:
    cells = {c["imgsz"]: c["mAP50"] for c in _cells(study, "yolov8n", "test")}
    assert cells[640] < 0.5 * cells[256]


# --------------------------------------------------------------------------
# the arithmetic the corrected paragraph rests on
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "size,upsamples", [(128, False), (160, False), (192, False), (224, True), (256, True)]
)
def test_which_grid_sizes_upsample_200px_sources(size: int, upsamples: bool) -> None:
    """256/200 = 1.28: the claim that 256 px does not upsample is false."""
    assert (size > 200) is upsamples


# --------------------------------------------------------------------------
# report text
# --------------------------------------------------------------------------


FALSE_CLAIMS = (
    "is the only size in the grid that does not",
    "except 256 upsamples the input",
)


def test_model_study_no_longer_asserts_the_false_upsampling_claim() -> None:
    """`model_study.md` is the flagship report; the claim must be gone outright."""
    if not MODEL_STUDY_MD.exists():
        pytest.skip("model_study.md not present")
    text = MODEL_STUDY_MD.read_text()
    for claim in FALSE_CLAIMS:
        assert claim not in text, f"model_study.md still contains: {claim!r}"


def test_the_resolution_study_only_ever_quotes_the_false_claim() -> None:
    """The study has to reproduce the wrong sentence in order to correct it.

    That is legitimate, but only as a marked quotation. Every line carrying the
    claim must be a Markdown blockquote, so no reader can mistake the erratum
    for the report's own position.
    """
    if not STUDY_MD.exists():
        pytest.skip("resolution_study.md not generated yet")
    for lineno, line in enumerate(STUDY_MD.read_text().splitlines(), 1):
        for claim in FALSE_CLAIMS:
            if claim in line:
                assert line.lstrip().startswith(">"), (
                    f"resolution_study.md:{lineno} states the false claim "
                    f"outside a blockquote: {line.strip()!r}"
                )


def test_model_study_states_the_correct_upsampling_arithmetic() -> None:
    if not MODEL_STUDY_MD.exists():
        pytest.skip("model_study.md not present")
    text = MODEL_STUDY_MD.read_text()
    assert "1.28" in text, "the 256/200 ratio should be stated explicitly"
    assert "224" in text, "the val peak at 224 px must be acknowledged"


def _leaf_floats(obj: Any) -> list[float]:
    if isinstance(obj, dict):
        return [v for x in obj.values() for v in _leaf_floats(x)]
    if isinstance(obj, list):
        return [v for x in obj for v in _leaf_floats(x)]
    if isinstance(obj, bool):
        return []
    return [float(obj)] if isinstance(obj, (int, float)) else []


def _derived_gaps(study: dict[str, Any]) -> dict[str, str]:
    """Differences the report quotes, recomputed from the cells rather than typed.

    These are the only four-decimal numbers in the body that are not themselves
    a measured cell. Recomputing them here means a change to the sweep either
    updates the report or fails this test -- it cannot silently disagree.
    """
    m = {
        (c["split"], c["imgsz"]): c["mAP50"]
        for c in study["runs"]
        if c["tag"] == "yolov8n"
    }
    plateau = (192, 224, 256, 320)

    def spread(split: str) -> float:
        vals = [m[(split, s)] for s in plateau]
        return max(vals) - min(vals)

    gaps = {
        f"{m[('val', 224)] - m[('val', 256)]:.4f}": "val 224 px over 256 px",
        f"{m[('test', 256)] - m[('test', 224)]:.4f}": "test 256 px over 224 px",
        f"{spread('val'):.4f}": "val spread across the 192-320 px plateau",
        f"{spread('test'):.4f}": "test spread across the 192-320 px plateau",
    }

    # Section 3 compares the 640 px model against the shipped one on test. Every
    # delta it quotes is recomputed here from the same cells, so a re-sweep that
    # moves a number cannot leave a stale one in the prose.
    def cell(tag: str, imgsz: int) -> dict[str, Any] | None:
        return next(
            (
                r
                for r in study["runs"]
                if r["tag"] == tag and r["split"] == "test" and r["imgsz"] == imgsz
            ),
            None,
        )

    shipped = cell("yolov8n", 256)
    for imgsz in (416, 512):
        other = cell("yolov8n_640", imgsz)
        if shipped is None or other is None:
            continue
        for metric in ("mAP50", "mAP50_95", "fitness"):
            gaps[f"{abs(other[metric] - shipped[metric]):.4f}"] = (
                f"test {metric}, 640 px model at {imgsz} px vs shipped at 256 px"
            )
        for name, value in other["per_class_AP50"].items():
            gaps[f"{abs(value - shipped['per_class_AP50'][name]):.4f}"] = (
                f"test AP50 delta on {name} at {imgsz} px"
            )
    return gaps


def test_every_number_quoted_in_the_report_exists_in_the_json(
    study: dict[str, Any], md_text: str
) -> None:
    """Guards against hand-edited numbers drifting away from the measurement.

    Every 0.xxxx in the report body must be either a value that appears
    somewhere in `resolution_study.json` -- a mAP, a per-class AP50, a fitness
    -- or one of the four gaps recomputed in `_derived_gaps`.
    """
    known = {f"{v:.4f}" for v in _leaf_floats(study)}
    known |= {f"{v:.3f}" for v in _leaf_floats(study)}
    known |= set(_derived_gaps(study))
    # The bootstrap half-width is quoted from model_study.md, not measured here.
    known.add("0.0254")
    body = md_text.split("## Provenance")[0]
    quoted = set(re.findall(r"\b0\.\d{4}\b", body))
    unknown = {q for q in quoted if q not in known}
    assert not unknown, f"numbers in the report with no JSON row: {sorted(unknown)}"


def test_the_val_peak_beats_256_by_less_than_the_measured_noise(
    study: dict[str, Any],
) -> None:
    """The reason 256 px survives a val peak at 224 px.

    `reports/model_study.md` measures a 95% bootstrap half-width of 0.0254 on
    mAP50 differences on these 180-image splits. If the 224-vs-256 val gap ever
    grows past that, the shipped size genuinely is wrong and section 5 of the
    study no longer holds.
    """
    m = {c["imgsz"]: c["mAP50"] for c in _cells(study, "yolov8n", "val")}
    gap = m[224] - m[256]
    assert 0 < gap < 0.0254, f"val 224-vs-256 gap is {gap:.4f}"


def test_256px_wins_every_criterion_except_val_mAP50(study: dict[str, Any]) -> None:
    """Five of six criteria choose 256 px; that is what makes it defensible."""
    losers = []
    for split in ("val", "test"):
        cells = _cells(study, "yolov8n", split)
        for metric in ("mAP50", "mAP50_95", "fitness"):
            argmax = max(cells, key=lambda c: c[metric])["imgsz"]
            if argmax != 256:
                losers.append((split, metric, argmax))
    assert losers == [("val", "mAP50", 224)], f"unexpected argmaxes: {losers}"


def test_report_records_the_measured_640px_epoch_cost(md_text: str) -> None:
    """The gap this study exists to close: the '11 min/epoch' claim was false."""
    assert "11 min" in md_text or "11 minutes" in md_text
    assert "s/epoch" in md_text


# --------------------------------------------------------------------------
# the 640 px training run
# --------------------------------------------------------------------------

RUN_640 = PROJECT_ROOT / "models/yolov8n_640"
RESULTS_640 = RUN_640 / "results.csv"
AUDIT_PROBE = PROJECT_ROOT / "docs/audit/yolov8n_640_7epochs.csv"

# ultralytics writes cumulative seconds in `time`; every other column is a
# per-epoch metric and must match the audit probe exactly.
_TIMING_COLUMNS = {"time"}


def _epoch_rows(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open() as handle:
        return list(csv.DictReader(handle))


def _per_epoch_seconds(rows: list[dict[str, str]]) -> list[float]:
    """Per-epoch wall seconds, tolerating a resumed run.

    `time` is cumulative, but ultralytics restarts that clock on resume, so a
    backwards step marks a new segment rather than a negative epoch. The 640 px
    run was resumed at epoch 28 after a session reap.
    """
    cum = [float(r["time"]) for r in rows]
    per_epoch: list[float] = []
    previous = 0.0
    for index, value in enumerate(cum):
        per_epoch.append(value if (index == 0 or value < previous) else value - previous)
        previous = value
    return per_epoch


def test_the_640_run_is_the_same_run_as_the_audit_probe() -> None:
    """The load-bearing evidence that 56 s/epoch describes *this* job.

    The audit timed seven epochs of this exact command on an idle machine and
    got 55.5 s/epoch median. This project's run of it, on a contended laptop,
    took 145 s/epoch. Those two numbers only belong in the same table if the
    two jobs are computing the same thing -- so every metric column of the
    first seven epochs must agree. If they ever diverge, the seed, the split or
    the recipe changed and the idle-machine timing may no longer be quotable.
    """
    if not (RESULTS_640.exists() and AUDIT_PROBE.exists()):
        pytest.skip("640 px run or audit probe not present")
    probe, run = _epoch_rows(AUDIT_PROBE), _epoch_rows(RESULTS_640)
    assert run, "the 640 px run logged no epochs"
    columns = [c for c in probe[0] if c not in _TIMING_COLUMNS]
    assert len(columns) >= 10, f"probe has too few metric columns: {columns}"
    for epoch, (a, b) in enumerate(zip(probe, run), start=1):
        for column in columns:
            assert round(float(a[column]), 5) == round(float(b[column]), 5), (
                f"epoch {epoch} column {column!r} diverges: "
                f"probe {a[column]} vs run {b[column]}"
            )


def test_the_640_run_is_far_cheaper_than_the_briefed_11_minutes() -> None:
    """The claim being corrected: >11 min/epoch, i.e. >660 s.

    Asserted against the *contended* wall time, which is the slowest honest
    number this project has. Even that is comfortably under the briefed figure,
    so the correction does not depend on the idle-machine measurement.
    """
    if not RESULTS_640.exists():
        pytest.skip("640 px run not present")
    per_epoch = _per_epoch_seconds(_epoch_rows(RESULTS_640))
    ordered = sorted(per_epoch)
    median = ordered[len(ordered) // 2]
    assert median < 660, f"median epoch {median:.1f}s does not refute the claim"


def test_the_json_training_block_matches_the_run_log(study: dict[str, Any]) -> None:
    """Stops the report's training numbers drifting from ultralytics' own csv."""
    if not RESULTS_640.exists():
        pytest.skip("640 px run not present")
    block = study.get("training_640")
    if block is None:
        pytest.skip("training_640 block not written yet")
    rows = _epoch_rows(RESULTS_640)
    assert block["epochs_completed"] <= len(rows), (
        "the JSON claims more epochs than results.csv contains"
    )
    if block["epochs_completed"] != len(rows):
        pytest.skip("training still running; block is an interim snapshot")
    per_epoch = _per_epoch_seconds(rows)
    assert block["epoch_seconds_mean"] == pytest.approx(
        sum(per_epoch) / len(per_epoch), abs=0.05
    )
    m50 = [float(r["metrics/mAP50(B)"]) for r in rows]
    m5095 = [float(r["metrics/mAP50-95(B)"]) for r in rows]
    assert block["max_val_mAP50"] == pytest.approx(max(m50), abs=5e-5)

    # `best.pt` is chosen by fitness, so the value describing the kept
    # checkpoint is the one at the best-fitness epoch -- not the run's maximum.
    # Conflating the two would quietly overstate the checkpoint by 0.01 here.
    fitness = [0.1 * a + 0.9 * b for a, b in zip(m50, m5095)]
    best = max(range(len(rows)), key=lambda i: fitness[i])
    assert block["best_epoch_by_fitness"] == best + 1
    assert block["val_mAP50_at_best_fitness_epoch"] == pytest.approx(
        m50[best], abs=5e-5
    )


def test_the_640_run_ran_its_whole_schedule() -> None:
    """A truncated run would make section 3's comparison meaningless."""
    if not RESULTS_640.exists():
        pytest.skip("640 px run not present")
    import yaml  # ultralytics dependency; present wherever the model trained

    args = yaml.safe_load((RUN_640 / "args.yaml").read_text())
    assert args["imgsz"] == 640 and args["batch"] == 16
    assert len(_epoch_rows(RESULTS_640)) == args["epochs"], (
        "the 640 px run did not complete the schedule it was configured for"
    )


def test_the_640_checkpoint_is_swept_over_the_same_grid(study: dict[str, Any]) -> None:
    """Comparing one hand-picked size against a swept model would be rigged."""
    if not any(r["tag"] == "yolov8n_640" for r in study["runs"]):
        pytest.skip("640 px checkpoint not swept yet")
    for split in ("val", "test"):
        sizes = [c["imgsz"] for c in _cells(study, "yolov8n_640", split)]
        assert sizes == list(REQUIRED_SIZES), f"{split}: swept {sizes}"


def test_the_640_model_also_peaks_below_its_training_size(
    study: dict[str, Any],
) -> None:
    """Section 2's headline mechanism, re-tested on an independent checkpoint.

    The 320 px model peaking at 224-256 px could be one checkpoint's quirk. A
    model trained at 640 px peaking below 640 px is the same effect measured
    again at a different training resolution, which is what licenses stating it
    as a property of the dataset rather than of the weights.
    """
    if not any(r["tag"] == "yolov8n_640" for r in study["runs"]):
        pytest.skip("640 px checkpoint not swept yet")
    for split in ("val", "test"):
        cells = _cells(study, "yolov8n_640", split)
        peak = max(cells, key=lambda c: c["mAP50"])
        assert peak["imgsz"] < 640, (
            f"{split}: the 640 px model peaks at its training size ({peak['imgsz']})"
        )


def test_640_is_reported_against_the_shipped_baseline_whichever_way_it_lands(
    study: dict[str, Any],
) -> None:
    """The credibility claim: the comparison exists and the report states it.

    Deliberately does not assert a direction -- the point of running 640 px was
    to report the number honestly either way. It asserts only that the two are
    measured on the same split and that the gap is inside the bootstrap noise
    the project measured, which is what section 3 actually claims.
    """
    if not any(r["tag"] == "yolov8n_640" for r in study["runs"]):
        pytest.skip("640 px checkpoint not swept yet")
    shipped = next(
        c for c in _cells(study, "yolov8n", "test") if c["imgsz"] == 256
    )
    best_640 = max(_cells(study, "yolov8n_640", "test"), key=lambda c: c["mAP50"])
    assert abs(best_640["mAP50"] - shipped["mAP50"]) < 0.0254, (
        "the 640 px gap has grown past the measured noise band; section 3's "
        "'indistinguishable' conclusion no longer holds"
    )
    assert f"{best_640['mAP50']:.4f}" in STUDY_MD.read_text(), (
        "the 640 px test result is not quoted in the report"
    )


def test_the_audit_probe_timing_is_recorded_separately_from_this_runs(
    study: dict[str, Any],
) -> None:
    """Two wall-time measurements of one job must never be silently merged."""
    probe = study.get("training_640_audit_probe")
    if probe is None:
        pytest.skip("audit probe block not present")
    mine = study["training_640"]
    assert probe["epoch_seconds_median"] < mine["epoch_seconds_median"], (
        "the idle-machine probe should be the faster of the two"
    )
    assert study["contention"]["load1_median"] > study["contention"]["cores"], (
        "a slower contended figure is only explainable with the load recorded"
    )
