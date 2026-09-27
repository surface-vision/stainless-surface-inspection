"""Tests for the command-line entry points and for report-file ownership.

Two real defects motivated this file, and both of them shipped because nothing
executed the code path:

1. `src/explain.py` called `load_detector(imgsz=args.imgsz)` one line *before* the
   `if args.imgsz is None` fallback, so the command the README documents died with
   `TypeError: int() argument must be a string ... not 'NoneType'`. It lived under
   `if __name__ == "__main__": # pragma: no cover`, which is exactly the shape of
   code that no test reaches.
2. `src/evaluate.py` wrote `reports/operating_point.json` unconditionally, while
   `src/false_alarm.py` only writes it behind a flag. Running the documented
   `make eval` therefore reset the console's default confidence from the shipped
   0.15 back to 0.05 -- a documented command silently regressing the shipped
   system, with a green test suite the whole time.

So: every CLI here is importable and every argument path is exercised, and the
ownership of `operating_point.json` is asserted rather than assumed.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _extra in (PROJECT_ROOT / "src",):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

os.environ.setdefault("YOLO_AUTOINSTALL", "False")

import evaluate  # noqa: E402
import explain as explain_mod  # noqa: E402
from inference import DEFAULT_IMGSZ  # noqa: E402

REPORTS = PROJECT_ROOT / "reports"
MAKEFILE = PROJECT_ROOT / "Makefile"
README = PROJECT_ROOT / "README.md"


# --------------------------------------------------------------------------- #
# 1. src/explain.py -- the CLI that crashed on its own documented defaults
# --------------------------------------------------------------------------- #


class _FakeExplanation:
    """Just enough of `Explanation` for `main` to run without a GPU."""

    def __init__(self) -> None:
        self.heatmap = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape(32, 32)

    def to_dict(self) -> dict[str, Any]:
        return {"method": "fake", "layer": None, "stats": {}, "elapsed_ms": 0.0, "warnings": []}


@pytest.fixture()
def cli_spy(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the two expensive calls in `explain.main` with recorders."""
    seen: dict[str, Any] = {}

    def fake_load_detector(weights: Any = None, **kw: Any) -> str:
        seen["load_detector_kwargs"] = dict(kw)
        seen["weights"] = weights
        # This is the assertion the original bug failed: `load_detector` casts
        # imgsz with int(), and None gets there if the fallback runs too late.
        assert kw.get("imgsz") is not None, "imgsz reached load_detector as None"
        int(kw["imgsz"])
        return "detector-sentinel"

    def fake_explain(detector: Any, image: Any, method: str = "eigencam", **kw: Any) -> _FakeExplanation:
        seen["detector"] = detector
        seen["image"] = image
        seen["method"] = method
        seen["explain_kwargs"] = dict(kw)
        return _FakeExplanation()

    monkeypatch.setattr(explain_mod, "load_detector", fake_load_detector)
    monkeypatch.setattr(explain_mod, "explain", fake_explain)
    return seen


def test_the_explain_parser_never_hands_on_a_none_imgsz() -> None:
    args = explain_mod.build_parser().parse_args(["frame.jpg"])
    assert args.imgsz == DEFAULT_IMGSZ
    assert isinstance(args.imgsz, int)


def test_explain_cli_runs_with_only_the_documented_defaults(cli_spy: dict[str, Any]) -> None:
    assert explain_mod.main(["frame.jpg"]) == 0
    assert cli_spy["load_detector_kwargs"]["imgsz"] == DEFAULT_IMGSZ
    assert cli_spy["method"] == "eigencam"
    assert cli_spy["explain_kwargs"] == {}


@pytest.mark.parametrize("imgsz", [128, 320, 640])
def test_explain_cli_honours_an_explicit_imgsz(cli_spy: dict[str, Any], imgsz: int) -> None:
    assert explain_mod.main(["frame.jpg", "--imgsz", str(imgsz)]) == 0
    assert cli_spy["load_detector_kwargs"]["imgsz"] == imgsz


def test_explain_cli_forwards_the_layer_only_where_it_means_something(
    cli_spy: dict[str, Any]
) -> None:
    explain_mod.main(["frame.jpg", "--method", "eigencam", "--layer", "9"])
    assert cli_spy["explain_kwargs"] == {"layer": 9}
    # Occlusion sensitivity has no layer to target, so passing one must not reach it.
    explain_mod.main(["frame.jpg", "--method", "occlusion", "--layer", "9"])
    assert cli_spy["explain_kwargs"] == {}


def test_explain_cli_writes_the_overlay_it_promises(
    cli_spy: dict[str, Any], tmp_path: Path
) -> None:
    import cv2

    frame = tmp_path / "frame.png"
    cv2.imwrite(str(frame), np.full((64, 64, 3), 120, dtype=np.uint8))
    out = tmp_path / "nested" / "cam.png"
    assert explain_mod.main([str(frame), "--out", str(out)]) == 0
    assert out.is_file() and out.stat().st_size > 0


def test_the_readme_command_for_explain_parses(cli_spy: dict[str, Any]) -> None:
    """The exact invocation in README.md, lifted from the file rather than retyped."""
    # Join shell line-continuations first, so the command reads as one line.
    text = README.read_text().replace("\\\n", " ")
    match = re.search(r"python src/explain\.py\s+(?P<args>[^\n]+)", text)
    assert match, "the README no longer documents an explain.py command"
    args = explain_mod.build_parser().parse_args(match.group("args").split())
    assert isinstance(args.imgsz, int)
    assert args.image.endswith(".jpg")


def test_the_module_docstring_records_what_a_cam_costs() -> None:
    """The demo needs an on-demand button, and the reason has to be findable."""
    doc = explain_mod.__doc__ or ""
    assert "on demand" in doc
    assert "spinner" in doc


# --------------------------------------------------------------------------- #
# 2. operating point ownership
# --------------------------------------------------------------------------- #


def _fake_point(source: str, conf: float) -> dict[str, Any]:
    return {"conf_threshold": conf, "source": source, "iou_threshold": 0.45}


def test_evaluate_stamps_its_own_name_on_the_point_it_writes(tmp_path: Path) -> None:
    operating = evaluate.OperatingPoint(
        threshold=0.05,
        point=evaluate.SweepPoint(threshold=0.05, tp=1, fp=1, fn=1, precision=0.5, recall=0.5,
                                  f1=0.5, fp_per_image=1.0, false_alarm_rate=0.5,
                                  defect_detection_rate=0.9, class_accuracy=1.0, images_flagged=1.0),
        rationale="test", sensitivity=[], f1_optimal_threshold=0.35, constraint_met=True,
        min_detection_rate=0.9, miss_cost_ratio=12.0, at_sweep_edge=True,
    )
    meta = {"nms_iou": 0.45, "tuning_split": "val", "weights": "w", "model_name": "m",
            "generated_at": "2026-09-09T00:00:00+00:00"}
    written = evaluate.write_operating_point(tmp_path / evaluate.OPERATING_POINT_OWN, operating, meta)
    payload = json.loads(written.read_text())
    assert written.name == "operating_point.evaluate.json"
    assert payload["source"] == evaluate.OPERATING_POINT_SOURCE
    assert "src/false_alarm.py" in payload["superseded_by"]


def test_evaluate_refuses_to_overwrite_a_false_alarm_derived_operating_point(
    tmp_path: Path,
) -> None:
    own = tmp_path / evaluate.OPERATING_POINT_OWN
    demo = tmp_path / evaluate.OPERATING_POINT_DEMO
    own.write_text(json.dumps(_fake_point(evaluate.OPERATING_POINT_SOURCE, 0.05)))
    demo.write_text(json.dumps(_fake_point("src/false_alarm.py", 0.15)))
    before = demo.read_bytes()

    message = evaluate.update_demo_operating_point(own, demo)
    assert message.startswith("refused:")
    assert demo.read_bytes() == before, "the shipped operating point was clobbered"


def test_the_refusal_can_be_overridden_but_only_deliberately(tmp_path: Path) -> None:
    own = tmp_path / evaluate.OPERATING_POINT_OWN
    demo = tmp_path / evaluate.OPERATING_POINT_DEMO
    own.write_text(json.dumps(_fake_point(evaluate.OPERATING_POINT_SOURCE, 0.05)))
    demo.write_text(json.dumps(_fake_point("src/false_alarm.py", 0.15)))

    evaluate.update_demo_operating_point(own, demo, force=True)
    assert json.loads(demo.read_text())["conf_threshold"] == 0.05


def test_evaluate_may_refresh_a_point_it_wrote_itself(tmp_path: Path) -> None:
    own = tmp_path / evaluate.OPERATING_POINT_OWN
    demo = tmp_path / evaluate.OPERATING_POINT_DEMO
    own.write_text(json.dumps(_fake_point(evaluate.OPERATING_POINT_SOURCE, 0.07)))
    demo.write_text(json.dumps(_fake_point(evaluate.OPERATING_POINT_SOURCE, 0.05)))
    assert evaluate.update_demo_operating_point(own, demo).startswith("overwrote")
    assert json.loads(demo.read_text())["conf_threshold"] == 0.07


def test_a_missing_demo_file_is_created_rather_than_refused(tmp_path: Path) -> None:
    own = tmp_path / evaluate.OPERATING_POINT_OWN
    demo = tmp_path / evaluate.OPERATING_POINT_DEMO
    own.write_text(json.dumps(_fake_point(evaluate.OPERATING_POINT_SOURCE, 0.05)))
    assert evaluate.update_demo_operating_point(own, demo).startswith("overwrote")
    assert demo.is_file()


def test_the_documented_eval_command_does_not_touch_the_demo_file_by_default() -> None:
    args = evaluate.build_parser().parse_args(["--split", "test", "--device", "auto"])
    assert args.update_operating_point is False
    assert args.force_operating_point is False


def test_the_shipped_operating_point_is_still_the_false_alarm_one() -> None:
    path = REPORTS / evaluate.OPERATING_POINT_DEMO
    if not path.is_file():
        pytest.skip("no shipped operating point on disk")
    payload = json.loads(path.read_text())
    assert payload["source"] == "src/false_alarm.py"
    assert payload["conf_threshold"] == pytest.approx(0.15)


# --------------------------------------------------------------------------- #
# 3. the Makefile is the documented interface, so it is tested like one
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "target", ["eval", "false-alarm", "study", "calibrate", "cross-domain", "bench", "test"]
)
def test_make_help_lists_every_assessment_target(target: str) -> None:
    listed = subprocess.run(["make", "help"], cwd=PROJECT_ROOT, capture_output=True,
                            text=True, check=True).stdout
    assert re.search(rf"^\s*(\x1b\[1m)?{re.escape(target)}\b", listed, re.MULTILINE), (
        f"`make help` does not list `{target}`"
    )


def test_clean_reports_does_not_delete_the_shipped_operating_point() -> None:
    body = MAKEFILE.read_text().split("clean-reports:", 1)[1]
    recipe = "\n".join(line for line in body.splitlines() if not line.lstrip().startswith("#"))
    removed = re.findall(r"reports/[\w.*-]+", recipe)
    assert "reports/operating_point.json" not in removed
    assert "reports/operating_point.evaluate.json" in removed


def test_eval_and_calibrate_run_at_the_served_input_size() -> None:
    text = MAKEFILE.read_text()
    for target in ("eval", "calibrate"):
        recipe = text.split(f"\n{target}:", 1)[1].split("\n\n", 1)[0]
        assert "$(INFER_IMGSZ)" in recipe, (
            f"`make {target}` does not pin the served input size; the console runs at 256 px"
        )
