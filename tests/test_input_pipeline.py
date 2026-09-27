"""Tests for the input-pipeline decision: 256 px, bilinear, and why not lossless.

The project premise this guards against is an attractive one that turned out to be
wrong, so it will be proposed again: NEU-DET is natively 200x200, every YOLO input
size is a multiple of 32, therefore every input size resamples the source,
therefore a path that pads instead of scaling -- keeping every pixel byte-exact --
should be more accurate. It is not. Three separate things have to stay pinned for
that conclusion to remain readable:

1. **The constant.** `inference.DEFAULT_IMGSZ` is 256 and the comment above it
   names the reports and the numbers that kept it there, so the next person to
   open the file does not have to rediscover this.
2. **The instrument.** `src/input_pipeline_probe.py` substitutes pre-resampled
   images on disk for the framework's own in-loader resize. That substitution is
   only meaningful if it is neutral, which is testable: the on-disk *bilinear*
   arm must reproduce the framework's own scaled number exactly. If it ever
   stops doing so, every other arm in the probe is measuring the substitution
   rather than the kernel.
3. **The finding.** Losslessness does not predict accuracy. The nearest-neighbour
   arm is *exactly* invertible -- the 200x200 source is recovered byte-for-byte
   out of the 256x256 image by indexing, which is asserted here on a synthetic
   frame rather than taken from the report -- and it is the worst-scoring arm of
   the five, beaten by an arm that destroys 36% of the source samples.

`reports/input_pipeline_decision.md` is the prose; these tests are what stop it
drifting away from `reports/input_pipeline_probe.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS = PROJECT_ROOT / "reports"
PROBE_JSON = REPORTS / "input_pipeline_probe.json"
DECISION_MD = REPORTS / "input_pipeline_decision.md"
INFERENCE_PY = PROJECT_ROOT / "src" / "inference.py"

for _extra in (PROJECT_ROOT / "src",):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import inference  # noqa: E402

# The measured claims the decision rests on. Each one is checked twice: against
# the JSON the probe wrote, and against the Markdown that quotes it. A number that
# drifts in either place fails here rather than in a judge's question.
#
# (checkpoint, split, arm, mAP50)
KEY_CLAIMS: tuple[tuple[str, str, str, float], ...] = (
    ("yolov8n_neudet", "test", "scale", 0.7524),  # the shipped held-out number
    ("yolov8n_neudet", "test", "bilinear", 0.7524),  # instrument identity
    ("yolov8n_neudet", "test", "nearest", 0.6252),  # lossless, and the worst arm
    ("yolov8n_neudet", "test", "decimated", 0.6910),  # 36% destroyed, still better
    ("yolov8n_neudet", "test", "pad", 0.7091),  # lossless, still loses
    ("yolov8n_joint", "test", "scale", 0.7642),
    ("yolov8n_joint", "test", "nearest", 0.6257),
    ("yolov8n_joint", "test", "decimated", 0.7296),
    ("yolov8n_joint", "test", "pad", 0.7164),
)

# Arms that preserve every source pixel, and arms that do not.
LOSSLESS_ARMS = frozenset({"pad", "nearest"})


def _load(path: Path) -> Any:
    if not path.exists():
        pytest.skip(f"{path.name} not generated yet; run src/input_pipeline_probe.py")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def probe() -> dict[str, Any]:
    return _load(PROBE_JSON)


@pytest.fixture(scope="module")
def decision_md() -> str:
    if not DECISION_MD.exists():
        pytest.skip("input_pipeline_decision.md not written yet")
    return DECISION_MD.read_text()


def _run(probe: dict[str, Any], checkpoint: str, split: str, arm: str) -> dict[str, Any]:
    for row in probe["runs"]:
        if row["checkpoint"] == checkpoint and row["split"] == split and row["pipeline"] == arm:
            return row
    raise AssertionError(f"no {arm} run for {checkpoint}/{split} in {PROBE_JSON.name}")


# --------------------------------------------------------------------------
# 1. the constant
# --------------------------------------------------------------------------


def test_default_imgsz_is_still_256() -> None:
    """Two studies and one probe looked for a reason to move this. There is none."""
    assert inference.DEFAULT_IMGSZ == 256


def test_default_imgsz_is_a_legal_yolo_input_size() -> None:
    assert inference.DEFAULT_IMGSZ % 32 == 0


def test_the_constant_names_the_evidence_that_kept_it_there() -> None:
    """A bare 256 invites someone to 'fix' it to the framework default."""
    source = INFERENCE_PY.read_text()
    head = source[: source.index("DEFAULT_IMGSZ = 256")]
    for citation in (
        "reports/model_study.json",
        "reports/input_study.md",
        "reports/input_pipeline_probe.json",
        "reports/input_pipeline_decision.md",
    ):
        assert citation in head, f"the DEFAULT_IMGSZ comment does not cite {citation}"


def test_the_constant_records_why_lossless_lost() -> None:
    """The counter-intuitive half is the half that gets deleted as noise."""
    source = INFERENCE_PY.read_text()
    head = source[: source.index("DEFAULT_IMGSZ = 256")]
    assert "0.6252" in head and "0.6257" in head, "the nearest-neighbour result is not recorded"
    assert "byte-for-byte" in head or "byte-exact" in head


def test_no_derived_study_dataset_leaked_into_the_runtime() -> None:
    """data/input_study/* exists to be measured, never to be served or trained from.

    The reports are cited by name in the comment above DEFAULT_IMGSZ, which is the
    point of them; what must never appear is the dataset directory itself.
    """
    source = INFERENCE_PY.read_text()
    assert "data/input_study" not in source
    for derived in ("pad224", "pad256", "nearest256", "bilinear256", "decimated256"):
        assert derived not in source, f"{derived} is a measurement input, not a runtime path"


# --------------------------------------------------------------------------
# 2. the instrument
# --------------------------------------------------------------------------


def test_the_probe_ran_at_the_shipped_default_size(probe: dict[str, Any]) -> None:
    assert probe["imgsz"] == inference.DEFAULT_IMGSZ
    # rect=True, pad=0.5 -> ceil(imgsz/32 + 0.5) * 32. The tensor is never `imgsz`.
    assert probe["network_tensor_px"] == 288


@pytest.mark.parametrize("checkpoint,split", [("yolov8n_neudet", "test"), ("yolov8n_neudet", "val")])
def test_the_on_disk_bilinear_arm_reproduces_the_framework_path(
    probe: dict[str, Any], checkpoint: str, split: str
) -> None:
    """If pre-resampling on disk is not neutral, no other arm means anything.

    `bilinear` runs the identical `cv2.resize` call `BaseDataset.load_image` makes
    for an upscale, then hands the loader a file it will not touch. Same tensor,
    same metric, to the last digit.
    """
    framework = _run(probe, checkpoint, split, "scale")
    on_disk = _run(probe, checkpoint, split, "bilinear")
    for metric in ("mAP50", "mAP50_95", "precision", "recall"):
        assert framework[metric] == pytest.approx(on_disk[metric], abs=1e-9), (
            f"{checkpoint}/{split}: on-disk substitution changed {metric} "
            f"({framework[metric]} vs {on_disk[metric]}); the probe is measuring itself"
        )


def test_every_derived_split_reached_the_network_unscaled(probe: dict[str, Any]) -> None:
    """The resampling under test happened in the probe, not a second time in the loader."""
    assert probe["tensor_checks"], "no tensor checks recorded"
    for key, check in probe["tensor_checks"].items():
        assert check["content_is_unscaled"], f"{key}: loader rescaled the file"
        assert check["tensor_matches_prediction"], f"{key}: unexpected tensor size"
        assert check["border_is_pad_value"], f"{key}: border is not the framework's 114"


def test_the_nearest_upscale_really_is_lossless() -> None:
    """Re-proved here on a synthetic frame, not quoted from the report.

    200 -> 256 is an upscale, so `floor(j * 200/256)` hits every source index at
    least once and the map can be inverted by picking the first output index for
    each source index. That is what makes the arm a fair test of 'lossless':
    it invents nothing, discards nothing, and still loses.
    """
    from input_pipeline_probe import verify_nearest_is_invertible

    rng = np.random.default_rng(1337)
    frame = rng.integers(0, 256, size=(200, 200, 3), dtype=np.uint8)
    record = verify_nearest_is_invertible(frame, 256)
    assert record["every_source_index_survives"]
    assert record["source_recovered_byte_exact_by_indexing"]
    assert record["max_abs_error_after_inversion"] == 0


def test_the_probe_records_the_invertibility_check_it_ran(probe: dict[str, Any]) -> None:
    checks = [b for b in probe["builds"] if b["kernel"] == "nearest"]
    assert checks, "the nearest arm was never built"
    for build in checks:
        assert build["png_round_trip_byte_exact"] == build["images_written"]
        if "invertibility" in build:
            assert build["invertibility"]["source_recovered_byte_exact_by_indexing"]


# --------------------------------------------------------------------------
# 3. the finding
# --------------------------------------------------------------------------


@pytest.mark.parametrize("checkpoint,split,arm,expected", KEY_CLAIMS)
def test_key_measurements_are_what_the_decision_quotes(
    probe: dict[str, Any], decision_md: str, checkpoint: str, split: str, arm: str, expected: float
) -> None:
    measured = _run(probe, checkpoint, split, arm)["mAP50"]
    assert measured == pytest.approx(expected, abs=5e-5), (
        f"{checkpoint}/{split}/{arm}: probe says {measured}, this test pins {expected}"
    )
    assert f"{expected:.4f}" in decision_md, (
        f"{expected:.4f} ({checkpoint}/{split}/{arm}) is not quoted in the decision report"
    )


@pytest.mark.parametrize("checkpoint,split", [("yolov8n_neudet", "test"), ("yolov8n_joint", "test")])
def test_losslessness_does_not_predict_accuracy(
    probe: dict[str, Any], checkpoint: str, split: str
) -> None:
    """The load-bearing result. Both lossless arms lose to the lossy shipped path.

    Stated as an ordering rather than as a threshold so it survives a re-measure:
    the best arm must be a lossy one, and a lossless arm must be the worst.
    """
    scores = {
        arm: _run(probe, checkpoint, split, arm)["mAP50"]
        for arm in ("scale", "nearest", "decimated", "pad")
    }
    best = max(scores, key=lambda a: scores[a])
    worst = min(scores, key=lambda a: scores[a])
    assert best not in LOSSLESS_ARMS, f"{checkpoint}/{split}: a lossless arm won ({best})"
    assert worst in LOSSLESS_ARMS, f"{checkpoint}/{split}: the worst arm was lossy ({worst})"
    assert scores["decimated"] > scores["nearest"], (
        f"{checkpoint}/{split}: destroying 36% of the samples no longer beats "
        "preserving all of them badly; the decision's mechanism has changed"
    )


@pytest.mark.parametrize("checkpoint,split", [("yolov8n_neudet", "test"), ("yolov8n_joint", "test")])
def test_the_padded_path_still_loses_to_the_scaled_path(
    probe: dict[str, Any], checkpoint: str, split: str
) -> None:
    pad = _run(probe, checkpoint, split, "pad")["mAP50"]
    scale = _run(probe, checkpoint, split, "scale")["mAP50"]
    assert pad < scale, f"{checkpoint}/{split}: pad {pad} >= scale {scale}; reopen the decision"


def test_the_nearest_versus_bilinear_contrast_is_resolved(probe: dict[str, Any]) -> None:
    """A finding this counter-intuitive has to clear the bootstrap, not just the mean."""
    table = probe["contrasts"]["yolov8n_neudet/test"]
    entry = table["nearest256_minus_bilinear256"]["mAP50"]
    assert entry["ci_excludes_zero"], f"nearest vs bilinear is not resolved: {entry}"
    assert entry["ci95_high"] < 0.0, "nearest is not resolvably worse than bilinear"


def test_the_instrument_identity_contrast_is_exactly_zero(probe: dict[str, Any]) -> None:
    entry = probe["contrasts"]["yolov8n_neudet/test"]["bilinear256_minus_scale256"]["mAP50"]
    assert entry["point_difference"] == pytest.approx(0.0, abs=1e-9)
    assert entry["ci95_low"] == pytest.approx(0.0, abs=1e-9)
    assert entry["ci95_high"] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# 4. the decision document
# --------------------------------------------------------------------------


def test_the_decision_states_what_was_not_changed(decision_md: str) -> None:
    lowered = decision_md.lower()
    for phrase in ("default_imgsz", "256", "not changed"):
        assert phrase in lowered, f"the decision report never mentions {phrase!r}"


def test_the_decision_explains_why_no_model_was_trained(decision_md: str) -> None:
    """The brief offered a training budget conditionally. Declining needs a reason."""
    lowered = decision_md.lower()
    assert "train" in lowered
    assert "pad224" in decision_md or "padded" in lowered


def test_the_decision_has_the_plain_english_section_for_judges(decision_md: str) -> None:
    lowered = decision_md.lower()
    assert "plain english" in lowered or "plain-english" in lowered
    assert "640" in decision_md, "the 640 px collapse is the headline for a non-specialist"
    assert "tiling" in lowered and "2048" in decision_md, "the line-scan tiling case is missing"
