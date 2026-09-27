Final integration pass complete. All work verified by execution.

## Coherence audit — what was wrong and what I changed

Read all 9 source files (`src/*.py`, `demo/app.py`, ~5,300 lines). Found and fixed **five** real inconsistencies:

1. **`inference.py`** — `_VALID_SEVERITIES` was dead (defined, never used), while `report.py` and `demo/app.py` each carried their own copy of the same four bands, and `app.py` reimplemented the bucketing cutoffs in `SEVERITY_CUTOFFS` + `severity_band()`. Replaced with a single public `SEVERITY_BANDS` + `_SEVERITY_CUTOFFS` in `inference.py`; `_TIER_BASE_SCORE` now derives from it; both consumers import it. Deleted ~15 lines of duplicated logic from `app.py`.
2. **`prepare_data.py`** — held a second copy of the class list (`CLASSES`), which is the one constant that *must not* drift (it writes `data.yaml`, which fixes the trained head's index order). Now `from inference import CLASS_NAMES as CLASSES`. There is no second copy of the class list in the tree.
3. **`evaluate.py` docstring claimed "the single file the demo reads to pick its default confidence slider"** — the demo did **not** read `operating_point.json`; it hardcoded `0.25`. Wired it up: `demo/app.py` gained `load_operating_point()` (snaps onto the slider grid, degrades to the built-in default on a missing/malformed/old-schema file) and a caption stating where the default came from. The console now defaults to **conf 0.05**, the threshold `evaluate.py` chose on `val`.
4. **Real bug in `inference._build_detections`** — a box the head placed wholly off the top-left corner produced an **inverted** bbox (`x2 < x1`), violating the contract every consumer relies on. Found by mutation-testing my own suite. Clamp now pins the far corner behind the near one; such a box collapses to zero area at the edge.
5. **Dead code** — `report._build_figures` re-imported `_to_rgb` locally though `inference` was already imported at module scope; `report.py`/`explain.py` lacked the `sys.path` bootstrap the other modules have. Deleted `scripts/_smoke_inference.py`, `_latency_bench.py`, `_contract_checks.py` (hardcoded-absolute-path dev scratch, superseded by the test suite and `benchmark.py`) and four `reports/_smoke_*.png`.

No unused imports, no debug prints, no TODO/FIXME anywhere. Class names/order and the `conf 0.25 / iou 0.45 / imgsz 640` defaults agree across all modules.

## Deliverables

**`tests/test_smoke.py`** — 101 tests, **101 passed in 7 s**. Covers split integrity (counts, stratification, label well-formedness, no leakage by name *or* by MD5), the inference contract (types, ranges, internal consistency, JSON round trip, boxes clamped in-frame), batch-vs-single agreement, tiled inference (fall-through, coordinate lifting, global NMS across seams, overlap clamping), annotation shape/dtype/non-mutation, severity monotonicity in all four dimensions, greyscale/RGBA/float/1x1/multi-megapixel handling, the console's decode and operating-point path, disposition rules, and one test that runs the whole Streamlit script through `AppTest` and asserts it renders clean. I mutation-tested the suite twice; it caught the severity-cutoff mutation immediately and the bbox-clamp mutation only after I added a direct test for it — which is how bug #4 surfaced.

**`requirements.txt`** — 18 packages, every pin verified against the venv with `pip install --dry-run` (all "already satisfied"). Only what the project imports; `seaborn` and `python-pptx` are installed but imported nowhere, so they are excluded.

**`README.md`** — business problem, dataset, setup/train/eval/demo commands, project layout, and a 10-point limitations section. Every number is from `reports/`, including the uncomfortable ones: mAP50 0.660 vs 97.8% frame-level accuracy, crazing AP50 0.330, an 81% image-level false alarm rate at the recommended threshold, and an explicit statement that the operating point is not deployable as it stands.

**`Makefile`** — `setup data train eval bench demo test` plus `export report clean-reports` and a `help` default; every target dry-run verified.

## End-to-end check (fresh shell, no inherited env)

```
=== 1/4  make test ===   101 passed in 6.95s
=== 2/4  make eval ===   mAP50=0.6598 mAP50-95=0.3618 P=0.6107 R=0.6475
                         recommended conf=0.05 (constraint met: True)
=== 3/4  make demo ===   GET / -> HTTP 200   GET /healthz -> HTTP 200
                         killed; port now: refused
=== 4/4  training    ===  1 process still running, epoch 80 (was 43)
```

## Repository state

Nothing under `models/` was read-modified or written; the `yolov8n_neudet` job ran undisturbed throughout and advanced from epoch 43 to 80. No training run was started. `reports/` was refreshed by the two `evaluate.py` runs against the now-finished `yolov8s_neudet/best.pt` (the previous `evaluation.json` was stale — generated at epoch ~86 against `last.pt`, reporting mAP50 0.596). The repo is consistent: one contract module, one class list, one severity definition, one operating point flowing from evaluator to console.