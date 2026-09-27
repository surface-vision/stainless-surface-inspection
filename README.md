# Surface defect detection for stainless steel strip

## It is live: **<https://surface-vision.github.io>**

**No install, no upload, no account.** The real trained detector runs inside your
browser via ONNX Runtime Web. Drop in a steel frame and it localises and
classifies every defect, in 30-50 ms, on your own machine.

---

An operator watching strip pass at 45-900 m/min misses defects, notices them
hundreds of metres late, and leaves no record that traces back to a heat or a
roll change. This repository is an automated inspector for that job, built as a
line-side system rather than a notebook: a 6.22 MB YOLOv8 detector that reaches
**mAP50 0.7524** on a held-out test split it was never tuned on, a threshold
chosen by an explicit mill cost model instead of by F1, confidence numbers that
are **calibrated** so a "0.8" means something, a **plausibility gate** that
refuses to answer on frames that are not steel, a latency benchmark that turns
milliseconds into cameras-per-accelerator, and a deployment path verified
numerically against the PyTorch reference. Every defect comes back with its
metallurgical root cause and standing corrective action, because a red box on a
screen is not an action. Its own error rates are measured and stated, including
the one that is bad.

### The headline numbers

Every figure is traceable to a file in `reports/`; the map is one section down.

| | measured | where |
|---|---|---|
| **mAP50**, held-out test split, 180 frames never trained or tuned on | **0.7524** | `model_study.json` |
| mAP50-95 / precision / recall | 0.3967 / 0.696 / 0.687 | `model_study.json` |
| defect-type accuracy, dominant detection | **98.9%** (178/180) | `evaluation.json` |
| shipped checkpoint | **6.22 MB**, 3.01 M params | `model_study.json` |
| **calibration error** (ECE), before → after isotonic | **0.1418 → 0.0461**, mAP unchanged | `calibration.json` |
| **out-of-distribution gate** false rejections | **0** across 3,200 genuine steel frames | `ood_guard.json` |
| **browser parity** vs the PyTorch reference | **40/40 detections matched**, max box delta **0.14 px**, max confidence delta **0.0016** | `site/verify/parity_report.json` |
| browser inference | **30-50 ms** per frame in real Chrome (WASM, single thread) -- medians of 31.7 ms and 49.1 ms in two recorded runs of the same 12 frames, so it is quoted as a band; the same graph under onnxruntime-node is **7.7 ms** | `site/verify/parity_browser.json`, `deck/build/live_demo_check.json`, `site/verify/parity_report.json` |

The second checkpoint (`yolov8n_joint`, NEU-DET + Severstal) is the honest answer
to this system's worst weakness — the shipped model fires on **93.7%** of verified
defect-free strip. Jointly trained, that falls to **32.5%**, cross-domain ROC AUC
goes **0.608 → 0.958** and Severstal defect recall **72.0% → 86.6%**, while
in-domain mAP50 *improves* to 0.7642 (`gap1_cross_domain.json`,
`gap1_detection_metrics.json`). It ships, it is selectable, and it is deliberately
not the default — the four conditions for promoting it are written out in
`inference.resolve_weights()`.

### Two ways to run it

| | what it is | where |
|---|---|---|
| **Browser demo** | The real detector, client-side, zero setup. Tiled inference for wide strips, calibrated confidence, severity scoring, per-class operator guidance. No OOD gate -- that is console-only, and the page says so. | **<https://surface-vision.github.io>** — source in `site/` |
| **Operator console** | The full Streamlit station: batch inspection over a coil, per-coil ACCEPT/DOWNGRADE/HOLD reports, line simulation, EigenCAM explanations, checkpoint comparison. | `make demo` locally, or deploy free — <https://github.com/surface-vision/surface-vision-console> |

### Why the browser path is a result, not a fallback

Strip imagery is process data. Mills are reluctant to ship it to a cloud, and that
reluctance is usually where a vision pilot stalls. Running the detector
client-side removes the question: **no image leaves the plant**, there is no cloud
dependency, no per-inference cost, no cold start, and no inference API to secure.
The same 12 MB artefact that runs in a browser tab runs on a mill-floor industrial
PC, and it is verified to produce the *same detections* as the PyTorch model it
was exported from — 40/40, to 0.14 px.

**Read that as a demonstration of deployability, not as an architecture proposal.**
A production installation is GigE cameras, a PLC interface and a marking gun, not
a web page. What the browser build establishes is that the artefact is small
enough, fast enough and portable enough to be deployed at the edge without a
GPU — and that the exported model is numerically the model that was evaluated.

---

Everything below is reproducible from this repository with the commands given.
Numbers quoted are from the artefacts in `reports/`, not from the literature.

---

## Evidence: which file backs which claim

Nothing in this README is an estimate. Each claim below names the artefact that
produced it, and each artefact is regenerated by the command in its section.

| claim | artefact | key |
|---|---|---|
| mAP50 0.7524, mAP50-95 0.3967, P 0.696, R 0.687 | `reports/model_study.json` | `runs[5]` |
| 256 px beats 320/416/512/640 px at serve time | `reports/model_study.json` | `runs[].imgsz` |
| Retraining at 640 px does not recover it (0.7338, inside the ±0.0254 bootstrap band) | `reports/resolution_study.json` | `runs[52].mAP50`; `selection.*.bootstrap_half_width_from_model_study` |
| Defect-type accuracy 98.9% (178/180) | `reports/evaluation.json` | `image_level.accuracy` |
| Operating point conf 0.15, chosen on `val` by cost model | `reports/operating_point.json` | `operating_point` |
| Clean-crop false alarm ≤ 23.7% (an upper bound) | `reports/false_alarm.json` | `clean_patch_false_alarm` |
| ECE 0.1418 → 0.0461, mAP numerically unchanged | `reports/calibration.json` | `metrics.ece` |
| OOD gate: 0 false rejections / 3,200 real steel frames | `reports/ood_guard.json` | `ood_guard.real_steel.*.reject` |
| Joint model: clean-frame false alarms 93.7% → 32.5% | `reports/gap1_cross_domain.json` | `runs[0].sweep[2].clean_frame_fa` → `runs[2]...` |
| Joint model: cross-domain ROC AUC 0.608 → 0.958 | `reports/gap1_cross_domain.json` | `runs[*].separation_cross_domain.auc` |
| Joint model: Severstal defect recall 72.0% → 86.6% | `reports/gap1_cross_domain.json` | `runs[*].sweep[2].defect_crop_recall` |
| Joint model in-domain mAP50 0.7642 | `reports/gap1_detection_metrics.json` | `neu_det.256.mAP50` |
| GC10-DET overall mAP50 0.5821 | `reports/gc10_coverage.json` | `test_evaluation.overall.mAP50` |
| Roll marks are weak: AP50 0.203 on 24 instances | `reports/gc10_coverage.json` | `brief_relevant_families.roll_marks` |
| Edge cracks have no public dataset — geometric proxy only, AP50 0.880 | `reports/gc10_coverage.json` | `brief_relevant_families.edge_defects_proxy` |
| Latency, and cameras per accelerator | `reports/benchmark.json` | `mill.scenarios` |
| Browser build matches PyTorch 40/40, ≤0.14 px, ≤0.0016 conf; onnxruntime-node median 7.7 ms | `site/verify/parity_report.json` | `summary` |
| Browser build matches in a real Chrome, 40/40; 30-50 ms per inference across runs | `site/verify/parity_browser.json`, `deck/build/live_demo_check.json` | `summary.session_run_ms`, `browser_infer_ms` |

Two figures in that table are the ones to read first if you are checking whether
this project is honest with itself: the **23.7% clean-crop false alarm rate**,
which is reported as an upper bound and explained in Limitation 1, and the
**93.7% clean-frame false alarm rate** of the shipped checkpoint on real
defect-free strip, which is the largest gap in the system and is why the joint
checkpoint exists.

---

## The problem

On a hot rolling or finishing line, surface defects are found by an operator
watching strip pass at 45 to 900 m/min. That has three failure modes the mill
pays for:

* **Escapes.** A defect that is not seen ships as prime. Found downstream or by
  the customer, it becomes a prime-to-secondary downgrade on a ~20 t coil plus
  claim handling.
* **Late detection.** By the time a defect is noticed, hundreds of metres of
  strip carrying the same process fault have already been coiled. The value is
  in catching the *onset*, which fixes the process rather than sorting the
  product.
* **Inconsistency.** Two inspectors disagree, and neither produces a record that
  can be traced back to a heat, a cast sequence or a roll change.

An automated inspector addresses all three only if it is honest about its own
error rate. This system therefore reports the operating cost of every threshold
it could run at, and picks one by an explicit cost model instead of by F1.

Each detection is returned with the metallurgical root cause and the corrective
action for that defect family (`DEFECT_INFO` in `src/inference.py`), because a
red box on a screen is not an action.

---

## Dataset

**NEU-DET** (Northeastern University), hot-rolled steel strip surface defects:
200x200 greyscale-toned JPEG crops, six classes, one dominant defect per image.

The upstream mirror ships 1620 train / 180 test. Selecting a model on the test
set would leak, so `src/prepare_data.py` carves a stratified 180-image
validation set out of train and leaves the original test split untouched:

| split | images | per class | role |
|---|---|---|---|
| `train` | 1440 | 240 | fitting |
| `val`   | 180  | 30  | model selection and threshold tuning |
| `test`  | 180  | 30  | reported once, never tuned on |

Classes, in the index order fixed by `data/neu-det/data.yaml` and by the trained
head:

| id | class | base severity | typical root cause |
|---|---|---|---|
| 0 | `crazing`         | high     | low hot ductility; reheat/finishing temperature off schedule, tramp Cu/Sn |
| 1 | `inclusion`       | critical | slag or mould-flux carryover, refractory erosion |
| 2 | `patches`         | medium   | patchy descaling, burner temperature streaks |
| 3 | `pitted_surface`  | high     | over-oxidation in the reheat furnace, over-pickling |
| 4 | `rolled-in_scale` | high     | incomplete descaling, scale baked onto the work rolls |
| 5 | `scratches`       | medium   | seized table rollers, worn side guides, coil handling |

The split is verified by `tests/test_smoke.py`: counts, per-class stratification,
label well-formedness, and no image appearing in two splits either by name or by
MD5 of its bytes.

---

## Setup

Requires Python 3.11 and about 4 GB of disk.

```
git clone <this repository>
cd JSW-PS1
make setup
```

`make setup` creates `.venv` and installs the pins in `requirements.txt`. To do
it by hand:

```
python3.11 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

`torch` resolves to the Metal (MPS) build on Apple silicon and to the CPU build
elsewhere. On an NVIDIA line PC, install `torch`/`torchvision` from the CUDA
index first, then run the requirements file. Every script takes
`--device auto|mps|cuda|cpu` and defaults to `auto` (MPS, then CUDA, then CPU).

Rebuild the split from the raw download:

```
make data
# equivalently
.venv/bin/python src/prepare_data.py --raw data/NEU-DET-raw --out data/neu-det
```

Verify the installation:

```
make test
# equivalently
.venv/bin/python -m pytest tests/ -v
```

---

## Training

```
make train
# equivalently
.venv/bin/python src/train_detector.py \
    --model yolov8s.pt --name yolov8s_neudet \
    --epochs 150 --imgsz 320 --batch 32 --device auto
```

Writes to `models/<name>/`, then runs the held-out test split once and writes
`reports/train_summary_<name>.json`.

Two choices worth defending:

* **`--imgsz 320`, not 640.** Source images are natively 200x200 and the median
  labelled box is ~70 px on its short side at 320, so 640 upsamples without
  adding information. Measured on this machine: ~35 s/epoch at 320 against
  >11 min/epoch at 640.
* **Augmentation.** Horizontal and vertical flips, mild rotation, scale, shear
  and mosaic are label-preserving on strip texture. Hue jitter is disabled
  (`hsv_h=0.0`): mill imagery is effectively greyscale, so colour rotation
  invents a variation the camera cannot produce.

Overrides: `make train MODEL=yolov8n.pt NAME=yolov8n_neudet EPOCHS=100 IMGSZ=640`.

### Result on the held-out test split

Two checkpoints were trained: **yolov8n** (3.01 M parameters, 2.05 GFLOPs at
320 px, early-stopped at epoch 135 of 150) and **yolov8s** (11.14 M parameters,
150 epochs). `inference.resolve_weights()` serves the **nano** one — see
`reports/model_study.json` for why, and the caveat below.

Held-out `test`, ultralytics validator defaults, MPS:

| model | input | mAP50 | mAP50-95 | P | R |
|---|---|---|---|---|---|
| **yolov8n (served)** | **256 px** | **0.7524** | **0.3967** | 0.6960 | 0.6867 |
| yolov8n | 320 px (training size) | 0.7286 | 0.3926 | 0.7322 | 0.6018 |
| yolov8n | 416 px | 0.6230 | 0.2919 | 0.4920 | 0.6556 |
| yolov8n | 640 px | 0.3438 | 0.1495 | 0.4454 | 0.3892 |
| yolov8s | 256 px | 0.7343 | 0.4061 | 0.7087 | 0.6807 |
| yolov8s | 320 px | 0.6598 | 0.3618 | 0.6107 | 0.6475 |

Reproduce any row with
`.venv/bin/python -c "from ultralytics import YOLO; print(YOLO('models/yolov8n_neudet/weights/best.pt').val(data='data/neu-det/data.yaml', split='test', imgsz=256, device='mps').box.map50)"`.

**Input size is an accuracy control, not a quality-of-rendering control.** The
model was fine-tuned at 320 px and more than half its mAP50 is gone by 640 px.
256 px was chosen on `val` (`src/model_study.py`) and is what the console, the
export and the coil report now default to.

Per-class AP at the 320 px training size, where the two models can be compared
against the published NEU-DET literature:

| class | instances | yolov8n AP50 | yolov8s AP50 |
|---|---|---|---|
| `crazing`         | 79 | 0.395 | 0.330 |
| `inclusion`       | 89 | 0.842 | 0.805 |
| `patches`         | 99 | 0.945 | 0.921 |
| `pitted_surface`  | 46 | 0.748 | 0.704 |
| `rolled-in_scale` | 69 | 0.647 | 0.438 |
| `scratches`       | 64 | 0.795 | 0.762 |

**How much of the nano-over-small gap is real?** At 320 px the paired bootstrap
(2000 resamples, `reports/model_study.json`) puts the mAP50 difference at +0.069,
95 % CI [+0.031, +0.099] — excludes zero. Give each model its own best input size
and the same difference is +0.018, 95 % CI [−0.008, +0.042] — *contains* zero, and
small is marginally ahead on mAP50-95. So the defensible claim is not "nano is the
better detector"; it is "**nano is not worse, and it is 3.7x smaller and 3.5x
cheaper in FLOPs**". That is what the deployment choice rests on.

`reports/evaluation.md` and `reports/evaluation.json` are the **served** nano
checkpoint at its **served** 256 px, regenerated 2026-09-09. Per-class AP50 there
reads `crazing` 0.444, `inclusion` 0.827, `patches` 0.949, `pitted_surface` 0.756,
`rolled-in_scale` 0.630, `scratches` 0.909 — the 320 px table above is kept only
because it is the size at which the two checkpoints are comparable with the
published literature.

The spread is the honest headline. `patches`, `inclusion` and `scratches` have
crisp boundaries and are localised well. `crazing` and `rolled-in_scale` are
diffuse textures whose extent is a judgement call even for the annotator, and
the detector's boxes are correspondingly loose — which costs AP50 far more than
it costs the operator, who needs to know *that* the frame is crazed, not the
exact polygon. That gap between the box-level score and the frame-level decision
— for the served nano checkpoint at 256 px, **0.7524 mAP50 against 98.9 %
image-level accuracy** — is the most important number in this report and is why
the evaluation reports both.

---

## Evaluation

```
make eval
# equivalently
.venv/bin/python src/evaluate.py --split test --device auto
```

Answers three separate questions and keeps them separate:

1. **How good is the detector?** COCO-style mAP from the ultralytics validator
   on its own protocol defaults, so the headline numbers are comparable with
   published NEU-DET results.
2. **How good is it as a classifier of coils?** Each image scored by its
   dominant detection, giving a confusion matrix, accuracy and macro-F1.
3. **What does it cost to run?** A 19-point confidence sweep on the **validation**
   split, priced by a mill cost model. This produces a threshold, but **not the
   shipped one** — see the next section for why, and for the one that ships.

Split discipline is enforced in code: `--split` and `--tuning-split` must differ,
or the run refuses to start.

Artefacts written to `reports/`: `evaluation.md` (the document a person signs
off), `evaluation.json`, `operating_point.evaluate.json`, `confusion_matrix.png`,
`pr_curves.png`, `f1_vs_threshold.png`, `false_alarms_vs_recall.png` and
`error_gallery/` (worst misses and most confident false positives per class).

`make eval` evaluates the checkpoint at its **served** size (256 px), not at the
320 px it was trained at, because an evaluation report exists to describe the
detector that is actually deployed.

---

## The operating point

```
make false-alarm
# equivalently
.venv/bin/python src/false_alarm.py
```

The shipped threshold is **conf = 0.15**, chosen on `val` by `src/false_alarm.py`
and recorded in `reports/operating_point.json`, which the console reads for its
default confidence slider — so the screen and the signed-off report cannot drift
apart.

**Why `src/evaluate.py` does not decide this.** Every NEU-DET image contains a
defect. The only "false alarm" `evaluate.py` can see is therefore *a spurious
extra box on a coil that was already going to be flagged* — a nuisance-box rate,
not the event a mill buys on. `src/false_alarm.py` builds the missing population
instead: 771 crops mined from `test` with **zero** intersection with any
ground-truth box (each grown 6 px before the test, so a crop that merely grazes a
label is discarded), at four scales, thinned to IoU 0.25 against each other and
capped per image so no single frame can dominate. Each crop is run at the
pixels-per-millimetre the network was trained on rather than at the full
deployment `imgsz`, which would measure a resampling artefact instead of the
detector.

At conf = 0.15, measured on that population:

| | value |
|---|---|
| clean-crop false alarm rate | **23.7 %**, 95 % CI [17.7 %, 30.1 %] |
| boxes per clean crop | 0.27 |
| image-level defect detection rate (`val`) | **90.6 %** |
| detection rate on defect-containing control crops | 98.0 % flagged, 90.9 % localised |
| expected mill cost (patch basis) | 0.2822 — the lowest of every threshold clearing the 90 % floor |

The interval is **image-clustered**, not binomial: the 771 crops come from 106
source images with a measured design effect of 4.3, so the effective sample size
is 179, and the Wilson interval that assumes independence is about half as wide
and wrong. The report tests the independence assumption directly and **rejects**
it — the false alarm rate is flat across a 4x span of crop area (constant-rate
model chi2 0.35 on 3 dof; Poisson-in-area 34.6, and 19.3 after the design-effect
correction, p = 2.4e-4). A false alarm on clean steel is a property of the local
surface texture, not a per-unit-area event, so the per-crop rate is the frame-level
estimate and the 87.1 % area extrapolation is a pessimistic bound rather than a
prediction.

**23.7 % is itself an upper bound.** 78 % of the boxes raised on clean crops carry
the source image's *own* defect class, against 17 % by chance — the signature of
the detector firing on unlabelled continuation of a real defect rather than
inventing one on sound metal. Separating the two needs re-annotation.

**0.15 is chosen by the detection floor, not by the economics, and the report says
so.** On cost alone the curve keeps falling past it: conf 0.35 scores 0.2178 on the
same patch basis against 0.2822 at 0.15, and cuts the clean-crop false alarm rate
from 23.7 % to 9.6 %. It is rejected because its image-level detection rate is
0.789 — below the 0.90 floor the line will accept — which means 2.2x as many coils
shipping with an undetected defect. 0.35 is also the F1-optimal threshold, and F1
weights a miss and a false alarm equally, which is not how a mill is paid. Both are
reported; neither is followed. Move the floor and the recommendation moves with it,
which is the honest way round.

Two files, two questions, and they must not be confused:

| file | written by | measures | threshold |
|---|---|---|---|
| `reports/operating_point.json` | `src/false_alarm.py` | false alarms on **clean steel** | **conf 0.15 — shipped** |
| `reports/operating_point.evaluate.json` | `src/evaluate.py` | spurious extra boxes on **already-defective** frames | conf 0.05 — superseded |

Until 2026-09-09 `evaluate.py` wrote `operating_point.json` unconditionally, so
running the documented `make eval` silently reset the console's default confidence
from 0.15 back to 0.05. It now writes its own filename and will not touch the
console's file without `--update-operating-point --force-operating-point`;
`tests/test_reporting.py` pins that behaviour.

---

## Confidence calibration

```
make calibrate
# equivalently
.venv/bin/python src/calibrate.py
```

The brief asks for a verdict "ideally with a confidence score". The raw detector
score is not a probability: on `test` at conf >= 0.05 it is **monotonically
under-confident in every bin**, with ECE **0.1418**. A detection displayed as 0.60
is right 84 % of the time, 0.70 → 93 %, 0.80 → 98 %.

`src/calibrate.py` fits `score -> P(true positive)` **on `val` only** and persists
it to `reports/calibration.json`; the console imports `calibrate_confidence(raw)`.
Two families are fitted and chosen between by 5-fold cross-validation on `val`
**grouped by source image**, scoring out-of-fold ECE — so the more flexible family
gets no in-sample advantage:

| family | parameters | out-of-fold ECE on `val` |
|---|---|---|
| **isotonic (shipped)** | non-parametric, monotone | **0.0250** |
| Platt (reference only) | 2 | 0.0614 |
| temperature | 1 | 0.0961 |
| raw score | — | 0.1008 |

Temperature scaling barely moves it, and that is a structural result rather than a
tuning failure: one temperature is a *sharpening* knob, and the measured bias is
under-confidence at **both** ends, which is an additive shift. On held-out `test`:

| | before | after |
|---|---|---|
| ECE | 0.1418 | **0.0461** |
| ECE at the shipped conf 0.15 (n = 481) | 0.1654 | **0.0559** |
| Brier score | 0.1754 | 0.1576 |
| max per-bin gap | 0.2417 | 0.1159 |

**mAP is unchanged, and that is measured rather than argued.** Every calibrator
here is strictly increasing, so the greedy matcher and the PR sweep see the
identical ranking; isotonic's plateaus would otherwise create ties, so a vanishing
`1e-6 * s` term restores a strict order. Recomputing AP50 on `test` from raw and
from calibrated scores gives **bit-identical values in all six classes**. Plateau
values are Jeffreys-smoothed `(k + 0.5) / (n + 1)`, so a top plateau where all 99
`val` detections happened to be correct calibrates to 0.995 rather than printing a
flat `1.00` that 99 samples cannot support.

Full write-up and reliability diagram: `reports/calibration.md`,
`reports/calibration.png`.

---

## Speed and line capacity

```
make bench
# equivalently
.venv/bin/python src/benchmark.py
```

Times the shipping code path (`DefectDetector.predict` / `predict_batch`), not a
bare forward pass, so letterboxing, host/device transfer, NMS and record
building are all included. Checkpoints are snapshotted and SHA-256'd before they
are measured, because a training job may be rewriting `last.pt` while it runs.

Measured on an Apple M5 (MPS), **yolov8n** (the served checkpoint), median of
three passes, no training running. Read every MPS figure with the +/-25 %
between-invocation band in limitation 6 &mdash; these move by tens of percent
between runs of the same script on the same idle machine:

| input | single-frame p50 | peak throughput | plan at 70 % utilisation | test mAP50 |
|---|---|---|---|---|
| 256 px | 10.9 ms | 329.3 FPS (batch 4) | 230.5 FPS | 0.752 |
| **320 px** | **5.2 ms** | **256.6 FPS (batch 16)** | **179.6 FPS** | 0.729 |
| 416 px | 6.5 ms | 193.6 FPS (batch 16) | 135.5 FPS | 0.623 |
| 640 px | 9.2 ms | 106.9 FPS (batch 8) | 74.8 FPS | 0.344 |

Batch-1 latency is not monotone in input size on this backend &mdash; 256 px is
reproducibly *slower* single-frame than 320 px, on independent invocations. The
throughput column, not the latency column, is what the capacity plan uses.

On **CPU**, which repeats far better than MPS (within ~10 % run to run), the same
checkpoint costs 10.3 ms/frame at 256 px and 13.7 ms at 320 px at batch 1, and
7.4 / 9.9 ms at batch 8. Never submit 16 frames at once on CPU: throughput
collapses 4-5x at that batch boundary (`protocol.cpu_batch_note`). This is not
this codebase &mdash; it reproduces on a bare two-layer `Conv2d`/`SiLU` stack,
where 320 px costs 1.10 ms/frame at batch 8 and 8.45 ms/frame at batch 16.

The second half converts that into an installation, from the assumptions in
`MILL_CONFIG` at the top of `src/benchmark.py`: 1.28 m strip, 0.20 mm/px optical
resolution, 2048 px area-scan sensors. That gives a 0.41 m field of view, four
cameras across the width, and one frame every 0.348 m of travel.

Quoted at **320 px**, the cheapest input still within 95 % of the best val mAP50
&mdash; the reference size is chosen on `val`, never on test:

| line | speed | FPS/camera | tiled @320 | downscaled @320 |
|---|---|---|---|---|
| Bright annealing | 45 m/min | 2.1 | 552 inf/s -> 4 GPU | 8.6 inf/s -> 83 cam/GPU |
| Hot-band anneal and pickle | 90 m/min | 4.3 | 1,103 inf/s -> 7 GPU | 17.2 inf/s -> 41 cam/GPU |
| Skin-pass / inspection | 250 m/min | 12.0 | 3,064 inf/s -> 18 GPU | 47.9 inf/s -> 15 cam/GPU |
| 20-hi reversing cold mill | 420 m/min | 20.1 | 5,147 inf/s -> 29 GPU | 80.4 inf/s -> 8 cam/GPU |
| Continuous pickling exit | 500 m/min | 23.9 | 6,128 inf/s -> 35 GPU | 95.7 inf/s -> 7 cam/GPU |
| Hot strip mill finishing exit | 900 m/min | 43.1 | 11,029 inf/s -> 62 GPU | 172.3 inf/s -> 4 cam/GPU |
| Tandem cold mill exit, thin gauge | 1500 m/min | 71.8 | 18,382 inf/s -> 103 GPU | 287.2 inf/s -> 2 cam/GPU |

An earlier version of this table quoted the tiled column at 640 px, which needs
only 16 tiles per frame instead of 64 and so reports roughly a quarter of the
accelerator count &mdash; at an input size where this checkpoint scores 0.344
mAP50. Tiling at the size the model can actually see at is four times more
expensive, and that is the real number.

The two columns are the real architecture decision. "Tiled" cuts each 2048 px
camera frame into windows the size of the network input, so one source pixel is
one network pixel and the stated 0.20 mm/px survives — 64 tiles per frame at
320 px input. "Downscaled" squashes the frame into one inference and degrades
the effective resolution to 1.28 mm/px at 320 px input, which will not resolve a
1 mm defect. **One M5 is not a deployment target for full-resolution inspection
above the slowest line.** Sizing a real installation means a proper accelerator,
a smaller backbone, or an accepted resolution trade; the arithmetic here is what
makes that choice explicit rather than assumed.

Within a 200 ms photon-to-verdict budget, 3.0 m of strip passes the camera at
900 m/min, which is how far downstream a marking gun or diverter has to sit.

---

## The operator console

```
make demo
# equivalently
.venv/bin/streamlit run demo/app.py
```

Serves on <http://localhost:8501>. Four work areas:

* **Single frame** — PASS/DEFECT verdict, source next to the annotated overlay, a
  per-detection table, and the root cause and corrective action for every defect
  family present.
* **Batch inspection** — many frames through `predict_batch`, a sortable frame
  table, defect-family distribution, and CSV exports for the coil file.
* **Line simulation** — N frames scored one at a time, showing true single-shot
  latency, a running defect rate and the strip speed the station can sustain.
* **Defect atlas** — the standing knowledge base, useful before anything is
  uploaded.

The sidebar picks any checkpoint under `models/`, retunes confidence, NMS IoU,
network input size and tiled inference in place. It opens on the configuration
the studies selected: **`yolov8n_neudet/best.pt`, 256 px input, conf 0.15** —
the checkpoint from `inference.resolve_weights()`, the input size from
`src/model_study.py` (chosen on `val`) and the threshold from
`reports/operating_point.json` (`src/false_alarm.py`, also chosen on `val`).
The console defaulted to 640 px until 2026-09-09, which cost it more than half
its mAP50; `tests/test_smoke.py` now pins the default to the study's choice so it
cannot drift back. One coupling is asserted rather than measured: conf 0.15 was
tuned at the 320 px training magnification, not at 256 px. Uploads are decoded through a path
that handles greyscale, palette, RGBA, CMYK and 16-bit TIFF sources, honours EXIF
orientation, downscales past 60 MP, and rejects aspect ratios that would
letterbox the short edge to zero pixels.

Everything above the `# UI` divider in `demo/app.py` is free of rendering calls
and is exercised headlessly by the test suite.

**Deploying it.** A trimmed, self-contained copy of this console -- both
checkpoints, 18 held-out frames, one real 2048x1000 line-scan capture, and the
`src/` modules it needs vendored in -- lives at
<https://github.com/surface-vision/surface-vision-console> and deploys free on
Streamlit Community Cloud in about two minutes of clicking; that repository's
`DEPLOY.md` is the click-by-click. It is not a Hugging Face Space because Hugging
Face now requires a PRO subscription for Docker and Gradio Spaces on free CPU
hardware. The move cost 15x the RAM (a free CPU Space gave 16 GB; a Streamlit
Community Cloud container gives roughly 1 GB), so the app was measured against the
new ceiling rather than assumed to fit: `tools/measure_memory.py` in that
repository runs it under a real Streamlit runtime and reports **peak 585-618 MB
against ~1024 MB**, after two reductions that are documented where they are made.

---

## Other entry points

```
# Deployment export: ONNX (opset 17) and CoreML, verified against PyTorch
make export
.venv/bin/python src/export_model.py --imgsz 256

# Per-coil inspection report -> reports/coil_report.html and .json
make report
.venv/bin/python src/report.py --images data/neu-det/test/images \
    --coil-id JS-HR-2609-0417 --shuffle-seed 20260909 --imgsz 256

# Explain one frame (EigenCAM, or causal occlusion sensitivity)
.venv/bin/python src/explain.py data/neu-det/test/images/inclusion_271.jpg \
    --method eigencam --out /tmp/cam.png

# Fit the confidence calibrator (val only) -> reports/calibration.{json,png,md}
make calibrate
```

`--shuffle-seed` is not decoration. A directory listing of NEU-DET is alphabetical,
which means sorted by defect class, so an unshuffled coil position map shows six
clean blocks that read as sustained process upsets that are not there. The seed
makes the shuffle reproducible.

The export is verified, not just produced. At the shipping input size of 256 px
(`reports/export_summary.json`): 233-node graph, `onnx.checker` **pass**, and the
head tensor from ONNX Runtime matches PyTorch on the same preprocessed frame at
cosine similarity 1 - 1.2e-13, max absolute deviation 7.1e-4, decoding to
**identical detections** (2 vs 2, worst box delta 2.3e-05 px, worst confidence
delta 5.4e-07). Every one of those numbers is bit-for-bit reproducible across
runs. The ONNX artefact is 12.13 MB and the CoreML one 12.16 MB.

The one number here that is *not* reproducible is latency: the forward pass costs
3.1-3.3 ms in ONNX Runtime against 5.7-5.9 ms in PyTorch, each at its own best
thread count, so the backend ratio lands between 1.76x and 1.82x. **That is a
forward pass on one pre-letterboxed tensor, not a frame rate** &mdash; no decode,
no letterbox, no NMS, no record building &mdash; and it must never be quoted as
one. The end-to-end CPU figure for the same checkpoint is the benchmark's:
10.3 ms/frame at 256 px batch 1, 7.4 ms/frame at batch 8.

Nothing under `models/` is written: the checkpoint is staged into `export/` first.

`src/explain.py` uses EigenCAM rather than a gradient CAM, and says why in its
module docstring: a served ultralytics detector has `requires_grad` cleared on
every parameter, so Grad-CAM cannot run without mutating the process-wide cached
model the UI and the report generator share. EigenCAM needs only a forward pass
and therefore explains exactly the network in production — at the cost of being
class-agnostic. A slower, causal occlusion-sensitivity fallback ships alongside
it, and any heatmap that comes back NaN, flat or structureless raises rather
than being presented as an explanation.

The CLI produces the CAM at the size the model is *served* at (256 px by default),
because a CAM explains one specific forward pass. Cost, measured on this host at
256 px: the first CAM in a fresh process is **1,236 ms** in-process and 1,998 ms
through the CLI — MPS kernel compilation, not the algorithm — and every CAM after
that is **~11 ms**, against 5.9 ms for the detection it explains. A console should
therefore put it behind an on-demand button with a spinner: the steady-state cost
would be affordable inline, but the first click is the one an operator forms an
opinion on.

---

## Project layout

```
data/
  NEU-DET-raw/            upstream download (1620 train / 180 test)
  neu-det/                the 1440/180/180 split, data.yaml, manifest.json
src/
  prepare_data.py         build the stratified split, with a leakage guard
  prepare_severstal.py    Severstal masks -> YOLO boxes; the joint NEU-DET+Severstal set
  prepare_gc10.py         GC10-DET ingest: roll-mark and edge-defect coverage
  train_detector.py       fine-tune YOLOv8 and score the held-out test split
  inference.py            THE CONTRACT: DefectDetector, InferenceResult,
                          class names, colours, defect knowledge base,
                          severity scoring, tiled inference, annotation
  evaluate.py             mAP, image-level confusion matrix, confidence sweep,
                          cost model, error gallery
  false_alarm.py          THE OPERATING POINT: clean-steel false alarm proxy with
                          image-clustered intervals, a tested spatial-independence
                          model, a positive control, and the cost re-derivation
                          that selects the shipped conf 0.15
  model_study.py          THE SELECTION STUDY: nano-vs-small paired bootstrap,
                          input-size sweep, TTA pricing, per-class failure
                          taxonomy regressed against measured image statistics
  calibrate.py            score -> P(true positive), fitted on val, rank-preserving
                          so mAP is untouched; exports calibrate_confidence()
  cross_domain.py         the shipped detector on steel it has never seen
  domain_shift.py         photometric normalisation for off-rig frames
  benchmark.py            latency/throughput sweep + mill capacity plan
  export_model.py         ONNX/CoreML export with numerical verification
  explain.py              EigenCAM and occlusion sensitivity
  report.py               per-coil roll-up, disposition rules, HTML + JSON
demo/
  app.py                  Streamlit operator console
tests/
  test_smoke.py           end-to-end suite
  test_calibration.py     calibrator invariants and the mAP-invariance proof
  test_reporting.py       CLI entry points and the operating-point ownership guard
models/                   training runs; <run>/weights/{best,last}.pt
reports/                  generated evaluation, benchmark and coil artefacts
export/                   staged checkpoint, .onnx, .mlpackage
scripts/run_training.sh   the training driver used for the runs in models/
Makefile                  setup / data / train / eval / false-alarm / study /
                          calibrate / cross-domain / bench / demo / test / export
requirements.txt          pinned dependencies
```

`src/inference.py` is the single integration point. Class names and their index
order, the display colours, the defect knowledge base, the severity bands and
their cutoffs are all defined there once, and every other module — the console,
the evaluator, the reporter, the exporter, even the data preparation script —
imports them. There is no second copy of the class list anywhere in the tree.

The public contract is `DefectDetector.predict/predict_batch/predict_tiled ->
InferenceResult`, a JSON-serialisable record with a verdict, per-detection boxes
in original-image pixel coordinates, a 0-100 severity score, and per-stage
latency. `tests/test_smoke.py` asserts every invariant that contract promises.

---

## Testing

```
make test
.venv/bin/python -m pytest tests/ -v
```

312 tests at the time of writing (2026-09-09), about 27 s. Anything that needs the
GPU shares one detector at 320 px, so the suite stays polite to a training job on
the same accelerator.

`tests/test_smoke.py` (111) covers: split integrity and leakage; the inference
contract (types, ranges, internal consistency, JSON round trip, boxes clamped
inside the frame); batch scoring agreeing with single-frame scoring; tiled
inference (fall-through, coordinate lifting, global NMS across seams, overlap
clamping); annotation shape, dtype and non-mutation of the caller's array;
severity monotonicity in confidence, area, defect count and seriousness; graceful
handling of greyscale, RGBA, float, 1x1 and multi-megapixel frames; the console's
decode, banding and operating-point path; the coil disposition rules and the
evaluator's matching; and one test that executes the whole Streamlit script
through Streamlit's own harness and asserts it renders without raising.

`tests/test_calibration.py` pins the properties a calibrator has to hold for the
rest of the system to stay true — strict monotonicity on every family (a tie makes
mAP tie-break dependent), the JSON round trip, the fit/score split discipline, and
loud degradation to the identity map when `calibration.json` is missing or corrupt.

`tests/test_reporting.py` exists because two real defects shipped under a green
suite. Both were in code no test could reach: the `explain.py` CLI crashed on its
own documented defaults from inside a `# pragma: no cover` block, and `evaluate.py`
overwrote the console's operating point from a code path nothing exercised. Every
CLI is now importable and every argument path runs, the README's own explain
command is lifted out of the file and parsed rather than retyped, and the
ownership of `operating_point.json` is asserted in both directions.

---

## Limitations

Stated plainly, because a surface inspection system that oversells itself gets
switched off in week two.

1. **The clean-steel false alarm rate is a proxy, and it is an upper bound.**
   Every NEU-DET image contains a defect, so the population `src/false_alarm.py`
   uses is *crops with no annotated defect mined from defective frames*, not
   genuinely clean strip. 78 % of the boxes it counts carry the source image's own
   defect class, so an unknown share of them are unlabelled continuation of the
   real defect rather than hallucination on sound metal — which is why 23.7 % is
   reported as a bound. The pooled figure is also a mixture whose weights are set
   by NEU-DET's annotation habits: across source classes with at least 20 crops it
   runs from 5.8 % (`patches`) to 48.8 % (`crazing`), a spread of 8x with
   non-overlapping intervals. A mill inspecting a different product mix would see
   a different number from the same detector. Real clean-strip frames remain the
   thing to measure.

2. **The cost model's constants are placeholders, and the threshold moves with
   them.** conf = 0.15 is the correct answer *to the cost model as parameterised*.
   The 12:1 miss-to-false-alarm ratio and the 5 % assumed defect prevalence are
   defensible orders of magnitude, not Jindal Stainless numbers, and they enter the
   argmin only through the single product `prevalence x ratio / (1 - prevalence)`
   = 0.63 — so arguing about the two separately is arguing about one number. The
   sensitivity table in `reports/false_alarm.md` shows where the threshold moves
   across that weight; the quality department's real downgrade and re-inspection
   costs would replace it.

3. **200x200 crops are not strip captures.** The model has never seen a real
   line image: no motion blur, no water or emulsion on the strip, no lighting
   gradient across a 1.28 m width, no roll marks, no weld seams, no edge of
   strip. `predict_tiled` exists for full-width frames and is tested for
   geometric correctness, but its detection quality on mill imagery is unmeasured.

4. **Two classes are weak.** On the served nano checkpoint at 320 px, `crazing`
   (AP50 0.395) and `rolled-in_scale` (AP50 0.647) are the two worst classes, and
   on yolov8s they are worse still (0.330 and 0.438). Both are diffuse textures
   with ambiguous extent. Frame-level accuracy stays high because the dominant
   detection is usually the right class, but the boxes should not be trusted for
   sizing or for area-based severity on those two.

5. **Severity scoring is a heuristic, not a standard.** The 0-100 score is a
   class base tier scaled by confidence and box area, with a damped noisy-OR
   aggregation to frame level. It is monotone and bounded and its rule lives in
   one function, but it has not been calibrated against any Jindal Stainless
   acceptance criterion. The same is true of the ACCEPT/DOWNGRADE/HOLD thresholds
   in `report.DispositionRules`, which are deliberately isolated for the quality
   department to re-tune.

6. **Speed figures are from a laptop, and they do not repeat.** Everything in the
   benchmark was measured on an Apple M5. The kernel load average at the start and
   end of each sweep is recorded in `protocol.host_contention` rather than
   asserted in prose. The larger problem is reproducibility: MPS throughput on this
   host moves by roughly 25 % *between invocations of the same script on an idle
   machine*, which repeating passes inside one process does not fix, because those
   passes share the process's thermal and GPU-clock state. Two independent
   three-pass runs put the 320 px nano peak at 233.9 and 282.6 FPS. Every
   accelerator count therefore carries that band, and `mill.scenarios` reports the
   slowest-pass count beside the median one. The capacity arithmetic is sound; the
   throughput it is fed must be re-measured on the accelerator that will actually
   be installed.

7. **Trained at 320 px, and it does not survive being served at 640.** The input
   size sweep is measured, not assumed: test mAP50 runs 0.752 / 0.729 / 0.623 /
   0.484 / 0.344 at 256 / 320 / 416 / 512 / 640 px. 256 px was chosen on `val` and
   is now the default for the console, the export and the coil report. Anything
   quoting this model at 640 px — an earlier console default, an earlier coil
   report, the 640 px export — is quoting a detector running at less than half its
   accuracy. Whether 256 px remains the right size on real strip captures, which
   are not 200x200 crops, is unmeasured.

8. **Single frame, single camera, no memory.** There is no tracking of a defect
   across consecutive frames, no fusion across the four cameras spanning the
   width, and no registration against a real coil-length encoder. The coil
   position map in the HTML report is frame index, not metres.

9. **EigenCAM is class-agnostic.** It shows where the layer's features are
   strongest, not which class they voted for. Read it as "the model's attention",
   not as evidence for a specific box. Where a CAM and the occlusion map
   disagree, the occlusion map is the causal measurement and wins.

10. **No production plumbing.** No GigE camera ingest, no PLC or MES interface,
    no marking-gun trigger, no model versioning or drift monitoring, no
    authentication on the console. This is an inspection engine and its evidence,
    not a commissioned installation.

11. **The browser build proves portability, not that production should be a web
    page.** <https://surface-vision.github.io> runs the exported ONNX graph
    client-side and is verified to match the PyTorch reference detection-for-
    detection (40/40, max box delta 0.14 px, max confidence delta 0.0016, on 12
    images -- `site/verify/parity_report.json`). That establishes three things and
    only three: the export is numerically faithful, the artefact is 12 MB, and it
    infers in 30-50 ms per frame on a WASM backend without a GPU -- a band, not
    a figure, because two recorded runs of the same 12 frames returned medians
    1.6x apart (7.7 ms under onnxruntime-node on the same machine). It does not
    establish
    throughput under a real camera feed, and 12 images is a parity check, not an
    accuracy evaluation -- the accuracy numbers come from the 180-frame held-out
    split, not from the browser. The privacy argument (no image leaves the plant)
    is a genuine property of client-side inference; the mill-floor deployment it
    implies is an industrial PC running the same artefact, not a browser tab.
