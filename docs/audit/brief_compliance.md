## LENS: LITERAL COMPLIANCE, CLAUSE BY CLAUSE

Everything below was reproduced by execution on `/Users/prathmeshwalimbe/Downloads/JSW-PS1` with `.venv/bin/python`. The running demo on 8501 was left alone; I drove a separate instance through `streamlit.testing.v1.AppTest`.

---

## PART A — THE ACTION ITEM AS ACCEPTANCE CRITERIA

### "detects **and classifies**" — SATISFIED

Reproduced. `ultralytics` val on the held-out test split at 256 px returns exactly the claimed headline: **mAP50 0.7524, mAP50-95 0.3967, P 0.696, R 0.687**. Per-class AP50: patches 0.949, scratches 0.909, inclusion 0.827, pitted_surface 0.756, rolled-in_scale 0.630, crazing 0.444.

Classification is demonstrated distinctly from detection and it holds up: I scored all 180 test images at the shipped conf 0.15 and the top-confidence detection carries the correct class on **177/180 (98.3%)**; the only confusion is pitted_surface → patches (3 images). `reports/model_study.json` `failure_taxonomy` shows `class_confusion: 0` for every class at both thresholds — I confirmed that is a real property, not a reporting artefact.

**The caveat nobody states:** 1677 of 1800 NEU-DET label files contain exactly one class. Classification on this dataset is close to whole-image texture classification, so "zero class confusion" is partly a property of the benchmark, not evidence the model separates a scratch from scale *in the same frame*. Nothing in the repo tests a mixed-defect frame.

### "on **steel-strip images**" — PARTIALLY SATISFIED, and this is finding #1

`predict_tiled` exists, is correct, and is wired to the demo sidebar (`demo/app.py:544,568,596`). I ran it on a genuine 1600×256 strip (Severstal aspect) built from 8 test crops:

- plain `predict()`: 2 detections, one of them a wrong-class `scratches 0.39` over a pitted region
- `predict_tiled()`: 15 detections, correct families in the correct places, severity 91.7, 86 ms

So the mechanism is real and materially better than squashing. Three problems:

1. **A judge cannot reach it.** I flipped the "Tiled inference (wide strip)" toggle via AppTest: the metrics, detections table and boxes are **byte-identical** to the untiled run. Every sample frame is 200×200, the tile defaults to 256, so `predict_tiled` falls through to `predict()`. I scanned the entire repo — `assets/`, `data/`, `demo/` — for any image with either edge ≥600 px: **zero**. `assets/` is empty. Unless the judge brings their own strip file, the only code path that satisfies the brief's literal wording is a toggle that does nothing.

2. **Tiled accuracy is measured nowhere.** Every mAP, precision, recall, false-alarm and operating-point number in `reports/` is single-frame 200×200. I measured the cost directly — one test image repeated 5× horizontally, so ground truth is 5× the single-frame count:

   | frame | expected on 5× strip | tiled returned | ratio |
   |---|---|---|---|
   | inclusion_272 | 15 | 24 | **1.60** |
   | rolled-in_scale_275 | 20 | 27 | 1.35 |
   | scratches_271 | 30 | 34 | 1.13 |
   | patches_277 | 35 | 25 | **0.71** |

   Tiling moves detection count between −29% and +60%. Cause: seam-truncated fragments of one continuous texture defect have IoU below 0.45, so global NMS keeps both. In my 1600×256 run, `pitted_surface [200,1,256,255]` and `pitted_surface [205,1,401,255]` both survive — both clipped at the 205-px tile stride.

3. **The test that is supposed to catch this is vacuous.** `tests/test_smoke.py:557` (`test_global_nms_deduplicates_defects_across_tile_seams`) asserts only that no two surviving same-class boxes overlap above `detector.iou`. That is what NMS guarantees by construction. It cannot fail, and it does not test seam merging.

*What it would take:* ship one 1600×256 strip mosaic in `assets/` and default the sample picker to it (an hour); add a seam-aware box union or connected-component merge for texture classes and a test that counts instances against known ground truth (a day).

### "in **real time**" — PARTIALLY SATISFIED, and the two artefacts contradict each other

What is actually demonstrated: the Line Simulation tab feeds 200×200 crops one at a time. I ran it — mean 5.4 ms, p50/p95 4.3/10.9 ms, sustained 91.9 fps. That is honest single-shot latency on the crops it has.

The problem is the headline KPI it prints: **"Max line speed 1102 m/min"**. That number is `fps_p95 × footprint_mm / 1000 × 60` with a default 200 mm footprint. It implicitly assumes 1 mm/px optical resolution and ignores strip width entirely — a 200 mm frame covers 1/6 of the 1.28 m strip the project's own geometry assumes.

`reports/benchmark.json` `mill.scenarios`, which I read directly, says the opposite and says it honestly:

| line | m/min | accelerators, full optical res (0.2 mm/px) |
|---|---|---|
| Skin-pass / inspection | 250 | **21** (18 at 320 px) |
| Hot strip mill exit | 900 | **75** |
| Tandem cold mill exit | 1500 | **125** |

Deck slide 2 carries the honest version ("at 250 m/min... roughly 18 accelerators"). The demo carries 1102 m/min with no resolution or width qualifier. **Same submission, two speed stories, ~4× apart, and the flattering one is the one on screen.** I confirmed the honest one by measurement: tiled inference on a real full-resolution frame runs 2048×2048 → 287 ms (3.5 fps), 2048×512 → 105 ms (9.5 fps), 1600×256 → 62 ms (16 fps).

A judge who asks "so how fast, really?" gets two answers from you. That is worse than either answer alone.

*What it would take:* one line — either divide by `1280/200` cameras and state the mm/px, or relabel the metric "single-camera field-of-view advance rate" and put the accelerator count beside it. Ten minutes.

**Also not demonstrated: motion.** `UPLOAD_TYPES` is `jpg/jpeg/png/bmp/tif/tiff`. No video, no camera. "Real time" is argued by a plot of 40 sequential still frames. The confirmed gap #1 stands.

### "working demo that flags and labels defects on an uploaded image, ideally with a **confidence score and defect type**" — SATISFIED

Driven and verified. Default render produces `Verdict DEFECT / 3 detection(s)`, five KPI tiles, source-vs-overlay images, and a detections table with columns `# | Defect | Confidence | Severity | Score | Area px² | Area % | x1 y1 x2 y2 | W | H`, plus a per-class root-cause and corrective-action card. Both required fields are present, per detection, with the class name spelled out. Upload path exists alongside the sample picker. The console loads its default conf 0.15 from `reports/operating_point.json` so screen and report cannot drift. This clause is comfortably met.

---

## PART B — THE FIVE KEY CONSIDERATIONS

### 1. Detection accuracy and false-alarm rate — **SATISFIED**, and it is the project's strongest work

**Is what you report actually a false-alarm rate?** Yes, and this is the one place the project outperforms the brief. `reports/false_alarm.md` explicitly refuses the easy number. It states that NEU-DET's "false alarm rate" is *a spurious box on a coil that was already defective* (59.4% at conf 0.15) and builds a separate clean-steel proxy: 771 crops with zero intersection with any GT box, grown 6 px before the test, four scales, image-clustered bootstrap CI, a positive control (98.0% flag / 90.9% localise on defect-containing crops of the same size), a rejected spatial-independence model (χ² 34.56 vs 0.35 on 3 dof), a per-source-class mixture breakdown (5.8% to 48.8%), and the observation that 78% of the boxes carry the source frame's own class against 17% by chance — so 23.7% is declared an **upper bound**, not a rate.

It also names the three numbers it refuses to put on a slide. I could not find a soft claim in it.

One definitional note the deck should tighten: slide 3 says "92.2% of defective test frames are **flagged** (166 of 180)". I measured the actual flag rate at conf 0.15 — it is **180/180, 100%**. The 92.2% is `defect_detection_rate` from `src/evaluate.py:587`, which counts images with ≥1 box matching GT at IoU 0.5 *with the correct class*. That is the better number; "flagged" is the wrong word for it and undersells you.

### 2. Ability to handle multiple defect types — **PARTIALLY SATISFIED**

Six classes, all detected, all coloured, all with a root-cause/action card. But against the brief's own named list:

| brief names | covered? | evidence |
|---|---|---|
| **scratches** | yes | AP50 0.909 |
| **scale** | yes | `rolled-in_scale` AP50 0.630 |
| **roll marks** | **no** | no class; grep across the whole tree returns one hit, `README.md:510`, in a limitations list |
| **edge cracks** | **no** | no class, and structurally impossible — NEU-DET is a 200×200 *interior* crop; the strip edge appears in zero training frames |

**2 of 4.** The edge-crack miss is the sharper one: it is not a labelling gap you close by adding a class, it is a field-of-view gap. Nothing in the dataset, the tiling geometry (`mill.geometry` covers 1.507 m across a 1.28 m strip, i.e. the edge is in frame) or the demo has ever seen a strip edge. The deck never says this. If a Jindal panellist asks "what about edge cracks" — a defect they named first in the brief — there is currently no answer on any slide.

Also worth being honest about internally: crazing instance recall at the shipped threshold is **0.354** and rolled-in_scale **0.507** (`failure_taxonomy.tuned_conf`). "Handles six types" is true at frame level and much weaker at instance level. The deck discloses crazing; it does not disclose that the two weakest classes are recall-limited by more than half.

*What it would take:* add a "Not covered" row to slide 4's limitations naming roll marks and edge cracks explicitly, with the field-of-view reason for edge cracks. Twenty minutes, and it converts an ambush into a demonstration of rigour.

### 3. Inference speed suitable for a production line — **PARTIALLY SATISFIED**

Model side is fine and reproducible: 6.22 MB, 3.01 M params, 1.2 ms inference on M5 MPS, ONNX and CoreML exports verified (max abs deviation 7.1e-4, cosine 0.999999999999, `allclose atol 1e-3` true). `benchmark.json` carries a ±25% between-invocation repeatability warning it measured itself, and refuses to quote an absolute frame rate. That is the right posture.

Docked for the demo-vs-benchmark contradiction in Part A, plus one measured risk nobody has looked at: **motion blur**. I applied a 15-px horizontal blur to `inclusion_272.jpg` — what a real line at speed with an insufficient shutter produces. Detections fell 3 → 1 and top confidence 0.82 → **0.37**. At line speed the detector loses two thirds of its detections and lands near the operating threshold. There is no blur augmentation in the training recipe and no blur robustness figure anywhere in `reports/`. For a criterion literally named "suitable for a production line", that is the physical failure mode most likely to bite.

### 4. A usable demo interface — **SATISFIED** (with one live-demo hazard, see finding #3 below)

Four tabs (Single frame, Batch inspection, Line simulation, Defect atlas), operating point loaded from disk, in-place retuning under a process-wide lock so slider nudges don't reload an 85 MB checkpoint, graceful decode of greyscale/CMYK/16-bit, no-checkpoint screen, error handling that never blanks the page. It runs (HTTP 200 on 8501). 111 tests pass in 4.07 s.

Two omissions against the brief:

- **Explainability is invisible in the product a judge clicks.** Confirmed suspected gap #2 exactly. `src/explain.py` is imported by `src/report.py:568` only — never by `demo/app.py` (grep for `explain|cam|heat` in `demo/app.py` returns nothing). The EigenCAM overlays exist and are good (`reports/explainability/*.png`, 6 files; 18 `<img>` tags in `reports/coil_report.html`). But `coil_report.html` is a static file with no link from the console, and the console has no download or "open coil report" button. Deck slide 4 lists "Explainable, and routed" as differentiator #5 — a judge who clicks through the demo will not see it.
- **The deck contains no screenshot of the demo.** I extracted all five slides and all six figures: `fig1` is annotated test crops, `fig2` an architecture box diagram, `fig3`–`fig6` charts. Slide 2 mentions "a Streamlit inspection console" in one clause of a "Ships today" line. If the deck is ever read without the live demo — which is how decks are usually read — there is no evidence the interface exists.

*What it would take:* two `st.image` calls in the single-frame tab wrapping `generate_cam` (the function is already there and already fails-soft in `report.py`), plus one screenshot pasted onto slide 2. Under two hours for both.

### 5. A clear path to scaling across grades and products — **PARTIALLY SATISFIED; concrete plan, zero transfer evidence**

The plan is genuinely concrete, not hand-waving: slide 5 separates "transfers unchanged" (tiling and serving architecture, severity/MES plumbing, training recipe, backbone features) from "needs re-labelling" (each grade, finish, illumination), and adds active learning, an anomaly channel for unseen defects, a cited self-supervised result (SimSiam 0.768 vs ImageNet 0.773 vs random 0.280), and a monitoring triple. The 0.6 h per-line recalibration figure is measured. Slide 4 says outright "Nothing here has been measured on stainless, on your cameras, or under your illumination."

**On the specific question — stainless vs carbon steel — the evidence that anything transfers is exactly zero, and there is no experiment in the repo that could have produced any.** Confirmed suspected gap #4, and it is worse than stated: **Severstal is not mentioned anywhere in the tree.** I grepped every `.md`, `.py`, `.json` and `.txt` outside `.venv` — no hits. It is not in `docs/research_notes.md` (whose sections 1a–1e catalogue NEU-DET baselines in detail), not in the limitations, not in the roadmap. So the deck cannot even say "a second public dataset exists and here is why we did not use it".

This matters against the brief's own words — "Teams **may use publicly available steel surface-defect datasets**", plural. A cross-dataset generalisation number is the single cheapest available proof that the "scales across grades" claim is more than a slide. Severstal would additionally supply the two things this project's own reports say it most lacks: real 1600×256 strip geometry, and ~6,000 genuine defect-free frames.

*What it would take:* zero-shot the current checkpoint on Severstal's 4 classes and publish the number however bad it is, then fine-tune on a few hundred and publish the delta. One day. Even a bad zero-shot number is a stronger slide than none, because it sizes the domain gap the whole rollout plan is designed to close.

---

## PART C — RANKED FINDINGS

**1. The system has never seen sound steel, and a judge can trigger it in one click.** Confirmed suspected gap #3, with reproducible cases far more damaging than the 23.7% ceiling. At the shipped conf 0.15 / imgsz 256:

| input | result |
|---|---|
| pure white 200×200 | `pitted_surface` **conf 0.74**, severity 86 |
| pure black | `pitted_surface` 0.34, severity 66 |
| smooth luminance gradient (60→200) | `pitted_surface` 0.30, severity 64 |
| uniform grey 200 | `pitted_surface` 0.29 |
| **real frame + specular water streak** | **`scratches` conf 0.87** — higher than the real inclusion in the same frame |

The gradient case is a lighting gradient across the strip. The streak case is emulsion or water. These are the two most common non-defects on a real line, and both fire at high confidence — the water streak *outranks the genuine defect*. `false_alarm.md` already identified scratches as 42% of all false positives and named the fix ("hard negatives from clean strip — exactly the data this report says the project does not yet have"); this is that prediction confirmed on synthetic mill artefacts. Practical risk: any judge who drags in a photo, a logo or a white PNG gets a confident critical-severity call. *Fix:* Severstal's defect-free half as negatives, or at minimum a guard in the demo that warns when input statistics fall outside the training distribution. Half a day for the guard.

**2. The one clause about strip geometry is served by an unreachable, unmeasured code path.** Section A above. Highest ratio of judge-visible impact to effort in the whole audit: ship one wide sample image.

**3. The demo's headline speed number contradicts the deck's own capacity arithmetic by ~4×.** Section A above. This is the finding most likely to be caught live, and it damages credibility disproportionately because the rest of the work is so careful about not overstating.

**4. `README.md` and `reports/evaluation.md` still carry the superseded operating point, and `make eval` will silently regress the shipped demo.** This one is self-inflicted:
- `README.md:226-249` ("The operating point") reports **conf = 0.05**, 81.1% false alarm rate, 3.18 false boxes/image, and states in bold **"This operating point is not deployable as it stands"**. Limitations #1 and #2 (lines 487-505) repeat the superseded legacy-false-alarm framing.
- `reports/evaluation.md` — the file named "evaluation report", "the document a person signs off" per the README — headlines **mAP50 0.6598** for `yolov8s` at 320 px. The deck headlines 0.7524 for `yolov8n` at 256.
- The README never mentions `reports/false_alarm.md`, `src/false_alarm.py` or the clean-steel study at all. `src/false_alarm.py` and `src/model_study.py` are **both absent from the README's own project-layout listing** — the two files carrying the project's strongest evidence.
- `src/evaluate.py:1550` writes `reports/operating_point.json` unconditionally; `src/false_alarm.py` overwrites it only under `--update-operating-point`. There is no `make` target for `false_alarm.py` or `model_study.py`. So running the documented `make eval` resets the demo's default confidence from 0.15 back to 0.05.

A judge's first click is `README.md`. It currently tells them the false alarm rate is 81.1% and the threshold is not deployable — both superseded, both wrong about the shipped system, and both worse than the truth. *Fix:* rewrite that one README section, add a supersession banner to `evaluation.md`, add `make false-alarm` / `make study`. Two hours, and it is the highest-leverage two hours available.

**5. Roll marks and edge cracks — 2 of the brief's 4 named defects are absent, and one is structurally unreachable.** Section B2. Cheap to disclose, expensive to be ambushed on.

**6. Explainability and the coil report are invisible from the demo; the deck has no demo screenshot.** Section B4.

**7. Minor, verified:** `tests/test_smoke.py:557` seam-dedup test cannot fail; README says "101 tests, about 7 s" (actual 111, 4.07 s); deck slide 3's "flagged" should read "found and correctly classified" (actual flag rate 100%, the honest number is 92.2%).

---

## PART D — DECK VS SUBMISSION GUIDELINES

Extracted all five slides from `/Users/prathmeshwalimbe/Downloads/JSW-PS1/deck/JSW_Surface_Defect_Detection.pptx`. All seven required topics are **genuinely present**, none merely nominal:

| topic | slide | verdict |
|---|---|---|
| Problem Understanding | 1 | Genuine — sourced quote, stainless-specific nickel economics, three named failure modes |
| Objectives | 1 | Genuine — six-row target-vs-measured table with explicit targets, not aspirations |
| Proposed Solution | 2 | Genuine — five-stage architecture + a numbered methodology in decision order |
| Validation & Feasibility | 3 | Genuine — held-out numbers, operating point, and a "what we cannot claim" block |
| Innovation & Differentiation | 4 | Genuine — six numbered points, each with a number attached |
| Assumptions & Limitations | 4 | Exceptionally strong — ten blocks including "Not built" and "Money" |
| Expected Impact | 5 | Genuine — technical / operational / sustainability + a swept ROI matrix with the ask named |

Two structural notes:

- **Slide 4 carries two of the seven topics and is at the density limit.** 3,823 characters at 8.7 pt on a 13.33×7.5 in slide — 77% more text than the next densest slide, in the smallest body font in the deck. Both topics are well argued; neither will be read in a five-minute review. If anything gets cut to make room for a demo screenshot, it should come from here.
- **The deck's discipline is its actual differentiator and it knows it** (slide 4 title: "What is different here is the discipline, not the architecture"). Which is precisely why findings #3 and #4 cost more than they look: an internally contradicted speed figure and a README that contradicts the deck's own headline are the two things that undercut a discipline claim fastest.

---

## SUMMARY SCORECARD

| clause | verdict |
|---|---|
| detects and classifies | **satisfied** |
| steel-strip images | **partially** — path exists, unreachable, unmeasured |
| in real time | **partially** — stills only; demo and deck disagree ~4× |
| demo flags + labels uploaded image | **satisfied** |
| confidence score and defect type | **satisfied** |
| detection accuracy and false-alarm rate | **satisfied** — exceeds the brief |
| multiple defect types | **partially** — 6 classes, 2 of 4 named defects |
| inference speed for a production line | **partially** |
| usable demo interface | **satisfied**, explainability not surfaced |
| path to scaling across grades/products | **partially** — concrete plan, zero transfer evidence |
| deck: all seven topics | **satisfied** |

The engineering and the statistical honesty are well above the bar. What is missing is almost entirely **presentation of what already exists** (one wide sample image, one speed-metric caption, one README section, one CAM call, one screenshot) plus **one genuinely absent experiment** (Severstal — for defect-free negatives, real strip geometry, and the only available cross-dataset evidence for the scaling claim). The first group is under a day in total and moves four clauses from partial to satisfied. The second is the one that would move the "scaling across grades" clause and close the false-alarm root cause.