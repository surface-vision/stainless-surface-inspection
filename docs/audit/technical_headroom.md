Audit complete. All project files verified untouched; the Streamlit server on 8501 is still up.

---

# AUDIT: JSW-PS1 surface defect detection

**Verdict up front.** The engineering discipline here is genuinely top-decile — the reports are more honest than most published papers, and the headline number reproduces to the digit. The weakness is not rigour, it is **evidence coverage**: the two things the deck declares unmeasurable (clean-steel false alarms, domain transfer) are both measurable in one afternoon with a free public dataset, and when I measured them the answers were bad. That is the gap that would move a judge's score.

## 0. What reproduces (verified by execution, not by reading)

| Claim | Result |
|---|---|
| yolov8n@256 test mAP50 0.7524 | **0.7524475543783158** — exact, P 0.696, R 0.687, mAP50-95 0.3967 |
| 1440/180/180 split, zero defect-free images | Confirmed: 0 empty label files in any split; ultralytics reports `180 images, 0 backgrounds` |
| 111 tests | 111 passed in 7.54 s |
| ONNX/CoreML exports at 256 | Present and verified (max abs dev 7.1e-4, identical detections) |
| The 640 px inference collapse | Reproduced independently on a second architecture (YOLO11n: 0.776 val @256 → 0.337 @640) |
| Demo running, tiled inference, aspect guard | HTTP 200; `predict_tiled` wired; `MAX_ASPECT_RATIO`=512 so a 1600×256 strip upload works |

The brief's "~0.8 ms inference" does not reproduce (validator reports 1.4 ms at 256); `reports/model_study.md` already forbids quoting absolute ms, so treat this as a brief-side typo, not a project claim.

---

## 1. THE FINDING THAT MATTERS — the model fails closed on real, genuinely defect-free steel

**Impact/hour: highest by a wide margin.** Hits *detection accuracy and false-alarm rate*, *path to scaling across grades and products*, and *usable demo*, simultaneously.

I verified Severstal is downloadable and then used it as the clean-steel population the project says it does not have.

**Severstal (`Voxel51/severstal_steel_defects`), verified by inspecting `samples.json` (25 MB, no auth):**
- 18,074 images, **every one 1600×256×3** — real strip geometry
- train 12,568 → **6,666 defective / 5,902 verified defect-free**; test 5,506 (labels not public)
- 4 classes, image counts 897 / 247 / 5,150 / 801; 19,958 pixel masks
- Masks decode from base64+zlib to `(256,1600)` uint8 numpy with **class ids 1–4 in the pixel values** → boxes are a `scipy.ndimage.label` away (mean 3.17 components/image)
- Full repo 1.68 GB; the 5,902 clean frames alone 0.61 GB. Measured 16-thread throughput **1.04 MB/s → ~10 min for the clean set, ~27 min for everything.** I pulled 800 frames in 61 s.

**Result — shipped `yolov8n_neudet/best.pt`, imgsz 256, iou 0.45, on 400 genuinely defect-free frames (3,200 200×200 crops):**

| conf | clean **crop** FA | clean **FRAME** FA | boxes/crop | real-defect crop recall (class-agnostic) |
|---|---|---|---|---|
| 0.15 *(recommended op point)* | 54.7 % | **95.8 %** | 1.96 | 0.724 |
| 0.25 *(`DefectDetector` default)* | 44.3 % | **89.5 %** | 1.19 | 0.583 |
| 0.40 | 28.3 % | 79.8 % | 0.57 | 0.415 |

**Discrimination is near chance.** ROC AUC of max-box-confidence, real defect crops vs real clean crops = **0.628**. Robust to pixel scale (0.628 / 0.631 / 0.607 / 0.540 at whole-frame zoom 1.0 / 0.78 / 0.5 / 0.25), so this is not a magnification artefact. The in-domain equivalent, integrated from the project's own `reports/false_alarm.json` ROC points, is **0.957**.

**End-to-end, through the shipped disposition chain.** `src/report.py` on a 60-frame coil of genuinely defect-free strip at conf 0.15:

```
coil SEVERSTAL-CLEAN-PROBE: 60 frames, defect rate 51.7%, 143 detections, p95 severity 85.1 -> HOLD
  - Defect rate 51.7% exceeds the hold limit of 25.0%.
  - 95th-percentile severity 85.1 reaches the hold limit of 70.0.
  - 8 frame(s) in the critical severity band (limit 0).
  - 18 frame(s) contain 'inclusion', a zero-tolerance defect (limit 0)
```

All four hard triggers fire. A mill running this holds every prime coil.

**Two sharp sub-findings the project could not have seen:**

1. **`reports/false_alarm.md` identifies the wrong confusion to attack.** It concludes *"The single largest failure mode is scratches at 42 % of all false positives … the fix is hard negatives from clean strip."* On genuinely clean strip the false alarms are **inclusion 39–42 % and patches 34–36 %; scratches is only 10–12 %**. And `inclusion` is the one class `DispositionRules.hold_classes` treats as zero-tolerance — the dominant hallucination is precisely the class that forces an unconditional HOLD.
2. **A quarter of the gap is photometric, three quarters is textural.** NEU-DET grey is mean 130.2 σ 41.1; Severstal is mean 86.5. Histogram-matching the clean crops to NEU-DET cuts crop FA from 44.1 % → 33.6 % at conf 0.25 (−24 % relative) for one line of preprocessing. The remaining 33.6 % needs data, not normalisation.

**Fairness control:** flat grey and Gaussian-noise inputs produce **zero** detections. This is not "fires on anything" — it fires on unfamiliar *steel texture*.

**I then ran the fix, and the result is more interesting than "add negatives".** Warm-started `best.pt` on NEU-DET + 1,440 real defect-free Severstal crops as YOLO background images, 40 epochs @320, **19.7 min measured**:

| | NEU-DET test mAP50 | mAP50-95 | held-out clean **FRAME** FA @0.15 | real-defect crop recall @0.15 | AUC |
|---|---|---|---|---|---|
| baseline | 0.7524 | 0.3967 | 94.0 % | 0.732 | 0.633 |
| + Severstal negatives | **0.7508** | **0.4040** | **5.8 %** | **0.057** | 0.583 |

Read this carefully. False alarms collapse **16×** at **zero in-domain cost** (mAP50 −0.0016, mAP50-95 **+0.0073**, crazing AP50 0.444→0.496). But defect recall on that domain collapses too, and AUC does not improve. **The model learned a domain classifier, not a defect classifier** — the predictable consequence of importing negatives from a domain with no positives. The correct experiment is joint training on Severstal **positives and negatives** (6,666 labelled frames with pixel masks are sitting there unused).

**Cost to close properly:** ~27 min download + ~1 h mask→box conversion + ~5.7 h for a 150-epoch joint run (or 1.5 h warm-start) + eval. **Call it 6–10 h of wall clock including write-up.** It converts slide 4's biggest stated limitation from "not measured, would need a plant capture campaign" into a measured cross-dataset generalisation result with a fix attached. Nothing else on this list comes close on impact per hour.

---

## 2. Accuracy headroom is small, and the reason 640 px was skipped is a bad measurement

**Architecture — MEASURED.** YOLO11n, identical recipe/seed/split, 320 px, 150 epochs, **45.3 min**:

| | test mAP50 | test mAP50-95 | fitness (0.1·mAP50+0.9·mAP50-95) | crazing | rolled-in_scale |
|---|---|---|---|---|---|
| yolov8n @256 (shipped) | 0.7524 | 0.3967 | 0.4323 | 0.444 | 0.630 |
| **YOLO11n @256** | **0.7598** | **0.4069** | **0.4422** | **0.542** | 0.560 |

+0.0074 mAP50, +0.0102 mAP50-95 — well inside the ±0.025 bootstrap half-width the project already measured, so **not a real gain**, though it does win on ultralytics' own composite and on the hard class (crazing +0.098). Training curves are statistically indistinguishable at every matched epoch (best-so-far val mAP50 at epoch 10/30/50/55: v8n .549/.654/.683/.709, y11n .552/.653/.693/.701). **Verdict: architecture is not where the accuracy is. Do not re-do the deck for this.**

**Resolution — the brief's stated reason for training at 320 is wrong.** I launched the real run: `yolov8n` @640, batch 16 → **56.1 s/epoch, 4.25 GB GPU, = 2.34 h for 150 epochs.** The brief says "640 px measured over 11 min/epoch" (≈27 h). That is off by 12×. `reports/model_study.md` already suspected this and projected 1.9 min/epoch; the true figure is even cheaper. **A 2.3-hour overnight experiment was skipped on a bad number.** (I stopped the run at epoch 7 rather than leave a 2.3 h job on your GPU; relaunch with `.venv/bin/python src/train_detector.py --model yolov8n.pt --imgsz 640 --batch 16 --epochs 150`.) Expected payoff is modest — published YOLOv8n@640 on NEU-DET brackets our result at 74.0 and 78.6 — so this is a **credibility** fix ("we tested 640 and here is the number") more than an accuracy fix.

**A grid-boundary artefact nobody checked.** The imgsz sweep in `src/model_study.py` starts at 256. I extended it downward:

| imgsz | 128 | 160 | 192 | 224 | **256** |
|---|---|---|---|---|---|
| val mAP50 | 0.581 | 0.687 | 0.737 | **0.7576** | 0.7466 |
| test mAP50 | 0.532 | 0.659 | 0.722 | 0.727 | **0.7524** |

**On val — the split the project's own protocol uses to choose imgsz — the peak is 224, not 256.** Applying their stated rule to a complete grid selects 224 and reports test 0.7270, i.e. −0.025 off the headline. The headline is not wrong (256 is defensible and test agrees), but "why does your grid start at 256?" is a question a sharp judge will ask, and the current answer in `reports/model_study.md` line 112 is **factually incorrect**: *"the source images are 200x200, so 320 px is already upsampling and 256 px is the only size in the grid that does not."* 256/200 = 1.28×, so 256 upsamples too. The only non-upsampling sizes are ≤192 — and the measurement shows upsampling to 256 *does* pay, contradicting the stated mechanism ("nothing above the training size can pay for its own compute"). **~1 h to re-run the sweep and fix the paragraph.**

**Ensemble — oracle ceiling MEASURED.** Union of yolov8n and yolov8s at their best sizes, conf 0.15, IoU 0.5 + class match, 446 test GT boxes:

| | recall |
|---|---|
| yolov8n alone | 0.711 |
| yolov8s alone | 0.756 |
| **oracle union** | **0.814** |
| found by neither | 83 boxes (18.6 %) |

Crazing 0.354 → **0.595** under union. So there is ~10 pp of real recall headroom concentrated on the hard class — but a working fusion recovers maybe half of it, costs precision, and **doubles inference cost, which directly contradicts the throughput slide**. ~3–4 h. Rank it low.

**TTA:** already measured negative (−0.006 mAP50 for 2.39× latency). **Longer training:** both runs converged (nano early-stopped at 135, best at 85). Neither is headroom.

**Net honest ceiling: 0.752 → ~0.79–0.80 (top of the published credible band), and it comes almost entirely from crazing and rolled-in scale — which the project's own failure analysis correctly says are optics and annotation problems, not model problems.**

---

## 3. The confidence score on screen is wrong — and the brief explicitly asks for one

The brief: *"ideally with a confidence score."* `render_verdict` prints **Peak confidence** to the operator. Nobody checked whether it is a probability. It is not. Measured on 790 test predictions at conf ≥ 0.05, **ECE = 0.142**, systematically and monotonically **under-confident in every bin**:

| conf bin | n | mean conf | empirical precision | gap |
|---|---|---|---|---|
| [0.15,0.25) | 116 | 0.195 | 0.336 | −0.141 |
| [0.55,0.65) | 69 | 0.599 | **0.841** | −0.242 |
| [0.65,0.75) | 55 | 0.697 | **0.927** | −0.231 |
| [0.75,0.85) | 50 | 0.801 | **0.980** | −0.179 |
| [0.85,1.00) | 41 | 0.885 | **1.000** | −0.115 |

An operator reading "0.60" is looking at a detection that is right 84 % of the time. Because the miscalibration is monotone, a one-parameter temperature/isotonic fit on val fixes it, mAP is unchanged (rank-preserving), and it yields a reliability diagram — a genuinely differentiating slide that costs **~2 h**. Very high impact/hour: it turns a brief clause the project currently satisfies nominally into one it satisfies properly.

---

## 4. Demo gaps (confirmed, with one refinement and one bug)

**Gap 1 confirmed — stills only.** Four tabs: Single frame / Batch / Line simulation / Defect atlas. Zero occurrences of video, `st.camera_input`, `cv2.VideoCapture`, webrtc, or mp4. "Line simulation" is 40 random *still test-split images* fed to `predict` one at a time. Against *"detects … in real time"* and *"a usable demo interface"*, the real-time claim is argued, never shown. **~3–5 h** for a video upload + frame loop + a simple IoU tracker so a defect persists across frames (tracking is also absent everywhere — no ByteTrack, no tracker; `README.md` line 551 already admits it).

**Gap 2 — refined, and it partly refutes the brief's framing.** `demo/app.py` does not import `src/explain.py`, correct. But `src/report.py` line 568 **does**, and `reports/coil_report.html` carries **18 EigenCAM figures** as JPEG data-URIs. So explainability ships — in a CLI-generated HTML file the judge never opens. Deck slide 4 differentiator #5 is *"Explainable, and routed. EigenCAM heat maps (src/explain.py) show what the detector fired on"*; if a judge says "show me", the demo cannot. Wiring it in is **~1.5 h** (EigenCAM measured at **2,505 ms/frame**, so it needs an on-demand button + spinner, not inline).

**Bug, unreported, uncovered by the 111 tests.** `src/explain.py` line 827 crashes with documented default arguments:
```python
det = load_detector(imgsz=args.imgsz)          # line 827 — args.imgsz is None
if args.imgsz is None:                          # line 828 — fallback runs too late
    args.imgsz = DEFAULT_IMGSZ
```
`TypeError: int() argument must be a string … not 'NoneType'`. It is inside `if __name__ == "__main__": # pragma: no cover`, so no test touches it. `--imgsz 256` works. **2-minute fix**, but a judge who clones and runs the documented command gets a traceback from the file the deck cites as a differentiator.

**Gap 3 — no physical units anywhere.** Zero occurrences of `mm` in `src/inference.py` or `src/report.py`. `reports/benchmark.json` models 0.2 mm/px and 1.28 m strip width, but a detection is never converted to defect length in mm or position across the width — which is exactly what an MES disposition and caster feedback need. **~1 h.**

---

## 5. Missing defect types — the brief names two the project has never heard of

`grep -rni "edge crack|roll mark"` across `src/`, `demo/`, `deck/deck_content.md`, `docs/`: **zero hits** (one incidental mention at `README.md:510`). The business context names scratches, scale, roll marks, edge cracks; NEU-DET covers the first two. The deck has no mitigation narrative for this at all — it is the most obvious "you didn't read the brief" attack surface in the submission.

**Concrete answer, verified.** `imaadd05/gc10-det` on HuggingFace, CC BY 4.0: **2,065 train images at 2048×1000** (real line-scan strip geometry), 3,206 COCO boxes, 139 MB, 10 classes:

| GC10 class | boxes | maps to brief's |
|---|---|---|
| `8_yahen` (rolled pit) | 64 | **roll marks** |
| `9_zhehen` (crease) | 72 | roll marks / handling |
| `3_yueyawan` (crescent gap) | 251 | **edge defect** |
| `10_yaozhed` (waist folding) | 143 | edge defect |
| `2_hanfeng` (weld line) | 472 | coil-join |
| + punching, water/oil/silk spot, inclusion | 2,204 | — |

No public dataset has a true *edge crack* class; crescent gap and waist folding are the honest proxies, and 64 rolled-pit boxes is thin. **~4–6 h** to ingest, train a 16-class model and report. Even if accuracy on the rare classes is poor, *having measured it* is worth far more on a slide than silence — and 2048×1000 doubles as a second cross-dataset and a genuine tiling test.

---

## 6. Edge deployment — measured, and the naive answer does not work

No INT8, no FP16, no TensorRT/OpenVINO export exists; `src/export_model.py` mentions them only in a comment justifying opset 17. Two measurements:

- **FP16 is free and unused.** `half=True` at 256: mAP50 **0.7529** vs 0.7524, mAP50-95 0.3955 vs 0.3967. Zero accuracy cost, halves the artefact. **~30 min** to add.
- **Naive INT8 does not pay.** `quantize_dynamic` on the shipped ONNX: 12.13 MB → **3.22 MB (3.77×)**, but CPU median **4.82 ms → 8.75 ms (1.8× *slower*)** at 4 threads, and max abs output deviation 23.8 against a peak of 268 (~9 %). The credible path is **static INT8 with a calibration set on the actual target box (Jetson / x86)**, not a one-liner — and nothing here has ever run on a real edge device. **~2–3 h**, and it is honest to say so on the slide rather than claim edge-readiness.

---

## 7. Everything a strong team would have that this does not (all absent, verified by grep)

Anomaly detection / PatchCore / autoencoder · segmentation head · self-supervised pretraining · domain adaptation · active learning · conformal or MC-dropout uncertainty · multi-frame tracking. **All appear only as prose in deck slides 2 and 5, correctly labelled "proposed".** The honesty is a credit, but a competitor who ships one working anomaly channel gets the "innovation and differentiation" marks. Cheapest credible one: **PatchCore or a simple feature-memory anomaly score fitted on the 5,902 Severstal defect-free frames** — it needs no labels, directly attacks the unclassified-defect bucket the deck already argues for, and reuses the download from Finding 1. **~1 day.**

Segmentation deserves a specific mention: `reports/model_study.md` itself asks *"whether crazing should be scored as a box at all rather than as a whole-frame label or a segmentation mask."* Severstal ships 6,666 pixel masks. The dataset needed to answer the project's own open question is the same download as Finding 1.

---

## 8. Checked and cleared — do not spend time here

- **No centre prior.** NEU-DET crops are defect-centred, so tiled deployment could have suffered. Recall by centre-offset quartile: 0.723 / 0.721 / 0.667 / 0.732; size-controlled, the peripheral half is *better* (0.712 vs 0.676). Hypothesis refuted.
- **Wide-strip uploads work.** `predict_tiled` + `MAX_ASPECT_RATIO`=512; a 1600×256 frame tiles correctly.
- **`reports/evaluation.md` is stale** (yolov8s @320, mAP50 0.660, and `operating_point.evaluate.json` says conf 0.05 vs the shipped 0.15) — but `README.md:189` already flags it. Presentation risk only; ~15 min to regenerate.
- **Multi-type frames:** n=12 in test, mean type-recall 0.819 vs 1.000 on single-type, exact set match 0.667. Suggestive of a real weakness against *"ability to handle multiple defect types"*, but n=12 cannot support a claim. Not a finding; a reason to want a second dataset.

---

## RANKING — impact on the brief ÷ hours on this hardware

| # | Action | Hours | Brief clause hit | Why |
|---|---|---|---|---|
| **1** | **Severstal joint train (positives + negatives) + cross-dataset report** | **6–10** | false-alarm rate · scaling across grades · accuracy | Converts the deck's biggest admitted unknown into a measured result. 95.8 % clean-frame FA → 5.8 % with negatives alone at zero in-domain cost, in 20 min of training. Nothing else is close. |
| **2** | Confidence calibration + reliability diagram | 2 | *"with a confidence score"* · false-alarm rate | ECE 0.142, every bin under-confident by up to 24 pp. Cheap, visible, differentiating. |
| 3 | Video input + frame-to-frame tracking in the demo | 3–5 | *"real time"* · usable demo | The only clause currently argued rather than shown. |
| 4 | EigenCAM into the demo + fix the `explain.py` CLI crash | 1.5 | usable demo · innovation | Slide 4 claims it; the clicked product does not have it; the CLI is broken on defaults. |
| 5 | GC10-DET → roll marks + edge defects | 4–6 | multiple defect types | Two named defects with literally zero coverage today. |
| 6 | `yolov8n` @640, 150 epochs | **2.34 (measured)** | detection accuracy | Answers the one question a judge will ask; the stated reason for skipping it is off by 12×. |
| 7 | Re-run imgsz sweep from 128; fix the model_study 256-upsampling error | 1 | detection accuracy | val peak is at 224, not 256; a factual error sits in the flagship report. |
| 8 | FP16 export | 0.5 | inference speed | Measured free. |
| 9 | mm-based defect sizing and cross-width position | 1 | usable demo · scaling | Needed for MES/caster feedback; currently pixels only. |
| 10 | Static INT8 on a real edge device | 2–3 | inference speed | Naive dynamic INT8 measured 1.8× *slower*; say so honestly. |
| 11 | Anomaly channel on Severstal negatives | ~8 | innovation | Reuses the Finding-1 download. |
| 12 | YOLO11n swap | 0.75 (done) | — | **Measured +0.007 mAP50. Don't bother.** |
| 13 | 2-model ensemble | 3–4 | accuracy | +10 pp oracle recall ceiling but doubles inference — contradicts the throughput argument. |

**If only one thing gets done: #1.** It is the difference between a submission that says "we could not measure domain shift" and one that says "we measured it, here is the AUC, here is the fix, here is the residual". On the brief's own words — *"a clear path to scaling across grades and products"* — the current deck asserts *"Transfers unchanged: … the backbone features."* The measurement says the backbone features do **not** transfer: AUC 0.628 against 0.957 in domain. That single sentence on slide 5 is the one claim in this otherwise scrupulous submission that the evidence contradicts.

---

**Audit artefacts** (all under `/private/tmp/claude-501/-Users-prathmeshwalimbe-Downloads-JSW-PS1/f88a3d3d-c8be-40c8-8600-132f500095d3/scratchpad/`, nothing written into the project — verified by `find -newermt`):
`xdomain.json`, `xdomain2.json`, `xdomain3.json` (cross-domain sweeps) · `neg_eval.json` (baseline vs negatives) · `audit_runs/yolo11n_320.json` · `audit_runs/v8n_neg.json` · `audit_runs/yolov8n_640/results.csv` (7 epochs, 56.1 s/epoch) · `ens.json` (ensemble oracle) · `calib.json` (calibration) · `stepcost.json` (320 vs 640 step cost) · `clean_coil_report.html` (the HOLD-on-clean-steel coil report) · `severstal/` (1,400 real strip frames + masks) · `neudet_plus_neg/` (the negatives dataset).