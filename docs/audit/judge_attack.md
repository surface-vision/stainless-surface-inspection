## AUDIT: hostile-judge findings, JSW-PS1

**Verified first.** `mAP50 0.7524 / mAP50-95 0.3967 / P 0.696 / R 0.687` reproduces exactly from `models/yolov8n_neudet/weights/best.pt` at imgsz 256 on `split=test` (I ran it). Per-class AP50 matches to 4 dp. `111 passed` reproduces. `deck/build/verify_deck.py` passes. The pptx is 5 slides and its text matches `deck_content.md`. `0.6 h` = 15.24 s × 135 epochs ✓. `92.2%` = `false_alarm.json → holdout_detection` ✓. **The headline numbers are clean. The attack surface is everywhere else.**

---

### THE TEN QUESTIONS, HARDEST FIRST

**1. Your demo tells me this line runs at ~1,400 m/min. Your slide 2 tells me 250 m/min needs 18 accelerators. Which number do I believe?**
I ran `simulation_tab`'s exact arithmetic (`demo/app.py:1442`, `line_speed = fps_p95 × footprint / 1000`): 40 held-out frames, p95 8.65 ms → 115.6 fps → **1,387 m/min at the default 200 mm footprint**. That figure treats one 200×200 crop as a whole 1.28 m × 200 mm frame — 6.25 mm/px, **31× coarser** than the 0.2 mm/px the deck's own geometry assumes. `benchmark.json → mill.streams_curve` says one accelerator serves **1.0 stream at 45 m/min** tiled at 256 px and falls below 1.0 by 48 m/min. The demo's headline KPI overstates the deck's own engineering answer by ~30×, and the demo is what is on screen during Q&A.
**Answerable: NO.** The honest arithmetic is in `reports/benchmark.json` and slide 2; nothing in the repo reconciles it with the demo KPI, and the demo carries no caption saying so.

**2. Let me upload an actual strip frame. [drags a wide image] — why does it say CLEAN?**
`tiled = st.toggle("Tiled inference (wide strip)", value=False)` — `demo/app.py:1097`. Measured: a 2400×256 strip assembled from 12 test frames spanning all six classes returns **0 detections** whole-frame, while the same 12 frames scored individually give **12/12 flagged, 40 detections**. Turn tiling on: 52 detections, all six classes recovered. The brief's core deliverable is "flags and labels defects on an uploaded image"; the default setting silently fails on the one image shape a steel judge will hand you.
**Answerable: NO.** No aspect-ratio warning, no auto-tiling, no note in `single_frame_tab`.

**3. [uploads their company logo] It says "scratches, 47% confidence." What did you train it to say when there is no defect?**
Measured on the deployed model at conf 0.15: `esds-logo-light.png` → scratches 0.47; a phone photo → 13 boxes, inclusion 0.63; a plain white 200×200 → **pitted_surface 0.739**; a heavily blurred steel frame → inclusion 0.45. Root cause is confirmed empirically: `find data/neu-det/*/labels -name '*.txt' -size -1c` returns **0 empty label files across all 1,800 images**. There is no background class, no abstention, no OOD gate anywhere in `src/inference.py`.
**Answerable: PARTIALLY.** The limitation is stated (slide 4 "Dataset"; `reports/false_alarm.md:296-300`), but no measurement of OOD behaviour and no mechanism exists. The 23.7% ceiling does not cover this case at all.

**4. Slide 3 prints "precision 0.696" one line above "conf 0.15". What is precision at conf 0.15?**
It is **0.596**. `reports/false_alarm.json → holdout_detection.per_threshold[t=0.15]`: tp 325, **fp 220**, fp_per_image 1.22, `false_alarm_rate 0.6611`, `images_flagged_rate 0.9944`. The 0.696 on the slide is the ultralytics validator's max-F1 point at conf 0.001 — a different threshold from the operating point stated four lines below it. Two of five boxes an operator is shown at the shipping threshold are unmatched, and 99.4% of frames raise at least one box.
**Answerable: YES**, `reports/false_alarm.json → holdout_detection.per_threshold[t=0.15]` — but the deck puts the flattering number next to the operating point and the honest one nowhere.

**5. Your objective table says "< 10 MB: 6.22 MB". What does the file you would actually deploy weigh?**
`export/yolov8n_neudet_best.onnx` = **12,128,540 bytes (12.13 MB)**; the `.mlpackage` is 12 MB. `export_summary.json → sizes_mb` confirms `onnx_mb 12.13`, `coreml_mb 12.16`. The 6.22 MB is the training `.pt`. Slide 5 puts them in one sentence — "A 6.22 MB … detector with a verified ONNX and CoreML export" — so the objective on slide 1 is **failed by the deployable artefact, 2× over**. There is also no INT8, no TensorRT and no OpenVINO export, which is what a line PC would actually run.
**Answerable: YES**, `reports/export_summary.json → sizes_mb` — and it refutes the slide.

**6. You put a confidence interval on every difference in the deck. Where is the interval on 0.752?**
Nowhere. `grep -c ci95_low reports/model_study.json` = 4, all inside `bootstrap.*`. The ingredient exists — `bootstrap.at_deployment_size.mAP50.a_marginal_std = 0.01625` — giving a 95% CI on the headline of roughly **[0.720, 0.784]** on 180 images. That interval contains published YOLOv8n at 0.740 and YOLOv11n at 0.772. The slide title's number is the one number in the submission that is quoted bare.
**Answerable: YES**, from `reports/model_study.json → bootstrap.at_deployment_size.mAP50.a_marginal_std` — but it is never computed or stated, and stating it weakens slide 3's title.

**7. Every published NEU-DET number is at 640 px. You trained at 320. What does 640 give?**
Unknown. `models/*/args.yaml` both read `imgsz: 320`; `grep -l "imgsz: 640" models/*/args.yaml` returns nothing — **no 640 training run exists**. Slide 4 converts this into differentiation item 4 ("the same weights lose 54% at 640"), which is a statement about *inference* size on 320-trained weights, not about training at 640. A competing team running stock `yolo train model=yolov11n imgsz=640` on the same 1620/180 mirror lands in the 0.77–0.80 band on your own `docs/research_notes.md` s1b (YOLOv8n 78.6, YOLOv10n 79.2, YOLOv11n 77.2, LCED-YOLO 79.8) and beats 0.752 on the number a judge reads off a slide. Your defence is real — theirs is a validation number on a selection split — but it is a paragraph, and theirs is a bigger number.
**Answerable: NO for the measurement.** `model_study.json → positioning` explains *why* (3.2× cost/step, ~4.8 h projected) and pre-empts the framing, but there is no 640-trained result to point at.

**8. The brief names scratches, scale, roll marks and edge cracks. Where are roll marks and edge cracks?**
`grep -in "edge crack\|roll mark"` over `deck/deck_content.md` and `docs/research_notes.md`: **zero hits**. You cover 2 of the 4 defect families the customer named. Edge cracks in particular need strip-edge imaging, and NEU-DET is 200×200 interior crops with no edge in any frame. The brief's key consideration "ability to handle multiple defect types" is answered with six classes — none of which are two of the four the business context lists.
**Answerable: NO.** Not acknowledged anywhere in the deck, the reports or the README.

**9. Severstal is 12,568 real 1600×256 strip frames, roughly half of them defect-free, and it is free. Why is it not in this submission?**
`grep -rin "severstal"` over the whole repo: **zero hits**. It would supply exactly the three things this submission cannot produce — genuine defect-free negatives (killing finding 3 and turning the 23.7% ceiling into a measured rate), real strip aspect ratio (killing finding 2), and a cross-dataset generalisation number, which is the only evidence available for the domain-shift risk slide 4 names as the thing that decides whether the project is real.
**Answerable: NO.** Not mentioned, not evaluated, not listed as future work.

**10. Show me a coil your system releases.**
`reports/coil_report.json`: disposition **HOLD**, `defect_rate 0.9278`, 167 of 180 frames defective, reason "Defect rate 92.8% exceeds the hold limit of 25.0%". Every coil this system can be shown will HOLD, because every source frame contains a defect. The ACCEPT/RELEASE branch — the commercially important one — has never been exercised on real imagery; `tests/test_smoke.py:953` asserts it only on synthesised input.
**Answerable: PARTIALLY.** `coil_report.json → source_note` states the cause honestly, but there is no released-coil artefact to show.

---

### THE WEAKEST LINK

**The demo.** The modelling, the statistics and the reports are unusually disciplined — CIs, design effects, rejected extrapolations, pre-emptive counter-arguments. The demo is where all of that stops. It is the only artefact that (a) contradicts the deck by 30× on the brief's own "inference speed suitable for a production line" clause, (b) fails silently on the image shape the brief describes, (c) produces confident nonsense on any non-NEU input, and (d) contains none of the honesty the reports are full of. It is also the one artefact a judge is guaranteed to touch. Findings 1, 2 and 3 are all demo defects, and all three are the kind that happen live.

### SECONDARY (real, lower blast radius)

- **`reports/evaluation.md` is a different model.** Header: `yolov8s_neudet/best.pt`, imgsz 320, **mAP50 0.6598**. The file named "evaluation report" describes neither the deployed checkpoint nor the deck's number. Anyone who opens the repo looking for the evaluation finds 0.66.
- **`src/explain.py` CLI is broken.** The README command at `README.md:385-386` run verbatim crashes: `TypeError: int() argument must be … not 'NoneType'` at `src/inference.py:889` (`load_detector(imgsz=None)`). Works only with an explicit `--imgsz`. `grep -c explain tests/test_smoke.py` = 1 (a docstring). An 836-line module cited as differentiation item 5, not imported by `demo/app.py`, with a broken documented entry point and no test coverage.
- **Export "verified" on one image.** `export_summary.json → verification.detections`: one file, `pitted_surface_271.jpg`, 2 detections matched. That is the entire basis for "identical detections".
- **"Per-line recalibration inside one shift: 0.6 h"** is GPU time only. Labelling the new line's frames — the actual schedule driver — is not in the objective and appears nowhere as a duration.
- **The 640 option is live in the demo sidebar** (`IMGSZ_OPTIONS` includes 640). A judge who moves that slider watches the model drop to `mAP50 0.3438` (`model_study.json → runs[]`, yolov8n/test/640) with no warning in the UI.
- **A landmine to keep off the stage:** `reports/evaluation.md §2` reports **97.8% image-level accuracy**. That is the NEU-CLS-shaped number your own `research_notes.md` s1a and s1d call the hallmark of unserious work. If anyone on the team reaches for it under pressure, slide 3 collapses.
- **Slide 4's speaker note says "the shipping code defaults to 640".** It does not: `src/inference.py:45`, `DEFAULT_IMGSZ = 256`. Stale note asserting a defect that no longer exists.
- **Split-leakage check is exact-hash only.** I ran a calibrated near-duplicate check (32×32 high-pass, illumination removed, against the 99.9th percentile of 12,000 random same-class train pairs): **10 of 180 test images** exceed that threshold, top pairs all `scratches_*`, max cosine 0.447. That is *not* evidence of leakage — raw-thumbnail similarity up to 0.98 turned out to be global tone, not texture — but the repo cannot currently rule it out, because `content-hash disjoint` catches only byte-identical files. One within-train exact duplicate exists (`patches_101.jpg` / `patches_105.jpg`), which the deck already discloses.