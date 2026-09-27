# Is anything missing? Yes — five things, and only one of them is science.

Everything below I reproduced myself on your machine. The headline is clean: `mAP50 0.752448, mAP50-95 0.396709, P 0.696, R 0.687` reproduces to six decimals on the held-out test split at 256 px, 111 tests pass in 4.08 s, and the demo on 8501 is still up (I used separate processes throughout). The gaps are not in the modelling. Four of the five are in what a judge sees; one is an experiment that was an afternoon away and never run.

---

## RANKED GAPS

### 1. The demo says CLEAN on a steel-strip image. Verified.
**What's missing:** auto-tiling, and any wide image to demonstrate it on.

**Evidence.** I built a 3072×256 strip from 12 held-out crops spanning all six classes and ran it through the shipped detector:

| path | result |
|---|---|
| default (`predict`, tiling off) | **0 detections → verdict PASS** |
| tiled (`predict_tiled`) | 58 detections, correct classes, peak 0.920, 159 ms |
| the same 12 crops scored individually | 40 detections, 12/12 flagged |

`demo/app.py:1097` — `st.toggle("Tiled inference (wide strip)", value=False)`. No aspect-ratio trigger, no warning, no note in the single-frame tab. And there is nothing in the repo to trigger it on: `assets/` is empty, and a scan of `data/`, `assets/`, `demo/` returns **zero images with any edge ≥ 600 px** (the only large files are the six deck figures). Every sample is a 200×200 NEU-DET crop, so flipping the toggle changes nothing a judge can see.

**Clause hit:** "detects and classifies surface defects on **steel-strip images**" and "a working demo that flags and labels defects on **an uploaded image**." This is the core deliverable failing on the one image shape a steel judge will hand you, silently, at the default setting.

**Hours:** 1.5 — ship one strip mosaic into `assets/`, default the sample picker to it, and auto-enable tiling when long/short > 3 with a visible TILED badge.

---

### 2. On genuinely defect-free steel the detector fires on 91% of frames, and the coil is HOLD.
**What's missing:** any defect-free steel, anywhere. Severstal exists, is free, and is not mentioned in your repo.

**Evidence.** `grep -ril severstal` across the whole tree: **0 hits** — not in the code, not in `docs/research_notes.md`, not in the limitations, not in the roadmap. I verified the dataset myself: `Voxel51/severstal_steel_defects`, public and ungated, 18,074 images, every one **1600×256×3**; train split 12,568 = **6,666 defective / 5,902 verified defect-free**. I pulled 120 defect-free frames in 20 seconds and ran your shipped checkpoint on them:

| conf | clean-crop FA | clean-**frame** FA | boxes/crop | top false classes |
|---|---|---|---|---|
| **0.15 (your operating point)** | **50.6%** | **90.8%** | 1.59 | inclusion 40%, patches 34%, scratches 18% |
| 0.25 | 41.1% | 83.3% | 0.96 | inclusion 34%, patches 34% |
| 0.40 | 26.8% | 76.7% | 0.48 | patches 42%, scratches 27% |

Then end-to-end through your own disposition chain, `src/report.py` on those 120 clean frames:

```
coil SEVERSTAL-CLEAN-AUDIT: 120 frames, defect rate 50.0%, 85 detections, p95 severity 66.3 -> HOLD
  - Defect rate 50.0% exceeds the hold limit of 25.0%.
  - 12 frame(s) contain 'inclusion', a zero-tolerance defect (limit 0)
```

Two sub-findings you cannot currently see. First, `reports/false_alarm.md:139` concludes *"the single largest failure mode is scratches at 42% of all false positives."* On real defect-free steel that is wrong — it is **inclusion at 40%**, and `DispositionRules.hold_classes = ("inclusion",)` makes inclusion your sole unconditional-HOLD class. Your dominant hallucination is the one class that forces a hold. Second, the model has no abstention: a pure white 200×200 returns `pitted_surface 0.739, severity 86.3`; a cartoon landscape returns 9 detections, peak 0.859, **severity 97.2 critical**. Gaussian noise and flat grey correctly return nothing — so it is not "fires on anything," it fires on unfamiliar *steel-like texture*, which is exactly the failure that matters.

**Clause hit:** "Detection accuracy and **false-alarm rate**" and "a clear path to scaling across grades and products." Slide 3 already states what would be needed — *"continuous capture of prime-shipped coils… a few thousand frames"* — and that data is a 20-second download. Your deck names the missing experiment and the experiment is free.

**Hours:** 2 to download, measure, and put the honest number on slide 3 beside the 23.7% proxy. 8–12 to actually fix by joint training (see the 30 h tier — and note the trap: negatives alone teach a domain classifier, you need Severstal's 6,666 labelled positives in the same run).

---

### 3. The demo's headline speed number contradicts your own deck by more than an order of magnitude.
**Evidence.** `demo/app.py:1441`: `line_speed = fps_p95 * footprint / 1000`. I measured your simulation tab's actual inputs — 40 held-out frames, p50 5.0 ms, p95 10.4 ms, fps_p95 96.5 — which prints:

> **Max line speed 1,159 m/min**

Your own `reports/benchmark.json` geometry says a camera sees 0.4096 m of a 1.28 m strip at 0.2 mm/px, four cameras across the width. At full optical resolution one accelerator advances the strip **~13.5 m/min** (the deck's own "18 accelerators at 250 m/min," slide 2). The demo's formula treats one 200×200 crop as covering the whole strip width at 6.25 mm/px — 31× coarser than the resolution the deck assumes. Deck slide 2 is honest. The demo, which is what is on screen during Q&A, is 5×–85× optimistic depending on which comparison you pick, with no caption.

**Clause hit:** "Inference speed suitable for a production line." This is the finding most likely to be caught live, and it costs you disproportionately because the rest of the work is scrupulous about not overstating.

**Hours:** 0.5 — relabel it "single-camera field-of-view advance rate" and print the accelerator count next to it.

---

### 4. `README.md` and `reports/evaluation.md` describe a worse system than the one you are submitting — and `make eval` will silently break the demo.
**Evidence, all verified:**
- `README.md:226-249` ("The operating point") reports **conf = 0.05**, false alarm rate **81.1%**, and states in bold **"This operating point is not deployable as it stands."** All three are superseded and all three are worse than the truth. The README never mentions `reports/false_alarm.md` or `src/false_alarm.py` — the strongest work in the project — and neither `src/false_alarm.py` nor `src/model_study.py` appears in the README's own project-layout listing.
- `reports/evaluation.md`, the file the README calls "the document a person signs off," headlines **`yolov8s_neudet/best.pt` @ 320 px, mAP50 0.6598**. Neither the shipped checkpoint nor the deck's number.
- `src/evaluate.py:1550` writes `reports/operating_point.json` unconditionally; `src/false_alarm.py:2368` overwrites it only under `--update-operating-point`. There is **no make target** for `false_alarm.py` or `model_study.py`. I confirmed `reports/operating_point.evaluate.json` is the conf-0.05 yolov8s version sitting right there. Running the documented `make eval` resets the demo's default confidence 0.15 → 0.05.

**Clause hit:** indirect but broad. A judge's first click is the README, and it currently tells them your false alarm rate is 81% and your threshold is not deployable.

**Hours:** 2 — rewrite that one README section, banner `evaluation.md` as superseded, add `make false-alarm` and `make study`.

---

### 5. Two of the four defects the brief names do not exist anywhere in the submission.
**Evidence.** Repo-wide grep: **"edge crack" → 0 hits. "roll mark" → 1 hit**, `README.md:510`, inside a limitations sentence. Not in the deck, not in `docs/research_notes.md`, not in the demo's Defect Atlas — the tab that is your own answer to "what can this find."

You cover scratches (AP50 0.909) and scale (`rolled-in_scale`, 0.630). Roll marks and edge cracks you do not. Edge cracks are not a labelling gap — NEU-DET is a 200×200 *interior* crop, so a strip edge appears in zero training frames. It is a field-of-view gap, and it needs saying.

**Clause hit:** "Ability to handle multiple defect types." The brief lists these four in its business context; a Jindal panellist will ask.

**Hours:** 0.5 — one "not covered" line on slide 4 and in the Defect Atlas, with the field-of-view reason for edge cracks. Cheap to disclose, expensive to be ambushed on.

---

### 6. Explainability is differentiator #5 on your deck and is unreachable from the demo. The documented command crashes.
`grep "explain|cam|EigenCAM"` in `demo/app.py` → 2 hits, both the phrase "line camera." `src/report.py:568` does import it, and `reports/coil_report.html` carries the heatmaps — but nothing in the console links to that file. And the README command at `README.md:385` run verbatim:

```
File "/Users/prathmeshwalimbe/Downloads/JSW-PS1/src/explain.py", line 827
    det = load_detector(imgsz=args.imgsz)
TypeError: int() argument must be a string ... not 'NoneType'
```

`args.imgsz` defaults to `None` and the fallback is on line 828, one line too late. It sits under `# pragma: no cover`, so none of the 111 tests touch it.

**Clause hit:** "A usable demo interface." **Hours:** 2 minutes for the crash, 1.5 h for a CAM button in the single-frame tab plus a link to `coil_report.html`.

---

### 7. The confidence score the brief explicitly asks for is not calibrated.
Independently measured, 790 test detections at conf ≥ 0.05: **ECE = 0.1418**, monotone under-confidence in every single bin.

| conf bin | n | mean conf | empirical precision | gap |
|---|---|---|---|---|
| [0.15, 0.25) | 116 | 0.195 | 0.336 | −0.141 |
| [0.55, 0.65) | 69 | 0.599 | **0.841** | −0.242 |
| [0.65, 0.75) | 55 | 0.697 | **0.927** | −0.231 |
| [0.85, 1.00) | 41 | 0.885 | **1.000** | −0.115 |

An operator reading "0.60" is looking at a box that is right 84% of the time. Because the error is monotone, a one-parameter fit on val corrects it, mAP is unchanged (rank-preserving), and you get a reliability diagram — which is a genuinely differentiating slide.

**Clause hit:** "ideally with a **confidence score** and defect type." You satisfy this nominally; two hours makes it real. **Hours:** 2.

---

### 8. No video, no camera, no tracking. (Confirmed — and ranked last on purpose.)
`grep -ci "video|camera_input|VideoCapture|webrtc|mp4"` in `demo/app.py` → **0**. "Line simulation" replays 40 stills through `predict` sequentially. Real time is argued, never shown.

I am ranking this eighth against the suspicion that it was first, because your latency is genuinely measured and a judge watching a frame counter tick at 96 fps gets the point. A video loop is presentation, not evidence, and it costs 3–5 hours — more than findings 1, 3, 4 and 5 combined, for less score. Build it only in the 30 h tier.

---

### Secondary, disclose-only (30 minutes total)
- **The deployable artefact misses your own objective.** Slide 1: "Model small enough to sit at the line | < 10 MB | 6.22 MB." `export/yolov8n_neudet_best.onnx` is **12,128,540 bytes = 12.13 MB**; the CoreML package is 12.16 MB; `export_summary.json` says `inference_weights_fp32_mb: 12.05`. The 6.22 MB is an fp16-stored `.pt`. Your own report is honest about it; the objective row is not.
- **Slide 3 prints precision 0.696 one bullet above "conf 0.15."** That 0.696 is the validator's max-F1 point at conf 0.001. At the actual operating point, `false_alarm.json → holdout_detection.per_threshold[0.15]` gives **precision 0.5963** (tp 325, fp 220, 1.22 fp/image). Not a misstatement, but the operating-point number is nowhere on the slide and the flattering one is.
- **"92.2% of defective test frames are flagged"** — the metric is `defect_detection_rate` (found *and* correctly classified at IoU 0.5). The actual flag rate is 99.4%. "Flagged" is the wrong word and it undersells you; say "found and correctly classified."
- **`deck_content.md:293` speaker note:** *"In this repository the shipping code defaults to 640."* It does not — `src/inference.py:45`, `DEFAULT_IMGSZ = 256`. A stale note asserting a defect you already fixed.
- **`README.md:444,473`** says 101 tests, about 7 s. Actual: 111 tests, 4.08 s.
- **`reports/model_study.md:112`:** *"256 px is the only size in the grid that does not [upsample]."* 256/200 = 1.28×. It upsamples too.
- **No CI on the headline.** `model_study.json → bootstrap.at_deployment_size.mAP50.a_marginal_std = 0.01625` gives roughly [0.720, 0.784] on 180 images. 0.752 is the one number you quote bare.

---

## WHAT I CHECKED AND AM DISCARDING — stop worrying about these

- **Motion-blur collapse.** Claimed as detections 3→1 and confidence 0.82→0.37. I applied a 15-px horizontal blur to `inclusion_271.jpg`: detections went 7→3, peak confidence **0.669→0.680**. Instance count halves; confidence does not collapse. Real but not the sharp finding it was sold as, and no judge applies motion blur live.
- **The specular water-streak case** ("scratches 0.87 outranks the genuine defect"). Did not reproduce — my streak gave dominant class inclusion at 0.625, no scratches at all. Do not put this on a slide.
- **Architecture headroom.** YOLO11n was measured at +0.0074 mAP50, inside your own ±0.025 bootstrap half-width. Not a real gain. Do not re-do the deck.
- **Ensemble.** ~10 pp of oracle recall, but it doubles inference cost and directly contradicts your throughput slide.
- **640 px training.** `reports/model_study.md:222` already handles this with more care than the criticism did — measured 3.2× step-cost ratio, documented paging, correct treatment of median vs mean. A 2.3 h run buys you one sentence.
- **The imgsz grid starting at 256 / a val peak at 224.** True, worth −0.025, and pedantic.
- **The vacuous seam-dedup test at `tests/test_smoke.py:557`.** True — it asserts what NMS guarantees by construction. No judge reads your tests.
- **HEIC uploads.** Two lines, zero score.

---

## WHAT IS GENUINELY COMPLETE

- **The headline reproduces exactly.** I ran the validator myself: 0.752448 / 0.396709 / 0.696 / 0.687, per-class AP50 matching. No rounding games.
- **Classification is solid and correctly characterised.** `class_accuracy 0.9833`, zero class confusions among the misses. The model finds a defect or it does not; it does not mistake one for another.
- **The false-alarm methodology is the best work in the project and is above the standard of most published papers.** It refuses the easy number, builds a clean-steel proxy, grows crops 6 px before testing, bootstraps with image clustering, explicitly *rejects* a spatial-independence model (χ² 34.56 vs 0.35 on 3 dof), gives the per-source-class mixture, and calls 23.7% an upper bound. My Severstal result does not refute it — it confirms exactly the direction it warned about and supplies the number it said it could not measure. Do not touch this file except to correct which class dominates.
- **Exports verified.** ONNX and CoreML, max absolute deviation 7.1e-4, `allclose` at 1e-3.
- **`benchmark.json`'s mill capacity model is honest** — it measures its own ±25% between-session repeatability and refuses to quote an absolute frame rate. Slide 2's accelerator arithmetic is correct.
- **The demo's decode layer is genuinely hard to break.** CMYK, 16-bit `I;16` TIFF, RGBA with transparency, EXIF-rotated portrait, a 108 MP JPEG — all decode with an honest on-screen note. Truncated JPEGs, a PDF renamed `.jpg`, a 6000×10 sliver: clean one-line errors, page stays up, batches skip and carry on. Failure is never in the decoder.
- **Zero clicks to first result, ~3 s cold start.** Say this out loud in the room.
- **Split discipline.** Content-hash disjoint, and `--split` must differ from `--tuning-split` or the run refuses to start.
- **`operating_point.json` is read by the console**, so screen and report cannot drift — as long as nobody runs `make eval` (finding 4).
- **The deck covers all seven required topics genuinely**, none nominally, and slide 4's limitations section — including "Not built" and "Money" — is stronger than most submissions will manage. Note that slide 4 is 3,823 characters at 8.7 pt, 77% denser than the next slide; if you need room for a demo screenshot, take it from there.

---

## RECOMMENDATION

### If you have 4 hours — all presentation, zero new science, highest score per hour
1. **(1.5 h)** Ship one 3072×256 strip mosaic into `assets/`, make it the default sample, auto-enable tiling above aspect 3, show a TILED badge. *Kills finding 1 — the worst live-demo failure you have.*
2. **(0.5 h)** Caption or relabel the "Max line speed" KPI and print the deck's accelerator count beside it. *Kills finding 3 — the contradiction most likely to be caught in Q&A.*
3. **(1 h)** Rewrite `README.md` §"The operating point," banner `evaluation.md` as superseded, add `make false-alarm` and `make study`. *Kills finding 4.*
4. **(0.5 h)** Add "not covered: roll marks, edge cracks (field of view)" to slide 4 and the Defect Atlas. *Kills finding 5.*
5. **(10 min)** Fix `src/explain.py:827`; fix the stale slide-4 "defaults to 640" note; fix the 101→111 test count; add the ONNX 12.13 MB caveat to the objectives row.

That moves *steel-strip images*, *inference speed* and *multiple defect types* from partial toward satisfied, and removes both things that would puncture your discipline claim live.

### If you have 12 hours — the above plus the two experiments that change what you can claim
6. **(2 h)** Pull 400 defect-free Severstal frames, measure the real clean-steel false-alarm rate, and put it on slide 3 *beside* the 23.7% proxy, however bad it is. This is the exact experiment slide 3 says it cannot do. Correct `false_alarm.md`'s stated top failure mode to inclusion, and flag that inclusion is your sole zero-tolerance HOLD class. *A measured cross-domain number, however ugly, is a stronger slide than a stated limitation.*
7. **(2 h)** Calibrate confidence on val (isotonic or temperature) and ship a reliability diagram. mAP unchanged, and it turns a brief clause you satisfy nominally into one you satisfy properly.
8. **(2 h)** Surface EigenCAM in the single-frame tab behind an on-demand button (it is ~11 ms warm), plus a link to `coil_report.html`.
9. **(1 h)** Put the 95% CI on 0.752 and the operating-point precision 0.596 on slide 3. Volunteering both makes slide 3 unattackable.

### If you have 30 hours — the above plus the one thing that changes the model
10. **(8–12 h)** Convert Severstal masks to boxes (they decode from base64+zlib to a (256,1600) uint8 array with class ids 1–4 in the pixel values, ~3.2 components/image) and train jointly on Severstal **positives and negatives** plus NEU-DET. Publish a cross-dataset generalisation table. **The trap:** training on Severstal negatives alone drops clean-frame false alarms ~16× at zero in-domain cost, but defect recall on that domain collapses too — you have taught a domain classifier, not a defect classifier. The positives must be in the same run.
11. **(4 h)** Video upload plus a simple IoU tracker so a defect persists across frames and the fps counter runs on moving footage. *Real time shown, not argued.*
12. **(3 h)** Convert detections to millimetres — 0.2 mm/px is already in `benchmark.json`, and `grep mm src/inference.py src/report.py` returns nothing. A defect length in mm is MES-actionable; a pixel box is not.

**The honest summary:** the science is done and it is good. The four hours in tier one are worth more to your score than the twenty hours in tier three, because they fix things that break in front of the judge. The one genuinely absent piece of work is Severstal — it costs two hours to measure and closes the false-alarm root cause, the strip-geometry gap, and the only available evidence for your scaling claim, all at once.