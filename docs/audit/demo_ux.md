# DEMO AUDIT — `demo/app.py` driven under AppTest (own instance on 8555, now killed; 8501 untouched and still returning 200)

Method: read all 1,599 lines of `demo/app.py`; ran ~45 AppTest sessions against the real script (uploads, sidebar sweeps, batch, simulation, checkpoint/device switches); timed `src/explain.py` and `predict_tiled` directly; scored all 180 held-out frames to characterise the headline number. Zero uncaught exceptions in any run — the failure modes below are all *confident wrong answers*, not crashes.

---

## RANKED — what a judge hits, worst first

**1. Anything that is not steel is confidently dispositioned as a critical defect.**
A cartoon landscape (blue sky, yellow sun, green grass, red house) uploaded to Single frame renders: `DEFECT · 6 detection(s) · Dominant defect patches · Peak confidence 0.83 · Severity score 94.0/100 · critical`, a 6-row detection table, and four guidance cards instructing the operator to *"Verify descaler header pressure"* and *"Trace the coil back to its heat and cast sequence and quarantine it."* At imgsz 640 the same picture gives 12 detections and 95.0/100 critical. A pure white 400×400 frame (blown-out exposure) → `pitted_surface 0.74, severity 86.3 critical`. Pure black (lens cap) → `pitted_surface 0.34, 66.4 high`. An 8×8 px image → 3 `patches` boxes.
Against the brief: this *is* the false-alarm rate, made visceral, and it is the first thing a curious judge does. There is no OOD guard anywhere in `demo/` or `src/inference.py` (grep: 0 hits; the term appears only inside `evaluate.py`/`false_alarm.py`/`benchmark.py`).
Cost to close: ~40 lines for a plausibility gate before the verdict (min resolution, colour-saturation and Laplacian-variance check → "frame does not look like a strip capture; verdict withheld"), plus one honest banner. The real fix is defect-free negatives (Severstal).

**2. The zero-click landing screen shows `Peak confidence 0.22` next to `Severity score 79.8/100 · high`.**
Launch → no clicks → `crazing_271.jpg` is already scored: `DEFECT · 3 detections · crazing · peak conf 0.22 · 79.8/100 high`. The family picker defaults alphabetically to `crazing`, which is the model's weakest class: measured over all 180 test frames at the shipped operating point, crazing's median peak confidence is **0.29 and 93% of crazing frames land below 0.50** (patches 0.88 / 0% below 0.50, scratches 0.77 / 0%, all-classes median 0.67). So the demo's automatic first impression is its worst class, showing a headline confidence a judge will read as "the model is 22% sure" beside a severity that says "high". Nothing on screen reconciles the two.
Cost: one line — default the family selectbox to `patches` or order families by median confidence. Highest score-per-character change in the project.

**3. One image, four contradictory verdicts, driven by sidebar knobs a judge will touch — and the deck says those knobs shouldn't exist.**
Same 4032×3024 phone-style photo of steel:

| setting | verdict |
|---|---|
| default (256, single pass) | **PASS** — "no defect above threshold" |
| Network input size → 1280 | **DEFECT**, 12 × pitted_surface, severity 84.7 critical |
| Tiled inference on, tile 256 | **DEFECT, 308 detections**, severity 86.7 critical, headline **"Throughput 0.6 fps"** |
| 108 MP image + tiled | **1402 detections**, **"Throughput 0.2 fps"** |

Deck differentiator #4 reads *"Input size pinned as an accuracy parameter, not a config knob… the same weights lose 54% of mAP50 at 640 px, silently."* The console ships it as a config knob (320/416/512/640/960/1280) with only a help-tooltip warning, and the verdict panel carries no caveat. A judge who reads the deck and then moves the slider has caught the project contradicting itself, live. The 0.6 / 0.2 fps figures also sit directly under a brief clause about production-line inference speed.
Cost: ~20 lines — move imgsz and tiling behind an "experimental / survey mode" expander and stamp non-default results as "not the validated operating point".

**4. Explainability exists, is fast, is sold on the deck, and is unreachable from the UI.** Confirmed: `grep -c "explain" demo/app.py` = 0. `src/explain.py` is 836 lines with a validated sign-corrected EigenCAM; I measured **11 ms warm** per heatmap (1.8 s once for the first `pytorch_grad_cam` import), and `Explanation.overlay(image)` already returns an RGB array. `reports/explainability/eigencam_*.png` exist on disk. Deck slide, differentiator #5: *"Explainable, and routed. EigenCAM heat maps (src/explain.py) show what the detector fired on."* If a judge asks "show me what it looked at", the answer is a CLI.
Cost: **~15 lines** in `single_frame_tab` — a toggle, a third image column, a caption with `exp.method/layer/stats`. Same story for `src/report.py` → `reports/coil_report.html` (a real per-coil disposition report, also not linked from any tab).

**5. "Max line speed 1267 m/min" is a headline KPI with no floor and no width.** Line simulation, defaults: `Sustained rate 105.6 fps → Max line speed 1267 m/min`. Set the "Frame footprint" number input to its max 2000 mm and it prints **12,408 m/min** (207 m/s) with a straight face. The formula is `fps_p95 × footprint / 1000`; it assumes one camera whose frame covers the *full strip width*, which for a 1250 mm strip and a 200 mm frame is off by roughly 6 cameras. Presented as a bordered metric tile beside genuinely measured latencies, so it inherits their credibility.
Cost: ~15 lines — bound the input, state "single camera, frame covers full strip width", and derive the camera count a 1250 mm strip needs.

**6. "Defect rate 100.0 %" on both the batch tab and the line simulation.** Sample batch of 12 → `Defect rate 100.0 %`; simulation → `Final defect rate 100.0 %` and a running-defect-rate trace pinned at 100 across all 40 frames. This is gap #3 made operator-visible: every NEU-DET frame is defective by construction, so the demo can never show sound steel passing. The only PASS a judge can produce is by pushing conf to 0.95 or uploading something the model happens to ignore. Nothing on screen explains why a simulated production line rejects every coil.
Cost: a caption today; Severstal negatives to actually fix.

**7. Severity is the most authoritative number on screen and the least evidenced.** `Severity score 94.0 / 100 · critical`, colour-coded, repeated as a progress bar in every table and in both CSV exports. It is `class_tier_anchor × (0.5 + 0.5·conf) × (1 + 0.55·min(1, area_frac/0.25))`, noisy-OR aggregated with decay 0.6 — hand-set constants never validated against yield loss, downgrade rate or rejection. Its only on-screen definition is one sidebar caption ("Severity = class base tier, scaled by confidence and box area"). Because area fraction is frame-relative, the same physical defect scores differently on a 200×200 crop and a 4032×3024 photo; and because the aggregation rewards box count, six spurious boxes on a cartoon outrank one genuine inclusion. A mill manager will read 94/100 as a measurement.
Cost: ~10 lines — an expander next to the number stating the formula and "ordinal triage aid, not calibrated to mill outcomes".

**8. Batch KPIs go stale silently.** After a batch run, moving the confidence slider from 0.15 to 0.90 leaves the batch tab showing `Total detections 7 / Defect rate 100.0 %` while the Single-frame tab on the same screen re-scores to 0 detections instantly. The caption discloses it ("run the batch again to re-score"); the five large metric tiles do not. Cost: ~10 lines to grey out and stamp "stale".

**9. The header claims a line identity the demo does not have.** `Jindal Stainless · Hot Strip Mill · Line 2 Exit` sits above every screen while all data is public NEU-DET hot-rolled **carbon** steel. "NEU-DET" appears exactly once in the rendered UI — in the 4th tab. The deck is scrupulous about this ("a capability demonstration, not a prediction"); the demo is not. Cost: one subtitle.

**10. iPhone photos are rejected at the widget.** `UPLOAD_TYPES` = jpg/jpeg/png/bmp/tif/tiff. HEIC — the iPhone default — is refused by the browser with a generic message, and `pillow-heif` is not installed so it would fail anyway. WEBP decodes fine in this Pillow build but is not allow-listed. The most likely "let me try my own photo" gesture in the room fails before the app sees the bytes. Cost: 2 lines + 1 dependency.

**Missing per the brief, visible in the product:** "edge crack" and "roll mark" appear **0 times** in `demo/app.py` and `src/inference.py`. The Defect atlas — the demo's own answer to "what can this find" — lists six NEU-DET families and neither of the two defect types the brief names.

---

## Direct answers

**Real-world inputs.** Decoding is genuinely strong and I could not break it: RGBA (with fully transparent holes), CMYK JPEG, 16-bit `I;16` TIFF, EXIF-orientation-6 portrait, and a 108 MP / 16.8 MB JPEG all decode, convert, and print an honest on-screen note ("Converted from CMYK to RGB", "Downscaled from 12000x9000 to 8944x6708 (over the 60 MP single-frame limit)"), the big one in 2.0 s end-to-end. A truncated JPEG, a PDF renamed `.jpg`, and a 6000×10 strip all produce clear one-line `st.error`s with no traceback and the page stays up; a batch containing bad files warns "Skipped: …" and scores the rest. **Failure is never in the decoder — it is in the verdict.** The embarrassing cases are all well-formed images the model answers confidently and wrongly (items 1 and 3).

**Confidence and defect type.** Prominent and unambiguous as presentation: 2.05 rem PASS/DEFECT flag on a green/red field, then `Dominant defect`, `Peak confidence`, `Severity score`, latency, throughput. `dominant_class` and `max_confidence` come from the same detection (`_finalise`), so they can't disagree. Two soft spots: class names are raw dataset labels (`pitted_surface`, `rolled-in_scale`), not mill vocabulary; and "Peak confidence" is a per-box score presented as if it were a frame-level certainty (see item 2).

**Clicks to first result: zero.** `make demo` → the page opens on Single frame with a held-out sample already scored and annotated. Cold start is ~1.3 s of imports + model load + warmup (torch 0.44 s, load_detector 0.11 s, warmup 0.18 s), so under ~3 s to first paint; warm reruns are 0.06 s. This is genuinely excellent and worth saying out loud in the room — it's just aimed at the wrong sample.

**Misleading UI:** "Max line speed … m/min" (item 5), "Severity score /100" (item 7), "Defect rate 100.0%" without explanation (item 6), and stale batch tiles (item 8). To its credit, the sidebar caption is exemplary — it names the source file, the split, what chose the threshold, and calls the 23.7% figure "an upper bound" — but it is 0.78 rem grey text in a sidebar, below the fold of attention, while the things that mislead are 1.9 rem metric tiles.

**Heatmap reachable from the UI:** no, not at all — see item 4. ~15 lines to surface it, 11 ms to render it.

**Would a non-technical mill manager understand it?** The verdict strip, the colour-coded severity band and the root-cause/corrective-action cards are genuinely well-judged — that part reads like a quality station, not a notebook. What breaks the illusion for that reader: a 22%-confidence "high severity" opening screen, a 100% defect rate, a severity score with no stated basis, `pitted_surface` as a label, a detection table of `x1 y1 x2 y2 W H Area px²`, and a header naming a line the system has never seen.

---

## Suspected gaps — verdicts

1. **Video/stream: CONFIRMED.** No `st.camera_input`, no video widget, no frame-source-over-time anywhere. "Line simulation" replays 40 stills through `predict` sequentially; latency and fps are real, motion is not.
2. **Explainability invisible: CONFIRMED**, and worse than stated — the deck sells it as differentiator #5.
3. **No defect-free data: CONFIRMED and operator-visible** — the 100% defect rate KPIs and the white/black/cartoon criticals are its face in the product.
4. **Single dataset / Severstal unused: CONFIRMED** from the demo side — there is no PASS a judge can trust, because the console has never been shown sound steel; the only demonstrable PASSes come from pushing conf to 0.95 or from inputs the model happens to ignore.
5. **Edge cracks / roll marks absent: CONFIRMED** — 0 occurrences in the demo code; the atlas tab silently answers a different question than the brief asked.

Artefacts: `/private/tmp/claude-501/-Users-prathmeshwalimbe-Downloads-JSW-PS1/f88a3d3d-c8be-40c8-8600-132f500095d3/scratchpad/` — `make_assets.py`, `assets/` (14 fixtures), `drive.py`…`drive6.py`, `explain_check.py`, `tiled.py`.