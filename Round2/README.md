# Round 2 — Detailed Case Summary

Self-contained workspace for the Stainless Spark Round 2 submission (PS1).
Everything needed to submit, rebuild or defend the deck is in this folder.

## What to submit

**`out/JSW_Round2_Surface_Defect_Detection.pdf`** — 8 slides (cover + 7), the maximum allowed.

The brief allows a 120–150 sec video **or** a 6–8 slide PPT. The deck is the stronger
submission here: it carries the risk table, the KPIs and the claim sourcing, which a
150-second video has no room for. If you choose video instead, the script is in
`JSW_Round2_Video_Script.md`.

## Before you submit — one edit

Team details are still placeholders. They live in **one place**, at the top of
`build/build_round2.js`:

```js
const TEAM = '[TEAM NAME]';
const MEMBERS = ['[Member 1]', '[Member 2]', '[Member 3]', '[Member 4]'];
const INSTITUTION = '[INSTITUTION]';
```

Edit those three lines, then rebuild (below). The name flows to the cover and to the
header of all seven content slides.

## Rebuilding

```bash
cd build
npm install                                        # once: pptxgenjs, lucide-static, sharp
node gen_assets.js                                 # icons + cover photo crop  → img/
../../.venv/bin/python gen_detections.py           # defect gallery from our model → img/det_*.png
../../.venv/bin/python gen_proof.py                # proof-slide samples + right-type rate → img/
../../.venv/bin/python gen_qr.py                   # cover QR → qr.png
node build_round2.js                               # → ../out/*.pptx
```

Only the last step is needed to change text or team details — the assets are already
generated. Then export the PDF from PowerPoint (File → Export → PDF). Always check the
PDF, not the PPTX preview: PowerPoint applies text insets that other renderers do not,
and that is where overflow shows up.

## How the deck covers the brief

The five required sections are the navigation tabs across the top of every slide.
Each slide answers one question and ends in a decision band.

| Slide | Section | Main visual | Decision band |
|---|---|---|---|
| 1 | Cover | Coil photo, strip of six real detections, four headline numbers, live-demo QR | — |
| 2 | Problem | Value-chain ribbon (born here / seen today) under a rising value-added wedge; PAF arrow | Defects are found late, by sampling, with no named cause |
| 3 | Insights | Defect gallery — our model's real detections, each with born-at / owner / first fix; grade risk heatmap → pilot grade | Classify, don't just alarm; pilot on 300 series |
| 4 | Insights | Four findings: false-alarm bar chart, accuracy bars, processor pictogram, trust stats | The model is the cheap part |
| 5 | Solution | Six-stage ribbon with build status, live-output panel, innovations, build-vs-buy dots | Buy proven cameras, own the model |
| 6 | Solution (proof) | Three dataset cards with samples, per-defect accuracy bars vs published range, headline results, weak spots → fixes | Strong on four of six; every weak spot has a fix |
| 7 | Implementation | Gantt with go/no-go, cost doughnut, top-5 risks, six pilot KPIs | ₹2.5–4.2 cr, one line, seven months |
| 8 | Impact | Loss-pool heatmap, payback bars, tornado, impact icons, the ask | Approve the pilot |

Read the bottom bands in order and you have the elevator pitch.

Against the judging criteria: **depth** is slides 3, 4 and 6 (real detections, measured
findings, per-defect results and weak spots); **feasibility** is slide 7 (gated rollout, costed, nothing touches the mill
until P3); **innovation** is slide 5 (clean-strip training, root-cause routing,
unseen-defect check).

## Files

```
out/     JSW_Round2_Surface_Defect_Detection.pptx / .pdf   ← submit the PDF
build/   build_round2.js     the deck, as code (team config at the top)
         gen_assets.js       icons (Lucide, ISC licence) and cover photo crop
         gen_detections.py   runs our joint model on held-out test images, draws the gallery
         gen_proof.py        proof-slide samples (Severstal, GC10) and the 176/180 right-type rate
         gen_qr.py           cover QR, via the project's own encoder
         econ.py             the payback model behind slide 8
         img/                generated images, plus the source photo
evidence/CLAIMS.md           every number on every slide, traced to its source file
old/     JSW_Round2_v1..v4_*   earlier versions (detailed, clean, compact, synced)
         JSW_Round2_v1_Detailed.*   8 slides, full technical detail (25 Sep)
         JSW_Round2_v2_Clean.*      8 slides, plain language, no visuals (26 Sep)
         build_round2_v1_detailed.js / build_round2_v2_clean.js   their sources
JSW_Round2_Video_Script.md   148-second script, if you submit video instead
stainless_test/              photo guide + run_stainless_test.py (same model and settings as the live demo)
interviews/                  who to call, messages, questions, consent; quotes.json feeds the deck
```

## Image credits

- **Cover photo:** "Rolled coil in Oulu Jun2009 001.jpg", Methem (Mikko J. Putkonen),
  Wikimedia Commons, public domain. Cropped and desaturated. Credited on the slide.
- **Defect images:** held-out test frames from NEU-DET (Northeastern University) and
  Severstal steel defects (Kaggle); boxes and labels drawn by our model. GC10-DET frame:
  Lv et al., *Sensors* 2020, CC BY 4.0, raw (contrast-stretched).
- **Icons:** Lucide (lucide.dev), ISC licence.

## Version history

- **v14 (27 Sep), the software story — current.** Nothing removed; panels re-laid out to make room.
  Slide 5: every stage of the ribbon now names the software behind it (tiling, steel check, YOLOv8n detector,
  calibrated confidence, coil report, override log), plus a "two apps, one model" panel (browser demo and
  Python operator console) and the software stack in the source line. Slide 6: a seven-step "how we built it"
  pipeline (collect, split, train, select, calibrate, export, verify). Slide 7: a monthly retrain-and-release
  track in the roadmap. Slide 4: calibrated confidence as a trust point and the coil report's starting limits.
- **v13 (27 Sep), images on slides 4, 7, 8.** Slide 4: real mill-strip images beside the result rings
  (a flagged defect, a clean strip with no alarm) and the live demo's alarm view for "evidence on every alarm".
  Slide 7: mill photo heading the pilot scope, icons on the roadmap and KPI tiles, shorter labels. Slide 8: photo
  tiles for the six impact areas, proof points moved to the speaker notes.
- **v12 (27 Sep), more visuals, less text.** Text panels became pictures and diagrams: photo tiles on
  slide 2 (mill, coil, stainless lift), photo cards for the precedents and a Vande Bharat coach for the pilot grade
  on slide 3, icon tiles on slide 5, a risk heat-map on slide 7, and a value tree on slide 8. Photos from
  Wikimedia Commons (credits in `build/img/photos/CREDITS.txt` and the speaker notes).
- **v11 (27 Sep), consultants, not a price quote.** Cost predictions removed: no per-line capex,
  bill of materials, payback years, NPV, cash chart or commercial price guesses. The deck sizes the prize
  (value at stake, from JSL's public numbers) and shows how the pilot measures it: cover shows ~₹6.7 cr a year
  at stake per line and 7 months to shadow mode; slide 7 has the pilot scope (in / later) and decision owners;
  slide 8 has five value levers with their pilot measures. `build/econ.py` is kept for Q&A only.
- **v10 (27 Sep), Jindal Stainless scheme.** Restyled to match the team's Round 1 slide: orange
  header band with the Jindal Stainless and Stainless Spark logos, red active tab, charcoal left rail with a
  rotated section label, orange panel headers, charcoal tables and decision band, orange chart ramp.
  Colours sampled from the Round 1 slide (#E4803A, #B63831, #2B2B2B). Logos: `build/img/logo_jsl.jpg`
  (JSL-Black-1.jpg) and `build/img/logo_spark.png` (cropped from the Round 1 slide; no original file found).
- **v9 (27 Sep), no model internals.** The deck says what the system delivers, never how the
  model works: no model size, speed per image, calibration, training method or retraining. Slide 4's
  "compact model" panel is now "one verdict per coil" (coil → accept / downgrade / hold), and its trust
  panel shows how trust is earned (evidence, shadow mode, reviewed overrides). Fixed "six different owners".
- **v8 (27 Sep), Team Futuristic.** Team name on the cover and every header. Two hooks that
  fill themselves in: `stainless_test/` (drop photos, run one command → slide 6 gains a "Stainless (ours)"
  card) and `interviews/` (add real quotes to `quotes.json` → slide 2 gains a quote band and slide 3's
  "team hypothesis" labels become "checked with <role>"). With neither present, the deck is unchanged.
- **v7 (27 Sep), dense.** Same 8 slides, 2,267 words (max 345 per slide), a data visual
  or diagram added to every content slide: demand chart (2), stainless-only defect chips (3),
  resolution trade-off (4), where-each-stage-runs map (5), dataset split bar (6), gate diamonds and
  labelling volume (7), five-year cumulative cash chart and proof line per impact tile (8).
- **v6 (27 Sep), research-backed.** Same 8 slides. Adds a "Why now" panel (demand
  4.8 → 6.8 Mt, imports and the 2025 anti-dumping probe, mandatory BIS IS 6911), a precedent panel
  (Outokumpu, Tata Steel, Parsytec on stainless), stainless optics (bright/dark field, deflectometry),
  Pragati Level-2 integration, IEC 62443 security and gated retraining in the risks, synthetic rare
  defects, and customer and emissions context. Money in plain crore, international digit grouping.
  Sources in `evidence/RESEARCH_ADDENDUM.md`.
- **v5 (26 Sep), final-state only.** Same 8 slides as v4. No comparisons between our
  own versions: the cover and slide 4 state what the final system does (90% of defects caught,
  4.7% of clean strip flagged, 12 MB, ~55 ms, confidence within ±3 points). Comparisons that are
  choices for JSL (commercial vs in-house vs our hybrid, payback) stay.
- **v4 (26 Sep), synced to the live demo.** Same 8 slides as v3; every measured
  number now matches the detector live at surface-vision.github.io. That model was retrained on
  real clean and defective mill strip and chosen on validation across four training runs.
  Headline change: clean strip wrongly flagged 52% → 4.7%, mill defects caught 46% → 90%.
- **v3 (26 Sep), compact + visual.** Still 8 slides, words 2,732 → 2,056 (−25%),
  no slide over 310 words. Risk slide folded into Implementation, freeing a slide for a new
  proof slide (datasets, per-defect accuracy, weak spots). Added the value-chain ribbon, the
  real-detection gallery (replaces the root-cause table), native charts (false alarms, cost
  doughnut), a processor pictogram, icons throughout, and a cover photo.
- **v2 (26 Sep), clean.** Same 8 slides as v1 in plain language for an external audience:
  software and model names, test counts, RACI and the line-item BOM removed.
- **v1 (25 Sep), detailed.** Rebuilt the 22 Sep draft (`deck/round2/`, untouched) so it
  could be regenerated; team details moved to one config block; two numbers corrected;
  claim ledger added.

A video was rendered on 22 Sep (`Claude outputs/JSW_Round2_Video.mp4`) from the original
draft. It has not been re-cut against v3.
