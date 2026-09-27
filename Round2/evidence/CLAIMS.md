# Round 2 claim ledger

> **26 Sep, v4 (8 slides) — current.** The deck now quotes the model that is live at
> surface-vision.github.io, chosen on validation across four independent training runs
> (`reports/demo_seed_selection.json`) and exported at 320 px. Slide numbers below refer to
> the v1 layout: v1 slides 6-7 merge into slide 7, v1 slide 8 is slide 8, slide 6 is new (proof).
>
> **Shipped-model numbers, all on the untouched test split** (`reports/demo_eval_joint_seed1337.json`,
> `reports/demo_model.json`, produced by `scripts/eval_demo_model.py` and `scripts/build_demo_model.py`):
> - Clean mill strip wrongly flagged: 155/300 (51.7%) before → **14/300 (4.7%)** after — cover, slides 4, 5, 7
> - Mill-strip defects caught: 133/288 (46.2%) → **258/288 (89.6%)** — cover, slides 4, 7
> - Right defect type: **178 of 180** — slide 6
> - Box accuracy: lab **0.745**, real mill strip **0.566** — slide 6 ("0.57 vs 0.75")
> - Per defect (lab, held-out): patches 0.942, inclusion 0.795, scratches 0.794, pitted 0.763,
>   rolled-in scale 0.597, crazing 0.581 — slide 6 bars
> - Confidence error: 0.0822 → 0.0331 after calibration = **60% lower**, gap **3.3 points** — slides 4, 7
> - ~55 ms per image in a laptop browser (measured in Chrome on the live site) — slide 6
> - Seed spread, best validation score: live 0.699, seed1 0.673, seed2 0.666, seed3 0.663 —
>   the live model won, so nothing was swapped.
>
> **Superseded:** the earlier 93.7% → 32.5% pair measured clean *frames* (1600x256, ~6 crops each)
> under `reports/gap1_cross_domain.json`. It remains correct for that unit and is consistent with
> 4.7% per crop, but the deck and the site now both quote the per-image figure so they agree.


> **26 Sep:** the slides now say several of these in plain words, and some technical
> figures (test count, model size, parity check, raw metric names) are no longer on the
> slides at all. They stay here as backup for Q&A.

Every number on a slide, traced to the file it comes from. Re-verified 25 Sep 2026
against the repository, before this build. A judge who asks "where does that come
from?" should be answerable in one line.

Legend: **measured** = produced by our code, reproducible from `reports/*.json`.
**estimate** = our assumption, labelled as such on the slide. **external** = cited source.

## Slide 1 — cover

| Claim | Kind | Source | Verified value |
|---|---|---|---|
| mAP50 0.764, joint model | measured | `reports/gap1_cross_domain_fix.md:252` | 0.7642 |
| 93.7% to 32.5% clean-frame false alarms | measured | `reports/gap1_cross_domain_fix.md:341,351` | 93.7% → 32.5% at conf 0.15 |
| INR 2.5–4.2 cr per line | estimate | `build/econ.py` BOM, slide 6 | capex mid 3.35 cr |
| ~2.2 years payback | estimate | `build/econ.py` → `mid_payback` | 2.220 yr |

## Slide 2 — problem

| Claim | Kind | Source | Verified value |
|---|---|---|---|
| 300 m/min, human eye cannot inspect whole surface | external | AMETEK Surface Vision / Ternium case study (2020) | quoted |
| INR 1.26–1.65 lakh nickel per tonne of 304 | external + arithmetic | ASTM A240 (8.0–10.5% Ni); LME USD 16,402/t; INR 95.8/USD | `econ.py` → 125,705 / 164,988 |
| 2.57 Mt at INR 1,67,407/t | external | JSL FY26 results | 25,65,902 t |
| Cost of poor quality ~15% of sales (5–35%) | external | IISE, *Measuring the Cost of Quality* | quoted |

## Slide 3 — insights (stainless context)

| Claim | Kind | Source |
|---|---|---|
| 2B/BA finish damage cannot be polished out | external | ASSDA finish guidance |
| Cold rolling 2.05 → 2.67 MTPA by FY28; exports 11% | external | JSL Q1 FY27 earnings call and results (Aug 2026) |
| Grade × defect risk matrix (H/M/L) | **team hypothesis** | labelled on slide; to be re-scored in P0 against JSL defect records |
| Root-cause routing table | external + hypothesis | Leão et al. 2021 (tundish slivers); Steel in Translation 2023 (rolled-in scale) |

The matrix is the one place we assert something we have not measured. It is labelled
as a hypothesis on the slide, on purpose.

## Slide 4 — insights (our four findings)

| Claim | Kind | Source | Verified value |
|---|---|---|---|
| 93.7% of 505 clean Severstal frames flagged | measured | `reports/gap1_cross_domain_fix.md:31` | 93.7%, CI [91.5%, 95.6%] |
| Joint training → 32.5%, mAP50 0.752 → 0.764 | measured | same, lines 341, 564 | +0.0118, FA −61 points |
| Clean-only training collapses recall to 0.057 | measured | same, line 51 | 0.057 |
| YOLOv8n 256 px = 0.752 | measured | `reports/evaluation.json` → `detection_metrics.mAP50` | 0.75245 |
| YOLOv8s no gain (0.734), 3.7× params | measured | `reports/model_study.json` | val best 0.7355 |
| TTA −0.006 at 2.39× latency | measured | `model_study.json` → `latency/yolov8n/tta_paired` | 2.39× |
| Nano vs small 95% CI [−0.008, +0.042] | measured | `model_study.json` bootstrap | spans zero |
| 6.22 MB model | measured | `reports/export_summary.json` → `sizes_mb` | 6.22 |
| 3,064 inferences/s at 250 m/min, ~18 accelerators | measured | `reports/benchmark.json` → `mill/scenarios[2]` | 3,063.7; 18 |
| 48/s if downscaled to 1.28 mm/px | measured | same scenario | 47.87 |
| ECE 0.142 → 0.046 (−68%) | measured | `reports/calibration.json` → `metrics.ece` | 0.1418 → 0.0461 (−67.5%) |
| 0 of 3,200 real steel frames rejected by OOD gate | measured | `reports/ood_guard.json` → `ood_guard/real_steel` | 1440+180+180+1400 = 3,200; rejects 0 |

## Slide 5 — solution

| Claim | Kind | Source | Verified value |
|---|---|---|---|
| 548 tests | measured | `pytest --collect-only -q` | 548 collected (was 540 at the Sep 22 build) |
| Browser ONNX matches Python 40/40 detections | measured | `site/verify/parity_browser.json` | 12 images, 40 matched pairs, max box delta 0.14 px |
| SimSiam 0.768 vs ImageNet 0.773 mAP50 on steel | external | J. Mater. Inf. 2025 | quoted |
| Commercial systems USD 1–3M | external, **unverified** | industry range; labelled unverified on slide |

## Slide 6 — implementation

| Claim | Kind | Source | Verified value |
|---|---|---|---|
| Retrain in 0.6 h | measured | `model_study.json` → `seconds_per_epoch_median` | 15.24 s × 135 epochs = 0.57 h |
| Bill of materials, INR 250–420 lakh | **estimate** | team BOM at INR 95.8/USD; labelled on slide | to be replaced by vendor quotes in P0 |
| RACI, stage gates, integration layers | proposal | ours |

## Slide 7 — risk and KPIs

| Claim | Kind | Source | Verified value |
|---|---|---|---|
| Rolled-in scale AP50 0.63 | measured | `reports/evaluation.json` per-class | 0.6295 |
| Roll marks AP50 0.20 | measured | `reports/gc10_coverage.md:25` | mean 0.20 across 24 test instances |
| 92.2% defective frames flagged | measured | `reports/false_alarm.md:245` | 92.2% at conf 0.15 |
| ECE 0.046 | measured | `reports/calibration.json` | 0.0461 |
| KPI targets (≥90%, ≤25%, ≤200 ms) | **proposed** | to be agreed with JSL Quality in P0 |

## Slide 8 — impact

All from `build/econ.py`, re-run 25 Sep 2026:

| Claim | Verified value |
|---|---|
| Loss pool grid (3.3 … 26.8 cr) | matches `grid_line_cr` exactly |
| Base case 6.7 cr pool, 30% recovered = 2.0 cr, 1.5 cr net | 6.7 × 0.30 = 2.01; − 0.5 run cost = 1.51 |
| Payback 2.2 yr | 2.220 |
| Commercial system 12.7 yr | 12.698 (USD 2M at INR 95.8 = 19.16 cr) |
| 5-year NPV +INR 2.1 cr at 12% | 2.089 |
| Breakeven ≥0.11% of line revenue | 0.001083 |
| Fleet: 43 cr pool, 12.9 cr/yr recovered, ~6.4 lines | 42.95; 12.89; 6.41 |

## Corrections made in this build

Two numbers carried over from the 22 Sep build were changed:

1. **540 → 548 tests.** The suite grew; 548 is what `pytest --collect-only` reports today.
2. **"ONNX export verified 40/40 detections" → "browser ONNX matches Python 40/40 detections."**
   The 40/40 figure is browser-vs-Python parity across 12 images
   (`site/verify/parity_browser.json`), not a 40-image export check. The old wording
   overstated what was tested.

## What we deliberately do not claim

- No claim about JSL's actual downgrade rate, inspection coverage, or claim volume.
  None of it is public. Each is asked for as a P0 input instead of assumed.
- The grade × defect matrix is a hypothesis from metallurgy literature, not JSL data.
- Commercial system pricing is an unverified industry range, labelled as such.

## v7 additions (27 Sep) — derived numbers, shown with their working

- **Slide 2 demand chart:** ISSDA consumption FY21 2.61, FY22 3.46, FY23 3.94, FY24 4.49, FY25 4.80 Mt;
  FY30 6.8 Mt is a forecast, drawn lighter and labelled FY30F. Source in `RESEARCH_ADDENDUM.md` §1.
- **Slide 4 resolution trade-off:** processors scale with pixel count, i.e. with (1 / resolution)².
  Measured anchor: ~18 processors at 0.2 mm/px (`reports/benchmark.json`, 3,063.7 inferences/s).
  0.5 mm → 18 × (0.2/0.5)² = 2.9 ≈ **3**; 1.28 mm → 0.44 → **1** (matches the measured 47.9/s downscaled case).
  Smallest visible defect uses the machine-vision rule of thumb of 2-3 pixels: 0.5 / 1.3 / 3.2 mm.
- **Slide 6 dataset split:** 1,800 (NEU-DET) + 6,001 (Severstal crops used) + 2,294 (GC10-DET) = 10,095.
- **Slide 8 cumulative cash:** base case net benefit = 0.4 Mt × 1% × 10% × ₹167,407/t × 30% − ₹0.5 cr
  = **₹1.509 cr/yr**. Our solution: −3.35, −1.84, −0.33, +1.18, +2.69, +4.19 (Y0-Y5, ₹ cr).
  Commercial (USD 2M = ₹19.16 cr): −11.6 after five years. Same model as `build/econ.py`.
- **Slide 8 proof lines:** "HR AI: 95% self-served" = JSL's AI assistant resolves >95% of routine HR queries
  (People Matters Infini-T Awards 2026, `docs/research_notes.md` §3d, secondary). "Ternium's business case" =
  strip-break avoidance in the AMETEK/Ternium case study.

## v14 additions (27 Sep) — the software story, back on the slides

The user asked for the software to be described again (reversing the 26 Sep "no model internals" rule).
Every new figure, and where it comes from:

- **Training split 3,769 / 494 / 768** (train / tune / final test); **1,190 clean** training images;
  test = 180 lab + 588 mill (288 defective, 300 clean) — `data/joint_xdsafe/manifest.json` — slide 6.
- **YOLOv8n, ~3 M parameters (3,012,018 for the 6-class head), 10 classes, 320 px input** —
  `reports/export_summary_320.json`, `reports/demo_model.json` — slide 5.
- **Up to 200 training passes, early stop after 40 without gain, seed 1337** — `models/yolov8n_joint/args.yaml` — notes only.
- **4 training runs, best picked on tuning (validation) data** — `reports/demo_seed_selection.json` — slide 6.
- **Calibration 8% → 3%**: expected calibration error 0.0822 → 0.0331, isotonic, fitted on 1,181 validation
  detections, scored on 1,637 test detections — `reports/demo_model.json` — slides 5, 6.
- **12 MB model file** (12.1 MB ONNX) — `reports/demo_model.json`, `site/model/detector.onnx` — slides 5, 6.
- **37 of 37 detections match** browser runtime vs Python on 12 images; max box gap 0.23 px, max confidence gap
  0.0024 — `site/verify/parity_report.json` — slides 5, 6.
- **400 automated tests** — `grep -c "def test_" tests/*.py` (400 on 27 Sep) — slide 6.
- **~50 ms an image in the browser** — measured in Chrome on the live site 27 Sep: 41-67 ms
  (`site/verify/click_upload_check.mjs`, `site/verify/page_check.mjs`) — slide 5.
- **Wide frames split into up to 12 tiles** — `site/js/tiling.js` (`MAX_TILES = 12`) — notes.
- **Steel check first** — `hf_space/app.py` runs `ood_guard` before the detector; the browser demo has no steel
  check (the dry run in `stainless_test/README.md` shows it) — slide 5 names it only under the console.
- **Coil report starting limits**: accept ≤ 2% defective frames with nothing high or critical; hold ≥ 25%, any
  critical frame, or any inclusion — `reports/coil_report.json` `rules` — slide 4.
- **Operator console features** (steel check, whole-coil batch, coil verdict with reasons, line-speed simulation,
  EigenCAM heat map) — `hf_space/README.md`, `hf_space/app.py` — slide 5.
