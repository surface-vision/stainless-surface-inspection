# Stainless test — our own photos

Every public steel dataset is carbon steel. This test puts real **stainless** in front of the
model, so the first question from Jindal Stainless ("has it seen stainless?") has an answer.

## 1. What to collect (target: 60+ photos, about 1 hour)

| Aim | Count |
|---|---|
| Stainless with a **visible defect** (scratch, dent, pit, stain, rust spot, roll mark, weld heat tint) | 40+ |
| **Clean** stainless, no visible defect | 20+ |
| At least **two finishes**, ideally all three below | — |

| Finish | What it looks like | Where to find it |
|---|---|---|
| **2B** | dull, matte grey, slightly reflective | sheet offcuts at fabricators and steel stockists, sink undersides, ducting |
| **BA** | mirror-like | appliance trim, kitchen backsplashes, lift panels, BA sheet offcuts |
| **No4 / HL** | brushed, fine parallel lines | fridge and dishwasher doors, lift doors, counters, brushed sheet |

Good sources: sheet-metal fabricators and kitchen-equipment workshops (they have offcuts and
reject pieces), scrap dealers, the IIT Bombay central workshop. **Flat sheet is better than
curved objects.** Utensils are a last resort.

## 2. How to shoot

- Phone camera, **20–30 cm** from the sheet, the sheet **filling the frame**. No zoom, HDR off.
- **Soft, even light** — shade, an overcast window, or a white paper reflector. No direct sun.
- For mirror-like **BA**: tilt the phone **10–15°** so you don't photograph your own reflection.
- For every defective piece, take **one extra shot with a torch held low at a grazing angle** from the
  side (this is "dark-field" light, the way inspection lines light stainless). Name it with `-dark`,
  e.g. `BA_scratch_04-dark.jpg`.
- Keep a ruler or coin in a few shots for scale (optional).

## 3. How to name the files

```
<finish>_<label>_<nn>.jpg        finish: 2B, 2D, BA, No4, HL or other
                                 label:  clean, or what you see (scratch, dent, pit, stain, rust, rollmark, heattint)
examples:  2B_scratch_01.jpg   BA_clean_03.jpg   No4_dent_02.jpg   BA_scratch_04-dark.jpg
```

Put them all in `Round2/stainless_test/photos/`.

## 4. Run it

```bash
cd ~/Downloads/JSW-PS1
.venv/bin/python Round2/stainless_test/run_stainless_test.py
```

It runs the **same model the live demo runs**, the same way (greyscale, square windows, 320 px,
confidence 0.15), and writes to `Round2/stainless_test/out/`:

- `summary.json` — defects flagged / clean flagged, overall and per finish
- `annotated/` — every photo with boxes drawn
- `stainless_sheet.png` — 8 photos on one image, ready for a slide
- `results.json` — every detection

Then rebuild the deck (`cd Round2/build && node build_round2.js`): slide 6 **automatically gains a fourth
dataset card, "Stainless (ours)"**, with a real annotated photo and your numbers.

## 5. Rules that keep it credible

- **Keep every photo you take**, including ones where the model does badly. Judges trust a reported
  weakness; they do not trust a hand-picked set.
- Label from what you **see**, before running the model, not after.
- Expect weak results on mirror-like BA, and on defect types the model has never been taught.
  That is a finding, not a failure: it is exactly why the pilot trains on JSL's own strip.

## What a dry run showed

`_dryrun_commons/` ran three public-domain stainless images (Wikimedia Commons, CC0) through the tool
to prove it works end to end. Those images are not used anywhere in the deck. On worn stainless the
model raised alarms but called them "Unclassified defect" with wide boxes instead of naming the
scratches, and it also flagged a dishcloth, because the demo has no "is this steel?" check. Your photos
will show how large that gap is on real sheet.
