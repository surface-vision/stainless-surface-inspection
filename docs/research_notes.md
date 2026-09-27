# RESEARCH NOTES — AI Surface Defect Detection for Jindal Stainless
**Compiled 2026-09-09. Confidence legend: `[HARD]` = number appears verbatim in a named source · `[VENDOR]` = vendor/marketing self-claim, directionally useful, not independent · `[SECONDARY]` = reported via search summary, primary page not fully verified · `[DERIVED]` = my own arithmetic from sourced inputs, shown with the inputs · `[ESTIMATE]` = my judgement, not sourced · `[SUSPECT]` = published but implausible or unverifiable — do not put on a slide as-is**

---

## 1. PUBLISHED BENCHMARKS ON NEU-DET

### 1a. The honest headline: the credible band is mAP50 ≈ 0.70–0.80

This is the single most important framing fact for the deck. Well-run baseline detectors land in the low-to-mid 70s; strong published improvements land 78–85. Papers claiming 95%+ on NEU-DET *detection* are almost always measuring something else (classification on NEU-CLS, a non-standard split, or augmented data leaking between train and test).

### 1b. Master comparison table

| Method | mAP50 | mAP50-95 | Params (M) | FPS | Year | Source | Conf |
|---|---|---|---|---|---|---|---|
| **Faster R-CNN (ResNet34/50 baseline)** | 70.2 / 77.9 | — | — | — | 2020 | He et al., IEEE TIM 69(4):1493 | `[DERIVED]` from "DDN is 4.6/4.4 higher" |
| **DDN (MFN + ResNet34/50)** | **74.8 / 82.3** | — | — | — | 2020 | He, Song, Meng, Yan, IEEE TIM 69(4):1493 | `[HARD]` |
| SSD | 75.4 | — | 24.1 | — | 2025 | PMC11820220 Table 3 | `[HARD]` |
| Fast R-CNN | 76.7 | — | 137.3 | — | 2025 | PMC11820220 Table 3 | `[HARD]` |
| DETR | 66.6 | — | 36.7 | — | 2025 | PMC11820220 Table 3 | `[HARD]` |
| YOLOv7 | 71.5 | — | 37.2 | — | 2025 | PMC11820220 Table 3 | `[HARD]` |
| YOLOv8n | 78.6 | — | 3.20 | — | 2025 | PMC11820220 Table 3 | `[HARD]` |
| YOLOv10n | 79.2 | — | 3.00 | — | 2025 | PMC11820220 Table 3 | `[HARD]` |
| **YOLOv10n-SFDC** | **85.5** | — | 2.67 | — | 2025 | PMC11820220 (proposed) | `[HARD]` |
| SSD | 72.9 | 33.8 | 50.2 | 106 | 2026 | PMC12736593 | `[HARD]` |
| Faster R-CNN | 73.1 | 37.5 | 136.7 | 32 | 2026 | PMC12736593 | `[HARD]` |
| RT-DETR | 75.4 | 43.3 | 19.8 | 144 | 2026 | PMC12736593 | `[HARD]` |
| YOLOv5s | 77.6 | 39.2 | 7.1 | 153 | 2026 | PMC12736593 | `[HARD]` |
| YOLOv8n | 74.0 | 43.0 | 3.2 | 144 | 2026 | PMC12736593 | `[HARD]` |
| YOLOv11n | 77.2 | 45.8 | 2.6 | 156 | 2026 | PMC12736593 | `[HARD]` |
| RDD-YOLO | 78.3 | 43.3 | 74.2 | 57 | — | via PMC12736593 | `[HARD]` |
| WSS-YOLO | 79.1 | 46.9 | 4.5 | 69 | — | via PMC12736593 | `[HARD]` |
| **LCED-YOLO** | **79.8** | 46.8 | 2.1 | 151 | 2026 | PMC12736593 (proposed) | `[HARD]` |
| YOLOv8 baseline | 73.0 | — | — | — | 2025 | Dynamic-YOLO paper | `[SECONDARY]` |
| Dynamic-YOLO | 75.1 | — | — | — | 2025 | same | `[SECONDARY]` |
| YOLOv8l / n / s / m | 74.1 / 73.6 / 73.3 / 73.2 | 39.1–40.2 | — | — | 2025 | ScienceDirect S1566253525011030 | `[SECONDARY]` |
| **LESSDD-Net** (on *augmented* NEU-DET) | 75.37 | — | 1.55 | 131 | 2026 | PMC12899440 | `[HARD]` — note: augmented split, not comparable |
| Faster R-CNN + SimSiam self-supervised pretrain | 76.8 | 38.5 | — | — | 2025 | oaepublish jmi.2025.21 | `[HARD]` |
| Faster R-CNN + ImageNet pretrain | 77.3 | 38.0 | — | — | 2025 | same | `[HARD]` |
| Faster R-CNN, random init | 28.0 | 8.8 | — | — | 2025 | same | `[HARD]` |

### 1c. Per-class AP — the most useful table you have

From **Maity & Ghosh, "Comparative Analysis of Object Detection Algorithms for Surface Defect Detection", arXiv:2510.21811 (Oct 2025), Adamas University**. I extracted Table 1 directly from the PDF text stream, so these are verbatim. Note: 70/20/10 split, out-of-the-box models, no per-class tuning — which is exactly why the numbers are low and honest. Per-class values are AP@[.5:.95].

| Metric | YOLOv11 | Faster R-CNN | RetinaNet | RT-DETR | YOLOv8 | DETR |
|---|---|---|---|---|---|---|
| Overall AP (50:95) | **38.6%** | 9.7% | 21.1% | 21.0% | 35.9% | 11.6% |
| **AP@IoU 0.50** | **71.6%** | 30.1% | 46.2% | 55.0% | 68.7% | 25.2% |
| AP Crazing | 21.3% | 8.0% | 12.8% | 18.2% | 17.6% | 8.2% |
| AP Inclusion | 44.1% | 2.9% | 21.1% | 27.6% | 39.7% | 14.3% |
| AP Patches | 60.5% | 17.3% | 47.3% | 14.9% | 57.4% | 6.9% |
| AP Pitted Surface | 43.8% | 11.8% | 31.7% | 31.4% | 43.4% | 17.8% |
| AP Rolled-in Scale | 23.0% | 11.5% | 13.9% | 19.7% | 22.3% | 10.1% |
| AP Scratches | 39.1% | 6.9% | **0.0%** | 22.8% | 34.9% | 12.5% |

**Class difficulty ordering, consistent across sources** `[HARD]`: **patches** easiest (0.90–0.97 AP50), then **pitted surface** and **scratches**; **inclusion** mid; **crazing** and **rolled-in scale** consistently hardest. Both papers state this explicitly. This is a great slide — it maps directly onto physics (crazing = fine low-contrast crack network; rolled-in scale = low-contrast texture against a scaled background).

### 1d. Claims to treat as SUSPECT — do not cite

- **"99.5% mAP@50 with EfficientNet backbone in YOLOv8"** (IOS Press FAIA251752) `[SUSPECT]` — implausible for NEU-DET detection.
- **"98.5% mAP, >120 FPS on Jetson Orin"** (arXiv:2606.07659, "Industrial-YOLO") `[SUSPECT]` — I fetched the PDF twice and could not extract the results tables (font-subset encoded); the abstract does not break results out by dataset, and it mixes NEU with MVTec AD plus private automotive data. **Could not verify. Flagged.**
- The per-class table I got back from PMC12736593 via WebFetch listed crazing at 92–94% and patches at 57–74%, which inverts the ordering every other source reports — the row labels were almost certainly mis-associated during extraction. **Use the arXiv:2510.21811 per-class table instead; I extracted that one from raw PDF text myself.**

### 1e. Positioning guidance for the deck
Anything you achieve in **0.72–0.78 mAP50** with a stock YOLOv8s is *squarely competitive with published baselines* and roughly at the level of a 2020 IEEE TIM paper (DDN, 0.748 on ResNet34). Say that plainly. Claiming 0.95+ would immediately mark the work as unserious to anyone who knows the dataset.

---

## 2. STEEL SURFACE INSPECTION IN INDUSTRY

### 2a. Best primary source found — AMETEK / Ternium case study
`[HARD]` — all from AMETEK Surface Vision "Ensuring Steel Quality" metals case study (2020), PDF text extracted directly:

- Site: **Ternium Pesqueria, Monterrey, Mexico** — Pickling Line Tandem Cold Mill (PLTCM), **five stands, six-high**, automotive-focused.
- **Rolling capacity 6,000 tonnes per day.**
- **"The steel strip passes in front of the quality inspector at 300 metres per minute, so it is impossible for the human eye to inspect the whole surface."** ← *This is your single best quote for the deck. A vendor and a mill both saying human inspection does not work at line speed.*
- SmartView configuration: **four cameras, two each side, bright field mode, installed just after pickling.**
- Business case was **strip breakages**, not cosmetics: defects caused breakages that "caused significant damage to the mill equipment." System was wired to the **PLC to slow the mill before the first mill bite** when a severe defect was detected under certain thickness conditions, plus visual and acoustic operator alarms.
- Maintenance engineer, quoted: *"If the SmartView system was not in service for any reason, we simply couldn't operate the line for thin gauge or exposed quality steel — the risk would be too high."*
- Ternium context: 12.4 Mt/yr capacity, 17 sites.
- **Domain shift, admitted by the vendor** `[HARD]`: *"As the line processes material from different hot mills, the features for the same class of defect are different, so this variable can make detection more challenging."* ← Use this in section 6; it is far stronger than citing an academic paper for the same point.
- Operational reality: *"unclassified defects are not shown to the quality operators or inspectors"* — i.e. real systems have a meaningful unclassified bucket, owned by a process engineer.

### 2b. ISRA VISION / Parsytec (Atlas Copco)
- Matrix cameras at **170 µm resolution**, bright and dark field lighting `[VENDOR]` — isravision.com / AIST release.
- **PEARL** deep-learning module "increases optical inspection accuracy by **up to 10%**" under high temperature / complex surfaces (Salzgitter Flachstahl case) `[VENDOR]`.
- **"Up to 60% of all quality-relevant surface defects originate in the slab"** `[VENDOR]` — strong argument for inspecting *early and at multiple points*, not just at despatch.
- **450+ SURFACE MASTER installations worldwide** `[SECONDARY]` — from search summary; I could not confirm on an ISRA primary page.
- Claimed to be used by **all of the top 10 steel producers** `[VENDOR]`.
- Named steel customers in public releases: Hyundai Steel, Shandong Steel, Salzgitter, Tata Steel, POSCO, China Steel Corp, ArcelorMittal Eisenhüttenstadt, Samuel Steel Pickling.

### 2c. Cognex
- **In-Sight 9000**: 2,000 pixels per line, 32 MP images `[VENDOR]`. In-Sight 3800 line scan for high-speed lines.
- **ViDi / Deep Learning suite**: ViDi Red (segmentation + anomaly detection), ViDi Green (classification), ViDi Blue (localisation) `[VENDOR]`. Note ViDi Red does **unsupervised anomaly detection** — relevant to your section 6 story about unseen defects.
- No public pricing. Cognex positions as component vendor, not turnkey mill-wide ASIS.

### 2d. System cost
- **~USD 2.4M for a machine-vision surface inspection installation on cold-rolled/galvanised lines**, plus **USD 170k–370k** to integrate with plant systems; **ROI in 6–10 months** `[SUSPECT — SEO content farm]`. Source is oxmaint.com, which alongside ifactoryapp.com and ifactory.jrsinnovation.com is near-certainly AI-generated SEO content with fabricated case studies. **The $2.4M order of magnitude is plausible and consistent with my own expectation** `[ESTIMATE]` **for a multi-camera, both-surfaces, PLC-integrated ASIS on one line, but I could not source it to any vendor, mill, or analyst. Do not put a dollar figure on a slide attributed to a source.** If you need a number, present it as "order USD 1–3M per line, industry range, unverified."
- Market context `[SECONDARY]`: surface inspection market **USD 4.86bn (2024) → USD 9.0bn (2032)** (MarketsandMarkets); a second firm says USD 6.80bn (2024) → 14.70bn (2032). The spread between research houses is itself worth noting.

### 2e. Line speeds and strip widths `[HARD]` unless noted
- **Cold mill / pickling line inspection point: 300 m/min** (Ternium, AMETEK).
- **Hot strip mill exit: up to ~1,200 m/min** for thin gauge (Britannica gives "often 100 km/h" ≈ 1,667 m/min for small cross-sections; a Nippon Steel technical report cites **700 m/min for ~1.6 mm strip**). A patent cites ~250 m/min for packaging-grade band and <750 m/min preferred exit speed. **Use 250–1,200 m/min as the honest hot-mill band** `[ESTIMATE]` from these bracketing sources.
- Strip widths: typical flat-product band **1,000–1,600 mm** `[ESTIMATE]`; Jindal's own product brochure would confirm exact widths — I did not verify a Jindal-specific figure.

### 2f. Required inspection throughput — `[DERIVED]`, arithmetic shown
Inputs: line speed from AMETEK case study; 170 µm/pixel from ISRA. Formula: line rate = speed / pixel pitch; throughput = pixels-across × line rate.

| Scenario | Speed | Width | Px across | Line rate | Throughput (1 face) | Both faces | 200×200 tiles/s (both faces) |
|---|---|---|---|---|---|---|---|
| Ternium PLTCM (sourced speed) | 300 m/min | 1,250 mm | 7,353 | 29.4 kHz | **216 Mpix/s** | 433 Mpix/s | **10,813** |
| Fast cold/finishing line | 600 m/min | 1,600 mm | 9,412 | 58.8 kHz | 554 Mpix/s | 1,107 Mpix/s | 27,682 |
| Hot strip mill exit | 1,200 m/min | 1,600 mm | 9,412 | 117.6 kHz | 1,107 Mpix/s | 2,215 Mpix/s | 55,363 |

**This is the strongest technical slide in the deck.** At the *slowest*, best-sourced scenario, full-surface inspection means processing the equivalent of **~10,800 NEU-DET-sized 200×200 tiles every second**. A single 200×200 tile is exactly what our model consumes. That reframes the engineering problem honestly: the model is the easy part; **tiling, sharding across GPUs, and triage are the real system.** It also justifies choosing a small fast model (YOLOv8s, ~3–11M params, 140–160 FPS on full frames) over a heavy two-stage detector.

---

## 3. JINDAL STAINLESS

### 3a. Financials — all `[HARD]`, jindalstainless.com press releases

| Metric | FY25 | FY26 | Change |
|---|---|---|---|
| Net revenue (consolidated) | — | **INR 42,955 cr** | +9.3% YoY |
| Net revenue (standalone) | INR 40,182 cr | — | — |
| EBITDA (consolidated) | — | **INR 5,560 cr** | +19.2% |
| PAT (consolidated) | — | **INR 3,185 cr** | +27.4% |
| EBITDA / PAT (FY25) | 3,905 cr / 2,711 cr | — | — |
| Sales volume | 23,73,070 t (+9% over FY24) | **25,65,902 t** | +8.1% |
| Exports | — | 8% of sales (Q4: 7%) | — |

Q4 FY26: revenue INR 11,337 cr (+11.2%), EBITDA INR 1,455 cr (+37.1%), PAT INR 834 cr (+41.4%).
Annual turnover **USD 4.86 bn (FY26)** `[SECONDARY]` — Wikipedia.

**Derived per-tonne economics** `[DERIVED]` from FY26 consolidated figures above:
- **Blended realisation ≈ INR 167,400 per tonne**
- **EBITDA ≈ INR 21,670 per tonne** (12.9% margin)

These two numbers are the backbone of any ROI slide, and they come straight from the company's own reported figures — unattackable.

### 3b. Capacity and plants — `[HARD]`
- **Melt capacity now 4.2 MTPA** globally (3.0 MTPA India + 1.2 MTPA Indonesia JV, commissioned; announced Mar 2026).
- **Jajpur, Odisha: 2.1 MTPA melt**, scalable to 3.2 MTPA. Integrated plant.
- **Hisar, Haryana: 0.8 MTPA melt.** Founded 1970 as Jindal Strips Ltd — the group's origin site.
- India total melt **2.9 MTPA** `[SECONDARY]` (pre-Indonesia figure; now 3.0).
- **Cold rolling: 2.05 → 2.67 MTPA by FY28** (INR 900 cr, Hisar + Kharagpur, commissioning Q2 FY28).
- Other assets: **250,000 tpa ferro chrome plant at Jajpur** with captive chrome ore mines at Sukinda; ferro manganese/silico manganese complex (1×27.6 MVA FeMn + 2×27.6 MVA SiMn, 100,000 tpa); **Chromeni Steels, Mundra, Gujarat — 0.6 MTPA cold rolling** (54% equity, ~INR 1,340 cr).
- **16 manufacturing/processing facilities** across India, Spain and Indonesia (as of Mar 2025); presence in 12+ countries `[SECONDARY]`.
- **India's largest stainless steel producer; top 3–5 globally** `[SECONDARY]` — Wikipedia gives both "top 5" and "top 3" in different places; say "among the world's largest" to be safe.
- India's largest producer of coin blanks `[SECONDARY]`.

### 3c. Strategy and capex — `[HARD]`
- **Target 3.5 MTPA sales volume by FY29** (from 2.57 MTPA FY26). MD Abhyuday Jindal.
- **INR 2,600 cr capex in FY27.**
- **INR 5,400 cr strategic investment package**: Indonesia SMS >700 cr, Jajpur downstream ~1,900 cr, infrastructure/sustainability ~1,450 cr, Chromeni ~1,340 cr.
- Stated focus on **cold rolled and value-added products** — directly relevant: value-added and exposed-quality product is exactly where surface defects destroy the most value.

### 3d. Digitalisation / AI — the hook for the deck
**Abhyuday Jindal, MD, on AI (Economic Times 16 Jul 2024, republished on jindalstainless.com 24 Jul 2024)** `[HARD]`, direct quotes:
- *"AI isn't just a technological upgrade; it's a strategic move that will keep our industry competitive in the global market and help us become safer, smarter, and more sustainable."*
- **"With AI, machine learning (ML), robotics, and the Internet of Things (IoT), the highest quality can be ensured, minimising the wastage of raw materials."** ← *Open the deck with this. The MD has already publicly committed to AI for quality assurance and material waste reduction. Your project is the literal execution of his sentence.*
- *"AI will not replace; it will instead renew and re-energise."* ← useful for pre-empting the operator-displacement objection.

Other `[SECONDARY]` / `[HARD]`:
- **Partnership with Capgemini** to assess and design the digital roadmap (IoT, analytics, AI/ML) supporting the expansion.
- **Stainless Mart** — B2B platform, "first of its kind in the Indian stainless steel sector."
- **People Matters Infini-T Awards 2026**: Gold in HR Operations, Silver in Digital Adoption & Change Leadership. Covers 17 entities, 12,700+ employees; employee satisfaction **45% → 87%**, helpdesk tickets **−91%**, AI assistant resolves **>95%** of routine queries. Useful proof they can actually land an AI deployment and measure it.
- **Caveat / gap** `[FLAGGED]`: I found **no public statement of an existing automated surface inspection system at Jajpur or Hisar**, and no public AI-for-quality deployment. The digitalisation record is real but concentrated in HR, customer platforms and roadmapping. Treat the shop-floor vision opportunity as **greenfield** — this is an argument *for* the project, but do not assert they have nothing (absence of public evidence ≠ absence).
- Sector context `[SECONDARY]`: Tata Steel, JSW and SAIL all have active AI programmes; JSW targets being "India's most digitally advanced steel business" by 2026 (ARC Advisory — page returned 403, so `[SECONDARY]` via search summary only).

---

## 4. COST OF POOR QUALITY — AND A DEFENSIBLE ROI MODEL

### 4a. What is genuinely sourced
- **COPQ in manufacturing averages ~15% of sales, range 5–35%**, depending on product complexity `[HARD]` — IISE, "Measuring the Cost of Quality."
- **ASQ: most manufacturers spend 15–20% of total sales revenue on quality-related costs** `[SECONDARY]` — widely quoted; I saw it via multiple secondary pages, not on asq.org directly.
- Mature quality systems can hold COPQ **below 10%**; troubled operations exceed 30% `[SECONDARY]`.
- PAF model framing (Prevention / Appraisal / Internal Failure / External Failure) is the right structure for the slide `[HARD]`.
- **Stainless 304 pricing, India, 2026** `[SECONDARY]`, two independent ranges that partially disagree — report both:
  - 304 CR coil **INR 220–230/kg** (~INR 2.20–2.30 lakh/t); 304 HR coil **INR 210–215/kg** — Nexizo/BigMint-type trade sources, Q1 FY2026-27.
  - 304 CR/HR **INR 1.80–1.98 lakh/t** — Accurate Steels / trade listings.
  - Nickel is 8–10% of 304 by mass at INR 1,500–2,000/kg, "over half the raw material cost" — explains why *yield* loss on stainless is far more painful than on carbon steel. **This is a real differentiator for a stainless pitch: a downgraded tonne of 304 destroys ~3× the value of a downgraded tonne of commodity HRC.** `[ESTIMATE]` on the multiple; the input prices are sourced.

### 4b. Numbers circulating that you must NOT cite
All of the following come from **oxmaint.com / ifactoryapp.com / ifactory.jrsinnovation.com**, which are AI-generated SEO content farms with fabricated case studies. They are seductive because they are perfectly shaped for a slide. `[SUSPECT — DO NOT USE]`:
- "Surface defects cost steel manufacturers 15–20% of revenue"
- "Surface quality defects drive 2–5% of production to secondary/reject, costing $3M–$12M annually"
- "A single defect downgrades a $900/ton prime coil to a $600/ton secondary"
- "84% reduction in quality escapes", "40% more defects than manual graders", "customer claims down 55%", "ROI exceeding 1900%"
- "A producer lost a $4.2 million coil order..."

I attempted to source a real replacement (ArcelorMittal's own coiled-tubing surface inspection page) — it is a genuine primary source but is **qualitative only**; the only hard numbers are 7,000+ m coiled tubing lengths, **50 tonnes of steel potentially scrapped if defects are found in one hot-rolled coil** `[HARD]`, and "almost seven times the pixelation" versus the previous system `[HARD]`. The 50-tonne figure is genuinely useful as a unit-of-loss anchor.

### 4c. A defensible ROI model built only on Jindal's own numbers
`[DERIVED]` — every input is either a Jindal reported figure or an explicitly-flagged assumption. This is the honest way to build the slide: **make the uncertain input visible and sweep it**, rather than importing a fabricated $/ton.

Fixed, sourced: FY26 volume **2,565,902 t**, realisation **INR 167,407/t**, EBITDA **INR 21,669/t**.
Assumed, swept: prime→secondary discount (10/15/20%) × downgrade rate (0.5/1.0/2.0% of volume).

**Annual value at risk from surface-driven downgrade, INR crore/yr:**

| Downgrade rate ↓ / Discount → | 10% | 15% | 20% |
|---|---|---|---|
| 0.5% of volume | 21 | 32 | 43 |
| 1.0% of volume | 43 | 64 | 86 |
| 2.0% of volume | 86 | 129 | 172 |

**Worked mid-case:** 1.0% downgrade rate, 10% discount → **INR 43 cr/yr loss pool**. If earlier and more consistent detection avoids **30%** of it → **INR 13 cr/yr recovered**, against roughly **INR 21 cr** capex for one ~USD 2.4M line system at INR 88/USD.

**→ Payback ≈ 1.6 years on a single line at the conservative mid-case.** Say this. It is a *good but not miraculous* number, and presenting it that way is what will make the room believe the rest of the deck. The upside cases (2% downgrade, 20% discount) reach payback in under 5 months, but rest on the unsourced downgrade rate — **flag the downgrade rate as the number Jindal must supply from their own MIS.** That is also a natural, credible ask to close the deck on.

**Secondary benefit streams to name but not monetise** (no defensible source): avoided strip breakage and mill damage (the actual Ternium business case, `[HARD]` qualitative), reduced customer claims, inspector redeployment, and root-cause feedback to the caster/descaler.

---

## 5. THE SIX NEU-DET CLASSES — ROOT CAUSE AND CORRECTIVE ACTION

Operator-facing text. Sourcing is mixed and I have marked each. Metallurgy for crazing and patches is the weakest-sourced — flagged.

**0 — CRAZING** `[PARTLY ESTIMATE]`
- *Appearance:* fine, dense network of shallow interconnected cracks; low contrast — the hardest class for every published detector (best AP@[.5:.95] only 21.3%).
- *Root cause:* thermal-mechanical fatigue cracking of the surface layer. Two mechanisms: (a) craze/fire-crack network on the **work roll** surface from repeated heating–quenching cycles, imprinted onto the strip; (b) surface cracking of the strip from excessive thermal gradients or rolling outside the correct temperature window.
- **Flag:** I could not find a primary metallurgical source that ties NEU-DET "crazing" specifically to roll thermal fatigue. Roll fire-cracking is standard mill knowledge, but treat the attribution as `[ESTIMATE]` and have a metallurgist review before it goes to operators.
- *Corrective action:* review work-roll cooling uniformity and roll change interval; inspect rolls for fire-crack networks; check finishing/coiling temperature control.

**1 — INCLUSION** `[HARD]`
- *Appearance:* elongated dark streaks/slivers along the rolling direction; thin metallic splinters partially attached to the surface.
- *Root cause:* **non-metallic inclusions entrapped upstream in the caster**, rolled out and exposed during hot rolling. Documented sources: **mould flux entrainment** and **Al₂O₃ / SiO₂ / Al–Ti–O reoxidation products from tundish flux**; also **low steel level in the tundish** (Leão et al., *Ironmaking & Steelmaking* 48(8), 2021, on sliver defects in ULC Al-killed steel); Li et al. (2022) on mould flux entrainment causing slivers on hot-dip galvanised auto exposed panel. Also arises from rolled-in scrap fragments.
- *Corrective action:* **this is a caster problem, not a mill problem** — control tundish steel level and avoid low-level casting; review mould flux melting behaviour and viscosity; minimise reoxidation; tighten start-up practice (dry/semi-molten flux is readily entrapped in the first slab). Feed detection back to the caster heat ID.

**2 — PATCHES** `[ESTIMATE]`
- *Appearance:* irregular large-area regions of differing reflectance/texture. Easiest class for detectors (AP50 typically 0.90–0.97).
- *Root cause:* locally non-uniform surface condition — uneven descaling coverage leaving residual oxide, uneven pickling (under-pickling leaving scale, over-pickling attacking the base), or local segregation/adherent scale patches.
- **Flag:** the loosest of the six. NEU-DET's own documentation does not define patches metallurgically, and I found no primary source tying it to a specific mechanism. Present as "non-uniform surface condition, typically descaling/pickling coverage" and mark it as requiring plant confirmation.
- *Corrective action:* check descaling header coverage across width and pickling line acid concentration/temperature/line speed.

**3 — PITTED SURFACE** `[HARD]`
- *Appearance:* small irregular pits/indentations, often in patches ("scale pits").
- *Root cause:* **oxide scale rolled into and then detached from the surface, leaving shallow irregular depressions.** Driven by inadequate descaling before the roll stand and by inconsistent reheat furnace temperatures promoting excessive oxide formation. Can also be pickling-related pitting.
- *Corrective action:* verify descaling water pressure and nozzle condition; stabilise reheat furnace temperature and residence time; reduce furnace atmosphere oxidation.

**4 — ROLLED-IN SCALE** `[HARD]` — best-sourced of the six
- *Appearance:* dark embedded oxide, low contrast against an already-scaled background — second-hardest class for detectors (best AP@[.5:.95] 23.0%).
- *Root cause:* three documented sub-types — **furnace-born (primary) scale**, **rolling-induced (secondary) scale**, and **oxide film detached from the roll surface and embedded in the strip**. Direct drivers: **low high-pressure descaling water pressure and blocked nozzles**; **inadequate descaling nozzle arrangement**; **fast rolling rhythm with poor roll cooling** causing the roll oxide film to detach.
- *Sources:* "Improvement of Hot Rolling Technology to Reduce the 'Rolled-in Scale' Defect," *Steel in Translation* (Springer, 2023), DOI 10.3103/S0967091223040150; Utsunomiya et al., "Formation mechanism of surface scale defects in hot rolling process," *CIRP Annals* (S0007850614000250) — finds scale behaviour (uniform deformation / cracking / fragmentation / indentation into the matrix) depends strongly on **rolling temperature and pre-rolling scale thickness**, with **temperature drop from contact with cold rolls** causing the cracking.
- *Corrective action:* regular checks of the high-pressure descaling system — **maintain descaler pressure (>180 bar cited)**, clear blocked nozzles, verify pump output is not 10–15% below design; control rolling rhythm; ensure adequate roll cooling.

**5 — SCRATCHES** `[HARD, thin]`
- *Appearance:* sharp linear marks, usually along the rolling/handling direction.
- *Root cause:* mechanical abrasion from **tooling and handling** (IspatGuru states this directly, without elaboration) — side guides, roller table debris, coiler/mandrel contact, transport and slitting.
- *Corrective action:* inspect and clean guide faces and roller tables; check side guide alignment and wear; review coil handling, strapping and transport practice. Note RetinaNet scored **0.0% AP on scratches** in the arXiv comparison — thin high-aspect-ratio objects break anchor/FPN assumptions, worth a technical footnote.

---

## 6. TECHNICAL DIFFERENTIATION — THE HARD PROBLEMS

### 6a. Known hard problems, with evidence

1. **Domain shift between mills, grades and lighting — the #1 deployment killer.** Best evidence is the **vendor's own admission** `[HARD]`: *"As the line processes material from different hot mills, the features for the same class of defect are different, so this variable can make detection more challenging."* (AMETEK/Ternium). Academic corroboration: "cross-scenario generalization degradation caused by domain shift among production lines"; "interference information such as blur and shadow leads to domain shift between source and target domains" (ScienceDirect S1474034624006153, cross-supervised contrastive domain adaptation for steel defect segmentation, 2024); ASME JCISE 24(1):011006, cross-domain transfer learning for galvanized steel strip defects. **Slide implication: a model trained on NEU-DET is a capability demonstration, not a deployable asset. Say so, and propose a per-line calibration/fine-tune protocol.**

2. **Low-contrast defects.** Crazing and rolled-in scale sit at 21.3% and 23.0% AP@[.5:.95] for the best model tested — roughly *one third* of what patches achieve. Steel images suffer "highly variant illumination, large amounts of noise due to surface scale" `[SECONDARY]`. This is a physics/optics problem as much as a model problem — bright-field vs dark-field illumination choice matters more than architecture.

3. **Class imbalance and scale variation.** NEU-DET is nominally balanced (300 images/class) but *instance* counts and *object sizes* are not. Papers address this with **Focaler-IoU**, **Shape-IoU**, and **Focal-CIoU** losses specifically to "mitigate class imbalance" and "dynamically weight different samples" `[HARD]`, PMC12736593 / PMC11820220. Real mill data is far more imbalanced than NEU-DET — rare defects are the expensive ones.

4. **Small objects on a cluttered background.** "Small targets within images that contain significant background interference"; "irregular shapes with indistinct boundaries"; "considerable scale variations between different classes" `[HARD]`, PMC12899440.

5. **Dataset limitations of NEU-DET itself** `[HARD]`: only 1,800 images, 200×200, grayscale. Papers explicitly note it "poses several challenges due to its limited dataset size and grayscale nature." Be honest that our 1440/180/180 split is small.

6. **Labelling cost.** "The labor-intensive nature of data labeling complicates the deployment of supervised learning models in practical scenarios" `[HARD]`, oaepublish jmi.2025.21.

7. **Unseen / novel defect types.** "Dynamic changes in defect types at industrial sites where new defects often lack labeled samples, making it difficult for traditional supervised learning to adapt quickly" `[HARD]`, Sci Rep s41598-025-29871-w. Corroborated operationally by the Ternium unclassified-defect bucket.

8. **Throughput.** See §2f — 10,800–55,000 tiles/s. Not usually treated as a research problem, but it is the binding constraint in a mill.

### 6b. What recent papers propose — and what actually works

| Approach | Representative work | Result | Honest read |
|---|---|---|---|
| **Attention + efficient conv modules** | YOLOv10n-SFDC (DualConv, SlimFusionCSP, Shape-IoU) | 79.2 → **85.5** mAP50, params 3.00 → **2.67M** `[HARD]` | Best single reported gain. Accuracy *and* size improve together. |
| **Feature-segmentation / partial-connection lightweighting** | LESSDD-Net | +3.19% mAP over YOLO11n, **−39.9% params**, −20.6% FLOPs, 131 FPS `[HARD]` | Right direction for edge deployment at line speed. |
| **Multi-level hierarchical feature fusion** | DDN / MFN (IEEE TIM 2020) | +4.6/+4.4 mAP over Faster R-CNN `[HARD]` | The original insight; multi-scale fusion is now standard. |
| **Transformer backbones** | DETR, RT-DETR, SH-DETR, multi-scale deformable transformer w/ iterative query refinement | DETR **25.2** and RT-DETR **55.0** mAP50 out-of-the-box vs YOLOv11 **71.6** `[HARD]` | **Transformers substantially underperform on NEU-DET without heavy tuning** — small dataset, small objects. A genuinely contrarian, defensible slide: we chose YOLO deliberately, not by default. |
| **Self-supervised pretraining** | SimSiam → Faster R-CNN, on SSDD (20,272 unlabeled imgs) | mAP50 **0.768** vs ImageNet **0.773** vs random init **0.280** `[HARD]` | SSL *matches* ImageNet pretraining **with zero labels**. The headline is not accuracy, it is that a mill's own unlabeled coil imagery is a free asset. Strongest labelling-cost answer. |
| **Anomaly detection for unseen defects** | PatchCore, ViDi Red, MVTec AD family; Real-IAD D3 (CVPR 2025); UniADC (arXiv:2511.06644) | PatchCore robust to lighting change (≤3pp AU-PRO drop) `[SECONDARY]` | The right architecture for the "unclassified bucket." Propose supervised detector **+** unsupervised anomaly channel in parallel. |
| **Diffusion-based few-shot defect generation** | PMC12115093 (few-shot steel defect generation); Sci Rep s41598-025-29871-w (diffusion + zero-shot) | — | Addresses rare-class scarcity; still research-grade. |
| **Active learning** | — | — | **Flag: I found no NEU-DET-specific active-learning paper.** Mention as a proposed operating practice, not as cited prior work. |

### 6c. Suggested differentiation narrative
Everyone benchmarks on NEU-DET; almost nobody addresses **(i)** the domain shift the vendors themselves admit to, **(ii)** the unclassified-defect bucket that real mills run, and **(iii)** the throughput arithmetic. Proposing a **small fast supervised detector + parallel unsupervised anomaly channel + self-supervised pretraining on Jindal's own unlabeled imagery + a per-line recalibration protocol** is both technically current and specifically responsive to what the AMETEK/Ternium deployment reveals about how these systems actually fail.

---

## GAPS AND THINGS I COULD NOT VERIFY

1. **No sourced price for a commercial ASIS.** The ~USD 2.4M figure traces only to an SEO content farm. Order of magnitude is plausible; attribution is not available.
2. **No sourced vendor detection-rate claim.** ISRA's PEARL "+10% accuracy" is a *relative* improvement, not an absolute detection rate. No vendor publishes an absolute detection/classification rate. Treat any "99% detection" claim in a competitor deck as unsourceable too.
3. **Jindal's actual downgrade / rejection / yield-loss rate.** Not public. This is the single input the ROI model most needs — make it the explicit ask.
4. **Whether Jindal already has surface inspection at Jajpur/Hisar.** No public evidence either way.
5. **Jindal-specific line speeds, strip widths, and camera infrastructure.** Not public; the deck's throughput slide uses Ternium's 300 m/min as a sourced proxy — label it as such.
6. **arXiv:2606.07659 ("Industrial-YOLO", 98.5% mAP, 120 FPS Jetson Orin)** — PDF text not extractable; results not broken out by dataset. Unverified.
7. **Metallurgical root cause for "crazing" and "patches"** as NEU-DET defines them — my attributions are reasoned, not sourced. Needs a Jindal metallurgist's sign-off before operator-facing use.
8. **ARC Advisory "How AI Is Reforging India's Steel Industry"** returned HTTP 403; JSW/Tata/SAIL AI claims are `[SECONDARY]` via search summary only.
9. **ASQ's "15–20% of sales" figure** — quoted everywhere, not verified on asq.org. The IISE "5–35%, average ~15%" figure *is* directly sourced and is the safer citation.

---

## SOURCE LIST

**Benchmarks**
- He, Song, Meng, Yan, "An End-to-End Steel Surface Defect Detection Approach via Fusing Multiple Hierarchical Features," IEEE Trans. Instrum. Meas. 69(4):1493, 2020 — https://ui.adsabs.harvard.edu/abs/2020ITIM...69.1493H/abstract · https://repository.lboro.ac.uk/articles/journal_contribution/An_end-to-end_steel_surface_defect_detection_approach_via_fusing_multiple_hierarchical_features/12249215
- Maity & Ghosh, "Comparative Analysis of Object Detection Algorithms for Surface Defect Detection," arXiv:2510.21811, 2025 — https://arxiv.org/abs/2510.21811 (per-class table extracted from PDF)
- "A Novel YOLOv10-Based Algorithm for Accurate Steel Surface Defect Detection" (YOLOv10n-SFDC), 2025 — https://pmc.ncbi.nlm.nih.gov/articles/PMC11820220/
- "An Efficient Lightweight Method for Steel Surface Defect Detection" (LCED-YOLO), 2026 — https://pmc.ncbi.nlm.nih.gov/articles/PMC12736593/
- "LESSDD-Net," 2026 — https://pmc.ncbi.nlm.nih.gov/articles/PMC12899440/
- "Application of self-supervised learning in steel surface defect detection," J. Mater. Inf. 2025 — https://www.oaepublish.com/articles/jmi.2025.21
- "A real-time surface defect detection model based on adaptive feature information selection and fusion," 2025 — https://www.sciencedirect.com/science/article/pii/S1566253525011030
- NEU-DET dataset — https://ieee-dataport.org/documents/neu-det
- `[SUSPECT]` arXiv:2606.07659 — https://arxiv.org/abs/2606.07659 · IOS Press FAIA251752 — https://ebooks.iospress.nl/DOI/10.3233/FAIA251752

**Industry / vendors**
- AMETEK Surface Vision, Ternium case study (2020) — https://www.ameteksurfacevision.com/-/media/ameteksurfacevisionv2/documentation/casestudies/ametek_surface_vision_ternium_case_study_rev1_en.pdf
- ISRA VISION metals/steel — https://www.isravision.com/en-en/industries/metals/steel · https://www.isravision.com/en-en/industries/metals/steel-surface-quality-inspection
- AIST ISRA/Parsytec release archive — https://www.aist.org/isra-vision-parsytec-installing-surface-inspection-unit-for-italian-flats-producer
- Cognex In-Sight 9000 / 3800 line scan, metal inspection — https://www.cognex.com/en/applications/automated-defect-detection/material-quality-inspection/metal-inspection-and-quality-control
- ArcelorMittal, surface inspection for coiled tubing — https://industry.arcelormittal.com/market-segments/steel-for-energy/QualityTubingsurfaceinspection
- Britannica, hot strip exit speeds — https://www.britannica.com/technology/steel/Hot-strip · Nippon Steel technical report — https://www.nipponsteel.com/en/tech/report/nssmc/pdf/111-03.pdf
- SMS group, cold rolling mills for stainless — https://www.sms-group.com/plants/cold-rolling-mills-for-stainless-steel
- Surface inspection market — https://www.marketsandmarkets.com/Market-Reports/surface-inspection-market-192440286.html

**Jindal Stainless**
- FY26 results — https://www.jindalstainless.com/press-releases/jindal-stainless-announces-financial-results-for-the-quarter-and-financial-year-ended-march-31-2026/
- FY25 results — https://www.jindalstainless.com/press-releases/jindal-stainless-announces-financial-results-for-the-quarter-and-financial-year-ended-march-31-2025/
- INR 5,400 cr investment / 4.2 MTPA — https://www.jindalstainless.com/press-releases/jindal-stainless-announces-inr-5400-crore-strategic-investments-capacity-expansion-to-4-2-mtpa/
- **Abhyuday Jindal on AI** — https://www.jindalstainless.com/blog/ai-will-keep-indias-manufacturing-sector-globally-competitive-abhyuday-jindal-md-jindal-stainless/
- 3.5 MTPA by FY29, INR 2,600 cr FY27 capex — https://aninews.in/news/business/jindal-stainless-targets-35-mtpa-sales-by-fy29-lines-up-rs-2600-crore-capex-for-fy2720260813211536/
- INR 900 cr cold rolling / 2.67 MTPA by FY28 — https://stainlesstoday.com/jindal-stainless-to-invest-%E2%82%B9900-crore-on-cold-rolling-expansion-targets-2-67-mtpa-capacity-by-fy28/
- Digital transformation — https://www.jindalstainless.com/annualreport-2023/corporate-overview/embracing-digital-transformation · Infini-T Awards 2026 — https://www.tipranks.com/news/company-announcements/jindal-stainless-wins-top-honours-for-digital-hr-transformation-at-infini-t-awards-2026
- Wikipedia — https://en.wikipedia.org/wiki/Jindal_Stainless · Jajpur plant — https://www.gem.wiki/Jindal_Stainless_Jajpur_steel_plant

**Cost of quality / pricing**
- IISE, "Measuring the Cost of Quality" — https://www.iise.org/details.aspx?id=22118
- Quality Digest, COPQ — https://www.qualitydigest.com/inside/quality-insider-article/what-your-company-s-cost-poor-quality-082304.html
- SS 304 India pricing — https://nexizo.ai/stainless-steel-category · https://accuratesteels.com/stainless-steel-cost/ · https://www.bigmint.co/intel/detail/india-leading-stainless-steel-producer-raises-coil-prices-on-cost-pressures-and-strong-dollar-36230

**Metallurgy**
- "Improvement of Hot Rolling Technology to Reduce the 'Rolled-in Scale' Defect," Steel in Translation, 2023 — https://link.springer.com/article/10.3103/S0967091223040150
- Utsunomiya et al., "Formation mechanism of surface scale defects in hot rolling," CIRP Annals — https://www.sciencedirect.com/science/article/abs/pii/S0007850614000250
- Leão et al., "Sliver defects in ULC Al-killed steel caused by low steel level in the tundish," Ironmaking & Steelmaking 48(8), 2021 — https://www.tandfonline.com/doi/full/10.1080/03019233.2020.1848306
- Li et al., mould flux entrainment slivers on galvanised auto panel, 2022 — https://journals.sagepub.com/doi/abs/10.1080/03019233.2021.2004534
- Continuous casting inclusions review (Thomas, UIUC) — https://ccc.illinois.edu/PDF%20Files/Publications/03_Mexico_Nov_Inclusion_review_v5a_updated.pdf
- IspatGuru — https://www.ispatguru.com/rolling-of-hot-strip-and-strip-defects/ · https://www.ispatguru.com/rolling-process-for-steel/typical-surface-defects-on-hot-rolled-steel-strip/

**Hard problems / recent methods**
- Cross-supervised contrastive domain adaptation for steel defect segmentation — https://www.sciencedirect.com/science/article/abs/pii/S1474034624006153
- Cross-Domain Transfer Learning for Galvanized Steel Strips, ASME JCISE 24(1):011006 — https://asmedigitalcollection.asme.org/computingengineering/article/24/1/011006/1166156/
- Diffusion + zero-shot for unseen defect categories — https://www.nature.com/articles/s41598-025-29871-w
- Few-shot steel defect generation via diffusion — https://pmc.ncbi.nlm.nih.gov/articles/PMC12115093/
- Survey: DL image anomaly detection in industrial inspection, Frontiers 2025 — https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1554196/full
- awesome-industrial-anomaly-detection — https://github.com/m-3lab/awesome-industrial-anomaly-detection
- UniADC — https://arxiv.org/pdf/2511.06644

**Unreliable — flagged, do not cite:** oxmaint.com, ifactoryapp.com, ifactory.jrsinnovation.com (AI-generated SEO content with fabricated case studies and $/ton figures).