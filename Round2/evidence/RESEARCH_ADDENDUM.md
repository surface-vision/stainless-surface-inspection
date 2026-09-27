# Research addendum — Round 2 (27 Sep 2026)

New research added to the deck, with sources, where each fact is used, and what was
deliberately left out. Use this as the Q&A backup: if a judge asks "where is that from?",
the answer is here.

## 1. Why now (slide 2)

| Fact | Source | Used on |
|---|---|---|
| India stainless consumption **4.80 Mt in FY25**, up 84% in five years (2.61 Mt in FY21), +8% YoY | ISSDA via [Business Standard, Jun 2025](https://www.business-standard.com/industry/news/domestic-stainless-steel-use-grows-84-in-5-yrs-to-4-8-mt-in-fy25-issda-125060500433_1.html) | Slide 2 |
| Demand forecast **6.8 Mt by FY30**; 7-8% a year FY26-FY28, led by rail (Vande Bharat, metro coaches), infrastructure, auto, elevators | [Free Press Journal](https://www.freepressjournal.in/business/indias-stainless-steel-consumption-grows-84-in-5-years-driven-by-infra-railways-and-green-energy-projects); [IBEF](https://www.ibef.org/news/india-s-stainless-steel-demand-to-grow-7-8-annually-over-two-to-three-years-issda) | Slide 2 |
| Per-capita use 3.4 kg in India vs world average above 6 kg | same ISSDA coverage | Notes only |
| Stainless imports **1.73 Mt in FY25** (China, Indonesia, Vietnam, Korea) | [SteelOrbis](https://www.steelorbis.com/steel-news/latest-news/indias-steel-imports-fall-13-percent-year-on-year-in-cy25-on-safeguard-duty-impact-1431657.htm) | Slide 2 |
| DGTR opened an **anti-dumping investigation on cold-rolled 300/400 series** from China, Indonesia, Vietnam on **29 Sep 2025** | [UNI Customs Consulting](https://unicustomsconsulting.com/en/india-opens-anti-dumping-investigation-into-300-400-series-stainless-steel-from-china-indonesia-vietnam/) | Slide 2 |
| **IS 6911:2017 BIS certification mandatory** for stainless plate, sheet and strip from **30 Aug 2024**, under the Steel & Steel Products (Quality Control) Order 2024 | Certification bodies, e.g. [Omega QMS](https://omegaqms.co.in/blogs/understanding-is-6911-stainless-steel-sheet-strips-and-the-mandatory-bis-certification/) — **secondary**; confirm against the BIS gazette before quoting a date in Q&A | Slide 2 |

**The argument:** demand is growing fastest in surface-critical uses (coaches, metros, elevators),
cheap imports are under trade scrutiny, and certification is now mandatory. Consistent,
documented surface quality is the domestic producer's moat.

## 2. Precedent — who already does this (slide 3)

| Who | What | Source |
|---|---|---|
| **Outokumpu** (Europe's largest stainless producer) | AI-based Advanced Surface Inspection System at **Tornio**, switchover from **Oct 2024**; hybrid-cloud architecture rebuilt to run **without internet**; designed to scale across sites. Quote: *"A defect in hot rolling can extend further down to cold rolling, and a scratch can elongate to several meters."* No accuracy figures published. | [Nortal case study](https://nortal.com/insights/ai-in-action-surface-defect-detection-in-outokumpu-stainless-steel) |
| **Tata Steel Kalinganagar** | India's first WEF Global Lighthouse plant; *"in-process video analytics is being used to identify surface defects on cold rolled products"* (2019) | [Tata group newsroom](https://www.tata.com/newsroom/business/tata-steel-kalinganagar-indias-industrial-lighthouse) |
| **ISRA Parsytec on stainless** | Systems for stainless annealing-and-pickling lines (Sandvik): matrix cameras at **170 µm**, **bright- and dark-field lighting**, automatic "Coil Decision", pre-built defect catalogue | [Stainless Steel World](https://stainless-steel-world.net/parsytec-surface-inspection-system-orders/) |
| POSCO (Pohang) | WEF Lighthouse using AI for productivity and quality; line-scan camera with linear light source | WEF Lighthouse coverage — not used on slides |

## 3. JSL's own programmes the pilot can ride on (slide 7)

| Fact | Source |
|---|---|
| **Project Pragati** launched **14 May 2025** at Hisar with Dassault Systèmes and Capgemini. **Phase 2: Jajpur with Level-2 integration** (real-time process parameters from machines). Reported: lead times −10-15%, inventory −8-10%, capacity utilisation +~5% | [JSL press release](https://www.jindalstainless.com/press-releases/jindal-stainless-launches-industry-first-supply-chain-digitalisation-project/) |
| **Why it matters:** the pilot's P0 links to Pragati's Level-2 feed instead of building a new plant integration, so each defect is tied to its heat and coil with less risk and cost. | — |

## 4. Stainless-specific imaging (slides 5, 7)

| Fact | Source |
|---|---|
| Mirror-like (specular) surfaces hide defects under direct light; **combining dark- and bright-field illumination** improves detection | [Optics and Lasers in Engineering, 2016](https://www.sciencedirect.com/science/article/abs/pii/S0143816616301634) |
| **Deflectometry** (reading distortions in a reflected stripe pattern) is the established method for specular surfaces | [Fraunhofer IOSB](https://www.iosb.fraunhofer.de/en/projects-and-products/deflectometry-for-the-inspection-of-specular-surfaces.html) |
| Recent review of vision inspection on highly reflective metal surfaces | [ScienceDirect review, 2026](https://www.sciencedirect.com/science/article/pii/S2215098626002181) |
| Stainless-specific defects: slivers (mould powder entrapment), **roping and ridging (ferritic grades)**, orange peel (coarse grains), anneal colour, lamination | [Analysis of Surface Defects on Stainless Steel Cold Rolling Strip](https://www.researchgate.net/publication/258478772_Analysis_of_Surface_Defects_on_Stainless_Steel_Cold_Rolling_Strip) |

## 5. Innovation — rare defects (slides 6, 7)

| Fact | Source |
|---|---|
| Few-shot **steel surface defect generation with diffusion models** (Sensors, 2025) — synthetic images for classes with few real examples | [PMC12115093](https://pmc.ncbi.nlm.nih.gov/articles/PMC12115093/) |
| Training-free industrial defect generation with diffusion models (ICCV 2025) | [ICCV 2025 paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Xu_Training-Free_Industrial_Defect_Generation_with_Diffusion_Models_ICCV_2025_paper.pdf) |
| SteelDefectX: vision-language dataset for steel defects — basis for auto-written defect reports (future phase) | [arXiv 2603.21824](https://arxiv.org/pdf/2603.21824) |

## 6. Security and model governance (slide 7)

| Fact | Source |
|---|---|
| **IEC 62443** is the standard for securing industrial control systems: plants split into security zones joined by monitored conduits; every connected camera is a new entry point | [Fortinet overview](https://www.fortinet.com/resources/cyberglossary/iec-62443); [Cisco](https://www.cisco.com/c/en/us/products/collateral/security/isaiec-62443-3-3-wp.html) |
| Our own finding: across four training runs, the clean-strip false-alarm rate ranged from 4.7% to 17% on the same test images, so **every retrained model must beat the live one on a fixed test set before it ships** | `reports/demo_seed_selection.json` |

## 7. Customers and sustainability (slide 8)

| Fact | Source |
|---|---|
| JSL supplies **high-strength 301L** for **Vande Bharat sleeper coaches** (ICF, BEML), and stainless for Vande Metro, Kolkata's underwater metro, RRTS, Mumbai Metro; 1,031 t of 301N for Bangalore Metro Phase 2; supplier to Indian Railways since 1998 | [JSL press release](https://www.jindalstainless.com/press-releases/jindal-stainless-supplies-stainless-steel-for-vande-bharat-sleeper-train/); [Metro Rail News](https://metrorailnews.in/jsl-supplies-stainless-steel-for-vande-bharat/) |
| JSL emission intensity **2.15 → 1.76 tCO2e per tonne** (FY23 → FY26); Scope 1+2 down 5% in FY26; **net zero by 2050**, 50% intensity cut by 2035; 70.12% scrap use in FY26 | [ScanX](https://scanx.trade/stock-market-news/companies/jindal-stainless-cuts-emissions-5-uses-70-12-scrap-in-fy26/47750003); [JSL ESG](https://www.jindalstainless.com/jindal-stainless-ambitious-esg-efforts-plans-to-commit-to-net-zero-by-2050/) |
| Non-prime (secondary) coil trades **10-30% below prime** — so the deck's 10-20% discount assumption is conservative | [Metal Center News, Secondary Steel Report](https://www.metalcenternews.com/editorial/current-issue/secondary-steel-report/44788) and trader listings — **secondary**, industry range |

## Deliberately NOT used

- **"99.2% / 99.9% accuracy", "91% of escapes eliminated", "$85,000 per rejection"** — from oxmaint.com
  SEO pages with no primary source. A judge who checks would find nothing behind them.
- **WEF "41% decrease in product defects"** — an aggregate across all Lighthouse sites and industries,
  seen only in a search summary (the WEF page returned 403). Not specific to steel inspection.
- **Tata Steel / JSW accuracy figures** — none published. Only the qualitative Tata Kalinganagar statement is cited.

## Still unknown — ask JSL (these are the P0 inputs)

1. Does any JSL line already run automated surface inspection? No public evidence either way.
2. Downgrade and claim rates by line and grade — the number that decides the business case.
3. Finish mix on the pilot line (2B vs BA vs 2D) — decides whether deflectometry is needed on day one.
4. Timing of Pragati Phase 2 at Jajpur — decides whether P0 links to it or to the existing Level 2 directly.
