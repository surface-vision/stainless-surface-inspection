# Interviews — first-hand evidence for the deck

Five short conversations turn the deck's "team hypothesis" labels into "checked with …", and put real
voices from the plant floor on slide 2. Judges score first-hand evidence highly; finals panels ask
"who did you talk to?"

**Before contacting anyone at Jindal Stainless itself, check the competition rules.** Talking to people
at other plants, vendors and faculty is normally fine.

## 1. Who to talk to (aim: 5 calls of 15-20 minutes)

| # | Who | Why | Where to find them |
|---|---|---|---|
| 1 | **Metallurgy professor** (steel processing, rolling, surface quality) | Check the defect → owner mapping on slide 3 and the grade-risk matrix | IIT Bombay MEMS department; email or office hours |
| 2 | **Quality or process engineer at a steel plant** (Tata Steel, JSW, SAIL, AM/NS, or a stainless mill) | How inspection works today, what an acceptable false-alarm rate is, who acts on an alarm | IIT Bombay alumni on LinkedIn; the alumni association |
| 3 | **Cold-rolling or annealing-line operator or supervisor** | What inspectors actually do on a shift; what would make them trust an alarm | Same; or a fabricator's shop floor |
| 4 | **Machine-vision vendor representative** (ISRA/Parsytec, AMETEK Surface Vision, Cognex India) | Ballpark system cost per line, install time, what is hard about stainless and BA finish | Company "contact sales" pages, LinkedIn India sales and application engineers |
| 5 | **Stainless fabricator or buyer** (kitchen, lift, rail-coach or appliance maker) | What a surface defect costs them, and what they reject a coil for | Local fabricators; the same place you collect stainless offcuts |

## 2. Messages to send

**LinkedIn (under 300 characters):**
> Hi [Name], we're IIT Bombay students (Team Futuristic) in the Jindal Stainless "Stainless Spark"
> case competition, working on AI surface inspection for steel strip. Could we have 15 minutes of your
> view on how surface defects are caught today? Happy to share our work.

**Email:**
> Subject: 15 minutes on steel surface inspection — IIT Bombay student team
>
> Dear [Name],
> We are Krishna Kanta Mondal and Prathmesh Walimbe from IIT Bombay (Team Futuristic), taking part
> in Jindal Stainless's "Stainless Spark" case competition. We have built a working surface-defect detector
> (live at surface-vision.github.io) and want to check our assumptions with people who know the shop floor.
> Would you have 15 minutes this week for a call? We would only quote you with your permission.
> Thank you, [names]

## 3. What to ask (pick 5-6; stop at 20 minutes)

**Everyone**
1. Where on the line are surface defects found today, and by whom?
2. What happens to a coil when a defect is found late? (downgrade, re-work, scrap, claim)
3. If an automatic system raised alarms, how many false alarms per shift would people tolerate before ignoring it?

**Metallurgist**
4. Looking at our six defect types and the process we blame for each (slide 3), which would you change?
5. For 200, 300, 400-series and duplex, which defects matter most on exposed finishes? (our grade matrix)
6. Which stainless-only defects should a system learn first: roping, ridging, orange peel, anneal colour?

**Plant engineer / operator**
7. Roughly what share of coils is downgraded for surface reasons? A range is fine ("under 1%", "1-3%").
8. Who would own an alarm: quality, the line, or the upstream process? How fast can they act?
9. What would make you trust it? What would make you switch it off?

**Vendor**
10. What does a two-sided inspection system for a 1.3-1.6 m line typically cost, installed? A range is fine.
11. What is hardest about mirror-like BA stainless, and how do you light it?

**Fabricator / buyer**
12. What surface defect makes you reject or claim against a coil? What does one claim cost you?

**Always end with consent:** "May we quote you in our competition deck, by name and role, by role only,
or not at all?"

## 4. How to record it

After each call, add an entry to `quotes.json` (see `quotes.example.json` for the format):

| Field | What to write |
|---|---|
| `role` | "QA engineer, integrated steel plant" — specific, but no company if they prefer |
| `name` | only if they agreed to be named |
| `consent` | `"name"`, `"role"`, or leave the entry out entirely if they said no |
| `quote` | their words, under ~110 characters, about the **problem** (not praise for your project) |
| `validates` | `["owners"]` if a metallurgist confirmed the defect → owner mapping, `["grades"]` for the grade matrix |

Then rebuild the deck (`cd Round2/build && node build_round2.js`):

- slide 2 gains a **"voices from the plant floor"** band with up to three quotes;
- slide 3's header and source line change from **"team hypotheses"** to **"checked with <role>"** for whatever was validated.

Also note every number they give you (downgrade share, system cost, tolerable false alarms) — send them to
me and I will fold them into the economics and the Q&A bank.

## Rules

- Never paraphrase a quote into something stronger than what was said.
- No quote without consent. Role-only is fine and common.
- Keep notes of every call, including the ones that disagree with you: disagreement you have answered is
  the strongest thing you can bring to the finals.
