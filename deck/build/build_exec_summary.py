"""Build the executive-summary submission: a cover plus two content slides.

The round allows 1-2 content slides plus a cover and no appendix, so the full
study in JSW_Surface_Defect_Detection.pptx is compressed by changing its
encoding, not by dropping claims (the case-deck-density system):

  * a navigation strip across the top, the active sections highlighted
  * a rotated left rail naming the slide's mode
  * four panels per slide, each with a header bar, a source, and one exhibit
  * a bottom band stating the decision the slide produces
  * nothing below 9 pt; green / amber / red are used only for status

Every number is a key in reports/ or deck/build/exec_summary_checks.json, and
each panel's header bar names where its numbers come from.

Run:
    .venv/bin/python deck/build/build_exec_summary.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.util import Emu, Inches, Pt

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_deck as bd  # noqa: E402  -- Arial metrics and the overflow ledger

OUT = bd.DECK / "JSW_Executive_Summary.pptx"
QR = bd.FIGDIR / "fig7_live_demo_qr.png"
FAMILIES = bd.FIGDIR / "fig1_defect_families.png"

SW, SH = 13.333, 7.5
RAIL_X, RAIL_W = 0.18, 0.30
BODY_X, BODY_R = 0.62, 12.98
BODY_W = BODY_R - BODY_X
BODY_TOP, BODY_BOTTOM = 1.18, 6.52
BAND_Y, BAND_H = 6.62, 0.66
STRIP_H = 0.46
GAP = 0.12

FLOOR_PT = 9.0
BODY_PT, SMALL_PT, HEAD_PT = 10.0, 9.0, 10.5

# palette: one primary, one accent, one neutral, and status colours kept for status
NAVY = RGBColor(0x14, 0x3A, 0x5A)
NAVY_SOFT = RGBColor(0x2B, 0x5B, 0x80)
ON_NAVY = RGBColor(0xBF, 0xD2, 0xE2)
ACCENT = RGBColor(0x00, 0x86, 0xC3)
INK = RGBColor(0x14, 0x1C, 0x24)
SLATE = RGBColor(0x33, 0x41, 0x4F)
MUTED = RGBColor(0x5F, 0x6B, 0x78)
NEUTRAL = RGBColor(0xF1, 0xF4, 0xF7)
EDGE = RGBColor(0xD5, 0xDC, 0xE3)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GREEN = RGBColor(0x2E, 0x7D, 0x32)
AMBER = RGBColor(0xB7, 0x79, 0x00)
RED = RGBColor(0xC6, 0x28, 0x28)

TABS = ["Problem", "Solution", "Validation", "Roadmap and value"]
TEAM = "Futuristic"

FLOOR_BREACHES: list[str] = []
WORDS: dict[int, int] = {}
_current = {"slide": 0}

L, C, R = PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.RIGHT
TOP, MIDDLE = MSO_ANCHOR.TOP, MSO_ANCHOR.MIDDLE


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------
def _count(text: str) -> None:
    WORDS[_current["slide"]] = WORDS.get(_current["slide"], 0) + len(text.split())


def _style(piece, text, size, bold, colour) -> None:
    if size < FLOOR_PT:
        FLOOR_BREACHES.append(f"slide {_current['slide']}: {size} pt '{text[:30]}'")
    piece.text = text
    piece.font.size = Pt(size)
    piece.font.bold = bold
    piece.font.name = bd.FONT
    piece.font.color.rgb = colour
    _count(text)


def write(slide, x, y, w, h, paragraphs, label, anchor=TOP):
    """paragraphs: [(runs, align, space_after_pt)], runs: [(text, size, bold, colour)].

    Height is checked with the real Arial metrics. A paragraph that mixes bold
    and regular runs is measured as bold, which errs on the side of reporting
    an overflow that PowerPoint would not show.
    """
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = shape.text_frame
    frame.word_wrap = True
    frame.auto_size = MSO_AUTO_SIZE.NONE
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    frame.vertical_anchor = anchor
    measure = []
    for i, (runs, align, after) in enumerate(paragraphs):
        paragraph = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        paragraph.alignment = align
        paragraph.space_after = Pt(after)
        for text, size, bold, colour in runs:
            _style(paragraph.add_run(), text, size, bold, colour)
        measure.append(("".join(r[0] for r in runs), max(r[1] for r in runs),
                        any(r[2] for r in runs), after))
    bd.check_fit(label, bd.block_height_in(measure, w), h)
    return shape


def one(slide, x, y, w, h, text, size, bold, colour, label, align=L, anchor=TOP):
    return write(slide, x, y, w, h, [([(text, size, bold, colour)], align, 0)], label, anchor)


def shape(slide, kind, x, y, w, h, fill, edge=None, radius=None, weight=0.75):
    item = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    if radius is not None:
        item.adjustments[0] = radius
    if fill is None:
        item.fill.background()
    else:
        item.fill.solid()
        item.fill.fore_color.rgb = fill
    if edge is None:
        item.line.fill.background()
    else:
        item.line.color.rgb = edge
        item.line.width = Pt(weight)
    item.shadow.inherit = False
    return item


def rect(slide, x, y, w, h, fill, edge=None):
    return shape(slide, MSO_SHAPE.RECTANGLE, x, y, w, h, fill, edge)


def labelled(slide, kind, x, y, w, h, fill, text, size, bold, colour, label,
             radius=None, edge=None, inset=0.04):
    item = shape(slide, kind, x, y, w, h, fill, edge, radius)
    frame = item.text_frame
    frame.word_wrap = True
    frame.auto_size = MSO_AUTO_SIZE.NONE
    frame.margin_left = frame.margin_right = Inches(inset)
    frame.margin_top = frame.margin_bottom = 0
    frame.vertical_anchor = MIDDLE
    paragraph = frame.paragraphs[0]
    paragraph.alignment = C
    _style(paragraph.add_run(), text, size, bold, colour)
    bd.check_fit(label, bd.block_height_in([(text, size, bold, 0)], w - 2 * inset), h + 0.02)
    return item


def pill(slide, x, y, w, text, fill, label, h=0.21):
    return labelled(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, fill,
                    text, SMALL_PT, True, WHITE, label, radius=0.5, inset=0.02)


def hline(slide, x, y, w, colour, weight=0.75):
    return rect(slide, x, y, w, weight / 72.0, colour)


def dot(slide, cx, cy, d, fill, edge=None):
    return shape(slide, MSO_SHAPE.OVAL, cx - d / 2, cy - d / 2, d, d, fill, edge, weight=1.5)


def notes(slide, text: str) -> None:
    slide.notes_slide.notes_text_frame.text = " ".join(text.split())


# ---------------------------------------------------------------------------
# master: strip, rail, question, band, panel
# ---------------------------------------------------------------------------
def new_slide(prs, number: int):
    _current["slide"] = number
    return bd.blank_slide(prs)


def top_strip(slide, active: tuple[int, ...] | None, right_text: str) -> None:
    rect(slide, 0, 0, SW, STRIP_H, NAVY)
    if active is None:
        return
    x = BODY_X
    for i, name in enumerate(TABS):
        text = f"{i + 1}  {name}"
        w = bd.text_width_in(text, 9.5, True) + 0.36
        on = i in active
        labelled(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, 0.09, w, 0.28,
                 WHITE if on else NAVY_SOFT, text, 9.5, True, NAVY if on else ON_NAVY,
                 f"tab {name}", radius=0.3)
        x += w + 0.08
    write(slide, x + 0.2, 0.13, BODY_R - x - 0.2, 0.22,
          [([(right_text, SMALL_PT, True, ON_NAVY), (TEAM, SMALL_PT, True, WHITE)], R, 0)],
          "strip right")


def rail(slide, mode: str) -> None:
    top = STRIP_H + 0.14
    height = BAND_Y + BAND_H - top
    rect(slide, RAIL_X, top, RAIL_W, height, NEUTRAL)
    cx, cy = RAIL_X + RAIL_W / 2, top + height / 2
    box = slide.shapes.add_textbox(Inches(cx - height / 2), Inches(cy - RAIL_W / 2),
                                   Inches(height), Inches(RAIL_W))
    box.rotation = 270
    frame = box.text_frame
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    frame.vertical_anchor = MIDDLE
    paragraph = frame.paragraphs[0]
    paragraph.alignment = C
    _style(paragraph.add_run(), mode.upper(), 10.0, True, NAVY)


def question(slide, text: str, page: str) -> None:
    holder = slide.shapes.title
    holder.left, holder.top = Inches(BODY_X), Inches(0.58)
    holder.width, holder.height = Inches(BODY_W - 1.0), Inches(0.48)
    frame = holder.text_frame
    frame.word_wrap = True
    frame.auto_size = MSO_AUTO_SIZE.NONE
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    frame.vertical_anchor = MIDDLE
    paragraph = frame.paragraphs[0]
    paragraph.alignment = L
    _style(paragraph.add_run(), text, 20.0, True, INK)
    bd.check_fit("question", bd.block_height_in([(text, 20.0, True, 0)], BODY_W - 1.0), 0.48)
    one(slide, BODY_R - 0.9, 0.70, 0.9, 0.24, page, SMALL_PT, True, MUTED, "page", align=R)


def band(slide, text: str) -> None:
    rect(slide, BODY_X, BAND_Y, BODY_W, BAND_H, NAVY)
    rect(slide, BODY_X, BAND_Y, 0.09, BAND_H, ACCENT)
    write(slide, BODY_X + 0.30, BAND_Y + 0.06, BODY_W - 0.55, BAND_H - 0.12,
          [([(text, 12.0, True, WHITE)], L, 0)], "band", anchor=MIDDLE)


def panel(slide, x, y, w, h, title, source):
    """Draw a panel and return its inner box (x, y, w, h)."""
    rect(slide, x, y, w, h, WHITE, EDGE)
    rect(slide, x, y, w, 0.30, NAVY)
    title_w = bd.text_width_in(title, HEAD_PT, True)
    one(slide, x + 0.12, y + 0.045, title_w + 0.05, 0.22, title, HEAD_PT, True, WHITE,
        f"panel {title}")
    room = w - 0.24 - title_w - 0.25
    if bd.text_width_in(source, SMALL_PT) > room:
        bd.OVERFLOWS.append(f"panel {title} source: needs {bd.text_width_in(source, SMALL_PT):.2f} "
                            f"in, has {room:.2f} in")
    one(slide, x + w - 0.12 - room, y + 0.06, room, 0.2, source, SMALL_PT, False, ON_NAVY,
        f"panel {title} source", align=R)
    return x + 0.14, y + 0.40, w - 0.28, h - 0.50


def table_header(slide, x, y, cols, labels, label):
    cursor = x
    for (w, align), text in zip(cols, labels):
        one(slide, cursor, y, w - 0.06, 0.18, text, SMALL_PT, True, MUTED, f"{label} {text}",
            align=align)
        cursor += w
    hline(slide, x, y + 0.20, sum(w for w, _ in cols), EDGE)


# ---------------------------------------------------------------------------
# cover
# ---------------------------------------------------------------------------
def cover(prs) -> None:
    slide = new_slide(prs, 0)
    top_strip(slide, None, "")
    one(slide, BODY_X, 0.12, 8.0, 0.24,
        "JINDAL STAINLESS   |   CASE COMPETITION   |   PROBLEM STATEMENT 1",
        SMALL_PT, True, WHITE, "cover strip")
    one(slide, BODY_R - 3.0, 0.12, 3.0, 0.24, "EXECUTIVE SUMMARY", SMALL_PT, True, ON_NAVY,
        "cover strip right", align=R)

    one(slide, BODY_X, 1.00, 8.3, 0.26, "AI-POWERED SURFACE DEFECT DETECTION", 12.0, True,
        ACCENT, "cover kicker")
    holder = slide.shapes.title
    holder.left, holder.top = Inches(BODY_X), Inches(1.34)
    holder.width, holder.height = Inches(8.4), Inches(1.30)
    frame = holder.text_frame
    frame.word_wrap = True
    frame.auto_size = MSO_AUTO_SIZE.NONE
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    frame.vertical_anchor = TOP
    title = "Catch every strip defect at the line, not at the customer"
    paragraph = frame.paragraphs[0]
    paragraph.alignment = L
    _style(paragraph.add_run(), title, 34.0, True, INK)
    bd.check_fit("cover title", bd.block_height_in([(title, 34.0, True, 0)], 8.4), 1.30)

    one(slide, BODY_X, 2.74, 8.3, 0.56,
        "A 6.22 MB computer-vision model that detects, classifies and scores surface defects "
        "in real time, calibrated to what a wrong call costs the mill, and running live today.",
        14.0, False, SLATE, "cover standfirst")

    hline(slide, BODY_X, 3.56, 8.3, EDGE, 1.0)
    write(slide, BODY_X, 3.72, 8.3, 1.40, [
        ([("TEAM", SMALL_PT, True, MUTED)], L, 2),
        ([(TEAM, 22.0, True, NAVY)], L, 4),
        ([("Krishna Kanta Mondal   ·   Prathmesh Walimbe", 12.0, False, SLATE)],
         L, 3),
        ([("IIT Bombay", 12.0, True, SLATE)], L, 0),
    ], "cover team")

    # live demo card
    cx, cy, cw, ch = 9.55, 1.00, BODY_R - 9.55, 3.62
    rect(slide, cx, cy, cw, ch, WHITE, NAVY)
    rect(slide, cx, cy, cw, 0.34, NAVY)
    one(slide, cx, cy + 0.07, cw, 0.22, "TRY THE LIVE DEMO", HEAD_PT, True, WHITE,
        "cover demo head", align=C)
    side = 2.05
    slide.shapes.add_picture(str(QR), Inches(cx + (cw - side) / 2), Inches(cy + 0.52),
                             width=Inches(side))
    write(slide, cx + 0.2, cy + 2.72, cw - 0.4, 1.20, [
        ([(bd.LIVE_HOST, 13.0, True, NAVY)], C, 5),
        ([("The trained detector, running in your browser. Upload a strip image; nothing "
           "leaves the page.", SMALL_PT, False, SLATE)], C, 0),
    ], "cover demo copy")

    strip = slide.shapes.add_picture(str(FAMILIES), Inches(BODY_X), Inches(5.45),
                                     width=Inches(9.2))
    keep = 300 / 424  # titles and frames only; the root-cause captions render under 9 pt
    strip.crop_bottom = 1 - keep
    strip.height = Inches(9.2 * 424 / 2046 * keep)
    write(slide, 9.95, 5.72, BODY_R - 9.95, 1.10, [
        ([("Six defect families shown on held-out test frames.", SMALL_PT, True, NAVY)], L, 4),
        ([("Slide 1: problem and solution. Slide 2: validation, roadmap and value.",
           SMALL_PT, False, SLATE)], L, 0),
    ], "cover caption")

    notes(slide, """
        Cover. Team details are placeholders to fill in before submission. The QR code opens
        https://surface-vision.github.io, the shipped detector running client-side.
    """)


# ---------------------------------------------------------------------------
# slide 1: problem and solution
# ---------------------------------------------------------------------------
PROBLEM_KPIS = [
    ("8-10%", "nickel in grade 304 by mass, so a stainless downgrade costs more"),
    ("300 m/min", "strip speed at which the human eye cannot inspect the full surface"),
    ("~15%", "of sales lost to poor quality in manufacturing, range 5-35%"),
]

ROUTE = [
    ("Casting", "inclusions"),
    ("Hot rolling", "scale, crazing, edge cracks"),
    ("Anneal, pickle", "pits, patches"),
    ("Cold rolling", "roll marks, scratches"),
    ("Finishing", "scratches"),
    ("Customer", ""),
]

OPTIONS = [
    ("YOLOv8n at 256 px", "0.752", "3.0 M params", "CHOSEN", GREEN),
    ("YOLOv8n + Severstal clean strip", "0.764", "same size", "READY", GREEN),
    ("YOLOv8s, 3.7x the parameters", "0.734", "3.5x FLOPs", "NO GAIN", AMBER),
    ("YOLOv8n retrained at 640 px", "0.734", "4x pixels", "NO GAIN", AMBER),
    ("Test-time augmentation", "-0.006", "2.39x latency", "REJECTED", RED),
]

STAGES = [
    ("1  Capture", "4 cameras across a 1.28 m strip at 0.2 mm/px", "TO BUILD", AMBER),
    ("2  Tile", "64 tiles per frame, so full resolution survives", "BUILT", GREEN),
    ("3  Detect", "YOLOv8n, 6.22 MB; threshold 0.15 set by a cost model", "BUILT", GREEN),
    ("4  Decide", "severity 0-100; coil ACCEPT, DOWNGRADE or HOLD", "BUILT", GREEN),
    ("5  Act", "HMI alert, PLC hold, MES record with root cause", "TO BUILD", AMBER),
]

RETURNS = ["Bounding box", "Defect type", "Calibrated confidence", "Severity 0-100",
           "Root cause and fix"]


def slide_one(prs) -> None:
    slide = new_slide(prs, 1)
    top_strip(slide, (0, 1), "JINDAL STAINLESS  |  PS1  |  ")
    rail(slide, "Problem and solution")
    question(slide, "Why is late detection costly, and what did we build to fix it?", "1 / 2")

    row_a, row_b = BODY_TOP, 3.72
    h_a, h_b = row_b - GAP - row_a, BODY_BOTTOM - row_b
    w_left = 6.95

    # --- P1: cost of late detection ---------------------------------------
    x, y, w, h = panel(slide, BODY_X, row_a, w_left, h_a, "Cost of Late Detection",
                       "research_notes s2a, s4a")
    step = w / len(ROUTE)
    for i, (stage, born) in enumerate(ROUTE):
        sx = x + i * step
        if born:
            one(slide, sx + 0.02, y - 0.03, step - 0.04, 0.35, born, SMALL_PT, False, MUTED,
                f"born {stage}", align=C, anchor=MSO_ANCHOR.BOTTOM)
        shape(slide, MSO_SHAPE.CHEVRON if i else MSO_SHAPE.PENTAGON, sx, y + 0.34,
              step + 0.06, 0.44, INK if stage == "Customer" else NAVY_SOFT)
        inset = 0.10 if i == 0 else 0.20
        one(slide, sx + inset, y + 0.34, step - inset - 0.10, 0.44, stage, SMALL_PT, True,
            WHITE, f"chevron {stage}", align=C, anchor=MIDDLE)
    bracket_y = y + 0.86
    hline(slide, x + 0.05, bracket_y, 4 * step - 0.15, ACCENT, 2.5)
    hline(slide, x + 4 * step + 0.05, bracket_y, 2 * step - 0.10, MUTED, 2.5)
    one(slide, x + 0.05, bracket_y + 0.06, 4 * step - 0.15, 0.20,
        "AI target: classify each defect where it is born", SMALL_PT, True, ACCENT, "target")
    one(slide, x + 4 * step + 0.05, bracket_y + 0.06, 2 * step - 0.10, 0.36,
        "Found today: by eye, or by the customer", SMALL_PT, True, MUTED, "today")

    kpi_y = y + 1.36
    kpi_w = (w - 2 * 0.16) / 3
    for i, (number, caption) in enumerate(PROBLEM_KPIS):
        kx = x + i * (kpi_w + 0.16)
        rect(slide, kx, kpi_y, 0.05, h - 1.36, ACCENT)
        write(slide, kx + 0.14, kpi_y - 0.02, kpi_w - 0.14, h - 1.34, [
            ([(number, 15.0, True, NAVY)], L, 0),
            ([(caption, SMALL_PT, False, SLATE)], L, 0),
        ], f"kpi {number}")

    # --- P2: model options tested ------------------------------------------
    px = BODY_X + w_left + GAP
    x, y, w, h = panel(slide, px, row_a, BODY_R - px, h_a, "Model Options Tested",
                       "model_study, gap1 reports")
    cols = [(2.30, L), (0.62, R), (1.07, R), (w - 3.99, C)]
    table_header(slide, x, y, cols, ["OPTION", "mAP50", "COST", "VERDICT"], "options")
    row_h = 0.285
    for i, (name, score, cost, verdict, colour) in enumerate(OPTIONS):
        ry = y + 0.26 + i * row_h
        if i % 2 == 0:
            rect(slide, x, ry - 0.03, w, row_h, NEUTRAL)
        one(slide, x + 0.06, ry, cols[0][0] - 0.1, 0.22, name, SMALL_PT, i < 2, INK,
            f"opt {name}")
        one(slide, x + cols[0][0], ry, cols[1][0] - 0.06, 0.22, score, SMALL_PT, True, NAVY,
            f"opt score {name}", align=R)
        one(slide, x + cols[0][0] + cols[1][0], ry, cols[2][0] - 0.06, 0.22, cost, SMALL_PT,
            False, SLATE, f"opt cost {name}", align=R)
        vx = x + cols[0][0] + cols[1][0] + cols[2][0]
        pill(slide, vx + 0.10, ry - 0.005, cols[3][0] - 0.16, verdict, colour, f"verdict {name}")
    one(slide, x, y + 0.26 + 5 * row_h + 0.04, w, 0.34,
        "Held-out test, scored once. Nano vs small: 95% CI [-0.008, +0.042] spans zero.",
        SMALL_PT, False, MUTED, "options note")

    # --- P3: solution architecture -------------------------------------------
    w_arch = 8.95
    x, y, w, h = panel(slide, BODY_X, row_b, w_arch, h_b, "Solution Architecture",
                       "src/inference.py, src/report.py")
    arrow = 0.20
    box_w = (w - arrow * (len(STAGES) - 1)) / len(STAGES)
    box_h = 1.02
    for i, (name, detail, status, colour) in enumerate(STAGES):
        bx = x + i * (box_w + arrow)
        rect(slide, bx, y, box_w, box_h, NEUTRAL, EDGE)
        write(slide, bx + 0.09, y + 0.08, box_w - 0.18, box_h - 0.14, [
            ([(name, BODY_PT, True, NAVY)], L, 3),
            ([(detail, SMALL_PT, False, SLATE)], L, 0),
        ], f"stage {name}")
        pill(slide, bx + 0.09, y + box_h + 0.07, box_w - 0.18, status, colour,
             f"stage status {name}")
        if i < len(STAGES) - 1:
            shape(slide, MSO_SHAPE.RIGHT_ARROW, bx + box_w + 0.03, y + box_h / 2 - 0.08,
                  arrow - 0.06, 0.16, NAVY)

    ry = y + box_h + 0.40
    one(slide, x, ry, w, 0.18, "EVERY DEFECT RETURNS", SMALL_PT, True, MUTED, "returns head")
    chip_gap = 0.10
    chip_w = (w - chip_gap * (len(RETURNS) - 1)) / len(RETURNS)
    for i, text in enumerate(RETURNS):
        labelled(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x + i * (chip_w + chip_gap), ry + 0.22,
                 chip_w, 0.30, WHITE, text, SMALL_PT, True, NAVY, f"chip {text}",
                 radius=0.25, edge=ACCENT)
    one(slide, x, ry + 0.60, w, 0.20,
        "Closed loop (proposed): operator overrides feed active learning; a per-line "
        "retrain takes 0.6 h.", SMALL_PT, False, SLATE, "closed loop")

    # --- P4: live demo --------------------------------------------------------
    dx = BODY_X + w_arch + GAP
    x, y, w, h = panel(slide, dx, row_b, BODY_R - dx, h_b, "Live Demo", "checked 13 Sep 2026")
    side = 1.22
    slide.shapes.add_picture(str(QR), Inches(x + (w - side) / 2), Inches(y), width=Inches(side))
    write(slide, x, y + side + 0.06, w, h - side - 0.06, [
        ([(bd.LIVE_HOST, HEAD_PT, True, NAVY)], C, 4),
        ([("Upload a strip image", SMALL_PT, False, SLATE)], C, 1),
        ([("Box, type, confidence, severity", SMALL_PT, False, SLATE)], C, 1),
        ([("30-50 ms per frame on CPU", SMALL_PT, False, SLATE)], C, 1),
        ([("No image leaves the browser", SMALL_PT, False, SLATE)], C, 0),
    ], "demo copy")

    band(slide, "Late, sampled inspection is the cost. A 6 MB detector that classifies every "
                "defect at line speed and names its process owner is the fix, and it already "
                "runs live.")

    notes(slide, """
        Problem: defects are born upstream (inclusions at the caster, scale and crazing at hot
        rolling) but found at finishing or by the customer, after every process step has been
        paid for. Sources: nickel share of 304 and the IISE cost-of-quality range from
        docs/research_notes.md s4a; 300 m/min from the AMETEK/Ternium case study, s2a.
        Options: reports/model_study.json (nano vs small bootstrap, TTA), resolution_study
        (640 px retrain), gap1_detection_metrics.json (joint model 0.7642). Architecture:
        tiling, detection and the coil decision are built; cameras and PLC/MES are not.
    """)


# ---------------------------------------------------------------------------
# slide 2: validation, roadmap and value
# ---------------------------------------------------------------------------
SCORECARD = [
    ("Detection accuracy",
     "mAP50 0.752 on held-out test; all 180 defective frames flagged",
     "MET", GREEN, "-"),
    ("False-alarm rate",
     "Clean strip flagged 93.7% -> 32.5% after joint training; confidence calibrated, "
     "ECE 0.142 -> 0.046",
     "PARTIAL", AMBER, "Jindal clean strip, P1"),
    ("Multiple defect types",
     "Type right on 177/180 frames. AP50: scratches 0.91, scale 0.63, roll marks 0.20; "
     "no public edge-crack data",
     "PARTIAL", AMBER, "Plant labelling, P1"),
    ("Line-speed inference",
     "6.22 MB, 30-50 ms per frame on CPU. At 250 m/min: 18 accelerators at 0.2 mm/px, "
     "or 1 downscaled",
     "MET", GREEN, "Resolution spec, P0"),
    ("Usable demo",
     "Live web demo: upload an image, get box, type, confidence, severity and root cause",
     "MET", GREEN, "-"),
    ("Scaling across grades",
     "Second dataset: cross-domain AUC 0.608 -> 0.958 and in-domain mAP50 up to 0.764; "
     "retrain 0.6 h",
     "PARTIAL", AMBER, "Stainless grades, P1-P4"),
]

DUMBBELL = [
    ("Clean strip flagged, %", 93.7, 32.5, "lower is better"),
    ("Defect recall, %", 72.0, 86.6, ""),
    ("Cross-domain AUC x100", 60.8, 95.8, ""),
    ("In-domain mAP50 x100", 75.2, 76.4, ""),
]

PHASES = [
    ("P0", "Instrument one line", 0.0, 1.5, "Frames captured at line speed"),
    ("P1", "Fine-tune on Jindal strip", 1.5, 4.0, "Clean-strip alarms under target"),
    ("P2", "Shadow mode, no interlock", 4.0, 7.0, "Operator-verified alarm rate"),
    ("P3", "PLC hold on severe defects", 7.0, 12.0, "Hold decisions trusted"),
    ("P4", "Second line and grade", 12.0, 20.0, "Fleet rollout decision"),
]

ROI = [
    ("0.5%", "21", "32", "43"),
    ("1.0%", "43", "64", "86"),
    ("2.0%", "86", "129", "172"),
]


def slide_two(prs) -> None:
    slide = new_slide(prs, 2)
    top_strip(slide, (2, 3), "JINDAL STAINLESS  |  PS1  |  ")
    rail(slide, "Proof and rollout")
    question(slide, "Does it meet the brief, and what should Jindal do first?", "2 / 2")

    row_a, row_b = BODY_TOP, 4.22
    h_a, h_b = row_b - GAP - row_a, BODY_BOTTOM - row_b
    w_left = 7.75

    # --- P1: scorecard against the brief -------------------------------------
    x, y, w, h = panel(slide, BODY_X, row_a, w_left, h_a, "Scorecard Against the Brief",
                       "reports/*.json, exec_summary_checks.json")
    cols = [(1.50, L), (4.05, L), (0.86, C), (w - 6.41, L)]
    table_header(slide, x, y, cols, ["CONSIDERATION", "EVIDENCE, HELD-OUT DATA", "STATUS",
                                     "NEXT STEP"], "score")
    row_h = (h - 0.26) / len(SCORECARD)
    for i, (topic, evidence, status, colour, closer) in enumerate(SCORECARD):
        ry = y + 0.26 + i * row_h
        if i % 2 == 0:
            rect(slide, x, ry - 0.02, w, row_h, NEUTRAL)
        cursor = x
        one(slide, cursor + 0.06, ry + 0.01, cols[0][0] - 0.1, row_h - 0.02, topic, SMALL_PT,
            True, INK, f"score topic {topic}")
        cursor += cols[0][0]
        one(slide, cursor, ry + 0.01, cols[1][0] - 0.1, row_h - 0.02, evidence, SMALL_PT,
            False, SLATE, f"score evidence {topic}")
        cursor += cols[1][0]
        pill(slide, cursor + 0.04, ry + (row_h - 0.25) / 2 - 0.02, cols[2][0] - 0.08, status,
             colour, f"score status {topic}")
        cursor += cols[2][0]
        one(slide, cursor + 0.06, ry + 0.01, cols[3][0] - 0.08, row_h - 0.02, closer, SMALL_PT,
            False, SLATE, f"score closer {topic}")

    # --- P2: joint training dumbbell --------------------------------------------
    px = BODY_X + w_left + GAP
    x, y, w, h = panel(slide, px, row_a, BODY_R - px, h_a, "Training on Real Clean Strip",
                       "gap1 reports, conf 0.15")
    dot(slide, x + 0.08, y + 0.09, 0.13, WHITE, MUTED)
    one(slide, x + 0.22, y, 1.6, 0.2, "Shipped, NEU-DET only", SMALL_PT, False, SLATE, "lg a")
    dot(slide, x + 2.02, y + 0.09, 0.13, ACCENT)
    one(slide, x + 2.16, y, w - 2.16, 0.2, "Joint, + Severstal", SMALL_PT, True, ACCENT, "lg b")

    label_w = 1.40
    plot_x, plot_w = x + label_w + 0.50, w - label_w - 0.95
    top = y + 0.36
    row_h = 0.40
    for i, (name, before, after, hint) in enumerate(DUMBBELL):
        cy = top + i * row_h + 0.22
        write(slide, x, cy - 0.20, label_w, 0.42, [
            ([(name, SMALL_PT, True, INK)], L, 0),
        ] + ([([(hint, SMALL_PT, False, MUTED)], L, 0)] if hint else []), f"db label {name}",
            anchor=MIDDLE)
        hline(slide, plot_x, cy, plot_w, EDGE, 0.75)
        bx, ax = plot_x + plot_w * before / 100, plot_x + plot_w * after / 100
        rect(slide, min(bx, ax), cy - 0.02, abs(ax - bx), 0.04, ON_NAVY)
        dot(slide, bx, cy, 0.15, WHITE, MUTED)
        dot(slide, ax, cy, 0.15, ACCENT)
        pairs = sorted(((before, bx, MUTED, False), (after, ax, ACCENT, True)), key=lambda v: v[1])
        for side, (value, vx, colour, bold) in zip(("left", "right"), pairs):
            lx = vx - 0.12 - 0.40 if side == "left" else vx + 0.12
            one(slide, lx, cy - 0.09, 0.40, 0.18, f"{value:.1f}", SMALL_PT, bold, colour,
                f"db value {name} {value}", align=R if side == "left" else L)
    axis_y = top + len(DUMBBELL) * row_h + 0.04
    hline(slide, plot_x, axis_y, plot_w, MUTED, 0.75)
    for tick in (0, 50, 100):
        one(slide, plot_x + plot_w * tick / 100 - 0.25, axis_y + 0.03, 0.5, 0.18, str(tick),
            SMALL_PT, False, MUTED, f"tick {tick}", align=C)
    one(slide, x, axis_y + 0.24, w, 0.2,
        "473 of 505 clean Severstal frames flagged before, 164 after", SMALL_PT, False, MUTED,
        "db note")

    # --- P3: pilot roadmap ---------------------------------------------------------
    x, y, w, h = panel(slide, BODY_X, row_b, w_left, h_b, "Pilot Roadmap, 20 Months",
                       "phase plan; 0.6 h retrain measured")
    name_w, gate_w = 2.30, 2.25
    bar_x, bar_w = x + name_w, w - name_w - gate_w - 0.40
    one(slide, x, y, name_w, 0.18, "PHASE", SMALL_PT, True, MUTED, "rm phase")
    one(slide, x + w - gate_w, y, gate_w, 0.18, "EXIT GATE", SMALL_PT, True, MUTED, "rm gate")
    for month in (0, 4, 8, 12, 16, 20):
        one(slide, bar_x + bar_w * month / 20 - 0.2, y, 0.4, 0.18, f"m{month}", SMALL_PT, False,
            MUTED, f"rm m{month}", align=C)
    hline(slide, x, y + 0.21, w, EDGE)
    row_h = (h - 0.27) / len(PHASES)
    for i, (code, name, start, end, gate) in enumerate(PHASES):
        ry = y + 0.27 + i * row_h
        write(slide, x, ry + 0.03, name_w - 0.1, row_h - 0.04,
              [([(f"{code}  ", SMALL_PT, True, ACCENT), (name, SMALL_PT, False, INK)], L, 0)],
              f"rm name {code}")
        rect(slide, bar_x, ry + row_h / 2 - 0.005, bar_w, 0.01, EDGE)
        rect(slide, bar_x + bar_w * start / 20, ry + 0.05, bar_w * (end - start) / 20,
             row_h - 0.10, NAVY if i < 3 else NAVY_SOFT)
        one(slide, x + w - gate_w, ry + 0.03, gate_w, row_h - 0.04, gate, SMALL_PT, False, SLATE,
            f"rm gate {code}")
    bracket = bar_x + bar_w * 7 / 20
    rect(slide, bracket, y + 0.23, 0.012, h - 0.25, ACCENT)
    one(slide, bracket + 0.08, y + 0.27 + 0.03, bar_w * 13 / 20 - 0.08, row_h - 0.04,
        "pilot go / no-go", SMALL_PT, True, ACCENT, "rm pilot")

    # --- P4: business case ---------------------------------------------------------------
    x, y, w, h = panel(slide, px, row_b, BODY_R - px, h_b, "Business Case",
                       "research_notes s3a, FY26")
    write(slide, x, y, w, 0.40, [
        ([("2.57 Mt x 1% downgraded x 10% discount x INR 167,407/t", SMALL_PT, False, SLATE)],
         L, 1),
        ([("= INR 43 cr a year at risk", BODY_PT, True, NAVY)], L, 0),
    ], "roi formula")
    gy = y + 0.44
    cols = [1.55, (w - 1.55) / 3, (w - 1.55) / 3, (w - 1.55) / 3]
    head = ["Downgraded \\ discount", "10%", "15%", "20%"]
    cursor = x
    for cw, text in zip(cols, head):
        one(slide, cursor, gy, cw - 0.06, 0.18, text, SMALL_PT, True, MUTED, f"roi h {text}",
            align=L if cursor == x else R)
        cursor += cw
    hline(slide, x, gy + 0.20, w, EDGE)
    for i, row in enumerate(ROI):
        ry = gy + 0.23 + i * 0.19
        cursor = x
        for j, (cw, text) in enumerate(zip(cols, row)):
            mid = i == 1 and j == 1
            one(slide, cursor, ry, cw - 0.06, 0.18, text if j else f"{text} of volume", SMALL_PT,
                mid, ACCENT if mid else SLATE, f"roi {i}{j}", align=L if j == 0 else R)
            cursor += cw
    write(slide, x, gy + 0.84, w, y + h - gy - 0.84, [
        ([("Recovering 30% is about INR 13 cr a year: payback 0.7-2 years on a USD 1-3M "
           "line system (unverified range).", SMALL_PT, False, SLATE)], L, 2),
        ([("Needed from Jindal: the real downgrade rate.", SMALL_PT, True, NAVY)], L, 0),
    ], "roi close")

    band(slide, "Recommendation: approve a 7-month pilot on one line, ending in shadow mode. "
                "The model already clears accuracy, speed and demo; only Jindal's own strip can "
                "close false alarms, roll marks and edge cracks.")

    notes(slide, """
        Scorecard sources: reports/model_study.json (mAP50 0.7524), gap1_cross_domain.json
        (473/505 clean frames flagged before, 164/505 after; defect crop recall 72.0% to
        86.6%; AUC 0.608 to 0.958), calibration.json (ECE), gc10_coverage.json (roll marks
        0.203), evaluation at 256 px (scratches 0.909, rolled-in scale 0.630),
        benchmark.json mill.scenarios at 250 m/min and 320 px (18 accelerators tiled, 1
        downscaled), deck/build/exec_summary_checks.json (177/180 type correct and 180/180
        flagged at conf 0.15). Business case: FY26 volume and realisation are Jindal
        reported figures; the downgrade rate and discount are swept, not known; the system
        cost is an unverified USD 1-3M industry range, so payback is quoted as a range.
    """)


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------
def verify(prs) -> list[str]:
    problems: list[str] = []
    if len(prs.slides) != 3:
        problems.append(f"expected cover + 2 slides, found {len(prs.slides)}")
    for index, slide in enumerate(prs.slides):
        for item in slide.shapes:
            if item.rotation:
                continue
            right = (item.left + item.width) / 914400.0
            bottom = (item.top + item.height) / 914400.0
            if item.left < 0 or item.top < 0 or right > SW + 0.01 or bottom > SH + 0.01:
                problems.append(f"slide {index}: shape off the page ({right:.2f}, {bottom:.2f})")
            if item.shape_type == MSO_SHAPE_TYPE.PICTURE:
                continue
        if not slide.notes_slide.notes_text_frame.text.strip():
            problems.append(f"slide {index}: no speaker notes")
    for number in (1, 2):
        if WORDS.get(number, 0) > 500:
            problems.append(f"slide {number}: {WORDS[number]} words, ceiling is 500")
    return problems


def main() -> None:
    prs = Presentation()
    prs.slide_width = Inches(SW)
    prs.slide_height = Inches(SH)
    cover(prs)
    slide_one(prs)
    slide_two(prs)
    prs.save(OUT)

    problems = verify(prs) + FLOOR_BREACHES + bd.OVERFLOWS
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
    print("words per content slide:", {k: v for k, v in WORDS.items() if k})
    if problems:
        print("\nPROBLEMS")
        for item in problems:
            print("  -", item)
        sys.exit(1)
    print("no overflow, floor, bounds or word-count problems")


if __name__ == "__main__":
    main()
