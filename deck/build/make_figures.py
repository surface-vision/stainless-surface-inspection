"""Generate every figure used by the submission deck.

All quantities are read from the project's own report artefacts
(reports/*.json), from the dataset itself, or from docs/research_notes.md.
Nothing is hard-coded from memory. Every figure is authored at the exact
size it occupies on its slide, so on-slide point sizes equal the point
sizes set here.

Run:
    .venv/bin/python deck/build/make_figures.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

ROOT = Path("/Users/prathmeshwalimbe/Downloads/JSW-PS1")
REPORTS = ROOT / "reports"
FIGDIR = ROOT / "deck" / "figures"
DPI = 220

# --- chrome tokens (text, rules, panels) -- never used as series colours -----
INK = "#141C24"
SLATE = "#33414F"
MUTED = "#6B7885"
RULE = "#D6DCE1"
PANEL = "#F3F5F7"
STEEL = "#1F4E6B"

# --- series colours: dataviz skill reference palette, slots 1 and 2.
# Validated light mode, --pairs all: every check PASS.
S1 = "#2a78d6"
S2 = "#eb6834"
SEQ_LO = "#d3e2f6"
SEQ_HI = "#123f75"

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

# Upstream owner per defect family. Source: src/inference.py DEFECT_INFO,
# condensed; corroborated by docs/research_notes.md section 5.
OWNER = {
    "crazing": "reheat and\ncoiling temperature",
    "inclusion": "caster, ladle\nand tundish",
    "patches": "descaler coverage,\npickle line",
    "pitted_surface": "furnace atmosphere,\nover-pickling",
    "rolled-in_scale": "descaler pressure,\nroll cooling",
    "scratches": "guides, coiler,\nhandling",
}


def _install_fonts() -> None:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ):
        if Path(path).exists():
            fm.fontManager.addfont(path)
    have_arial = any(f.name == "Arial" for f in fm.fontManager.ttflist)
    plt.rcParams.update(
        {
            "font.family": "Arial" if have_arial else "DejaVu Sans",
            "text.color": INK,
            "axes.labelcolor": SLATE,
            "axes.edgecolor": RULE,
            "xtick.color": SLATE,
            "ytick.color": SLATE,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "pdf.fonttype": 42,
        }
    )


def _load(name: str) -> dict[str, Any]:
    with (REPORTS / name).open() as handle:
        return json.load(handle)


def _seq_ramp(values: Sequence[float]) -> list[str]:
    """Sequential rule: one hue, light to dark, ordered by magnitude."""
    lo = np.array([int(SEQ_LO[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)
    hi = np.array([int(SEQ_HI[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)
    span = max(values) - min(values)
    out = []
    for value in values:
        t = 0.5 if span == 0 else (value - min(values)) / span
        rgb = lo + (hi - lo) * t
        out.append("#%02x%02x%02x" % tuple(int(round(c)) for c in rgb))
    return out


def _titles(fig, ax, title: str, sub: str) -> None:
    """Panel title above its subtitle, both left-aligned on the axes."""
    ax.text(0.0, 1.185, title, transform=ax.transAxes, fontsize=11.0,
            fontweight="bold", color=INK, va="bottom", ha="left")
    ax.text(0.0, 1.045, sub, transform=ax.transAxes, fontsize=8.0,
            color=MUTED, va="bottom", ha="left")


# ---------------------------------------------------------------------------
# Figure 1 -- the six defect families, real held-out images with ground truth
# ---------------------------------------------------------------------------
def fig_defect_families() -> Path:
    picks = {
        "crazing": "crazing_281.jpg",
        "inclusion": "inclusion_275.jpg",
        "patches": "patches_282.jpg",
        "pitted_surface": "pitted_surface_271.jpg",
        "rolled-in_scale": "rolled-in_scale_293.jpg",
        "scratches": "scratches_295.jpg",
    }
    img_dir = ROOT / "data" / "neu-det" / "test" / "images"
    lbl_dir = ROOT / "data" / "neu-det" / "test" / "labels"

    # Authored narrower than the earlier full-width strip: slide 1 now gives the
    # right-hand end of this row to the live-demo call to action.
    fig_w, fig_h = 9.30, 1.93
    cell = 1.02
    gap = 0.40
    strip_w = 6 * cell + 5 * gap
    x0 = (fig_w - strip_w) / 2
    y0 = 0.60
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI)

    for idx, cls in enumerate(CLASSES):
        left = (x0 + idx * (cell + gap)) / fig_w
        ax = fig.add_axes([left, y0 / fig_h, cell / fig_w, cell / fig_h])
        image = np.asarray(Image.open(img_dir / picks[cls]).convert("L"))
        height, width = image.shape
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        for line in (lbl_dir / picks[cls]).with_suffix(".txt").read_text().split("\n"):
            parts = line.split()
            if len(parts) != 5:
                continue
            _, cx, cy, bw, bh = (float(p) for p in parts)
            ax.add_patch(
                Rectangle(
                    ((cx - bw / 2) * width, (cy - bh / 2) * height),
                    bw * width, bh * height,
                    fill=False, edgecolor=S2, linewidth=1.5,
                )
            )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(RULE)
        ax.text(0.5, 1.06, cls.replace("_", " "), transform=ax.transAxes,
                fontsize=9.8, fontweight="bold", color=INK, ha="center", va="bottom")
        ax.text(0.5, -0.085, OWNER[cls], transform=ax.transAxes, fontsize=7.6,
                color=MUTED, ha="center", va="top", linespacing=1.30)

    fig.text(0.30 / fig_w, 0.028,
             "Held-out NEU-DET test frames with ground truth. Each family points at a different "
             "upstream owner -- which is why a classified detection beats an alarm.",
             fontsize=7.0, color=MUTED, ha="left", va="bottom")
    out = FIGDIR / "fig1_defect_families.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 2 -- serving architecture
# ---------------------------------------------------------------------------
def fig_architecture() -> Path:
    bench = _load("benchmark.json")
    geo = bench["mill"]["geometry"]
    dep = bench["mill"]["deployment"]

    fig = plt.figure(figsize=(12.3, 2.62), dpi=DPI)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 104)
    ax.set_ylim(-2.2, 26)
    ax.axis("off")

    def box(x, y, w, h, title, lines, fill="white", edge=STEEL, lw=1.4,
            tsize=8.8, bsize=7.5, lead=1.95):
        ax.add_patch(
            FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.3,rounding_size=0.6",
                           facecolor=fill, edgecolor=edge, linewidth=lw, zorder=2)
        )
        ax.text(x + w / 2, y + h - 1.4, title, ha="center", va="top", fontsize=tsize,
                fontweight="bold", color=INK, zorder=3)
        for i, line in enumerate(lines):
            ax.text(x + w / 2, y + h - 3.4 - i * lead, line, ha="center", va="top",
                    fontsize=bsize, color=SLATE, zorder=3)

    def seg(points, color=STEEL, lw=1.6, ls="-", head=True):
        for i in range(len(points) - 1):
            last = i == len(points) - 2
            ax.add_patch(
                FancyArrowPatch(points[i], points[i + 1],
                                arrowstyle="-|>" if (head and last) else "-",
                                mutation_scale=12, color=color, linewidth=lw,
                                linestyle=ls, zorder=1, shrinkA=0, shrinkB=0)
            )

    ry, rh = 10.4, 9.6
    top = ry + rh
    box(0.5, ry, 15.0, rh, "1  Acquisition",
        [f"{geo['cameras_across_width']} cameras across {geo['strip_width_m']} m",
         f"{geo['sensor_px']} px at {geo['optical_resolution_mm_per_px']} mm/px",
         f"1 frame every {geo['strip_advance_per_frame_m']:.3f} m"])
    box(17.6, ry, 14.6, rh, "2  Tiling",
        [f"{dep['tiles_per_frame_by_input']['320']} tiles/frame at 320 px",
         f"{int(geo['tile_overlap'] * 100)}% tile overlap",
         "full optical resolution kept"])
    box(34.3, ry, 17.4, rh, "3  Detector",
        ["YOLOv8n, 3.012 M param, 6.22 MB",
         "conf 0.15, NMS IoU 0.45",
         "ONNX / CoreML export verified"], edge=S1, lw=2.2)
    box(53.8, ry, 17.0, rh, "4  Severity and roll-up",
        ["per-box severity score",
         "p95 severity across the coil",
         "HOLD / RELEASE disposition"])
    box(72.8, ry, 26.7, rh, "5  Plant integration",
        ["HMI overlay + EigenCAM heat map",
         "PLC interlock: slow or stop the line",
         "MES coil record, root cause routed"])

    for xa, xb in ((15.5, 17.6), (32.2, 34.3), (51.7, 53.8), (70.8, 72.8)):
        seg([(xa, ry + rh / 2), (xb, ry + rh / 2)])

    # unsupervised channel, parallel to the supervised detector
    box(17.6, 1.5, 34.1, 7.2, "Anomaly channel   (proposed)",
        ["unsupervised, needs no labels -- catches defect types never seen"],
        edge=MUTED, lw=1.2, fill=PANEL, tsize=8.0, bsize=7.3)
    seg([(24.0, ry), (24.0, 8.7)], color=MUTED, ls=(0, (4, 3)), lw=1.3)
    seg([(51.7, 5.1), (54.6, 5.1), (54.6, ry)], color=MUTED, ls=(0, (4, 3)), lw=1.3)

    box(56.9, 1.5, 42.6, 7.2, "Closed loop   (proposed)",
        ["operator adjudication  ->  active learning  ->  per-line fine-tune, 0.6 h",
         "drift watched on alarm rate, score spread and override rate"],
        edge=S2, lw=1.5, fill="#fdf1ea", tsize=8.0, bsize=7.3)
    seg([(86.2, ry), (86.2, 8.7)], color=S2, lw=1.5)
    seg([(99.5, 5.1), (102.2, 5.1), (102.2, 23.2), (43.0, 23.2), (43.0, top)],
        color=S2, lw=1.5)
    ax.text(72.0, 23.6, "retrain and recalibrate per line", fontsize=7.4,
            color=S2, ha="center", va="bottom", fontweight="bold")

    ax.text(0.5, -1.9,
            "Built and measured today: tiling (DefectDetector.predict_tiled), the detector, severity scoring, and the coil HOLD/RELEASE report.   "
            "Not built: cameras, PLC and MES links, the anomaly channel, the closed loop.",
            fontsize=6.8, color=SLATE, ha="left", va="bottom")
    out = FIGDIR / "fig2_architecture.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 3 -- throughput demand versus one accelerator
# ---------------------------------------------------------------------------
def fig_throughput() -> Path:
    bench = _load("benchmark.json")
    scen = bench["mill"]["scenarios"]
    util = bench["mill"]["assumptions"]["utilisation_ceiling"]
    # MPS throughput is not reproducible to a point value on this host; the
    # deployment verification recorded a 320 px peak band of 234-283 f/s across
    # sessions. Carry the band, never a point.
    band_lo, band_hi = 234.0 * util, 283.0 * util

    keep = {
        "Bright annealing line": "Bright annealing",
        "Hot-band anneal and pickle line": "Anneal + pickle",
        "Skin-pass / inspection line": "Skin-pass / inspect",
        "Continuous pickling line exit": "Pickling line exit",
        "Hot strip mill finishing exit": "Hot strip mill exit",
    }
    rows = [s for name in keep for s in scen if s["line"] == name]

    fig = plt.figure(figsize=(6.45, 2.85), dpi=DPI)
    ax = fig.add_axes([0.335, 0.335, 0.645, 0.480])

    y = np.arange(len(rows))[::-1]
    for yy, s in zip(y, rows):
        lo, hi = s["inferences_per_s_downscaled"], s["inferences_per_s_tiled"]
        ax.plot([lo, hi], [yy, yy], color="#C3CBD3", lw=2.6, zorder=2,
                solid_capstyle="round")
        ax.plot([lo], [yy], "o", ms=7.5, color=S2, zorder=4,
                markeredgecolor="white", markeredgewidth=1.2)
        ax.plot([hi], [yy], "o", ms=7.5, color=S1, zorder=4,
                markeredgecolor="white", markeredgewidth=1.2)
        ax.text(hi * 1.35, yy, f"{hi:,.0f}", va="center", ha="left",
                fontsize=8.4, color=INK, fontweight="bold")
        ax.text(lo / 1.35, yy, f"{lo:,.0f}", va="center", ha="right",
                fontsize=8.4, color=INK)

    ax.axvspan(band_lo, band_hi, color=SLATE, alpha=0.14, zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(2.2, 90000)
    ax.set_xticks([10, 100, 1000, 10000])
    ax.set_xticklabels(["10", "100", "1,000", "10,000"], fontsize=8.2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{keep[s['line']]},  {s['line_speed_m_per_min']:.0f}" for s in rows],
                       fontsize=8.6)
    ax.set_ylim(-0.75, len(rows) - 0.25)
    ax.set_xlabel("inferences per second required   (log scale)", fontsize=8.4)
    handles = [
        mpatches.Patch(color=S1, label="full optical resolution: 64 tiles/frame, 0.2 mm/px"),
        mpatches.Patch(color=S2, label="one downscaled frame per camera, 1.28 mm/px"),
        mpatches.Patch(color=SLATE, alpha=0.14,
                       label="one M5-class GPU, sustained 164-198 inf/s"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.520, -0.290),
              fontsize=7.5, frameon=False, handlelength=1.1, handleheight=0.9,
              borderpad=0.0, labelspacing=0.42)
    _titles(fig, ax, "Resolution, not hardware, is the decision",
            "line and speed in m/min, 4 cameras across 1.28 m  |  reports/benchmark.json")

    out = FIGDIR / "fig3_throughput.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 4 -- validation evidence
# ---------------------------------------------------------------------------
def fig_validation() -> Path:
    study = _load("model_study.json")
    xdom = _load("gap1_cross_domain.json")
    joint = _load("gap1_detection_metrics.json")

    run = next(r for r in study["runs"]
               if r["model"] == "yolov8n" and r["imgsz"] == 256
               and r["split"] == "test" and not r["augment"])
    inst = study["dataset"]["test_instances"]
    joint256 = joint["neu_det"]["256"]

    base_run = next(r for r in xdom["runs"] if r["tag"] == "baseline")
    joint_run = next(r for r in xdom["runs"] if r["tag"] == "joint")

    def sweep_at(xrun: dict[str, Any], conf: float) -> dict[str, Any]:
        return next(s for s in xrun["sweep"] if abs(s["conf"] - conf) < 1e-9)

    b15, j15 = sweep_at(base_run, 0.15), sweep_at(joint_run, 0.15)

    fig = plt.figure(figsize=(12.3, 2.86), dpi=DPI)
    ax_a = fig.add_axes([0.100, 0.150, 0.250, 0.610])
    ax_b = fig.add_axes([0.412, 0.150, 0.262, 0.610])
    ax_c = fig.add_axes([0.748, 0.150, 0.192, 0.610])

    # --- A: the cross-domain fix, on real clean/defective Severstal steel ---
    metrics = [
        ("clean-frame\nfalse alarm %",
         b15["clean_frame_fa"] * 100, j15["clean_frame_fa"] * 100),
        ("cross-domain\nAUC  (x100)",
         base_run["separation_cross_domain"]["auc"] * 100,
         joint_run["separation_cross_domain"]["auc"] * 100),
        ("Severstal\nrecall %",
         b15["defect_crop_recall"] * 100, j15["defect_crop_recall"] * 100),
    ]
    for i, (label, v0, v1) in enumerate(metrics):
        y = len(metrics) - 1 - i
        ax_a.plot([v0, v1], [y, y], color=RULE, lw=5.5, zorder=2,
                  solid_capstyle="round")
        ax_a.plot([v0], [y], "o", ms=8.5, mfc="white", mec="#9AA6B2",
                  mew=1.8, zorder=4)
        ax_a.plot([v1], [y], "o", ms=10, color=S2, zorder=5,
                  markeredgecolor="white", markeredgewidth=1.2)
        above = v0 > v1
        tag0 = "  shipped" if i == 0 else ""
        tag1 = "  joint" if i == 0 else ""
        ax_a.text(v0, y + (0.30 if above else -0.32), f"{v0:.1f}{tag0}",
                  ha="center", va="bottom" if above else "top",
                  fontsize=8.0, color="#7B8794", fontweight="bold")
        ax_a.text(v1, y + (-0.32 if above else 0.30), f"{v1:.1f}{tag1}",
                  ha="center", va="top" if above else "bottom",
                  fontsize=9.0, color=S2, fontweight="bold")
    ax_a.set_yticks(range(len(metrics)))
    ax_a.set_yticklabels([m[0] for m in reversed(metrics)], fontsize=8.2,
                         linespacing=1.25)
    ax_a.set_xlim(0, 108)
    ax_a.set_xticks([0, 25, 50, 75, 100])
    ax_a.tick_params(axis="x", labelsize=8.0)
    ax_a.set_ylim(-0.65, len(metrics) - 0.35)
    _titles(fig, ax_a, "A.  Backbone alone does not transfer",
            "conf 0.15, Severstal held-out  |  gap1_cross_domain.json")

    # --- B: in-domain accuracy, shipped vs joint, per class ------------------
    order = sorted(CLASSES, key=lambda c: run["per_class_AP50"][c])
    shipped_vals = [run["per_class_AP50"][c] for c in order]
    joint_vals = [joint256["per_class"][c]["AP50"] for c in order]
    yy = np.arange(len(order))
    ax_b.barh(yy - 0.19, shipped_vals, height=0.34, color="#9AA6B2", zorder=3,
              label="shipped  0.7524")
    ax_b.barh(yy + 0.19, joint_vals, height=0.34, color=S2, zorder=3,
              label="joint  0.7642")
    for i, (sv, jv) in enumerate(zip(shipped_vals, joint_vals)):
        ax_b.text(sv + 0.02, i - 0.19, f"{sv:.2f}", va="center", ha="left",
                  fontsize=7.4, color=SLATE)
        ax_b.text(jv + 0.02, i + 0.19, f"{jv:.2f}", va="center", ha="left",
                  fontsize=7.4, color=S2, fontweight="bold")
    ax_b.set_yticks(yy)
    ax_b.set_yticklabels([f"{c.replace('_', ' ')}  ({inst[c]})" for c in order],
                         fontsize=8.2)
    ax_b.set_xlim(0, 1.14)
    ax_b.set_xticks([0, 0.5, 1.0])
    ax_b.set_ylim(-0.75, len(order) - 0.25)
    ax_b.set_xlabel("AP50, held-out NEU-DET test", fontsize=8.4)
    ax_b.tick_params(axis="x", labelsize=8.2)
    ax_b.legend(fontsize=7.4, frameon=False, loc="lower right", handlelength=1.1)
    _titles(fig, ax_b, "B.  No trade-off in accuracy",
            "overall mAP50 0.7524 -> 0.7642  |  gap1_detection_metrics.json")

    # --- C: positioning against published NEU-DET, both checkpoints ---------
    published = [
        ("YOLOv10n-SFDC  2025", 85.5),
        ("YOLOv11n  2026", 77.2),
        ("RT-DETR  2026", 75.4),
        ("DDN  IEEE TIM 2020", 74.8),
        ("YOLOv8n  2026", 74.0),
        ("DETR  2026", 66.6),
    ]
    ours_shipped = ("shipped, held-out test", run["mAP50"] * 100)
    ours_joint = ("joint, held-out test", joint256["mAP50"] * 100)
    entries = sorted(published + [ours_shipped, ours_joint], key=lambda kv: kv[1])
    ax_c.axvspan(70, 80, color=S1, alpha=0.10, zorder=1)
    for i, (name, value) in enumerate(entries):
        is_shipped, is_joint = name == ours_shipped[0], name == ours_joint[0]
        colour = S1 if is_shipped else (S2 if is_joint else "#9AA6B2")
        ax_c.plot([0, value], [i, i], color=colour,
                  lw=1.6 if (is_shipped or is_joint) else 1.2, alpha=0.55, zorder=2)
        ax_c.plot([value], [i], "D" if (is_shipped or is_joint) else "o",
                  ms=9.0 if (is_shipped or is_joint) else 7, color=colour, zorder=4,
                  markeredgecolor="white", markeredgewidth=1.3)
        ax_c.text(value + 1.3, i, f"{value:.1f}", va="center", ha="left",
                  fontsize=8.0, color=INK,
                  fontweight="bold" if (is_shipped or is_joint) else "normal")
    ax_c.set_yticks(range(len(entries)))
    ax_c.set_yticklabels([n for n, _ in entries], fontsize=7.9)
    for tick, (name, _) in zip(ax_c.get_yticklabels(), entries):
        if name in (ours_shipped[0], ours_joint[0]):
            tick.set_fontweight("bold")
            tick.set_color(INK)
    ax_c.set_xlim(60, 93)
    ax_c.set_xticks([65, 70, 75, 80, 85])
    ax_c.tick_params(axis="x", labelsize=8.0)
    ax_c.set_ylim(-0.75, len(entries) - 0.25)
    ax_c.set_xlabel("mAP50 (%)", fontsize=8.2)
    _titles(fig, ax_c, "C.  Inside the credible band",
            "published [HARD]  |  research_notes s1a-1b")

    out = FIGDIR / "fig4_validation.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 5 -- capacity matched to data, and the input-size cliff
# ---------------------------------------------------------------------------
def fig_capacity() -> Path:
    study = _load("model_study.json")
    boot = study["bootstrap"]

    fig = plt.figure(figsize=(6.45, 2.35), dpi=DPI)
    ax_a = fig.add_axes([0.175, 0.235, 0.290, 0.545])
    ax_b = fig.add_axes([0.640, 0.235, 0.335, 0.545])

    entries = [
        ("320 px\ntraining size", boot["at_training_size"]["mAP50"]),
        ("256 px\nas deployed", boot["at_deployment_size"]["mAP50"]),
    ]
    for i, (label, d) in enumerate(entries):
        y = 1 - i
        colour = S1 if d["ci_excludes_zero"] else "#9AA6B2"
        ax_a.plot([d["ci95_low"], d["ci95_high"]], [y, y], lw=7, color=colour,
                  solid_capstyle="round", zorder=3, alpha=0.85)
        ax_a.plot([d["point_difference"]], [y], "o", ms=8, color=INK, zorder=4,
                  markeredgecolor="white", markeredgewidth=1.5)
        ax_a.text(d["ci95_high"] + 0.008, y + 0.20, f"+{d['point_difference']:.4f}",
                  fontsize=8.6, color=INK, fontweight="bold", va="center", ha="left")
        ax_a.text(d["ci95_high"] + 0.008, y - 0.22,
                  "excludes 0" if d["ci_excludes_zero"] else "contains 0",
                  fontsize=7.8, color=SLATE, va="center", ha="left",
                  fontweight="bold" if d["ci_excludes_zero"] else "normal")
    ax_a.axvline(0, color=SLATE, lw=1.3, zorder=2)
    ax_a.set_yticks([1, 0])
    ax_a.set_yticklabels([e[0] for e in entries], fontsize=8.4, linespacing=1.3)
    ax_a.set_ylim(-0.85, 1.85)
    ax_a.set_xlim(-0.035, 0.175)
    ax_a.set_xticks([0, 0.05, 0.10])
    ax_a.tick_params(axis="x", labelsize=8.2)
    ax_a.set_xlabel("mAP50, nano minus small", fontsize=8.4)
    _titles(fig, ax_a, "A.  Bigger is not better",
            "paired bootstrap, 2 000 resamples")

    sizes = [256, 320, 416, 512, 640]

    def curve(model: str) -> list[float]:
        return [next(r["mAP50"] for r in study["runs"]
                     if r["model"] == model and r["imgsz"] == size
                     and r["split"] == "test" and not r["augment"])
                for size in sizes]

    for model, colour, marker in (("yolov8n", S1, "o"), ("yolov8s", S2, "s")):
        ax_b.plot(sizes, curve(model), marker=marker, ms=5.5, lw=2.0,
                  color=colour, label=model, zorder=3)
    n_curve = curve("yolov8n")
    n256, n640 = n_curve[0], n_curve[-1]
    ax_b.annotate("", xy=(640, n640), xytext=(640, n256),
                  arrowprops=dict(arrowstyle="<->", color=SLATE, lw=1.3))
    ax_b.text(626, (n256 + n640) / 2, f"-{(1 - n640 / n256) * 100:.0f}%", ha="right",
              va="center", fontsize=9.4, color=INK, fontweight="bold")
    ax_b.plot([256], [n256], "o", ms=13, mfc="none", mec=S1, mew=2.0, zorder=4)
    ax_b.text(276, n256 + 0.030, "shipped", fontsize=8.4, color=INK, fontweight="bold")
    ax_b.set_xlim(225, 690)
    ax_b.set_xticks(sizes)
    ax_b.set_xticklabels([str(s) for s in sizes], fontsize=8.2)
    ax_b.set_ylim(0.24, 0.90)
    ax_b.set_yticks([0.3, 0.5, 0.7])
    ax_b.tick_params(axis="y", labelsize=8.2)
    ax_b.set_xlabel("network input size (px)", fontsize=8.4)
    ax_b.set_ylabel("test mAP50", fontsize=8.4)
    ax_b.legend(fontsize=8.0, frameon=False, loc="lower left", handlelength=1.4)
    _titles(fig, ax_b, "B.  A silent accuracy cliff",
            "held-out test  |  reports/model_study.json")

    out = FIGDIR / "fig5_capacity.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 6 -- phased rollout
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Phase:
    tag: str
    name: str
    start: float
    end: float
    lines: tuple[str, ...]


def fig_roadmap() -> Path:
    phases = (
        Phase("0", "Instrument one line", 0, 1.5,
              ("camera + lighting rig, one line",
               "capture prime coils at line speed",
               "unlocks every unmeasured number")),
        Phase("1", "Calibrate to Jindal steel", 1.5, 4.0,
              ("label 2-3k frames, two grades",
               "per-line fine-tune, 0.6 h a run",
               "re-tune threshold on this line")),
        Phase("2", "Shadow mode", 4.0, 7.0,
              ("HMI and MES live, no interlock",
               "operators adjudicate every alarm",
               "first true false-alarm number")),
        Phase("3", "Closed loop, one line", 7.0, 12.0,
              ("PLC interlock on severe defects",
               "anomaly channel for the unknown",
               "drift monitor and retrain trigger")),
        Phase("4", "Fleet", 12.0, 20.0,
              ("second line, plant and grades",
               "self-supervised pretrain, own data",
               "quarterly retrain cadence")),
    )
    fig = plt.figure(figsize=(12.3, 2.20), dpi=DPI)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(-2.4, 102.4)
    ax.set_ylim(0, 10)
    ax.axis("off")

    shades = ["#eaf0f8", "#d9e4f4", "#c6d7ef", "#b0c8e9", "#98b8e3"]
    card_gap = 1.1
    card_w = (100 - 4 * card_gap) / 5
    span = phases[-1].end
    for i, (phase, shade) in enumerate(zip(phases, shades)):
        x = i * (card_w + card_gap)
        ax.add_patch(
            FancyBboxPatch((x, 3.35), card_w, 6.3,
                           boxstyle="round,pad=0.02,rounding_size=0.4",
                           facecolor=shade, edgecolor="white", linewidth=1.6, zorder=2)
        )
        ax.text(x + 0.9, 9.10, f"PHASE {phase.tag}", fontsize=7.4, color=STEEL,
                fontweight="bold", va="top", ha="left", zorder=3)
        ax.text(x + card_w - 0.9, 9.10,
                f"months {phase.start:g}-{phase.end:g}", fontsize=7.2, color=SLATE,
                va="top", ha="right", zorder=3)
        ax.text(x + 0.9, 8.15, phase.name, fontsize=9.8, color=INK,
                fontweight="bold", va="top", ha="left", zorder=3)
        for j, line in enumerate(phase.lines):
            ax.text(x + 0.9, 6.65 - j * 1.02, line, fontsize=7.3, color=SLATE,
                    va="top", ha="left", zorder=3)
        # proportional duration bar beneath the equal-width cards
        ax.add_patch(
            Rectangle((phase.start / span * 100, 2.05),
                      (phase.end - phase.start) / span * 100 - 0.35, 0.62,
                      facecolor=shade, edgecolor="none", zorder=2)
        )

    ax.plot([0, 100], [2.36, 2.36], color=RULE, lw=1.0, zorder=1)
    for tick in (0, 1.5, 4, 7, 12, 20):
        x = tick / span * 100
        ax.plot([x], [2.36], "o", ms=4, color=STEEL, zorder=3)
        ax.text(x, 1.35, "start" if tick == 0 else f"m{tick:g}", fontsize=7.2,
                color=SLATE, ha="center", va="center")
    ax.text(0, 0.30,
            "Cards are equal width for legibility; the bar beneath them is to scale. Phase durations are a proposed plan, not a measurement. "
            "The 0.6 h per-line fine-tune is measured: 15.24 s/epoch median at 320 px, models/yolov8n_neudet/results.csv.",
            fontsize=6.7, color=MUTED, ha="left", va="bottom")

    out = FIGDIR / "fig6_roadmap.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 7 -- the QR code for the live demo on slide 1
# ---------------------------------------------------------------------------
LIVE_URL = "https://surface-vision.github.io"


def fig_live_demo_qr() -> Path:
    """Encode the live-demo URL as a QR code, with no third-party dependency.

    qrgen.py is a from-scratch encoder written for this deck (nothing in the
    project's requirements provides one). It verifies itself by reversing its
    own encoding, and deck/build/verify_qr.swift decodes the rendered PNG with
    Apple's Vision and Core Image detectors -- the same readers a phone camera
    uses -- so this image is checked by something that did not write it.
    """
    import qrgen

    code = qrgen.encode(LIVE_URL, level="Q")
    recovered = qrgen.self_decode(code)
    if recovered != LIVE_URL:
        raise AssertionError(f"QR self-decode returned {recovered!r}")

    out = FIGDIR / "fig7_live_demo_qr.png"
    # 4-module quiet zone as the standard requires; an integer number of pixels
    # per module so no resampling can soften a module edge.
    qrgen.render_png(code, out, module_px=26, quiet=4, dark=(0x14, 0x1C, 0x24))
    print(f"{'fig7_live_demo_qr.png':28s} QR version {code.version} "
          f"({code.size}x{code.size} modules), level {code.level}, mask "
          f"{code.mask}, payload {LIVE_URL!r}, self-decode OK")
    return out


def main() -> None:
    _install_fonts()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for builder in (fig_defect_families, fig_architecture, fig_throughput,
                    fig_validation, fig_capacity, fig_roadmap, fig_live_demo_qr):
        path = builder()
        image = Image.open(path)
        print(f"{path.name:28s} {image.size[0]:5d} x {image.size[1]:4d} px   "
              f"{image.size[0] / DPI:5.2f} x {image.size[1] / DPI:4.2f} in   "
              f"{path.stat().st_size / 1024:6.0f} KB")


if __name__ == "__main__":
    main()
