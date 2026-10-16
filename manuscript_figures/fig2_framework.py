"""
Fig. 2 - Integrated simulation-optimization framework (manuscript Fig. 2).
Drawn at final print width (7.0 in); font sizes are true print sizes (>= 9 pt).
Palette: seaborn 'muted'. Text is editable in the PDF.
Usage (from the repository root):
    python manuscript_figures/fig2_framework.py [out_dir]   # default out_dir: figures_print/
"""
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "mathtext.fontset": "custom", "mathtext.rm": "sans", "mathtext.it": "sans:italic",
    "mathtext.bf": "sans:bold",
})

# seaborn muted + light tints
C = {"blue": "#4878d0", "green": "#6acc64", "red": "#d65f5f", "purple": "#956cb4",
     "gold": "#d5bb67", "gray": "#797979"}
TINT = {"blue": "#e4ebf8", "green": "#e5f4e3", "red": "#f8e4e4", "purple": "#eee6f4",
        "gold": "#f4ecd2", "gray": "#efefef"}
INK = "#222222"
FS_H, FS = 9.8, 9.0

W, H = 7.0, 8.4
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 70)
ax.set_ylim(0, 84)
ax.axis("off")


def box(x, y, w, h, key, title, body="", ls="-", lw=1.3, title_dy=1.35):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0",
                                fc=TINT[key], ec=C[key], lw=lw, ls=ls, zorder=2))
    cx = x + w / 2
    if body:
        ax.text(cx, y + h - title_dy, title, ha="center", va="top", fontsize=FS_H,
                fontweight="bold", color=INK, zorder=3)
        ax.text(cx, y + h - title_dy - 2.25, body, ha="center", va="top", fontsize=FS,
                color=INK, linespacing=1.35, zorder=3)
    else:
        ax.text(cx, y + h / 2, title, ha="center", va="center", fontsize=FS_H,
                fontweight="bold", color=INK, zorder=3)
    return (x, y, w, h)


def arr(x0, y0, x1, y1, color=INK, ls="-", lw=1.3, rad=0.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=4,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, mutation_scale=10,
                                shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}"))


# ---------------- inputs ----------------
box(3, 75.5, 64, 8.0, "gray", "Inputs",
    "9 building configurations (Table 2)  ·  200 L/capita/day  ·  24-h demand pattern\n"
    "Water balance: 65% greywater, 25% toilet, 10% potable-only")

# ---------------- optimization loop frame ----------------
ax.add_patch(FancyBboxPatch((1.2, 31.0), 67.6, 43.4, boxstyle="round,pad=0,rounding_size=1.5",
                            fc="none", ec=C["purple"], lw=1.2, ls=(0, (5, 3)), zorder=1))
ax.text(66.8, 73.2, "Optimization loop", ha="right", va="center", fontsize=FS,
        style="italic", color="#6d4a8c")

box(14, 65.0, 42, 7.5, "purple", "NSGA-II optimizer (pymoo)",
    "Population 100  ·  100 generations  ·  SBX (0.9, $\\eta_c$ = 15)  ·  PM ($\\eta_m$ = 20)")

box(3, 54.5, 31, 8.0, "blue", "24 hourly pump speeds $s_h$",
    "0.5 ≤ $s_h$ ≤ 1.0\nSpeed floor: $s_h \\geq Q_h\\,/\\,(1.5\\,Q_d)$")
box(36, 54.5, 31, 8.0, "green", "Greywater reuse fraction $r$",
    "0 ≤ $r$ ≤ 0.8\nSaturation: $r^*$ = 0.25 / 0.65 = 38.5%")

box(3, 40.0, 31, 10.5, "blue", "WNTR potable model (EPANET)",
    "24-h simulation, 15-min step\nFloor demand × [1 − min(0.65$r$, 0.25)]\n"
    "Output: $E_{pump}$ and top-floor pressure $P_{top}$")
box(36, 40.0, 31, 10.5, "green", "Greywater module (analytical)",
    "$V_r$ = $V$ · min(0.65$r$, 0.25)\n$E_{reuse}$ = $V_r$ × (0.4 + 0.1) kWh/m³\n"
    "$S$ = 100 · min(0.65$r$, 0.25) %")

box(3, 32.5, 64, 5.0, "purple",
    "min $E$ = $E_{pump}$ + $E_{reuse}$     max $S$     subject to $P_{top}$ ≥ 140 kPa (every hour)")

# arrows inside loop
arr(27, 65.0, 18.5, 62.5); arr(43, 65.0, 51.5, 62.5)
arr(18.5, 54.5, 18.5, 50.5); arr(51.5, 54.5, 51.5, 50.5)
arr(40.5, 54.5, 30.5, 50.5, color="#3a8f35", ls=(0, (3, 2)), lw=1.1)
ax.text(35.5, 52.5, "reduces potable demand", ha="center", va="center", fontsize=FS,
        color="#3a8f35", style="italic", linespacing=1.1,
        bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"), zorder=5)
arr(18.5, 40.0, 18.5, 37.5); arr(51.5, 40.0, 51.5, 37.5)
# loop-back arrow (left side)
ax.plot([3, 2.2, 2.2], [35.0, 35.0, 68.75], color=C["purple"], lw=1.3, zorder=1)
arr(2.2, 68.75, 14, 68.75, color=C["purple"])
ax.text(8.2, 71.2, "Next\ngeneration", ha="center", va="center", fontsize=FS,
        color="#6d4a8c", linespacing=1.1)
arr(35, 75.5, 35, 72.5)

# ---------------- post-processing ----------------
box(3, 18.5, 41, 10.0, "blue", "Exact pump-schedule refinement",
    "Each Pareto solution: lowest speed per hour with\n$P_{top}$ ≥ 140 kPa (bisection, tolerance 10$^{-4}$)\n"
    "Front augmented with $r$ = 0 and $r$ = $r^*$")
box(3, 6.5, 41, 9.5, "gold", "Pareto front and key solutions",
    "Minimum energy ($r$ = 0)\nBalanced: closest to ideal point (equal weights)\n"
    "Maximum savings ($r$ = $r^*$)")
box(3, 0.6, 64, 4.0, "gold", "Design guidelines (Table 4)")
arr(23.5, 32.5, 23.5, 28.5); arr(23.5, 18.5, 23.5, 16.0); arr(23.5, 6.5, 23.5, 4.6)

# ---------------- drainage (outside loop) ----------------
box(47, 6.5, 20, 22.0, "red", "Drainage velocity",
    "PySWMM discrete-event\nsimulation, outside the\noptimization loop\n"
    "100 mm collector and\n150 mm main drain (2%)\nToilet flushes (1.4 L/s)\nand greywater draws\n"
    "Diversion $d$ = 0–0.80\nCheck: daily peak\nvelocity ≥ 0.6 m/s",
    ls=(0, (4, 2)))
arr(57, 6.5, 57, 4.6, color=C["red"])

out_dir = sys.argv[1] if len(sys.argv) > 1 else "figures_print"
os.makedirs(out_dir, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(out_dir, f"fig2_framework.{ext}"), dpi=600 if ext == "png" else None,
                facecolor="white")
print(f"saved {out_dir}/fig2_framework.pdf/.png")
