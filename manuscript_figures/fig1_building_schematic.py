"""
Fig. 1 - Building water system schematic (manuscript Fig. 1).
Drawn at final print width (7.0 in), so font sizes are true print sizes.
Palette: seaborn 'muted' (same as Figs. 3-5). Text is editable in the PDF.
Usage (from the repository root):
    python manuscript_figures/fig1_building_schematic.py [out_dir]   # default out_dir: figures_print/
"""
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, Circle, Polygon, Rectangle
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "pdf.fonttype": 42, "ps.fonttype": 42,          # editable text
    "mathtext.fontset": "custom", "mathtext.rm": "sans", "mathtext.it": "sans:italic",
})

# seaborn muted
BLUE, GREEN, RED, GOLD, GRAY = "#4878d0", "#6acc64", "#d65f5f", "#d5bb67", "#797979"
# darker shades for text on white (legibility, grayscale)
BLUE_T, GREEN_T, RED_T = "#2d5aa8", "#3a8f35", "#b23a3a"
INK = "#222222"
FS, FS_S = 9, 9            # body / minor annotation (pt at print size)

W, H = 7.0, 5.9                # inches
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 14.0)
ax.set_ylim(0, 11.8)
ax.axis("off")
halo = [pe.withStroke(linewidth=5, foreground="white")]

# ---------------- geometry ----------------
xB0, xB1 = 1.9, 10.0           # building walls
yB0, yB1 = 0.12, 10.35
y_ground = 2.38
xP, xR, xC, xS = 3.6, 4.3, 8.3, 9.2      # potable riser, reuse riser, GW collection, stack
xF0, xF1 = 5.0, 7.6                      # fixture box
floors = [(3.0, "Floor 1"), (5.0, "Floor 2"), (7.6, "Floor n−1"), (9.6, "Floor n (top)")]
y_top = floors[-1][0]
y_base = 1.4                              # basement pipe level

# building + ground
ax.add_patch(FancyBboxPatch((xB0, yB0), xB1 - xB0, yB1 - yB0,
             boxstyle="round,pad=0,rounding_size=0.15",
             fc="#f6f6f6", ec="#9a9a9a", lw=1.0, zorder=0))
ax.plot([xB0, xB1], [y_ground, y_ground], color="#9a9a9a", lw=0.8, ls=(0, (4, 3)), zorder=1)
ax.text(xB0 + 0.12, y_ground - 0.22, "Basement", fontsize=FS_S, color=GRAY, va="top", style="italic")
for y, _ in floors[1:]:
    ax.plot([xB0 + 0.05, xB1 - 0.05], [y - 0.62, y - 0.62], color="#d8d8d8", lw=0.6, zorder=1)

# ---------------- risers (drawn first; branches cross over with a halo) ----------------
ax.plot([xP, xP], [y_base, y_top], color=BLUE, lw=2.4, zorder=2, solid_capstyle="round")
ax.plot([xR, xR], [y_base, y_top - 0.18], color=GREEN, lw=2.0, zorder=2)
ax.plot([xC, xC], [y_top + 0.15, y_base], color=GREEN, lw=2.0, ls=(0, (3, 2)), zorder=2)
ax.plot([xS, xS], [y_top - 0.18, y_base], color=RED, lw=2.6, zorder=2)

def arrow(x0, y0, x1, y1, color, lw=1.6, ls="-", z=4, halo_on=False):
    a = ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=z,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                    mutation_scale=9, shrinkA=0, shrinkB=0))
    if halo_on:
        a.arrow_patch.set_path_effects(halo)
    return a

# ---------------- floors ----------------
for y, lab in floors:
    ax.add_patch(FancyBboxPatch((xF0, y - 0.4), xF1 - xF0, 0.8,
                 boxstyle="round,pad=0,rounding_size=0.08",
                 fc="#f4ecd2", ec=GOLD, lw=1.1, zorder=3))
    ax.text((xF0 + xF1) / 2, y, "Showers, sinks,\nlaundry, toilets", ha="center", va="center",
            fontsize=FS_S, color=INK, linespacing=1.15, zorder=5)
    ax.plot(xP, y + 0.15, "o", ms=5, color=BLUE, zorder=6)
    arrow(xP, y + 0.15, xF0, y + 0.15, BLUE, halo_on=True)            # potable
    arrow(xR, y - 0.18, xF0, y - 0.18, GREEN)                          # treated GW to toilets
    arrow(xF1, y + 0.15, xC, y + 0.15, GREEN, ls=(0, (3, 2)))          # collected GW
    arrow(xF1, y - 0.18, xS, y - 0.18, RED, halo_on=True)              # blackwater + uncollected GW
    ax.text(xB0 + 0.12, y + 0.28, lab, fontsize=FS, color=INK, va="center")

ax.text(xB0 + 0.12, y_top - 0.12, r"$P \geq$ 140 kPa", fontsize=FS_S, color=BLUE_T, va="center")

# floor-break region
yb = (floors[1][0] + floors[2][0]) / 2
ax.plot([(xF0 + xF1) / 2] * 3, [yb + 0.42, yb + 0.22, yb + 0.02], "o", ms=2.5, color=GRAY)
ax.text((xF0 + xF1) / 2, yb - 0.55, "n = 5, 10, or 20\n(3 m per floor)", ha="center", va="center",
        fontsize=FS_S, color=INK, linespacing=1.15)
for x, t, c, dx in ((xP, "Potable riser", BLUE_T, -0.24), (xR, "Reuse riser", GREEN_T, 0.24),
                    (xC, "GW collection", GREEN_T, -0.24), (xS, "Drain stack", RED_T, 0.25)):
    ax.text(x + dx, yb, t, rotation=90, ha="center", va="center", fontsize=FS_S, color=c)

# ---------------- basement: supply side ----------------
ax.add_patch(Rectangle((0.55, y_base - 0.24), 0.48, 0.48, fc=BLUE, ec="none", zorder=5))
ax.text(0.79, y_base - 0.42, "City main\n(H = 30 m)", ha="center", va="top", fontsize=FS_S,
        color=INK, linespacing=1.15)
xpump = 2.75
ax.plot([1.03, xpump - 0.28], [y_base, y_base], color=BLUE, lw=2.4, zorder=2)
ax.plot([xpump + 0.28, xP], [y_base, y_base], color=BLUE, lw=2.4, zorder=2)
ax.add_patch(Circle((xpump, y_base), 0.28, fc="white", ec=BLUE, lw=1.8, zorder=5))
ax.add_patch(Polygon([[xpump - 0.1, y_base - 0.13], [xpump - 0.1, y_base + 0.13],
                      [xpump + 0.14, y_base]], closed=True, fc=BLUE, ec="none", zorder=6))
ax.text(xpump, y_base - 0.42, "VFD booster\npump", ha="center", va="top", fontsize=FS_S,
        color=INK, linespacing=1.15)

# ---------------- basement: greywater treatment ----------------
tx0, tx1, ty0, ty1 = 4.62, 7.98, 0.72, 2.02
ax.add_patch(FancyBboxPatch((tx0, ty0), tx1 - tx0, ty1 - ty0,
             boxstyle="round,pad=0,rounding_size=0.1", fc="#e5f4e3", ec=GREEN, lw=1.3, zorder=3))
ax.text((tx0 + tx1) / 2, ty1 - 0.2, "Greywater treatment", ha="center", va="top",
        fontsize=FS, fontweight="bold", color=INK, zorder=5)
ax.text((tx0 + tx1) / 2, ty1 - 0.62, "MBR: 0.4 kWh/m³\nDistribution: 0.1 kWh/m³",
        ha="center", va="top", fontsize=FS_S, color=INK, linespacing=1.25, zorder=5)
arrow(xC, y_base, tx1, y_base, GREEN, ls=(0, (3, 2)))
ax.plot([tx0, xR], [y_base, y_base], color=GREEN, lw=2.0, zorder=2)

# ---------------- basement: drainage (PySWMM) ----------------
ax.plot(xS, y_base, "s", ms=6, color=RED, zorder=6)
x_c1, x_out = 11.25, 13.15
ax.plot([xS, x_c1], [y_base, y_base], color=RED, lw=2.2, zorder=2)            # collector
ax.plot([x_c1, x_out], [y_base, y_base], color=RED, lw=3.4, zorder=2)         # main drain
ax.plot(x_c1, y_base, "o", ms=4, color=RED, zorder=6)
ax.add_patch(Polygon([[x_out, y_base - 0.22], [x_out, y_base + 0.22], [x_out + 0.38, y_base]],
                     closed=True, fc=RED, ec="none", zorder=6))
ax.text((xS + x_c1) / 2 + 0.35, y_base - 0.25, "Collector\n100 mm", ha="center", va="top",
        fontsize=FS_S, color=INK, linespacing=1.15,
        bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"))
ax.text((x_c1 + x_out) / 2, y_base - 0.25, "Main drain\n150 mm, 2%", ha="center", va="top",
        fontsize=FS_S, color=INK, linespacing=1.15)
ax.text(x_out + 0.2, y_base + 0.35, "Outfall", ha="center", va="bottom", fontsize=FS_S, color=INK)

# ---------------- legend ----------------
handles = [
    Line2D([], [], color=BLUE, lw=2.4, marker="o", ms=4, label="Potable supply (WNTR/EPANET)"),
    Line2D([], [], color=GREEN, lw=2.0, ls=(0, (3, 2)), label="Collected greywater (GW), fraction r"),
    Line2D([], [], color=GREEN, lw=2.0, label="Treated greywater to toilets"),
    Line2D([], [], color=RED, lw=2.4, label="Blackwater + uncollected GW (PySWMM)"),
]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2,
          fontsize=FS, frameon=False, handlelength=2.6, columnspacing=1.6, borderaxespad=0.3)

out_dir = sys.argv[1] if len(sys.argv) > 1 else "figures_print"
os.makedirs(out_dir, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(out_dir, f"fig1_building_schematic.{ext}"), dpi=600 if ext == "png" else None,
                facecolor="white")
print(f"saved {out_dir}/fig1_building_schematic.pdf/.png")
