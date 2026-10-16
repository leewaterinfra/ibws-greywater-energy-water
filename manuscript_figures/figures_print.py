"""
Manuscript Figs. 3-5 from the archived results (no re-simulation needed).
Reads the CSVs that ibws_greywater_optimization.py writes to results/.

  fig3_pump_schedules   Hourly pump speeds, medium-density buildings (S4, S5, S6)
  fig4_pareto_3x3       Pareto fronts, nine configurations
  fig5_reuse_savings    (a) reuse-to-savings mapping; (b) added energy, gross vs. net

Drawn at the final print width (7.0 in) with fonts >= 9 pt; seaborn 'muted'
colors and viridis for the reuse color scale. Text stays editable in the PDFs.

Usage (from the repository root):
  python manuscript_figures/figures_print.py                 # results/ -> figures_print/
  python manuscript_figures/figures_print.py results out_dir --fig5b gross
Options:
  --fig5b net    (default, as in the manuscript) panel (b): added energy per m3 of daily
                 demand, gross vs. net (pump savings subtracted) for each building
  --fig5b gross  panel (b): gross reuse energy and water savings
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.linewidth": 0.8, "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# ---- constants (same as ibws_greywater_optimization.py, v16e) ----
SCENARIO_IDS = [f"S{i}" for i in range(1, 10)]
PATTERN = np.array([0.3, 0.3, 0.3, 0.3, 0.4, 0.6, 1.2, 1.5, 1.3, 0.9, 0.8, 0.7,
                    0.8, 0.9, 0.8, 0.7, 0.9, 1.4, 1.6, 1.2, 0.9, 0.7, 0.5, 0.4])
GREYWATER_FRACTION, TOILET_FRACTION, REUSE_MAX = 0.65, 0.25, 0.8
R_STAR = TOILET_FRACTION / GREYWATER_FRACTION
REUSE_KWH_M3 = 0.4 + 0.1
CURVE_QMAX_FRAC, SPEED_MIN = 1.5, 0.5

MUTED = {"blue": "#4878d0", "orange": "#ee854a", "green": "#6acc64", "red": "#d65f5f",
         "purple": "#956cb4", "gray": "#797979", "gold": "#d5bb67"}
HEIGHT_COLORS = {5: MUTED["blue"], 10: MUTED["green"], 20: MUTED["red"]}
DENSITY_MARKERS = {"Low": "o", "Med": "s", "High": "^"}
DENSITY_NAMES = {"Low": "Low density", "Med": "Medium density", "High": "High density"}
REF = "#c0602a"          # darker orange for reference lines/text (legible in print)
W = 7.0                  # print width, inches


def reuse_share(r):
    return np.minimum(GREYWATER_FRACTION * np.asarray(r, float), TOILET_FRACTION)


def load_results(d):
    out = {}
    for s in SCENARIO_IDS:
        f = {k: os.path.join(d, f"{k}_{s}.csv") for k in ("summary", "pareto", "keysol")}
        if all(os.path.exists(p) for p in f.values()):
            sm = pd.read_csv(f["summary"]).iloc[0]
            out[s] = {"summary": sm,
                      "front": pd.read_csv(f["pareto"]).sort_values("savings_pct"),
                      "key": pd.read_csv(f["keysol"]).set_index("solution"),
                      "floors": int(sm["Floors"]), "density": sm["Density Class"],
                      "upf": int(sm["Units/Floor"])}
    missing = [s for s in SCENARIO_IDS if s not in out]
    if missing:
        print(f"NOTE: missing results for {missing}")
    return out


def save(fig, out_dir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=600 if ext == "png" else None,
                    facecolor="white", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"  saved {name}.pdf/.png")


# ======================================================================
# Fig. 3  hourly pump schedules (medium density, 5/10/20 floors)
# ======================================================================
def fig3_pump_schedules(res, out_dir):
    panels = [("S4", "(a) 5 floors (S4)"), ("S5", "(b) 10 floors (S5)"),
              ("S6", "(c) 20 floors (S6)")]
    hours = np.arange(24)
    cols = [f"speed_h{h:02d}" for h in hours]
    floor_r0 = np.clip(PATTERN / PATTERN.max() / CURVE_QMAX_FRAC, SPEED_MIN, 1.0)
    styles = {"min_energy": (MUTED["blue"], "o"), "balanced": (MUTED["green"], "s"),
              "max_savings": (MUTED["gold"], "^")}
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.75), sharey=True)
    bal = []
    for ax, (s, title) in zip(axes, panels):
        k = res[s]["key"]
        ax.fill_between(hours, 0.45, 0.45 + 0.55 * PATTERN / PATTERN.max(), step="mid",
                        color=MUTED["gray"], alpha=0.13, lw=0)
        ax.step(hours, floor_r0, where="mid", color="#555555", lw=1.0, ls=(0, (1.5, 1.5)))
        for key, (c, m) in styles.items():
            ax.step(hours, k.loc[key, cols].values.astype(float), where="mid", color=c,
                    lw=1.6, marker=m, ms=2.8)
        bal.append(k.loc["balanced", "reuse_pct"])
        ax.set_title(title, fontweight="bold", pad=4)
        ax.set(xlim=(-0.5, 23.5), ylim=(0.45, 1.02), xlabel="Hour of day")
        ax.set_xticks(range(0, 24, 6))
        ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 6)])
    axes[0].set_ylabel("Pump speed (fraction of rated)")
    handles = [
        Line2D([], [], color=MUTED["blue"], lw=1.6, marker="o", ms=3.5, label="Minimum energy (r = 0)"),
        Line2D([], [], color=MUTED["green"], lw=1.6, marker="s", ms=3.5,
               label=f"Balanced (r = {min(bal):.1f}\u2013{max(bal):.1f}%)"),
        Line2D([], [], color=MUTED["gold"], lw=1.6, marker="^", ms=3.5,
               label=f"Maximum savings (r = {100 * R_STAR:.1f}%)"),
        Line2D([], [], color="#555555", lw=1.0, ls=(0, (1.5, 1.5)), label="Speed floor (r = 0)"),
        Patch(fc=MUTED["gray"], alpha=0.25, ec="none", label="Demand pattern (scaled)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.01), handlelength=2.2, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=0.6)
    save(fig, out_dir, "fig3_pump_schedules")


# ======================================================================
# Fig. 4  Pareto fronts, 3 x 3 (rows: density; columns: height)
# ======================================================================
def fig4_pareto_grid(res, out_dir):
    fig, axes = plt.subplots(3, 3, figsize=(W, 6.6))
    fig.subplots_adjust(left=0.135, right=0.875, top=0.885, bottom=0.075,
                        wspace=0.34, hspace=0.42)
    grid = {(v["density"], v["floors"]): s for s, v in res.items()}
    sc = None
    for i, dens in enumerate(("Low", "Med", "High")):
        for j, fl in enumerate((5, 10, 20)):
            ax = axes[i, j]
            s = grid.get((dens, fl))
            if s is None:
                ax.text(0.5, 0.5, "no results", ha="center", va="center", transform=ax.transAxes)
                continue
            fr, key, sm = res[s]["front"], res[s]["key"], res[s]["summary"]
            sc = ax.scatter(fr["energy_total_kwh"], fr["savings_pct"], c=fr["reuse_pct"],
                            cmap="viridis", vmin=0, vmax=40, s=9, lw=0, zorder=3)
            ax.scatter(key.loc["balanced", "energy_total_kwh"], key.loc["balanced", "savings_pct"],
                       s=75, marker="*", c=MUTED["red"], edgecolors="white", lw=0.6, zorder=5)
            ax.scatter(sm["Baseline Energy (kWh/d)"], 0, s=32, facecolors="none",
                       edgecolors="black", lw=1.0, zorder=5, clip_on=False)
            ax.text(0.04, 0.95, f"{s}, {res[s]['upf']} u/fl", transform=ax.transAxes,
                    ha="left", va="top", fontsize=9)
            ax.set_ylim(-1.5, 27)
            ax.set_yticks([0, 5, 10, 15, 20, 25])
            ax.xaxis.set_major_locator(MaxNLocator(4))
            if i == 0:
                ax.set_title(f"{fl} floors", fontweight="bold", pad=5)
            if j == 0:
                ax.set_ylabel("Water savings (%)")
                ax.annotate(DENSITY_NAMES[dens], xy=(-0.52, 0.5), xycoords="axes fraction",
                            rotation=90, ha="center", va="center", fontweight="bold")
            if i == 2:
                ax.set_xlabel("Total energy (kWh/day)")
    if sc is not None:
        cax = fig.add_axes([0.895, 0.20, 0.018, 0.55])
        cb = fig.colorbar(sc, cax=cax, ticks=[0, 10, 20, 30, 40])
        cb.set_label("Greywater reuse fraction (%)")
        cb.outline.set_linewidth(0.6)
    fig.legend(handles=[
        Line2D([], [], marker="*", ls="none", markerfacecolor=MUTED["red"], markeredgecolor="white",
               markersize=11, label="Balanced solution"),
        Line2D([], [], marker="o", ls="none", markerfacecolor="none", markeredgecolor="black",
               markersize=6, label="Baseline (fixed full speed, no reuse)")],
        loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=2, frameon=False,
        columnspacing=2.0)
    save(fig, out_dir, "fig4_pareto_3x3")


# ======================================================================
# Fig. 5  (a) reuse-to-savings mapping; (b) gross energy or gross vs. net
# ======================================================================
def fig5_reuse_savings(res, out_dir, panel_b="net"):
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 3.25))
    r = np.linspace(0, 100 * REUSE_MAX, 400)
    sav = 100 * reuse_share(r / 100)
    rs = 100 * R_STAR
    bal = [(res[s]["key"].loc["balanced", "reuse_pct"], res[s]["key"].loc["balanced", "savings_pct"],
            s) for s in SCENARIO_IDS if s in res]

    def marker(ax, x, y, s, size=34):
        ax.scatter(x, y, s=size, marker=DENSITY_MARKERS[res[s]["density"]],
                   c=HEIGHT_COLORS[res[s]["floors"]], edgecolors="white", lw=0.5, zorder=5)

    # (a)
    a.plot(r, sav, color=MUTED["blue"], lw=1.8, zorder=3)
    a.axhline(100 * TOILET_FRACTION, color=REF, ls="--", lw=1.0)
    a.axvline(rs, color=REF, ls="--", lw=1.0)
    a.text(rs + 1.5, 26.0, "25% ceiling (toilet demand)", color=REF, va="bottom")
    a.text(rs + 1.5, 1.0, f"r* = {rs:.1f}%", color=REF)
    lo, hi = min(x[0] for x in bal), max(x[0] for x in bal)
    a.axvspan(lo - 0.5, hi + 0.5, color=MUTED["green"], alpha=0.18, lw=0)
    a.text(1.0, 22.0, f"Balanced range\n{lo:.1f}\u2013{hi:.1f}%", ha="left", va="center",
           color="#3a8f35", linespacing=1.15)
    for x, y, s in bal:
        marker(a, x, y, s)
    a.set(xlim=(-2, 82), ylim=(-1, 30), xlabel="Greywater reuse fraction (%)",
          ylabel="Water savings (%)")
    a.set_title("(a)", loc="left", fontweight="bold")

    # (b)
    if panel_b == "net":
        b.plot(r, REUSE_KWH_M3 * reuse_share(r / 100), color="black", lw=1.2, ls="--",
               label="Gross (0.5 kWh/m\u00b3 reused)", zorder=4)
        for s in SCENARIO_IDS:
            if s not in res:
                continue
            fr, sm = res[s]["front"].sort_values("reuse_pct"), res[s]["summary"]
            v = float(sm["Daily Demand (m3/d)"])
            de = (fr["energy_total_kwh"] - float(sm["Min Energy (kWh/d)"])) / v
            b.plot(fr["reuse_pct"], de, color=HEIGHT_COLORS[res[s]["floors"]], lw=1.3, alpha=0.9)
        for f, c in HEIGHT_COLORS.items():
            sl = np.mean([float(res[s]["summary"]["Net Slope (kWh/m3 reused)"])
                          for s in res if res[s]["floors"] == f])
            b.plot([], [], color=c, lw=1.3, label=f"Net, {f} floors ({sl:.2f} kWh/m\u00b3)")
        b.axvline(rs, color=REF, ls="--", lw=1.0)
        b.set(xlim=(-2, 82), ylim=(0, 0.14), xlabel="Greywater reuse fraction (%)",
              ylabel="Added energy (kWh per m\u00b3 of daily demand)")
        b.legend(loc="lower right", frameon=False, handlelength=1.8)
    else:
        pen = REUSE_KWH_M3 * reuse_share(r / 100)
        b.fill_between(r, 0, pen, color=MUTED["blue"], alpha=0.12, lw=0)
        b.plot(r, pen, color=MUTED["blue"], lw=1.8)
        b2 = b.twinx()
        b2.plot(r, sav, color=MUTED["green"], lw=1.8)
        b2.set_ylim(0, 30)
        b2.grid(False)
        b2.set_ylabel("Water savings (%)", color="#3a8f35")
        b.axvline(rs, color=REF, ls="--", lw=1.0)
        for x, _, s in bal:
            marker(b, x, REUSE_KWH_M3 * reuse_share(x / 100), s)
        b.set(xlim=(-2, 82), ylim=(0, 0.16), xlabel="Greywater reuse fraction (%)")
        b.set_ylabel("Gross reuse energy\n(kWh per m\u00b3 of daily demand)", color="#2d5aa8")
    b.set_title("(b)", loc="left", fontweight="bold")

    handles = ([Line2D([], [], marker="o", ls="none", color=HEIGHT_COLORS[f], ms=5,
                       label=f"{f} floors") for f in (5, 10, 20)] +
               [Line2D([], [], marker=DENSITY_MARKERS[d], ls="none", color=MUTED["gray"], ms=5,
                       label=DENSITY_NAMES[d]) for d in ("Low", "Med", "High")])
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, -0.01), handletextpad=0.3, columnspacing=1.1)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.2)
    save(fig, out_dir, "fig5_reuse_savings" + ("" if panel_b == "net" else "_gross"))


def main():
    ap = argparse.ArgumentParser(description="Manuscript Figs. 3-5 from archived results")
    ap.add_argument("results", nargs="?", default="results")
    ap.add_argument("out", nargs="?", default="figures_print")
    ap.add_argument("--fig5b", choices=["net", "gross"], default="net")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    res = load_results(args.results)
    print(f"Loaded {len(res)} scenarios from '{args.results}'")
    fig3_pump_schedules(res, args.out)
    fig4_pareto_grid(res, args.out)
    fig5_reuse_savings(res, args.out, args.fig5b)


if __name__ == "__main__":
    main()
