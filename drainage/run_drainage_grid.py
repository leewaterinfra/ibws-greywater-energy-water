"""
Main-drain velocity for all drainage cases: scenarios S1-S9 x greywater
diversion d (0, 0.18, 0.385, 0.80) x slope (2%, 1%) x random realizations.
Generalizes the original grid2.py (one realization) to several seeds.

Usage (from the repository root):
    python drainage/run_drainage_grid.py                     # seeds 1 2 3
    python drainage/run_drainage_grid.py --seeds 1 2 3 --out drainage/results/drainage_grid.csv

Output columns: scenario, d, slope_pct, seed, vmax (daily peak main-drain velocity,
m/s), v_med_flush / v_min_flush (median / minimum per-flush peak velocity, m/s),
pct_flush_ge06 / pct_flush_ge05 (% of flushes with peak >= 0.6 / 0.5 m/s),
n_flush, qpeak (peak inflow, L/s).
"""
import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flush_swmm import SC, run  # noqa: E402

DIVERSIONS = (0.0, 0.18, 0.385, 0.80)
SLOPES = (0.02, 0.01)


def main():
    ap = argparse.ArgumentParser(description="PySWMM main-drain velocity grid")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--out", default=os.path.join("drainage", "results", "drainage_grid.csv"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tmp = f"_swmm_{os.getpid()}.inp"
    rows, t0 = [], time.time()
    try:
        for seed in a.seeds:
            for sc in SC:
                for d in DIVERSIONS:
                    for slope in SLOPES:
                        r = run(sc, d, slope, seed=seed, tmp=tmp)
                        rows.append(dict(scenario=sc, d=d, slope_pct=slope * 100, seed=seed,
                                         **{k: float(v) for k, v in r.items()}))
                print(f"seed {seed} {sc} done ({time.time() - t0:.0f} s)", flush=True)
    finally:
        for ext in (".inp", ".rpt", ".out"):
            if os.path.exists(tmp.replace(".inp", ext)):
                os.remove(tmp.replace(".inp", ext))
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)

    # summary of the quantities reported in the manuscript
    s2 = df[df.slope_pct == 2.0]
    print(f"\n2% slope: daily peak velocity min {s2.vmax.min():.3f}, max {s2.vmax.max():.3f} m/s; "
          f"cases >= 0.6 m/s: {(s2.vmax >= 0.6).sum()}/{len(s2)}")
    base = s2[s2.d == 0].set_index(["scenario", "seed"]).vmax
    for d in DIVERSIONS:
        sub = s2[s2.d == d].set_index(["scenario", "seed"]).vmax
        print(f"  d = {d:<5}: min {sub.min():.3f} m/s, max drop vs. d = 0: "
              f"{(base - sub).max():.3f} m/s, cases < 0.7 m/s: {(sub < 0.7).sum()}")
    s1 = df[df.slope_pct == 1.0]
    low = s1[(s1.d == 0) & (s1.vmax < 0.6)].scenario.unique()
    print(f"1% slope, no reuse: scenarios with daily peak < 0.6 m/s in any realization: {list(low)}")
    print(f"Saved {a.out} ({len(df)} cases, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
