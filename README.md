# iBWS Greywater Reuse: Energy-Water Trade-off Optimization

Code and results for: Lee, J., Lansing, J., and Cole, D. "Energy-Water Trade-offs Govern Greywater Reuse in Building Water Systems: An Integrated 'One Water' Framework with Design Guidelines." Submitted to ASCE *Journal of Sustainable Water in the Built Environment*.

The model couples WNTR (EPANET) potable-supply hydraulics with NSGA-II multi-objective optimization (pymoo). For nine US residential building configurations, it finds the Pareto front between total daily energy (booster pumping plus greywater treatment and distribution) and potable water savings from greywater reuse for toilet flushing.

## Repository contents

| Path | Contents |
|---|---|
| `ibws_greywater_optimization.py` | Optimization model, Table 3, and diagnostic figures (internal version v16e) |
| `results/` | Archived outputs used in the manuscript (v16e, seed 42, run on 2026-10-15) |
| `drainage/` | PySWMM discrete-event drainage simulation and its archived results |
| `manuscript_figures/` | Scripts for manuscript Figs. 1–5 |
| `requirements.txt` | Python dependencies |

## Installation

```bash
pip install -r requirements.txt
```

## Reproducing the results

```bash
# Optimize scenarios (any subset of S1-S9)
python ibws_greywater_optimization.py run S1 S2 S3

# Rebuild Table 3 and diagnostic figures from saved results
python ibws_greywater_optimization.py figures

# Everything: run S1-S9, then Table 3 and figures
python ibws_greywater_optimization.py all
```

| Option | Default | Description |
|---|---|---|
| `--pop` | 100 | NSGA-II population size |
| `--gen` | 100 | NSGA-II generations |
| `--seed` | 42 | Random seed |
| `--results` | `results` | Output folder for CSV/JSON results |
| `--figures` | `figures` | Output folder for diagnostic figures |

Each scenario takes about 10–12 minutes on one CPU core. Several `run` commands can execute in parallel in the same folder; each process uses its own EPANET temporary files. For a quick test, use `--pop 20 --gen 5 --results test_results`, so that the archived results are not overwritten.

## Reproducing the manuscript figures

Run from the repository root; output goes to `figures_print/` (PDF and 600 dpi PNG, 7.0 in wide).

```bash
python manuscript_figures/fig1_building_schematic.py   # Fig. 1  Building water system schematic
python manuscript_figures/fig2_framework.py            # Fig. 2  Simulation-optimization framework
python manuscript_figures/figures_print.py             # Figs. 3-5 from results/
```

| Figure | Contents |
|---|---|
| Fig. 3 | Hourly pump speeds of the key solutions, medium-density buildings (S4, S5, S6) |
| Fig. 4 | Pareto fronts, nine configurations (3 × 3) |
| Fig. 5 | (a) Reuse-to-savings mapping; (b) added energy, gross vs. net, by building height |

## Drainage simulation (PySWMM)

Drainage velocity is assessed outside the optimization by discrete-event simulation of the main building drain (100 mm collector, 150 mm main drain) with stochastic toilet flushes and greywater draws.

```bash
python drainage/flush_swmm.py          # steady-state check: 1.4 L/s, 150 mm, 2% slope -> 0.69 m/s (Manning: 0.69)
python drainage/run_drainage_grid.py   # 9 scenarios x 4 diversions x 2 slopes x 3 realizations (about 6 min)
```

Archived output: `drainage/results/drainage_grid.csv` (216 cases; column definitions are in `run_drainage_grid.py`). Greywater diversion d = 0, 0.18, 0.385, 0.80; slopes 2% and 1%; random seeds 1, 2, 3.

## Scenarios

| ID | Floors | Density | Units/floor | Occupants/floor | Base demand (L/s/floor) |
|---|---|---|---|---|---|
| S1 | 5 | Low | 6 | 15 | 0.0430 |
| S2 | 10 | Low | 8 | 20 | 0.0573 |
| S3 | 20 | Low | 6 | 15 | 0.0430 |
| S4 | 5 | Medium | 11 | 20 | 0.0573 |
| S5 | 10 | Medium | 16 | 29 | 0.0831 |
| S6 | 20 | Medium | 13 | 26 | 0.0745 |
| S7 | 5 | High | 17 | 26 | 0.0745 |
| S8 | 10 | High | 23 | 35 | 0.1003 |
| S9 | 20 | High | 21 | 32 | 0.0917 |

Base demand = occupants/floor × 200 L/capita/day ÷ 86,400 s ÷ 0.808 (mean of the hourly demand pattern).

## Method summary

- **Decision variables (25):** 24 hourly pump speeds (0.5–1.0 of rated, and high enough to keep flow within the pump curve) and the reuse fraction r (0–0.8).
- **Water balance:** greywater = 65% of demand; toilet demand = 25% (reuse ceiling). Water savings = min(0.65 r, 0.25). Potable floor demands are reduced by the same fraction. Saturation occurs at r\* = 38.5%.
- **Energy:** WNTR pump energy (70% efficiency) plus 0.5 kWh per m³ reused (treatment 0.4 + distribution 0.1).
- **Constraint:** top-floor pressure ≥ 140 kPa in every hour.
- **Optimization:** NSGA-II (pop 100, gen 100, SBX 0.9/η = 15, PM η = 20, seeded population).
- **Exact pump-schedule refinement:** the network has no storage, so hours are hydraulically independent, and pump power rises with speed at fixed demand. The energy-minimal schedule for any r is therefore the lowest feasible speed in each hour, found by bisection (tolerance 10⁻⁴). This is applied to every Pareto solution; the front is augmented with r = 0 and r = r\*.
- **Balanced solution:** closest to the ideal point after min-max normalization of both objectives, with equal weights.

## Outputs

`results/` (per scenario `<S>`):

| File | Contents |
|---|---|
| `pareto_<S>.csv` | Refined Pareto front: reuse, savings, total/pump/reuse energy, reused volume, minimum pressure, 24 hourly speeds |
| `pareto_raw_<S>.csv` | NSGA-II front before refinement |
| `keysol_<S>.csv` | Minimum-energy, balanced, and maximum-savings solutions with schedules |
| `summary_<S>.csv` | Scenario summary (one row; Table 3 source) |
| `run_info_<S>.json` | Settings, software versions, runtime, simulation count |
| `table3_manuscript.csv` | Table 3 as in the manuscript |
| `table3_summary_full.csv` | All summary columns, including net slope (kWh per m³ reused) |

## Software versions

The archived results were produced with Python 3.13.5, wntr 1.4.0rc3 (as reported by `wntr.__version__`), pymoo 0.6.1.5, numpy 2.4.2, and pandas 3.0.1 (recorded in `results/run_info_*.json`). The NSGA-II search depends on the package versions, so a rerun with other versions can give slightly different Pareto points. A check run with wntr 1.5.0 and pymoo 0.6.2 gave balanced reuse fractions within 0.6 percentage points of the archived values. The archived drainage results were generated with pyswmm 2.2.0 (swmm-toolkit 0.17.0).

## Funding

National Science Foundation, Grant No. 2142131.

## License

MIT License (see `LICENSE`).
