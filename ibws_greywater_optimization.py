"""
iBWS greywater reuse: energy-water trade-off optimization
==========================================================

Couples WNTR (EPANET) potable-supply hydraulics with NSGA-II (pymoo) to find
the Pareto front between total daily energy and potable water savings for
greywater reuse (toilet flushing) in nine US residential building scenarios.

Internal version: v16e (2026-10-15)
Author: Juneseok Lee, Manhattan University (NSF Grant No. 2142131)

USAGE
-----
    python ibws_greywater_optimization.py run S1 S2 S3   # optimize scenarios
    python ibws_greywater_optimization.py figures         # Table 3 + figures
    python ibws_greywater_optimization.py all             # run S1-S9, then figures

    Options: --pop 100 --gen 100 --seed 42 --results results --figures figures
    Several "run" commands may execute in parallel in the same folder.

MODEL
-----
Decision variables (25): 24 hourly booster-pump speeds s_h in [0.5, 1.0] and
the greywater reuse fraction r in [0, 0.8]. Each s_h is also bounded below so
that hourly pump flow stays within the defined pump curve (Q_h <= 1.5 Q_d s_h).

    Daily demand        V   = q_base x n_floors x 86,400 x mean(pattern)  [m3/d]
    Reused volume       V_r = V x min(0.65 r, 0.25)
    Water savings  (%)  S   = 100 x min(0.65 r, 0.25)
    Potable demand at every floor node is reduced by the same fraction.
    Total energy        E   = E_pump (WNTR) + V_r x (0.4 + 0.1) kWh/m3
    Constraint          minimum top-floor pressure >= 140 kPa (every hour)

Saturation: r* = 0.25 / 0.65 = 38.5 %; beyond r*, savings and energy are
constant.

Solution procedure
  1. NSGA-II (pop 100, gen 100, SBX 0.9/15, PM 20, seeded population).
  2. Exact pump-schedule refinement of every Pareto solution. The potable
     network has no storage, so each hour is hydraulically independent, and
     at a fixed (demand-driven) flow, pump power increases with speed. The
     energy-minimal schedule for a given r is therefore the lowest speed in
     each hour that keeps the top floor at >= 140 kPa, found by bisection
     (tolerance 1e-4). The front is augmented with r = 0 and r = r*, then
     filtered for non-dominance.
  3. Key solutions: minimum energy; maximum savings (lowest energy, then
     lowest r); balanced = closest to the ideal point after min-max
     normalization of both objectives with equal weights.

CHANGES FROM v16d (all verified by test runs)
  1. Pump speed pattern now has 24 hourly entries. v16d used 96 entries on
     EPANET's global 1-h pattern step, so only speeds 1-6 were applied (each
     for 4 h) and speeds 7-24 had no effect.
  2. Energy is integrated over 96 x 15-min intervals. v16d also summed the
     t = 24 h report point (+~0.45 %).
  3. The daily volume includes mean(pattern). v16d omitted it, charging reuse
     energy on 1/0.808 = 1.24 x the simulated volume.
  4. Nelder-Mead polishing of three solutions is replaced by the exact
     per-hour schedule (step 2), applied to all Pareto solutions.
  5. A unique EPANET file prefix per process allows parallel runs.
  6. Results are written per scenario as CSV; Table 3 and the figures are
     rebuilt from the CSVs.
  7. Per-hour speed floor Q_h / (1.5 Q_d): in v16d the 5F scenarios ran at
     0.5 speed beyond the last curve point at peak hours (up to 1.33 x),
     where EPANET extrapolates the curve.
"""

import argparse
import json
import os
import platform
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import wntr
import pymoo
from pymoo.core.problem import Problem
from pymoo.core.population import Population
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.termination import get_termination

for _cat in (UserWarning, FutureWarning, DeprecationWarning, RuntimeWarning):
    warnings.filterwarnings("ignore", category=_cat)

VERSION = "v16e"

# ======================================================================
# PARAMETERS (Table 1)
# ======================================================================
PER_CAPITA_LCD = 200            # L/capita/day (DeOreo et al. 2016, REUWS2)
PATTERN = np.array([            # 24 hourly demand multipliers
    0.3, 0.3, 0.3, 0.3, 0.4, 0.6,      # 00-05
    1.2, 1.5, 1.3, 0.9, 0.8, 0.7,      # 06-11
    0.8, 0.9, 0.8, 0.7, 0.9, 1.4,      # 12-17
    1.6, 1.2, 0.9, 0.7, 0.5, 0.4,      # 18-23
])
PATTERN_MEAN_TABLE2 = 0.808     # rounded mean used to normalize base demand (Table 2 note)

GREYWATER_FRACTION = 0.65       # greywater generated / demand
TOILET_FRACTION = 0.25          # toilet flushing / demand (reuse ceiling)
R_STAR = TOILET_FRACTION / GREYWATER_FRACTION   # 0.3846 saturation reuse

TREATMENT_KWH_M3 = 0.4          # MBR (Krzeminski et al. 2012)
DISTRIBUTION_KWH_M3 = 0.1       # greywater pumping to toilet risers
REUSE_KWH_M3 = TREATMENT_KWH_M3 + DISTRIBUTION_KWH_M3

P_MIN_KPA = 140.0               # minimum top-floor pressure
FLOOR_HEIGHT_M = 3.0
CITY_MAIN_HEAD_M = 30.0
PUMP_EFFICIENCY = 0.70
SPEED_MIN, SPEED_MAX = 0.5, 1.0
REUSE_MAX = 0.8

HYD_STEP_S = 900                # 15-min hydraulic/report step
DURATION_S = 24 * 3600
STEPS_PER_HOUR = 3600 // HYD_STEP_S

SCENARIO_IDS = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"]

# ======================================================================
# PLOT STYLE (seaborn "muted" colors; viridis colormap)
# ======================================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.linewidth": 1.0,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})
MUTED = {"blue": "#4878d0", "orange": "#ee854a", "green": "#6acc64",
         "red": "#d65f5f", "purple": "#956cb4", "gray": "#797979",
         "gold": "#d5bb67"}
HEIGHT_COLORS = {5: MUTED["blue"], 10: MUTED["green"], 20: MUTED["red"]}
DENSITY_MARKERS = {"Low": "o", "Med": "s", "High": "^"}


# ======================================================================
# SCENARIOS (Table 2)
# ======================================================================
def define_scenarios():
    """Nine US residential configurations: 3 density classes x 3 heights.

    Sources: floor plates (Portland 2020; Spokane 2018; MDPI Buildings 2021),
    unit sizes (RentCafe 2024), construction types (IBC 2021).
    Base demand (L/s per floor) = occ/floor x 200 / 86,400 / 0.808.
    """
    s = {
        "S1": dict(floors=5, units_per_floor=6, occ_per_unit=2.5, occ_per_floor=15,
                   construction="Type V-A wood", building_type="Suburban family (2-3BR)",
                   density_class="Low"),
        "S2": dict(floors=10, units_per_floor=8, occ_per_unit=2.5, occ_per_floor=20,
                   construction="Type I/III concrete", building_type="Urban family condo",
                   density_class="Low"),
        "S3": dict(floors=20, units_per_floor=6, occ_per_unit=2.5, occ_per_floor=15,
                   construction="Type I steel/concrete", building_type="Luxury condo tower",
                   density_class="Low"),
        "S4": dict(floors=5, units_per_floor=11, occ_per_unit=1.8, occ_per_floor=20,
                   construction="Type V-A wood", building_type="Standard rental (1-2BR)",
                   density_class="Med"),
        "S5": dict(floors=10, units_per_floor=16, occ_per_unit=1.8, occ_per_floor=29,
                   construction="Type I/III concrete", building_type="Market-rate rental",
                   density_class="Med"),
        "S6": dict(floors=20, units_per_floor=13, occ_per_unit=2.0, occ_per_floor=26,
                   construction="Type I steel/concrete", building_type="Market-rate tower",
                   density_class="Med"),
        "S7": dict(floors=5, units_per_floor=17, occ_per_unit=1.5, occ_per_floor=26,
                   construction="Type III-A wood/podium",
                   building_type="Urban workforce (studio/1BR)", density_class="High"),
        "S8": dict(floors=10, units_per_floor=23, occ_per_unit=1.5, occ_per_floor=35,
                   construction="Type I/III concrete", building_type="Dense urban rental",
                   density_class="High"),
        "S9": dict(floors=20, units_per_floor=21, occ_per_unit=1.5, occ_per_floor=32,
                   construction="Type I steel/concrete", building_type="Dense urban tower",
                   density_class="High"),
    }
    for sid, c in s.items():
        c["id"] = sid
        c["total_units"] = c["units_per_floor"] * c["floors"]
        c["total_occupants"] = c["occ_per_floor"] * c["floors"]
        c["demand_lps"] = round(
            c["occ_per_floor"] * PER_CAPITA_LCD / 86400 / PATTERN_MEAN_TABLE2, 4)
    return s


# ======================================================================
# WATER BALANCE
# ======================================================================
def reuse_share(r):
    """Fraction of total demand met by reused greywater: min(0.65 r, 0.25)."""
    return np.minimum(GREYWATER_FRACTION * np.asarray(r, float), TOILET_FRACTION)


# ======================================================================
# BUILDING HYDRAULIC MODEL (WNTR / EPANET)
# ======================================================================
class BuildingModel:
    """Booster pump + vertical riser serving n floors (demand-driven)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.sid = cfg["id"]
        self.n_floors = cfg["floors"]
        self.q_base = cfg["demand_lps"] / 1000.0                      # m3/s per floor
        self.daily_demand_m3 = (self.q_base * self.n_floors * 86400
                                * PATTERN.mean())                    # simulated volume
        # Pump sizing: design flow = peak-hour demand; quadratic H-Q curve
        self.design_flow = self.q_base * self.n_floors * PATTERN.max()
        static = self.n_floors * FLOOR_HEIGHT_M
        friction = 3.0 + 0.4 * self.n_floors
        net = static - CITY_MAIN_HEAD_M + P_MIN_KPA / 9.81 + friction
        self.design_head = max(net, 10.0)
        self.shutoff_head = 1.33 * self.design_head
        self.curve_qmax_frac = 1.5                                   # last curve point
        self.file_prefix = f"_epanet_{self.sid}_{os.getpid()}"
        self.n_sims = 0

    def _network(self, r):
        wn = wntr.network.WaterNetworkModel()
        wn.add_pattern("Demand", list(PATTERN))
        wn.add_reservoir("CityMain", base_head=CITY_MAIN_HEAD_M)
        wn.add_junction("Header", base_demand=0.0, elevation=0.0)
        q_floor = self.q_base * (1.0 - float(reuse_share(r)))
        for f in range(1, self.n_floors + 1):
            wn.add_junction(f"Floor{f}", base_demand=q_floor,
                            elevation=f * FLOOR_HEIGHT_M, demand_pattern="Demand")
        pts = [(qf * self.design_flow, max(self.shutoff_head * (1 - 0.35 * qf ** 2), 0.1))
               for qf in (0.0, 0.25, 0.5, 0.75, 1.0, self.curve_qmax_frac)]
        wn.add_curve("PumpCurve", "HEAD", pts)
        wn.add_pump("BoosterPump", "CityMain", "Header",
                    pump_type="HEAD", pump_parameter="PumpCurve")
        wn.add_pipe("HeaderToF1", "Header", "Floor1",
                    length=5.0, diameter=0.075, roughness=100)
        for f in range(1, self.n_floors):
            wn.add_pipe(f"F{f}ToF{f+1}", f"Floor{f}", f"Floor{f+1}",
                        length=FLOOR_HEIGHT_M, diameter=0.065, roughness=100)
        t = wn.options.time
        t.duration = DURATION_S
        t.hydraulic_timestep = HYD_STEP_S
        t.report_timestep = HYD_STEP_S
        t.pattern_timestep = 3600            # global step for ALL patterns
        return wn

    def simulate(self, speeds, r):
        """Run 24 h. speeds: 24 hourly values. Returns dict with
        energy_kwh, p_hour_kpa (24 hourly minima), q_hour_max (m3/s)."""
        speeds = np.asarray(speeds, float)
        if speeds.shape != (24,):
            raise ValueError("speeds must have 24 hourly values")
        wn = self._network(r)
        wn.add_pattern("PumpSpeed", list(speeds))        # one entry per hour
        wn.get_link("BoosterPump").speed_pattern_name = "PumpSpeed"
        res = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=self.file_prefix)
        self.n_sims += 1

        flow = res.link["flowrate"]["BoosterPump"]
        keep = flow.index < DURATION_S                    # 96 intervals, drop t = 24 h
        q = flow[keep].values
        h = np.abs(res.link["headloss"]["BoosterPump"][keep].values)
        p_top = res.node["pressure"][f"Floor{self.n_floors}"][keep].values * 9.81
        if len(q) != 24 * STEPS_PER_HOUR:
            raise RuntimeError(f"unexpected number of report steps: {len(q)}")

        power_kw = 1000.0 * 9.81 * q * h / 1000.0 / PUMP_EFFICIENCY
        return {
            "energy_kwh": power_kw.sum() * HYD_STEP_S / 3600.0,
            "p_hour_kpa": p_top.reshape(24, STEPS_PER_HOUR).min(axis=1),
            "q_hour_max": q.reshape(24, STEPS_PER_HOUR).max(axis=1),
        }

    def reuse_energy(self, r):
        return self.daily_demand_m3 * float(reuse_share(r)) * REUSE_KWH_M3

    def reused_m3(self, r):
        return self.daily_demand_m3 * float(reuse_share(r))

    def speed_floor(self, r):
        """Hourly minimum speed keeping pump flow within the defined curve:
        s_h >= Q_h / (1.5 Q_d). Pump flow equals floor demand (no storage)."""
        q_h = self.q_base * self.n_floors * PATTERN * (1.0 - float(reuse_share(r)))
        return np.clip(q_h / (self.curve_qmax_frac * self.design_flow) * (1 + 1e-6),
                       SPEED_MIN, SPEED_MAX)

    def cleanup(self):
        for ext in (".inp", ".rpt", ".bin", ".hyd"):
            try:
                os.remove(self.file_prefix + ext)
            except OSError:
                pass


def exact_schedule(model, r, tol=1e-4):
    """Energy-minimal 24-h speed schedule for reuse fraction r.

    Valid because hours are hydraulically independent (no storage) and pump
    power rises with speed at fixed demand-driven flow: each hour's optimum is
    the lowest speed in [speed floor, 1.0] meeting P_MIN_KPA. All 24 hours are
    bisected simultaneously (one 24-h simulation per iteration).
    Returns (speeds, sim_result) or (None, None) if infeasible at full speed.
    """
    lo = model.speed_floor(r)
    hi = np.full(24, SPEED_MAX)
    if model.simulate(hi, r)["p_hour_kpa"].min() < P_MIN_KPA:
        return None, None
    ok_lo = model.simulate(lo, r)["p_hour_kpa"] >= P_MIN_KPA
    hi[ok_lo] = lo[ok_lo]
    while (hi - lo).max() > tol:
        open_ = (hi - lo) > tol
        mid = np.where(open_, 0.5 * (lo + hi), hi)
        feas = model.simulate(mid, r)["p_hour_kpa"] >= P_MIN_KPA
        hi = np.where(open_ & feas, mid, hi)
        lo = np.where(open_ & ~feas, mid, lo)
    out = model.simulate(hi, r)
    if out["p_hour_kpa"].min() < P_MIN_KPA - 1e-6:          # safeguard
        raise RuntimeError(f"{model.sid}: bisection result infeasible at r={r:.4f}")
    return hi, out


# ======================================================================
# NSGA-II PROBLEM
# ======================================================================
class GreywaterProblem(Problem):
    """min E_total, max water savings; s.t. top-floor pressure >= 140 kPa."""

    def __init__(self, model):
        super().__init__(n_var=25, n_obj=2, n_ieq_constr=1,
                         xl=np.array([SPEED_MIN] * 24 + [0.0]),
                         xu=np.array([SPEED_MAX] * 24 + [REUSE_MAX]))
        self.model = model
        self.n_failed = 0

    def _evaluate(self, X, out, *args, **kwargs):
        F = np.zeros((len(X), 2))
        G = np.zeros((len(X), 1))
        for i, x in enumerate(X):
            r = x[24]
            try:
                sim = self.model.simulate(np.maximum(x[:24], self.model.speed_floor(r)), r)
                F[i, 0] = sim["energy_kwh"] + self.model.reuse_energy(r)
                F[i, 1] = -100.0 * float(reuse_share(r))
                G[i, 0] = P_MIN_KPA - sim["p_hour_kpa"].min()
            except Exception:
                self.n_failed += 1
                F[i] = (1e3, 0.0)
                G[i, 0] = 1e3
        out["F"] = F
        out["G"] = G


def nondominated(E, S, r):
    """Indices of non-dominated points (min E, max S); ties -> lowest r."""
    order = np.lexsort((r, -S, E))
    keep, best_s = [], -np.inf
    for i in order:
        if S[i] > best_s + 1e-9:
            keep.append(i)
            best_s = S[i]
    return np.array(keep)


def closest_to_ideal(E, S):
    """Equal-weight, min-max normalized distance to (min E, max S)."""
    e = (E - E.min()) / (E.max() - E.min() + 1e-12)
    s = (S.max() - S) / (S.max() - S.min() + 1e-12)
    return int(np.argmin(np.hypot(e, s)))


# ======================================================================
# RUN ONE SCENARIO
# ======================================================================
def run_scenario(cfg, pop_size, n_gen, seed, results_dir):
    t0 = time.time()
    sid = cfg["id"]
    model = BuildingModel(cfg)
    print(f"\n{'=' * 68}\n{sid}: {cfg['floors']}F, {cfg['units_per_floor']} u/fl, "
          f"{cfg['demand_lps']:.4f} L/s/fl, {cfg['building_type']}")
    print(f"  Daily demand {model.daily_demand_m3:.1f} m3/d | pump Qd "
          f"{model.design_flow * 1000:.2f} L/s, H0 {model.shutoff_head:.1f} m")

    try:
        base = model.simulate(np.ones(24), 0.0)
        print(f"  Baseline (full speed, r=0): {base['energy_kwh']:.2f} kWh/d, "
              f"min top-floor P {base['p_hour_kpa'].min():.1f} kPa")
        if base["p_hour_kpa"].min() < P_MIN_KPA:
            print("  Infeasible at full speed; scenario skipped.")
            return None

        # ---- NSGA-II with seeded initial population --------------------
        rng = np.random.RandomState(seed)
        s_lo = float(np.clip(np.sqrt(P_MIN_KPA / max(base["p_hour_kpa"].min(), 141.0))
                             * 1.05, SPEED_MIN, 0.98))
        n_seed = 20
        X0 = np.zeros((pop_size, 25))
        X0[:n_seed, 24] = np.linspace(0.0, REUSE_MAX, n_seed)
        mid, half = s_lo + 0.5 * (1 - s_lo), 0.3 * (1 - s_lo)
        X0[:n_seed, :24] = np.clip(mid + rng.uniform(-half, half, (n_seed, 24)), s_lo, 1.0)
        X0[n_seed:, :24] = rng.uniform(s_lo, 1.0, (pop_size - n_seed, 24))
        X0[n_seed:, 24] = rng.uniform(0.0, REUSE_MAX, pop_size - n_seed)

        problem = GreywaterProblem(model)
        algorithm = NSGA2(pop_size=pop_size, sampling=Population.new("X", X0),
                          crossover=SBX(prob=0.9, eta=15), mutation=PM(eta=20),
                          eliminate_duplicates=True)
        print(f"  NSGA-II pop={pop_size}, gen={n_gen} ...")
        res = minimize(problem, algorithm, get_termination("n_gen", n_gen),
                       seed=seed, verbose=False)
        if res.F is None:
            print("  No feasible NSGA-II solutions; scenario skipped.")
            return None
        Xr = np.atleast_2d(res.X)
        Fr = np.atleast_2d(res.F)
        Gr = np.atleast_2d(res.G) if res.G is not None else np.zeros((len(Fr), 1))
        feas = Gr[:, 0] <= 0
        Xr, Fr = Xr[feas], Fr[feas]
        t_nsga = time.time() - t0
        print(f"  NSGA-II: {len(Fr)} feasible Pareto solutions, "
              f"{problem.n_failed} failed evaluations, {t_nsga:.0f} s")

        raw = pd.DataFrame({"reuse_pct": 100 * Xr[:, 24],
                            "savings_pct": -Fr[:, 1],
                            "energy_total_kwh": Fr[:, 0]})
        applied = np.array([np.maximum(x[:24], model.speed_floor(x[24])) for x in Xr])
        for h in range(24):
            raw[f"speed_h{h:02d}"] = applied[:, h]
        raw.sort_values("reuse_pct").to_csv(
            os.path.join(results_dir, f"pareto_raw_{sid}.csv"), index=False)

        # ---- Exact pump-schedule refinement of every Pareto solution ---
        r_set = np.unique(np.concatenate([Xr[:, 24], [0.0, R_STAR]]))
        rows, scheds = [], []
        for r in r_set:
            sp, sim = exact_schedule(model, r)
            if sp is None:
                continue
            e_pump = sim["energy_kwh"]
            e_reuse = model.reuse_energy(r)
            rows.append(dict(reuse_pct=100 * r,
                             savings_pct=100 * float(reuse_share(r)),
                             energy_total_kwh=e_pump + e_reuse,
                             energy_pump_kwh=e_pump,
                             energy_reuse_kwh=e_reuse,
                             reused_m3_day=model.reused_m3(r),
                             min_pressure_kpa=sim["p_hour_kpa"].min(),
                             curve_exceeded_hours=int(np.sum(
                                 sim["q_hour_max"] > model.curve_qmax_frac
                                 * model.design_flow * sp + 1e-12)),
                             source=("endpoint" if r in (0.0, R_STAR) else "nsga2")))
            scheds.append(sp)
        ref = pd.DataFrame(rows)
        for h in range(24):
            ref[f"speed_h{h:02d}"] = [s[h] for s in scheds]

        E = ref["energy_total_kwh"].values
        S = ref["savings_pct"].values
        R = ref["reuse_pct"].values
        front = ref.iloc[nondominated(E, S, R)].sort_values("savings_pct").reset_index(drop=True)
        front.to_csv(os.path.join(results_dir, f"pareto_{sid}.csv"), index=False)

        # ---- Key solutions ----------------------------------------------
        E, S, R = (front[c].values for c in ("energy_total_kwh", "savings_pct", "reuse_pct"))
        i_min = int(np.lexsort((R, E))[0])
        plateau = np.where(S >= S.max() - 1e-6)[0]
        i_max = int(plateau[np.lexsort((R[plateau], E[plateau]))[0]])
        i_bal = closest_to_ideal(E, S)
        key = front.iloc[[i_min, i_bal, i_max]].copy()
        key.insert(0, "solution", ["min_energy", "balanced", "max_savings"])
        key.insert(0, "scenario", sid)
        key.to_csv(os.path.join(results_dir, f"keysol_{sid}.csv"), index=False)

        # Net slope of the front below saturation (kWh per m3 reused)
        lin = front[front["savings_pct"] < 100 * TOILET_FRACTION - 1e-6]
        slope = (np.polyfit(lin["reused_m3_day"], lin["energy_total_kwh"], 1)[0]
                 if len(lin) >= 3 else np.nan)
        pump_slope = (np.polyfit(lin["reused_m3_day"], lin["energy_pump_kwh"], 1)[0]
                      if len(lin) >= 3 else np.nan)

        mn, bl, mx = (front.iloc[i] for i in (i_min, i_bal, i_max))
        sched_cols = [f"speed_h{h:02d}" for h in range(24)]
        summary = {
            "Scenario": sid, "Density Class": cfg["density_class"],
            "Floors": cfg["floors"], "Units/Floor": cfg["units_per_floor"],
            "Occ/Floor": cfg["occ_per_floor"], "Demand (L/s)": cfg["demand_lps"],
            "Total Units": cfg["total_units"], "Total Occupants": cfg["total_occupants"],
            "Building Type": cfg["building_type"],
            "Daily Demand (m3/d)": round(model.daily_demand_m3, 2),
            "Pump Qd (L/s)": round(model.design_flow * 1000, 3),
            "Pump H0 (m)": round(model.shutoff_head, 2),
            "Baseline Energy (kWh/d)": round(base["energy_kwh"], 3),
            "Baseline Min Pressure (kPa)": round(base["p_hour_kpa"].min(), 1),
            "Min Energy (kWh/d)": round(mn["energy_total_kwh"], 3),
            "Min Energy Reuse (%)": round(mn["reuse_pct"], 2),
            "Pump-only Reduction (%)": round(
                100 * (1 - mn["energy_total_kwh"] / base["energy_kwh"]), 1),
            "Min E Speed Range": f"{mn[sched_cols].min():.3f}-{mn[sched_cols].max():.3f}",
            "Max Savings (%)": round(mx["savings_pct"], 2),
            "Max Savings Reuse (%)": round(mx["reuse_pct"], 2),
            "Max Savings Energy (kWh/d)": round(mx["energy_total_kwh"], 3),
            "Balanced Energy (kWh/d)": round(bl["energy_total_kwh"], 3),
            "Balanced Savings (%)": round(bl["savings_pct"], 2),
            "Balanced Reuse (%)": round(bl["reuse_pct"], 2),
            "Bal Speed Range": f"{bl[sched_cols].min():.3f}-{bl[sched_cols].max():.3f}",
            "Net Slope (kWh/m3 reused)": round(slope, 4),
            "Pump Slope (kWh/m3 reused)": round(pump_slope, 4),
            "Gross Reuse Energy (kWh/m3)": REUSE_KWH_M3,
            "Pareto Points (raw)": len(Fr), "Pareto Points (refined)": len(front),
            "Curve-Exceeded Hours (max)": int(front["curve_exceeded_hours"].max()),
        }
        pd.DataFrame([summary]).to_csv(
            os.path.join(results_dir, f"summary_{sid}.csv"), index=False)

        elapsed = time.time() - t0
        info = dict(scenario=sid, version=VERSION, pop=pop_size, gen=n_gen, seed=seed,
                    n_simulations=model.n_sims, failed_evaluations=problem.n_failed,
                    runtime_s=round(elapsed, 1), python=platform.python_version(),
                    wntr=wntr.__version__, pymoo=pymoo.__version__,
                    numpy=np.__version__, pandas=pd.__version__,
                    date=time.strftime("%Y-%m-%d %H:%M:%S"))
        with open(os.path.join(results_dir, f"run_info_{sid}.json"), "w") as fh:
            json.dump(info, fh, indent=2)

        print(f"  Min energy : {mn['energy_total_kwh']:.2f} kWh/d at r={mn['reuse_pct']:.1f}% "
              f"(pump-only reduction {summary['Pump-only Reduction (%)']:.1f}%)")
        print(f"  Balanced   : {bl['energy_total_kwh']:.2f} kWh/d, "
              f"{bl['savings_pct']:.1f}% savings, r={bl['reuse_pct']:.1f}%")
        print(f"  Max savings: {mx['energy_total_kwh']:.2f} kWh/d, "
              f"{mx['savings_pct']:.1f}% savings, r={mx['reuse_pct']:.1f}%")
        print(f"  Net slope {slope:.3f} kWh/m3 reused "
              f"(gross {REUSE_KWH_M3:.2f}, pump {pump_slope:+.3f})")
        if summary["Curve-Exceeded Hours (max)"] > 0:
            print("  WARNING: pump flow beyond the last curve point in some hours "
                  "(EPANET extrapolation).")
        print(f"  Done in {elapsed / 60:.1f} min ({model.n_sims} simulations)")
        return summary
    finally:
        model.cleanup()


# ======================================================================
# LOAD RESULTS
# ======================================================================
def load_results(results_dir):
    out = {}
    for sid in SCENARIO_IDS:
        f_sum = os.path.join(results_dir, f"summary_{sid}.csv")
        f_par = os.path.join(results_dir, f"pareto_{sid}.csv")
        f_key = os.path.join(results_dir, f"keysol_{sid}.csv")
        if all(os.path.exists(f) for f in (f_sum, f_par, f_key)):
            out[sid] = {"summary": pd.read_csv(f_sum).iloc[0],
                        "front": pd.read_csv(f_par),
                        "key": pd.read_csv(f_key).set_index("solution")}
    return out


# ======================================================================
# TABLE 3
# ======================================================================
def build_table3(res, results_dir):
    df = pd.DataFrame([res[s]["summary"] for s in SCENARIO_IDS if s in res])
    df.to_csv(os.path.join(results_dir, "table3_summary_full.csv"), index=False)
    t3 = pd.DataFrame({
        "Scenario": df["Scenario"],
        "Config (floor, units/floor)": [f"{f}F, {u}u/fl" for f, u in
                                        zip(df["Floors"], df["Units/Floor"])],
        "Density": df["Density Class"],
        "Total Units": df["Total Units"],
        "Baseline E (kWh/day)": df["Baseline Energy (kWh/d)"].round(1),
        "Min E (kWh/day)": df["Min Energy (kWh/d)"].round(1),
        "Max Sav (%)": df["Max Savings (%)"].round(1),
        "Balanced E (kWh/day)": df["Balanced Energy (kWh/d)"].round(1),
        "Bal Sav (%)": df["Balanced Savings (%)"].round(1),
        "Bal Reuse (%)": df["Balanced Reuse (%)"].round(1),
    })
    t3.to_csv(os.path.join(results_dir, "table3_manuscript.csv"), index=False)

    print("\nTABLE 3 (manuscript columns)")
    print(t3.to_string(index=False))
    print("\nRANGES ACROSS SCENARIOS")
    for col in ("Balanced Reuse (%)", "Balanced Savings (%)", "Pump-only Reduction (%)",
                "Net Slope (kWh/m3 reused)", "Baseline Energy (kWh/d)",
                "Baseline Min Pressure (kPa)", "Max Savings (%)"):
        v = df[col].astype(float)
        print(f"  {col:32s} {v.min():8.3f} - {v.max():8.3f}")
    ok = df["Max Savings (%)"].max() <= 100 * TOILET_FRACTION + 1e-6
    print(f"  Water balance: r* = {100 * R_STAR:.1f}%, max savings <= 25%: "
          f"{'OK' if ok else 'VIOLATED'}")
    return df


# ======================================================================
# FIGURES
# ======================================================================
def _save(fig, fig_dir, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  Saved {name}.png/.pdf")


def fig3_pareto_grid(res, scen, fig_dir):
    """Fig 3: 3x3 Pareto fronts (rows = density, columns = height)."""
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    grid = {(c["density_class"], c["floors"]): sid for sid, c in scen.items()}
    labels = {"Low": "Low Density\n(Family/Condo)", "Med": "Medium Density\n(Standard Rental)",
              "High": "High Density\n(Dense/Workforce)"}
    sc = None
    for i, dens in enumerate(("Low", "Med", "High")):
        for j, fl in enumerate((5, 10, 20)):
            ax, sid = axes[i, j], grid[(dens, fl)]
            if sid not in res:
                ax.text(0.5, 0.5, f"{sid}: no results", ha="center", va="center",
                        transform=ax.transAxes, color=MUTED["gray"])
                ax.set_title(f"{fl}F ({sid})", fontsize=9, fontweight="bold")
                continue
            fr, key, sm = res[sid]["front"], res[sid]["key"], res[sid]["summary"]
            sc = ax.scatter(fr["energy_total_kwh"], fr["savings_pct"], c=fr["reuse_pct"],
                            cmap="viridis", vmin=0, vmax=100 * REUSE_MAX, s=40, alpha=0.8,
                            edgecolors="white", linewidth=0.3)
            ax.scatter(key.loc["balanced", "energy_total_kwh"],
                       key.loc["balanced", "savings_pct"], s=200, marker="*",
                       c=MUTED["red"], edgecolors="white", linewidth=1.2, zorder=5)
            ax.scatter(sm["Baseline Energy (kWh/d)"], 0, s=80, facecolors="none",
                       edgecolors="black", linewidth=1.5, zorder=5)
            ax.set_title(f"{fl}F, {scen[sid]['units_per_floor']}u/fl ({sid})",
                         fontsize=9, fontweight="bold")
            if j == 0:
                ax.set_ylabel("Water Savings (%)", fontsize=9)
            if i == 2:
                ax.set_xlabel("Total Energy (kWh/day)", fontsize=9)
    for i, d in enumerate(("Low", "Med", "High")):
        axes[i, 0].annotate(labels[d], xy=(-0.35, 0.5), xycoords="axes fraction",
                            fontsize=10, fontweight="bold", ha="center", va="center",
                            rotation=90)
    if sc is not None:
        cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        fig.colorbar(sc, cax=cax).set_label("Greywater Reuse (%)", fontsize=10)
    fig.legend(handles=[
        Line2D([0], [0], marker="*", color="w", markerfacecolor=MUTED["red"],
               markersize=14, label="Balanced solution"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor="black", markersize=10, label="Baseline (full speed, no reuse)")],
        loc="upper center", bbox_to_anchor=(0.45, 0.99), ncol=2, frameon=False)
    fig.suptitle("Energy-Water Trade-offs Across 9 US Building Configurations",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0, 0.91, 0.98])
    _save(fig, fig_dir, "fig3_pareto_3x3")


def fig4_reuse_curve(res, scen, fig_dir):
    """Fig 4: (a) reuse-to-savings mapping; (b) gross reuse energy and savings."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    r = np.linspace(0, 100 * REUSE_MAX, 500)
    sav = 100 * reuse_share(r / 100)
    rs = 100 * R_STAR
    bal = [(res[s]["key"].loc["balanced", "reuse_pct"],
            res[s]["key"].loc["balanced", "savings_pct"], s) for s in SCENARIO_IDS if s in res]

    ax = axes[0]
    ax.plot(r, sav, color=MUTED["blue"], lw=2.5, zorder=3)
    ax.axhline(100 * TOILET_FRACTION, color=MUTED["orange"], ls="--", lw=1.5, alpha=0.7)
    ax.axvline(rs, color=MUTED["orange"], ls="--", lw=1.5, alpha=0.7)
    ax.text(rs + 2, 26, "25% ceiling\n(toilet demand)", fontsize=8, color=MUTED["orange"])
    ax.text(rs + 1, 4, f"r* = {rs:.1f}%\n(saturation)", fontsize=8, color=MUTED["orange"])
    if bal:
        lo, hi = min(b[0] for b in bal), max(b[0] for b in bal)
        ax.axvspan(lo - 0.5, hi + 0.5, alpha=0.12, color=MUTED["green"])
        ax.text(lo - 0.5, 1, f"Balanced range\n{lo:.1f}-{hi:.1f}%", fontsize=8,
                color=MUTED["green"], style="italic", ha="right")
    for rb, sb, s in bal:
        ax.scatter(rb, sb, s=100, marker=DENSITY_MARKERS[scen[s]["density_class"]],
                   c=HEIGHT_COLORS[scen[s]["floors"]], edgecolors="white", lw=1, zorder=5)
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=HEIGHT_COLORS[f],
                              markersize=8, label=f"{f}F") for f in (5, 10, 20)] +
              [Line2D([0], [0], marker=DENSITY_MARKERS[d], color="w", markerfacecolor="gray",
                      markersize=8, label=f"{d} density") for d in ("Low", "Med", "High")],
              fontsize=8, ncol=2, loc="center right")
    ax.set(xlim=(-2, 82), ylim=(-1, 30))
    ax.set_xlabel("Greywater Reuse Fraction (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Water Savings (%)", fontsize=11, fontweight="bold")
    ax.set_title("(a) Reuse-to-Savings Mapping", fontsize=12, fontweight="bold")

    ax = axes[1]
    pen = REUSE_KWH_M3 * reuse_share(r / 100)       # kWh per m3 of total demand
    ax.fill_between(r, 0, pen, alpha=0.15, color=MUTED["blue"])
    ax.plot(r, pen, color=MUTED["blue"], lw=2)
    ax2 = ax.twinx()
    ax2.plot(r, sav, color=MUTED["green"], lw=2.5)
    ax2.axhline(100 * TOILET_FRACTION, color=MUTED["orange"], ls="--", lw=1.5, alpha=0.7)
    ax.axvline(rs, color=MUTED["orange"], ls="--", lw=1.5, alpha=0.7)
    for rb, _, s in bal:
        ax.scatter(rb, REUSE_KWH_M3 * reuse_share(rb / 100), s=100,
                   marker=DENSITY_MARKERS[scen[s]["density_class"]],
                   c=HEIGHT_COLORS[scen[s]["floors"]], edgecolors="white", lw=1, zorder=5)
    ax.text(48, 0.143, "Beyond r*: no further savings;\nreuse energy constant",
            fontsize=8, color=MUTED["red"], style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFF0F0", ec=MUTED["red"], alpha=0.8))
    ax.set(xlim=(-2, 82), ylim=(0, 0.16))
    ax2.set_ylim(0, 30)
    ax.set_xlabel("Greywater Reuse Fraction (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Gross Reuse Energy (kWh per m³ of total demand)", fontsize=10,
                  fontweight="bold", color=MUTED["blue"])
    ax2.set_ylabel("Water Savings (%)", fontsize=11, fontweight="bold", color=MUTED["green"])
    ax.set_title("(b) Energy-Savings Trade-off Structure", fontsize=12, fontweight="bold")
    fig.suptitle(f"Greywater Reuse: Mapping and Balanced Solutions "
                 f"(saturation at r* = {rs:.1f}%)", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, fig_dir, "fig4_reuse_curve")


def fig5_energy_vs_savings(res, scen, fig_dir):
    """Fig 5: total energy vs water savings, all scenarios."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for s in SCENARIO_IDS:
        if s not in res:
            continue
        fr = res[s]["front"].sort_values("savings_pct")
        c = scen[s]
        ax.plot(fr["savings_pct"], fr["energy_total_kwh"],
                marker=DENSITY_MARKERS[c["density_class"]], color=HEIGHT_COLORS[c["floors"]],
                lw=2, ms=4, alpha=0.75,
                label=f"{c['floors']}F, {c['units_per_floor']}u ({c['density_class']})")
    ax.set_xlabel("Water Savings (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Total Energy (kWh/day)", fontsize=12, fontweight="bold")
    ax.set_title("Energy Cost of Water Savings Across 9 US Building Configurations",
                 fontsize=13, fontweight="bold")
    ax.legend(title="Configuration", fontsize=8, ncol=3, loc="upper left")
    fig.tight_layout()
    _save(fig, fig_dir, "fig5_energy_vs_savings")


def figS1_pump_schedules(res, scen, fig_dir, pairs=(("S4", "5F, Standard Rental"),
                                                     ("S6", "20F, Market-rate Tower"))):
    """Fig S1: hourly pump speeds of the key solutions (low- vs high-rise)."""
    fig, axes = plt.subplots(1, len(pairs), figsize=(14, 5))
    hours = np.arange(24)
    cols = [f"speed_h{h:02d}" for h in hours]
    for ax, (s, label) in zip(np.atleast_1d(axes), pairs):
        if s not in res:
            ax.text(0.5, 0.5, f"{s}: no results", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        ax.fill_between(hours, 0.45, 0.45 + 0.55 * PATTERN / PATTERN.max(), step="mid",
                        alpha=0.12, color=MUTED["gray"], label="Demand pattern (scaled)")
        for key, color, mk, lab in (("min_energy", MUTED["blue"], "o", "Min energy"),
                                    ("balanced", MUTED["green"], "*", "Balanced"),
                                    ("max_savings", MUTED["gold"], "^", "Max savings")):
            k = res[s]["key"].loc[key]
            ax.step(hours, k[cols].values.astype(float), where="mid", color=color, lw=2,
                    marker=mk, ms=5, label=f"{lab} (r = {k['reuse_pct']:.0f}%)")
        ax.axhline(1.0, color="black", ls="--", lw=1.2, alpha=0.5, label="Full speed")
        ax.set(xlim=(-0.5, 23.5), ylim=(0.4, 1.06), xticks=hours[::3])
        ax.set_xticklabels([f"{h:02d}" for h in hours[::3]])
        ax.set_title(f"{label} ({s})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Pump Speed (fraction of rated)")
        ax.legend(loc="lower left", fontsize=7)
    fig.suptitle("Energy-Optimal Hourly Pump Speeds: Low-Rise vs High-Rise",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, fig_dir, "figS1_pump_schedules")


def make_figures(results_dir, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    scen = define_scenarios()
    res = load_results(results_dir)
    if not res:
        sys.exit(f"No complete results found in '{results_dir}'. Run scenarios first.")
    missing = [s for s in SCENARIO_IDS if s not in res]
    if missing:
        print(f"NOTE: missing scenarios {missing}; figures will be partial.")
    build_table3(res, results_dir)
    print("\nFIGURES")
    fig3_pareto_grid(res, scen, fig_dir)
    fig4_reuse_curve(res, scen, fig_dir)
    fig5_energy_vs_savings(res, scen, fig_dir)
    figS1_pump_schedules(res, scen, fig_dir)


# ======================================================================
# MAIN
# ======================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description="iBWS greywater energy-water optimization")
    ap.add_argument("command", choices=["run", "figures", "all"])
    ap.add_argument("scenarios", nargs="*", help="e.g. S1 S6 (default: all nine)")
    ap.add_argument("--pop", type=int, default=100)
    ap.add_argument("--gen", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--results", default="results")
    ap.add_argument("--figures", default="figures")
    a = ap.parse_args(argv)

    os.makedirs(a.results, exist_ok=True)
    scen = define_scenarios()
    ids = [s.upper() for s in a.scenarios] or SCENARIO_IDS
    bad = [s for s in ids if s not in scen]
    if bad:
        ap.error(f"unknown scenario(s): {bad}")

    print(f"iBWS greywater optimization {VERSION} | WNTR {wntr.__version__}, "
          f"pymoo {pymoo.__version__} | pop={a.pop}, gen={a.gen}, seed={a.seed}")
    if a.command in ("run", "all"):
        t0 = time.time()
        for sid in ids:
            run_scenario(scen[sid], a.pop, a.gen, a.seed, a.results)
        print(f"\nTotal runtime {(time.time() - t0) / 60:.1f} min")
    if a.command in ("figures", "all"):
        make_figures(a.results, a.figures)


if __name__ == "__main__":
    main()
