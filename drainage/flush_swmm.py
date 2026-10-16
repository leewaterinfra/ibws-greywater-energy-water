"""
Discrete-event PySWMM simulation of main-drain velocity for scenarios S1-S9.

Dynamic-wave routing (0.5 s routing step) with stochastic fixture events
(toilet flushes and greywater draws) generated from building occupancy and
timed by the 24-h diurnal pattern. Greywater reuse removes a fraction d of
greywater events upstream of the main drain; toilet flushes are unchanged.

Assumptions (manuscript Methods, Drainage Velocity Assessment):
  collector 100 mm (5 m, n = 0.011) -> main drain 150 mm (10 m, n = 0.013),
  slope S, free outfall; all fixture discharges enter at the upstream node;
  toilet 50 L/cap/d as flushes at 1.4 L/s (about 6 L each); greywater
  130 L/cap/d: showers 52 L/cap/d (0.13 L/s for 400 s) and short draws
  78 L/cap/d (0.10 L/s for 60 s).

Run this file directly for the steady-state check (1.4 L/s in a 150 mm drain
at 2% slope; Manning gives 0.69 m/s). run_drainage_grid.py runs all cases.
"""
import sys, os, numpy as np
from pyswmm import Simulation, Nodes, Links

PATTERN = np.array([0.3,0.3,0.3,0.3,0.4,0.6,1.2,1.5,1.3,0.9,0.8,0.7,
                    0.8,0.9,0.8,0.7,0.9,1.4,1.6,1.2,0.9,0.7,0.5,0.4])
SC = {'S1':(5,15),'S2':(10,20),'S3':(20,15),'S4':(5,20),'S5':(10,29),
      'S6':(20,26),'S7':(5,26),'S8':(10,35),'S9':(20,32)}   # (floors, occ/floor)
T = 86400
DMAIN = 0.150

def area_circ(y, D=DMAIN):
    y = min(max(y,0.0), D)
    th = 2*np.arccos(1-2*y/D)
    return D**2/8*(th-np.sin(th))

def vel(q_lps, y):
    a = area_circ(y)
    return (q_lps/1000.0)/a if a > 1e-9 else 0.0

def write_inp(path, slope, d_collector=0.100, d_main=0.150, L1=5.0, L2=10.0):
    z3 = 0.0; z2 = z3 + L2*slope; z1 = z2 + L1*slope
    inp = f"""[TITLE]
iBWS main drain

[OPTIONS]
FLOW_UNITS LPS
INFILTRATION HORTON
FLOW_ROUTING DYNWAVE
LINK_OFFSETS DEPTH
START_DATE 01/01/2026
START_TIME 00:00:00
REPORT_START_DATE 01/01/2026
REPORT_START_TIME 00:00:00
END_DATE 01/02/2026
END_TIME 00:00:00
SWEEP_START 01/01
SWEEP_END 12/31
DRY_DAYS 0
REPORT_STEP 00:01:00
WET_STEP 00:00:01
DRY_STEP 00:00:01
ROUTING_STEP 0.5
INERTIAL_DAMPING PARTIAL
NORMAL_FLOW_LIMITED BOTH
FORCE_MAIN_EQUATION H-W
VARIABLE_STEP 0
LENGTHENING_STEP 0
MIN_SURFAREA 0.01
MAX_TRIALS 8
HEAD_TOLERANCE 0.0015
MINIMUM_STEP 0.1
THREADS 1

[JUNCTIONS]
J1 {z1:.4f} 1.0 0 0 0
J2 {z2:.4f} 1.0 0 0 0

[OUTFALLS]
OUT {z3:.4f} FREE NO

[CONDUITS]
C1 J1 J2 {L1} 0.011 0 0 0 0
C2 J2 OUT {L2} 0.013 0 0 0 0

[XSECTIONS]
C1 CIRCULAR {d_collector} 0 0 0 1
C2 CIRCULAR {d_main} 0 0 0 1

[REPORT]
SUBCATCHMENTS NONE
NODES NONE
LINKS NONE

[COORDINATES]
J1 0 0
J2 5 0
OUT 15 0
"""
    open(path, 'w').write(inp)

def events(sc, d_reuse, rng):
    floors, occ_pf = SC[sc]; N = floors*occ_pf
    p = PATTERN/PATTERN.sum()
    q = np.zeros(T+400)
    starts_flush = []
    # toilet flushes: 50 L/cap/d, 6 L each
    nf = int(round(N*50/6.0))
    hrs = rng.choice(24, size=nf, p=p); t0 = hrs*3600 + rng.integers(0,3600,size=nf)
    for t in t0:
        q[t:t+4] += 1.4; q[t+4] += 0.56
        starts_flush.append(int(t))
    # greywater: showers 52 L/cap/d (400 s @0.13 L/s); short draws 78 L/cap/d (60 s @0.1 L/s = 6 L)
    for vol, dur, qq in ((52.0*N, 400, 0.13), (78.0*N, 60, 0.10)):
        ne = int(round(vol/(dur*qq)))
        keep = rng.random(ne) >= d_reuse           # reuse diverts a fraction d of GW events upstream
        hrs = rng.choice(24, size=ne, p=p); t0 = hrs*3600 + rng.integers(0,3600,size=ne)
        for t in t0[keep]:
            q[t:t+dur] += qq
    return q[:T], np.array(sorted(starts_flush))

def run(sc, d_reuse, slope, seed=1, tmp='tmp.inp'):
    rng = np.random.default_rng(seed)
    q, fl = events(sc, d_reuse, rng)
    write_inp(tmp, slope)
    v = np.zeros(2*T)
    with Simulation(tmp) as sim:
        j1 = Nodes(sim)['J1']; c2 = Links(sim)['C2']
        k = 0
        for step in sim:
            if k >= 2*T: break
            j1.generated_inflow(float(q[k//2]))
            v[k] = vel(c2.flow, c2.depth)
            k += 1
    # per-flush peak velocity at main drain within 60 s window
    vp = np.array([v[2*t:2*t+120].max() for t in fl if t+60 < T])
    return dict(vmax=v.max(), v_med_flush=np.median(vp), v_min_flush=vp.min(),
                pct_flush_ge06=100*np.mean(vp >= 0.6), pct_flush_ge05=100*np.mean(vp >= 0.5),
                n_flush=len(vp), qpeak=q.max())

if __name__ == '__main__':
    # sanity check: steady 1.4 L/s should give ~0.69 m/s (Manning) in 150 mm @2%
    write_inp('chk.inp', 0.02)
    with Simulation('chk.inp') as sim:
        j1 = Nodes(sim)['J1']; c2 = Links(sim)['C2']; vv = 0
        for k, step in enumerate(sim):
            if k > 1500: break
            j1.generated_inflow(1.4)
            if k > 1200: vv = vel(c2.flow, c2.depth)
    print(f"steady-state check, 1.4 L/s, 150 mm @2%: V = {vv:.2f} m/s (Manning: 0.69)")
