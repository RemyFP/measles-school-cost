"""Timing of ABM kernels vs school size, and parallel scaling analysis."""
import sys, time
import numpy as np
import numba
from measles_abm import run_abm
sys.path.insert(0, ".")
from seir_numba import seir_new_infections

PRM = {"mixing_weights": (0.5, 0.2, 0.3), "beta_multiplier": 1.22,
       "incubation_shape": 4.0, "infectious_shape": 4.0,
       "rash_fraction": 0.0, "detection_delay_days": 4.0, "isolation_effectiveness": 0.9, 
       "isolation": True, "quarantine": True, "vax_prop": 0.85}
REPS = 1000


def tm(fn, k=3):
    best = 1e9
    for _ in range(k):
        t = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t)
    return best


for alg in ("reference", "active", "group"):
    run_abm({}, 2, algorithm=alg); run_abm({}, 2, algorithm=alg, parallel=True)
seir_new_infections(2, 1, 500, 1, 0.85, 15., 10.5, 5., .05, .997)

print(f"1000 reps, 85% vax, structured mixing, isolation + class quarantine (best of 3, ms)")
print(f"{'N':>6} {'SEIR numba':>11} {'reference':>10} {'active':>8} {'group':>8} {'group ||':>9}")
for N in (250, 500, 1000, 2000, 4000):
    p = {**PRM, "population": N}
    row = [tm(lambda: seir_new_infections(REPS, 1, N, 1, 0.85, 15., 10.5, 5., .05, .997))]
    for alg in ("reference", "active", "group"):
        row.append(tm(lambda: run_abm(p, REPS, algorithm=alg), 1 if alg == "reference" else 3))
    row.append(tm(lambda: run_abm(p, REPS, algorithm="group", parallel=True)))
    print(f"{N:6d} " + " ".join(f"{1000*x:{w}.1f}" for x, w in zip(row, (11, 10, 8, 8, 9))))

print("\nParallel scaling (reference kernel, N=2000; M3 Pro = 5 performance + 6 efficiency cores)")
p = {**PRM, "population": 2000}
t1 = tm(lambda: run_abm(p, REPS, algorithm="reference"), 1)
for chunk in (0, 1):
    numba.set_parallel_chunksize(chunk)
    for nt in (1, 2, 5, 8, 11):
        numba.set_num_threads(nt)
        t = tm(lambda: run_abm(p, REPS, algorithm="reference", parallel=True), 2)
        print(f"  chunksize={'static' if chunk == 0 else 'dynamic(1)'} threads={nt:2d}: {t:6.3f}s  speedup {t1/t:4.1f}x")
numba.set_parallel_chunksize(0); numba.set_num_threads(11)

# per-rep time spread -> load imbalance
r = run_abm(p, REPS, algorithm="reference")
d = r["duration_days"]
print(f"\nsimulated duration per rep (days): median {np.median(d):.0f}, 90th pct {np.percentile(d,90):.0f}, max {d.max():.0f}")
