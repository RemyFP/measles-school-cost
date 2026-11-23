"""
Compare the ABM with the dashboard SEIR (measles_efficiency.compute_new_infections)
on outbreak statistics and wall-clock time.

Run from this folder:  ../.venv/bin/python benchmark.py
"""
import sys
import time
import numpy as np
from randomgen import PCG64

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "dashboard"))
import measles_efficiency as me  # noqa: E402

from measles_abm import run_abm  # noqa: E402

N_REPS = 1000
THRESH = 10


def summarize(unvax, vax, I0=1):
    tot = unvax + vax
    big = tot >= THRESH
    s = {"P(>=10)": big.mean(), "mean": tot.mean()}
    if big.any():
        lo, med, hi = np.percentile(tot[big] + I0, [2.5, 50, 97.5])
        s.update({"lo": lo, "med": med, "hi": hi})
    else:
        s.update({"lo": np.nan, "med": np.nan, "hi": np.nan})
    return s


def fmt(name, s, secs):
    return (f"{name:<42} P(>=10)={s['P(>=10)']:.3f}  mean={s['mean']:6.1f}  "
            f"cond. 95% range={s['lo']:5.0f}-{s['hi']:5.0f} (median {s['med']:4.0f})  "
            f"time={secs:6.3f}s")


def run_seir(vax_prop, N=500, R0=15.0):
    params = {**me.DEFAULT_MSP_EFFICIENCY_PARAMS, "vax_prop": vax_prop,
              "population": N, "R0": R0}
    sampler = me.build_transition_sampler(np.random.Generator(PCG64(seed=147125098488)))
    t = time.perf_counter()
    u, v = me.compute_new_infections(params, sampler, N_REPS)
    return summarize(u, v), time.perf_counter() - t


def run_abm_timed(params, n_reps=N_REPS):
    t = time.perf_counter()
    r = run_abm(params, n_reps)
    return summarize(r["new_unvax"], r["new_vax"]), time.perf_counter() - t, r


def calibrate_beta(base_params, target_mean, lo=0.5, hi=4.0, iters=12):
    """Bisection on beta_multiplier so the ABM mean outbreak size matches."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        r = run_abm({**base_params, "beta_multiplier": mid}, 2000, seed=99)
        m = (r["new_unvax"] + r["new_vax"]).mean()
        lo, hi = (mid, hi) if m < target_mean else (lo, mid)
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    run_abm({}, 2)  # JIT warm-up (cached on disk after first run)
    run_abm({}, 2, parallel=True)

    STRUCTURED = {"mixing_weights": (0.5, 0.2, 0.3)}  # class / grade / school

    for vax_prop in (0.85, 0.90, 0.95):
        print(f"\n===== vaccination {vax_prop:.0%}, N=500, R0=15, {N_REPS} reps =====")
        s, t = run_seir(vax_prop)
        print(fmt("Dashboard SEIR (measles_efficiency)", s, t))
        s, t, _ = run_abm_timed({"vax_prop": vax_prop})
        print(fmt("ABM well-mixed, exponential periods", s, t))
        s, t, _ = run_abm_timed({"vax_prop": vax_prop, **STRUCTURED})
        print(fmt("ABM class/grade/school, beta x1", s, t))

    # Calibrate structured ABM to SEIR mean at 85%
    seir85, _ = run_seir(0.85)
    m = calibrate_beta({"vax_prop": 0.85, **STRUCTURED}, seir85["mean"])
    print(f"\nCalibrated beta multiplier for structured mixing at 85%: {m:.3f}")
    CAL = {**STRUCTURED, "beta_multiplier": m}
    for vax_prop in (0.85, 0.90, 0.95):
        s, t, _ = run_abm_timed({"vax_prop": vax_prop, **CAL})
        print(fmt(f"ABM structured calibrated, vax {vax_prop:.0%}", s, t))

    # Realistic natural history + interventions
    print("\n===== interventions (85%, structured+calibrated, gamma periods) =====")
    # pre-2026-11-20 intervention defaults (detection 4 days after onset, 90% isolation)
    REAL = {**CAL, "vax_prop": 0.85, "incubation_shape": 4.0, "infectious_shape": 4.0,
            "rash_fraction": 0.0, "detection_delay_days": 4.0, "isolation_effectiveness": 0.9, }
    scen = {
        "no intervention": {},
        "isolation at rash": {"isolation": True},
        "isolation + class quarantine": {"isolation": True, "quarantine": True,
                                         "quarantine_scope": 0},
        "isolation + school-wide quarantine": {"isolation": True, "quarantine": True,
                                               "quarantine_scope": 2},
    }
    for name, extra in scen.items():
        s, t, r = run_abm_timed({**REAL, **extra})
        print(fmt(name, s, t) + f"  quarantine student-days (mean)={r['quarantine_days'].mean():7.1f}")

    # Reproducibility: serial and parallel must give identical replicates
    a = run_abm({**REAL, "isolation": True, "quarantine": True}, 500)
    b = run_abm({**REAL, "isolation": True, "quarantine": True}, 500, parallel=True)
    same = all(np.array_equal(a[k], b[k]) for k in ("new_unvax", "new_vax", "quarantine_days"))
    print(f"\nSerial and parallel replicates identical: {same}")

    # Scaling with school size
    print("\n===== scaling with school size, 1000 reps (85%, structured, isolation + class quarantine) =====")
    for N in (250, 500, 1000, 2000, 4000):
        s, t_seir = run_seir(0.85, N=N)
        prm = {**REAL, "population": N, "isolation": True, "quarantine": True}
        t = time.perf_counter(); run_abm(prm, N_REPS); t_ser = time.perf_counter() - t
        t = time.perf_counter(); run_abm(prm, N_REPS, parallel=True); t_par = time.perf_counter() - t
        print(f"N={N:5d}  SEIR {t_seir:6.3f}s   ABM serial {t_ser:6.3f}s   ABM parallel {t_par:6.3f}s")
