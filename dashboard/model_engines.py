"""
Simulation engines for the test dashboard.

Every engine takes the same inputs (dashboard parameters, list of vaccination
rates, ABM settings) and returns the same structure, so the rest of the app
(figures, statistics, costs) does not know which model produced the numbers:

    {"engine": key,
     "scenarios": [{"vax_rate_pct", "quarantine", "role", "new_unvax", "new_vax",
                    "first_detect_day", "last_detect_day",      # quarantine costs
                    "quarantine_days", "n_quarantined"}, ...],  # None unless simulated
     # role "rate": the vaccination scenarios with the selected quarantine
     # (current rate first); role "quarantine" (ABM only): the current rate
     # with each of the other quarantine approaches, for the cost table.
     "curves": [[E+I per day] x n_curves],     # baseline scenario only
     "curve_new_total": [new infections of each curve simulation],
     "timing": {"total_ms", "curves_ms", "scenario_ms": [...], "threads", "where"}}

Scenarios are cached individually: changing one counterfactual vaccination
rate only simulates that scenario (the others are identical, because each
scenario uses the same seed whatever the other rates are).

Browser engines (JavaScript) produce the same structure in
assets/measles_models.js + assets/engine_callbacks.js.
"""
import contextlib
import copy
import json
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from randomgen import PCG64

import measles_efficiency
import measles_single_population as msp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "abm"))
import numba  # noqa: E402
from measles_abm import run_abm  # noqa: E402
from seir_numba import seir_simulate  # noqa: E402

_PARALLEL_LOCK = threading.Lock()

# key -> (label, model, where)
ENGINES = {
    "seir_py":          ("Compartmental · Python/NumPy (server) — current dashboard", "seir", "server"),
    "seir_numba":       ("Compartmental · Python/Numba (server)", "seir", "server"),
    "seir_js":          ("Compartmental · JavaScript (browser)", "seir", "browser"),
    "abm_numba":        ("ABM · Python/Numba, 1 thread (server)", "abm", "server"),
    "abm_numba_par":    ("ABM · Python/Numba, all cores (server)", "abm", "server"),
    "abm_js":           ("ABM · JavaScript, 1 thread (browser)", "abm", "browser"),
    "abm_js_workers":   ("ABM · JavaScript, Web Workers (browser)", "abm", "browser"),
}
SERVER_ENGINES = [k for k, v in ENGINES.items() if v[2] == "server"]
DEFAULT_ENGINE = "abm_numba_par"

# ABM settings shown in the testing panel
# ---------------------------------------
# School structure. The structured school spreads less at the same beta
# (infections saturate within classrooms), so beta is multiplied by a factor
# calibrated (8000 simulations, abm/benchmark.calibrate_beta) so that the mean
# outbreak size matches the well-mixed school at 85% coverage. The factor
# depends on the period distributions.
ABM_STRUCTURES = {
    "well_mixed": {"mixing_weights": (0.0, 0.0, 1.0)},
    "school": {"mixing_weights": (0.5, 0.2, 0.3)},
}
ABM_SCHOOL_BETA_MULTIPLIER = {"exponential": 1.22, "natural_history": 1.20}

# Period distributions.
# - exponential: the dashboard SEIR. The 5-day infectious period is the time
#   infectious at school, i.e. it already ends with isolation; a case is
#   "detected" (contacts quarantined) when it ends. The isolation option does
#   not apply.
# - natural_history: incubation (infection -> prodromal symptoms = start of
#   infectiousness) gamma, mean = dashboard incubation period (10.5 d),
#   truncated to [5, 18] d; infectious 8 d on average (truncated to [6, 10]),
#   rash at the mid-point, detection and isolation 1 day after rash onset.
#   R0 keeps the dashboard definition: beta = R0 / dashboard infectious period
#   (5 d = 4 d before rash + 1 d), so with isolation this matches the SEIR's
#   R0 = 15; without isolation each case transmits for 8 days (R = 24).
#   Gamma shapes: 98% of the untruncated distribution inside the bounds
#   (incubation SD 3.0 d, infectious SD 1.0 d).
ABM_PERIODS = {
    "exponential": {"incubation_shape": 1.0, "infectious_shape": 1.0,
                    "rash_fraction": 1.0, "detection_delay_days": 0.0},
    "natural_history": {"incubation_shape": 12.0, "incubation_min_days": 5.0,
                        "incubation_max_days": 18.0,
                        "infectious_period": 8.0, "infectious_shape": 64.0,
                        "infectious_min_days": 6.0, "infectious_max_days": 10.0,
                        "rash_fraction": 0.5, "detection_delay_days": 1.0,
                        "isolation_effectiveness": 1.0},
}
ABM_QUARANTINE = {"none": None, "class": 0, "grade": 1, "school": 2}
QUARANTINE_LABELS = {"none": "No quarantine", "class": "Classmates quarantined",
                     "grade": "Grade quarantined", "school": "Whole school quarantined"}
ABM_SETTINGS_DEFAULT = {"structure": "school", "periods": "natural_history",
                        "isolation": True, "quarantine": "none"}

SCENARIO_CACHE_SIZE = 64


def _typed(params_dict: Mapping) -> dict:
    """Scalar model parameters with fixed types. Values that went through the
    browser come back as JSON numbers (15.0 -> 15), and Numba compiles a new
    specialization for each argument-type combination (seconds)."""
    return {
        "R0": float(params_dict["R0"]),
        "incubation_period": float(params_dict["incubation_period"]),
        "infectious_period": float(params_dict["infectious_period"]),
        "relative_infectiousness_vaccinated": float(params_dict["relative_infectiousness_vaccinated"]),
        "vaccine_efficacy": float(params_dict["vaccine_efficacy"]),
        "population": int(params_dict["population"][0]),
        "I0": int(params_dict["I0"][0]),
        "sim_duration_days": int(params_dict["sim_duration_days"]),
        "time_step_days": float(params_dict["time_step_days"]),
    }


def abm_params(params_dict: Mapping, vax_rate_pct: float, abm_settings: dict | None) -> dict:
    """Dashboard parameters + testing-panel ABM settings -> measles_abm params.
    (Mirrored in assets/engine_callbacks.js for the browser engines.)"""
    s = {**ABM_SETTINGS_DEFAULT, **(abm_settings or {})}
    q = ABM_QUARANTINE[s["quarantine"]]
    p = _typed(params_dict)
    periods = ABM_PERIODS[s["periods"]]
    out = {
        **p,
        "vax_prop": vax_rate_pct / 100,
        **ABM_STRUCTURES[s["structure"]],
        "beta_multiplier": ABM_SCHOOL_BETA_MULTIPLIER[s["periods"]]
        if s["structure"] == "school" else 1.0,
        "beta_days": p["infectious_period"],
        **periods,
        "isolation": bool(s["isolation"]) and s["periods"] != "exponential",
        "quarantine": q is not None,
        "quarantine_scope": 0 if q is None else q,
    }
    # dashboard incubation period outside the truncation bounds: widen them
    if "incubation_min_days" in out:
        out["incubation_min_days"] = min(out["incubation_min_days"], 0.5 * p["incubation_period"])
        out["incubation_max_days"] = max(out["incubation_max_days"], 1.5 * p["incubation_period"])
    return out


def _ms(t0: float) -> float:
    return round(1000 * (time.perf_counter() - t0), 1)


def _list(x):
    return None if x is None else np.asarray(x).tolist()


def _run_scenario(engine: str, params_dict: Mapping, rate: float, abm_settings: dict | None,
                  n_reps: int, n_curves: int, seed: int) -> dict:
    """One vaccination scenario (+ outbreak curves if n_curves > 0)."""
    curves = curve_new_total = curves_ms = None
    q_days = n_q = None

    if engine == "seir_py":
        if n_curves:
            # Exactly what the production dashboard does: curves from a
            # separate 20-simulation run of the object-oriented model.
            t0 = time.perf_counter()
            graph_params: Any = dict(copy.deepcopy(params_dict))
            graph_params["simulation_seed"] = seed
            exp = msp.DashboardExperiment(graph_params, n_curves)
            curves = exp.ma7_num_infected.data  # already a 7-day moving average
            curve_new_total = exp.total_new_cases.data
            curves_ms = _ms(t0)
        t0 = time.perf_counter()
        p: Any = dict(copy.deepcopy(params_dict))
        p["vax_prop"] = [rate / 100]
        sampler = measles_efficiency.build_transition_sampler(
            np.random.Generator(PCG64(seed=seed)))
        u, v, first, last = measles_efficiency.compute_new_infections(
            p, sampler, n_reps, return_detection_days=True)
        ms = _ms(t0)
        curves_are_ma7 = True

    elif engine == "seir_numba":
        p = _typed(params_dict)
        t0 = time.perf_counter()
        u, v, c, first, last = seir_simulate(
            n_reps, seed % 2**32, p["population"], p["I0"], rate / 100, p["R0"],
            p["incubation_period"], p["infectious_period"],
            p["relative_infectiousness_vaccinated"], p["vaccine_efficacy"],
            p["time_step_days"], p["sim_duration_days"], n_curves)
        ms = _ms(t0)
        if n_curves:
            curves, curve_new_total = c, (u + v)[:n_curves]
        curves_are_ma7 = False

    elif engine in ("abm_numba", "abm_numba_par"):
        parallel = engine == "abm_numba_par"
        prm = abm_params(params_dict, rate, abm_settings)
        # Numba's default thread pool ("workqueue") aborts the whole process
        # if two threads launch parallel kernels at once, which happens when
        # the server handles overlapping requests. Run them one at a time;
        # the timer starts after any wait so the timing is compute only.
        with _PARALLEL_LOCK if parallel else contextlib.nullcontext():
            t0 = time.perf_counter()
            r = run_abm(prm, n_reps, seed=seed, n_curve_reps=n_curves, parallel=parallel)
            ms = _ms(t0)
        u, v = r["new_unvax"], r["new_vax"]
        first, last = r["first_detect_day"], r["last_detect_day"]
        # always "simulated": zeros without quarantine -> no quarantine cost
        q_days, n_q = r["quarantine_days"], r["n_quarantined"]
        if n_curves:
            curves = r["curves"]
            curves[:, 0] = params_dict["I0"][0]
            curve_new_total = (u + v)[:n_curves]
        curves_are_ma7 = False

    else:
        raise ValueError(f"not a server engine: {engine}")

    return {
        "scenario": {"vax_rate_pct": rate, "new_unvax": _list(u), "new_vax": _list(v),
                     "first_detect_day": _list(first), "last_detect_day": _list(last),
                     "quarantine_days": _list(q_days), "n_quarantined": _list(n_q)},
        "curves": _list(curves), "curve_new_total": _list(curve_new_total),
        "curves_are_ma7": curves_are_ma7, "curves_ms": curves_ms, "ms": ms,
    }


_cache: OrderedDict = OrderedDict()
_cache_lock = threading.Lock()


def _cached_scenario(key_parts: list, compute):
    key = json.dumps(key_parts, sort_keys=True, default=str)
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key], True
    value = compute()
    with _cache_lock:
        _cache[key] = value
        while len(_cache) > SCENARIO_CACHE_SIZE:
            _cache.popitem(last=False)
    return value, False


def run_server_engine(engine: str, params_dict: Mapping, vax_rates_pct: list[float],
                      abm_settings: dict, n_reps: int, n_curves: int, seed: int,
                      rerun: int = 0, use_cache: bool = True) -> dict:
    """`rerun` is part of the cache key: the re-run button recomputes everything."""
    t_start = time.perf_counter()
    model = ENGINES[engine][1]
    settings = abm_settings if model == "abm" else None
    # (rate, settings, role): vaccination scenarios with the selected
    # quarantine, then (ABM) the current rate with the other quarantine options
    runs = [(rate, settings, "rate") for rate in vax_rates_pct]
    if model == "abm":
        runs += [(vax_rates_pct[0], {**settings, "quarantine": q}, "quarantine")
                 for q in ABM_QUARANTINE if q != settings["quarantine"]]
    results, reused = [], []
    for i, (rate, st, role) in enumerate(runs):
        # curves for every current-rate run, so switching the selected
        # quarantine reuses the cached scenarios
        nc = n_curves if i == 0 or role == "quarantine" else 0
        compute = lambda rate=rate, st=st, nc=nc: _run_scenario(  # noqa: E731
            engine, params_dict, rate, st, n_reps, nc, seed)
        if use_cache:
            r, hit = _cached_scenario(
                [engine, dict(params_dict), st, float(rate), n_reps, nc, seed, rerun], compute)
        else:
            r, hit = compute(), False
        results.append(r)
        reused.append(hit)
    scenarios = [{**r["scenario"], "role": role, "quarantine": st["quarantine"] if st else None}
                 for r, (_, st, role) in zip(results, runs)]

    base = results[0]
    return {
        "engine": engine,
        "scenarios": scenarios,
        "curves": base["curves"],
        "curves_are_ma7": base["curves_are_ma7"],
        "curve_new_total": base["curve_new_total"],
        "timing": {"total_ms": _ms(t_start), "curves_ms": base["curves_ms"],
                   "scenario_ms": [r["ms"] for r in results], "reused": reused,
                   "threads": numba.get_num_threads() if engine == "abm_numba_par" else 1,
                   "where": "server", "n_reps": n_reps},
    }


def warm_up():
    """Load / compile the Numba kernels at startup so timings exclude JIT."""
    t0 = time.perf_counter()
    params = {**msp.DEFAULT_MSP_PARAMS, "population": [100]}
    for engine in SERVER_ENGINES:
        if engine != "seir_py":
            for periods in ABM_PERIODS:
                run_server_engine(engine, params, [80.0], {**ABM_SETTINGS_DEFAULT, "periods": periods},
                                  2, 1, 1, use_cache=False)
    return time.perf_counter() - t0
