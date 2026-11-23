import copy

import numpy as np
import pytest

import measles_single_population as msp
from model_engines import SERVER_ENGINES, run_server_engine, ABM_SETTINGS_DEFAULT, abm_params


@pytest.mark.parametrize("engine", SERVER_ENGINES)
def test_server_engines_share_output_structure(engine):
    params = copy.deepcopy(msp.DEFAULT_MSP_PARAMS)
    res = run_server_engine(engine, params, [85.0, 90.0, 95.0], ABM_SETTINGS_DEFAULT,
                            n_reps=200, n_curves=20, seed=1)
    rate_scen = [s for s in res["scenarios"] if s["role"] == "rate"]
    assert [s["vax_rate_pct"] for s in rate_scen] == [85.0, 90.0, 95.0]
    for s in res["scenarios"]:
        assert len(s["new_unvax"]) == len(s["new_vax"]) == 200
    curves = np.asarray(res["curves"])
    assert curves.shape == (20, params["sim_duration_days"] + 1)
    assert len(res["curve_new_total"]) == 20
    assert len(res["timing"]["scenario_ms"]) == len(res["scenarios"])
    for s in res["scenarios"]:
        first, last = np.asarray(s["first_detect_day"]), np.asarray(s["last_detect_day"])
        assert len(first) == 200 and np.all(first >= 0) and np.all(last >= first)


def test_json_ints_do_not_break_engines():
    """Parameters that went through the browser come back as ints (15.0 -> 15)."""
    params = copy.deepcopy(msp.DEFAULT_MSP_PARAMS)
    params.update(R0=15, infectious_period=5, incubation_period=10)
    for engine in ("seir_numba", "abm_numba"):
        run_server_engine(engine, params, [85], ABM_SETTINGS_DEFAULT, 10, 2, 1)


def test_well_mixed_abm_matches_compartmental():
    params = copy.deepcopy(msp.DEFAULT_MSP_PARAMS)
    probs = {}
    for engine in ("seir_numba", "abm_numba_par"):
        res = run_server_engine(engine, params, [85.0],
                                {**ABM_SETTINGS_DEFAULT, "structure": "well_mixed",
                                 "periods": "exponential"}, 4000, 0, 3)
        s = res["scenarios"][0]
        probs[engine] = np.mean(np.add(s["new_unvax"], s["new_vax"]) >= 10)
    # SE of each estimate ~0.008
    assert abs(probs["seir_numba"] - probs["abm_numba_par"]) < 0.04


def test_parallel_engine_survives_concurrent_requests():
    """Two overlapping requests used to abort the process (Numba workqueue)."""
    import threading
    params = copy.deepcopy(msp.DEFAULT_MSP_PARAMS)
    errors = []

    def run():
        try:
            run_server_engine("abm_numba_par", params, [65.0, 75.0], ABM_SETTINGS_DEFAULT,
                              500, 5, 1)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=run) for _ in range(3)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors


def test_changing_one_rate_only_reruns_that_scenario():
    params = copy.deepcopy(msp.DEFAULT_MSP_PARAMS)
    a = run_server_engine("seir_numba", params, [85.0, 90.0, 95.0], ABM_SETTINGS_DEFAULT, 300, 5, 4)
    b = run_server_engine("seir_numba", params, [85.0, 92.0, 95.0], ABM_SETTINGS_DEFAULT, 300, 5, 4)
    assert b["timing"]["reused"] == [True, False, True]
    assert b["scenarios"][2] == a["scenarios"][2]
    c = run_server_engine("seir_numba", params, [85.0, 92.0, 95.0], ABM_SETTINGS_DEFAULT, 300, 5, 4,
                          rerun=1)
    assert c["timing"]["reused"] == [False, False, False]
    assert c["scenarios"] == b["scenarios"]   # same seeds -> same results


def test_natural_history_periods():
    """Truncated incubation / infectious periods, isolation 1 day after rash."""
    import sys
    from measles_abm import run_abm
    params = copy.deepcopy(msp.DEFAULT_MSP_PARAMS)
    p = abm_params(params, 85.0, {**ABM_SETTINGS_DEFAULT, "periods": "natural_history", "isolation": True})
    assert p["infectious_period"] == 8.0 and p["beta_days"] == 5.0 and p["isolation"]
    r = run_abm({**p, "population": 50, "vax_prop": 0.0, "R0": 0.0}, 2000, seed=1)
    # with R0 = 0 the index case is detected at day round(D/2 * 4)/4 + 1, D in [6, 10]
    first = r["first_detect_day"]
    assert np.all((first >= 3.0) & (first <= 6.0))
    assert abs(first.mean() - 5.0) < 0.1   # quarter-day steps: rash rounds up by ~0.06 d
    # exponential periods: isolation does not apply
    pe = abm_params(params, 85.0, {**ABM_SETTINGS_DEFAULT, "periods": "exponential", "isolation": True})
    assert not pe["isolation"]


def test_abm_without_quarantine_has_zero_quarantine_cost():
    import cost_model as cm
    params = copy.deepcopy(msp.DEFAULT_MSP_PARAMS)
    res = run_server_engine("abm_numba", params, [85.0], {**ABM_SETTINGS_DEFAULT, "quarantine": "none"},
                            300, 0, 2)
    s = res["scenarios"][0]
    assert s["quarantine_days"] is not None and not any(s["quarantine_days"])
    out = cm.compute_outbreak_costs(s["new_unvax"], s["new_vax"], 1, quarantine={
        "quarantine_days": s["quarantine_days"], "n_quarantined": s["n_quarantined"]})
    assert out["cost_caregiver_quarantine"].sum() == 0
    res_q = run_server_engine("abm_numba", params, [85.0], {**ABM_SETTINGS_DEFAULT, "quarantine": "class"},
                              300, 0, 2)
    assert sum(res_q["scenarios"][0]["quarantine_days"]) > 0


def test_timing_total_includes_reused_scenarios():
    from app_engines import timing_table
    params = copy.deepcopy(msp.DEFAULT_MSP_PARAMS)
    run_server_engine("seir_numba", params, [85.0, 90.0], ABM_SETTINGS_DEFAULT, 200, 2, 9)
    res = run_server_engine("seir_numba", params, [85.0, 90.0], ABM_SETTINGS_DEFAULT, 200, 2, 9)
    assert res["timing"]["reused"] == [True, True]
    text = str(timing_table(res))
    total = sum(res["timing"]["scenario_ms"])
    assert f"{total:,.0f} ms" in text and "Simulated in this update" in text


def test_abm_adds_other_quarantine_approaches_at_current_rate():
    params = copy.deepcopy(msp.DEFAULT_MSP_PARAMS)
    res = run_server_engine("abm_numba_par", params, [85.0, 90.0], ABM_SETTINGS_DEFAULT, 300, 5, 3)
    q_scen = [s for s in res["scenarios"] if s["role"] == "quarantine"]
    assert [s["quarantine"] for s in q_scen] == ["class", "grade", "school"]
    assert all(s["vax_rate_pct"] == 85.0 for s in q_scen)
    assert res["scenarios"][0]["quarantine"] == "none"
    # quarantine reduces outbreak size and has quarantine days; none has none
    assert sum(res["scenarios"][0]["quarantine_days"]) == 0
    assert all(sum(s["quarantine_days"]) > 0 for s in q_scen)
    # switching the selected quarantine reuses every scenario at the current rate
    res2 = run_server_engine("abm_numba_par", params, [85.0, 90.0],
                             {**ABM_SETTINGS_DEFAULT, "quarantine": "class"}, 300, 5, 3)
    assert res2["timing"]["reused"][0] is True
    assert res2["scenarios"][0]["new_unvax"] == q_scen[0]["new_unvax"]
    assert [s["quarantine"] for s in res2["scenarios"] if s["role"] == "quarantine"] == \
        ["none", "grade", "school"]
    # compartmental engines: no quarantine alternatives
    seir = run_server_engine("seir_numba", params, [85.0, 90.0], ABM_SETTINGS_DEFAULT, 100, 5, 3)
    assert [s["role"] for s in seir["scenarios"]] == ["rate", "rate"]
