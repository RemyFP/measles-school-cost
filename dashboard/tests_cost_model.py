import numpy as np
import pytest
from randomgen import PCG64

import cost_model as cm
import measles_efficiency


def test_config_loads_and_defaults_within_bounds():
    config = cm.load_cost_config()
    for key, spec in config["parameters"].items():
        assert spec["min"] <= spec["value"] <= spec["max"], key


@pytest.mark.parametrize("baseline, expected", [
    (85, [90.0, 95.0]),
    (70, [80.0, 90.0]),
    (100, [100.0, 100.0]),
    (0, [33.3, 66.7]),
])
def test_counterfactual_rates(baseline, expected):
    assert cm.counterfactual_vax_rates(baseline, 2) == expected


def test_zero_cases_zero_cost_when_initial_cases_excluded():
    out = cm.compute_outbreak_costs([0, 0], [0, 0], initial_infected=1,
                                    include_initial_cases=False)
    assert np.all(out["cost_total"] == 0)


def test_costs_linear_in_cases():
    params = {**cm.COST_DEFAULTS, "public_health_cost_per_case": 1000.0}
    one = cm.compute_outbreak_costs([1], [0], 0, params, include_initial_cases=False)
    ten = cm.compute_outbreak_costs([10], [0], 0, params, include_initial_cases=False)
    for key in ["cost_total", "hospitalizations", "school_days_missed"]:
        assert ten[key][0] == pytest.approx(10 * one[key][0])


def test_hand_computed_example():
    params = {
        "isolation_days_per_case": 7, "caregiver_work_loss_prob": 0.5,
        "caregiver_daily_wage": 100,
        "outpatient_visits_per_case": 2, "outpatient_visit_cost": 50,
        "hosp_prob_unvaccinated": 0.1, "hosp_prob_vaccinated": 0.0,
        "hospitalization_cost": 10000, "public_health_cost_per_case": 0,
    }
    # 9 new unvaccinated + 1 initial = 10 unvaccinated cases, 2 breakthrough
    out = cm.compute_outbreak_costs([9], [2], initial_infected=1, cost_params=params,
                                    include_initial_cases=True)
    assert out["cases"][0] == 12
    # 7 calendar days = 5 school days
    assert out["school_days_missed"][0] == pytest.approx(60)
    assert out["cost_caregiver_isolation"][0] == pytest.approx(12 * 5 * 0.5 * 100)
    assert out["cost_outpatient"][0] == 12 * 2 * 50
    assert out["hospitalizations"][0] == pytest.approx(1.0)
    assert out["cost_total"][0] == pytest.approx(3000 + 1200 + 10000)


def test_summary_expected_values_and_ranges():
    s = cm.summarize_scenario([0, 0, 20, 40], [0, 0, 0, 0], initial_infected=1, threshold=10)
    assert s["prob_outbreak"] == 0.5
    assert s["mean_cases"] == pytest.approx(np.mean([1, 1, 21, 41]))
    lb, ub = s["cases_range_if_outbreak"]
    assert 21 <= lb <= ub <= 41
    s_none = cm.summarize_scenario([0, 1], [0, 0], initial_infected=1, threshold=10)
    assert s_none["cases_range_if_outbreak"] is None


def test_expected_cost_decreases_with_vaccination():
    """With common random numbers, higher coverage -> fewer expected cases and lower cost."""
    params = {**measles_efficiency.DEFAULT_MSP_EFFICIENCY_PARAMS}
    costs = []
    for rate in [85.0] + cm.counterfactual_vax_rates(85.0, 2):
        params["vax_prop"] = rate / 100
        sampler = measles_efficiency.build_transition_sampler(np.random.Generator(PCG64(seed=1)))
        unvax, vax = measles_efficiency.compute_new_infections(params, sampler, 300)
        costs.append(cm.summarize_scenario(unvax, vax, 1, 10)["mean_cost_total"])
    assert costs[0] > costs[1] > costs[2]


def test_no_quarantine_costs_without_simulated_quarantine():
    """Compartmental engines pass no quarantine info -> no quarantine costs."""
    out = cm.compute_outbreak_costs([20], [1], 1)
    assert out["students_quarantined"][0] == 0
    assert out["cost_caregiver_quarantine"][0] == 0


def test_simulated_quarantine_used_when_available():
    out = cm.compute_outbreak_costs([3], [0], 1, quarantine={"quarantine_days": [70.0],
                                                             "n_quarantined": [4]})
    assert out["students_quarantined"][0] == 4
    assert out["quarantine_school_days_missed"][0] == pytest.approx(50.0)
    assert out["total_school_days_missed"][0] == pytest.approx(
        out["school_days_missed"][0] + 50.0)


def test_summary_has_ranges_over_all_simulations():
    s = cm.summarize_scenario(list(range(100)), [0] * 100, initial_infected=1, threshold=10)
    lb, ub = s["range_cases"]
    assert 1 <= lb < s["mean_cases"] < ub <= 100


def test_format_money_negative():
    assert cm.format_money(-76334) == "−" + cm.format_money(76334)
