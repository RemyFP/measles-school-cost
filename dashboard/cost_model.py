"""
Outbreak cost model and counterfactual vaccination scenarios.

Two kinds of school absence are costed separately:
- isolation of cases: per case, `isolation_days_per_case` calendar days
  converted to school / work days (x school_days_per_week / 7);
- quarantine (exclusion) of exposed unvaccinated students: per student-day,
  only when the engine simulates it (ABM with a quarantine scope). The
  compartmental engines do not model quarantine, so they have no quarantine
  costs.
Parameter defaults live in `cost_parameters.toml`; this module contains no
hard-coded parameter values.

Costs are computed per simulation as expected values given the number of
cases (e.g. hospitalizations = cases x probability), so the spread across
simulations reflects uncertainty in outbreak size.
"""

import tomllib
from pathlib import Path

import numpy as np

COST_CONFIG_PATH = Path(__file__).with_name("cost_parameters.toml")


def load_cost_config(path: Path = COST_CONFIG_PATH) -> dict:
    with open(path, "rb") as f:
        config = tomllib.load(f)
    for key, spec in config["parameters"].items():
        missing = {"value", "min", "max", "step", "label", "group"} - spec.keys()
        if missing:
            raise ValueError(f"cost parameter '{key}' is missing fields {missing}")
        if not spec["min"] <= spec["value"] <= spec["max"]:
            raise ValueError(f"cost parameter '{key}': default outside [min, max]")
    return config


COST_CONFIG = load_cost_config()
COST_PARAM_SPECS: dict = COST_CONFIG["parameters"]
COST_DEFAULTS: dict = {k: spec["value"] for k, spec in COST_PARAM_SPECS.items()}
N_COUNTERFACTUALS: int = COST_CONFIG["counterfactual"]["n_scenarios"]
INCLUDE_INITIAL_CASES: bool = COST_CONFIG["general"]["include_initial_cases"]
SCHOOL_DAYS_PER_WEEK: float = COST_CONFIG["general"]["school_days_per_week"]
CURRENCY: str = COST_CONFIG["general"]["currency"]

# Display names of the cost components (keys of compute_outbreak_costs output)
COST_COMPONENTS = {
    "cost_caregiver_isolation": "Parent lost productivity (isolation)",
    "cost_caregiver_quarantine": "Parent lost productivity (quarantine)",
    "cost_outpatient": "Outpatient visits",
    "cost_hospital": "Hospitalizations",
    "cost_public_health": "Public health response",
}


def counterfactual_vax_rates(baseline_pct: float, n: int = N_COUNTERFACTUALS) -> list[float]:
    """Rates (in %) k/(n+1) of the way from the baseline to 100%, k = 1..n.
    85 -> [90, 95] for n = 2. Rounded to 0.1%."""
    baseline_pct = float(np.clip(baseline_pct, 0.0, 100.0))
    return [round(baseline_pct + k / (n + 1) * (100.0 - baseline_pct), 1)
            for k in range(1, n + 1)]


def compute_outbreak_costs(new_unvax_cases, new_vax_cases, initial_infected: int,
                           cost_params: dict | None = None,
                           include_initial_cases: bool = INCLUDE_INITIAL_CASES,
                           quarantine: dict | None = None) -> dict:
    """
    Per-simulation outcomes and costs.

    new_unvax_cases, new_vax_cases: arrays (one value per simulation) of new
        infections among unvaccinated / vaccinated students (initial infected
        students are unvaccinated and NOT included in these arrays).
    initial_infected: number of students initially infected.
    cost_params: dict overriding COST_DEFAULTS (missing keys use defaults).
    quarantine: per-simulation quarantine information, either
        {"quarantine_days": [...], "n_quarantined": [...]} (simulated, in
        calendar student-days). None -> no quarantine costs.

    Returns a dict of numpy arrays, one value per simulation.
    """
    p = {**COST_DEFAULTS, **(cost_params or {})}
    unvax = np.asarray(new_unvax_cases, dtype=float)
    vax = np.asarray(new_vax_cases, dtype=float)
    if include_initial_cases:
        unvax = unvax + initial_infected
    cases = unvax + vax

    school_frac = SCHOOL_DAYS_PER_WEEK / 7.0      # calendar days -> school / work days
    q = quarantine or {}
    if q.get("quarantine_days") is not None:
        n_q = np.asarray(q["n_quarantined"], dtype=float)
        q_student_days = np.asarray(q["quarantine_days"], dtype=float)
    else:
        n_q = q_student_days = np.zeros_like(cases)
    q_school_days = q_student_days * school_frac
    iso_school_days = cases * p["isolation_days_per_case"] * school_frac

    out = {
        "cases": cases,
        "cases_unvaccinated": unvax,
        "cases_vaccinated": vax,
        "school_days_missed": iso_school_days,
        "caregiver_work_days_lost": iso_school_days * p["caregiver_work_loss_prob"],
        "students_quarantined": n_q,
        "quarantine_school_days_missed": q_school_days,
        "quarantine_caregiver_work_days_lost": q_school_days
                                               * p["quarantine_caregiver_work_loss_prob"],
        "outpatient_visits": cases * p["outpatient_visits_per_case"],
        "hospitalizations": unvax * p["hosp_prob_unvaccinated"]
                            + vax * p["hosp_prob_vaccinated"],
    }
    # all students (cases in isolation + quarantined), school days only
    out["total_school_days_missed"] = out["school_days_missed"] + out["quarantine_school_days_missed"]
    out["cost_caregiver_isolation"] = out["caregiver_work_days_lost"] * p["caregiver_daily_wage"]
    out["cost_caregiver_quarantine"] = out["quarantine_caregiver_work_days_lost"] \
        * p["caregiver_daily_wage"]
    out["cost_outpatient"] = out["outpatient_visits"] * p["outpatient_visit_cost"]
    out["cost_hospital"] = out["hospitalizations"] * p["hospitalization_cost"]
    out["cost_public_health"] = cases * p["public_health_cost_per_case"]
    out["cost_total"] = sum(out[k] for k in COST_COMPONENTS)
    return out


def summarize_scenario(new_unvax_cases, new_vax_cases, initial_infected: int,
                       threshold: int, cost_params: dict | None = None,
                       lb_pct: float = 2.5, ub_pct: float = 97.5,
                       quarantine: dict | None = None) -> dict:
    """
    Summary statistics for one vaccination scenario.

    - "mean_<x>": expected value over ALL simulations (including introductions
      that do not lead to an outbreak) = expected burden per introduction.
    - "range_<x>": (lb, ub) percentiles over ALL simulations.
    - "prob_outbreak": P(new infections >= threshold), as in the dashboard.
    - "<x>_range_if_outbreak": (lb, ub) percentiles among simulations with
      >= threshold new infections, or None if there are none.
    """
    per_sim = compute_outbreak_costs(new_unvax_cases, new_vax_cases,
                                     initial_infected, cost_params, quarantine=quarantine)
    new_total = np.asarray(new_unvax_cases) + np.asarray(new_vax_cases)
    is_outbreak = new_total >= threshold

    summary: dict[str, float | tuple[float, float] | None] = \
        {"prob_outbreak": float(is_outbreak.mean())}
    for key, values in per_sim.items():
        summary[f"mean_{key}"] = float(values.mean())
        lb, ub = np.percentile(values, [lb_pct, ub_pct])
        summary[f"range_{key}"] = (float(lb), float(ub))
    for key in ("cases", "cost_total"):
        if is_outbreak.any():
            lb, ub = np.percentile(per_sim[key][is_outbreak], [lb_pct, ub_pct])
            summary[f"{key}_range_if_outbreak"] = (float(lb), float(ub))
        else:
            summary[f"{key}_range_if_outbreak"] = None
    return summary


def format_money(x: float, currency: str = CURRENCY) -> str:
    if x < 0:                       # e.g. cost averted < 0 (costs more than current)
        return "−" + format_money(-x, currency)
    if x >= 1e6:
        return f"{currency}{x / 1e6:,.2f}M"
    if x >= 1e4:
        return f"{currency}{x / 1e3:,.0f}K"
    if x >= 1e3:
        return f"{currency}{x / 1e3:,.1f}K"
    return f"{currency}{x:,.0f}"
