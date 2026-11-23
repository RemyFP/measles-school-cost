#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rfp437

>(^_^)> ~~~~
"""

# TODO: streamline the single population / multiple population stuff?
#   Some confusing stuff... e.g. remembering that some parameters are actually
#   a LIST (but of 1 element, for 1 subpopulation)

from dash import Dash, html, dcc, callback, Output, Input, State  # Patch
import plotly.express as px
import pandas as pd
import numpy as np
import copy
import dash_bootstrap_components as dbc
import measles_single_population as msp
import measles_efficiency
import subprocess

from randomgen import PCG64

from app_static_graphics import navbar, bottom_info_section, \
    bottom_credits, school_outbreak_projections_header
from app_dynamic_graphics import results_header, spaghetti_plot_section, \
    dashboard_input_panel, cost_results_section
from app_computation_functions import dashboard_results_fig, \
    EMPTY_SPAGHETTI_PLOT_INFECTED_MA, \
    dashboard_exceedance_prob_str,\
    dashboard_percentiles_str, \
    cost_results_table, cost_bar_fig, EMPTY_COST_BAR_FIG
from app_selectors import SELECTOR_DEFAULTS, counterfactual_rate_inputs
from cost_model import COST_PARAM_SPECS, COST_DEFAULTS, counterfactual_vax_rates, \
    summarize_scenario
import dash

# Model-engine testing panel (compartmental vs ABM, Python vs JavaScript).
# The simulation callbacks (formerly update_graph and
# run_counterfactual_simulations) live in app_engines.py.
from app_engines import engine_stores, engine_testing_accordion, timing_section, \
    quarantine_input
from model_engines import warm_up, QUARANTINE_LABELS

DASHBOARD_CONFIG = {
    'num_simulations_graph': 20,
    'num_simulations_results': 1000,
    'simulation_seed': 147125098488,
    'spaghetti_curve_selection_seed': 12345,
}
STATE_DATA_SUBFOLDER = 'state_data/'

msp.DEFAULT_MSP_PARAMS["simulation_seed"] = DASHBOARD_CONFIG["simulation_seed"]

# TODO -- again, this can be streamlined... but here is
#   the initial stab at generalizing the code to multiple states...
#   We can also cache the dataframe subsets...
# TODO -- maybe... put state-to-CSV mapping in JSON file?

state_name_to_two_letter_code = {
    "Connecticut": "CT",
    "Maryland": "MD",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "Pennsylvania": "PA",
    "Texas": "TX",
    "Alabama": "AL",
    "Arizona": "AZ", 
    "California": "CA", 
    "Colorado": "CO",
    "Iowa": "IA", 
    "Massachusetts": "MA",  
    "Michigan": "MI",
    "Minnesota": "MN", 
    "Oregon": "OR", 
    "Washington": "WA",
    "Idaho": "ID",
    "Indiana": "IN",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Wisconsin": "WI",
}

df_sort_grades = pd.DataFrame({
    "Grade": [
        "Kindergarten", "1st Grade", "2nd Grade", "3rd Grade", "4th Grade",
        "5th Grade", "6th Grade", "7th Grade", "8th Grade", "9th Grade",
        "10th Grade", "11th Grade", "12th Grade"],
    "grade_sort_order": list(range(13))
})


state_to_df_map = {
    state_name: pd.read_csv(STATE_DATA_SUBFOLDER + state_code + '_MMR_vax_rate.csv')
    for state_name, state_code in state_name_to_two_letter_code.items()
}

# Formatting of state dataframes
for state_name, df in state_to_df_map.items():
    df['County'] = df['County'].str.title()
    df['School District or Name'] = df['School District or Name'].str.title()
    df['School District or Name'] = df['School District or Name'].str\
        .replace('\r', ' ', regex=True)\
        .replace('\n', ' ', regex=True)
    df['School District or Name'] = df['School District or Name'].str.\
        replace('  ', ' ')
    df['School District or Name'] = df['School District or Name'].str\
        .replace('Isd', 'ISD')\
        .replace('Csd', 'CSD')

def get_county_subset_df(state_str, county_str) -> pd.DataFrame:
    state_subset_df = state_to_df_map[state_str]
    county_subset_df = state_subset_df[state_subset_df["County"] == county_str]

    return county_subset_df


# Generalizing state/county/school selection for different states
#   After state selection, county selection options should update
#   After county selection, school selection options should update


@callback(
    [Output('dashboard_params', 'data')],
    [State('dashboard_params', 'data'),
     Input('school_size_selector', 'value'),
     Input('vax_rate_selector', 'value'),
     Input('I0_selector', 'value'),
     Input('R0_selector', 'value'),
     Input('latent_period_selector', 'value'),
     Input('infectious_period_selector', 'value'),
     Input('threshold_selector', 'value'),
     Input('vaccine_efficacy_selector', 'value'),
     Input('vaccinated_infectiousness_selector', 'value')]
)
def create_params_from_selectors(params_dict,
                                 school_size,
                                 vax_rate_percent,
                                 I0,
                                 R0,
                                 latent_period,
                                 infectious_period,
                                 outbreak_threshold,
                                 vaccine_efficacy,
                                 vaccinated_infectiousness):
    school_size = school_size if school_size is not None else SELECTOR_DEFAULTS['school_size']
    vax_rate_percent = vax_rate_percent if vax_rate_percent is not None else SELECTOR_DEFAULTS['vax_rate']
    I0 = I0 if I0 is not None else SELECTOR_DEFAULTS['I0']
    R0 = R0 if R0 is not None else SELECTOR_DEFAULTS['R0']
    latent_period = latent_period if latent_period is not None else SELECTOR_DEFAULTS['latent_period']
    infectious_period = infectious_period if infectious_period is not None else SELECTOR_DEFAULTS[
        'infectious_period']
    outbreak_threshold = outbreak_threshold if outbreak_threshold is not None else SELECTOR_DEFAULTS[
        'outbreak_threshold']
    vaccine_efficacy = vaccine_efficacy if vaccine_efficacy is not None else SELECTOR_DEFAULTS[
        'vaccine_efficacy_selector']
    vaccinated_infectiousness = vaccinated_infectiousness if vaccinated_infectiousness is not None else \
        SELECTOR_DEFAULTS[
            'vaccinated_infectiousness_selector']

    params_dict['population'] = [int(school_size)]
    params_dict['vax_prop'] = [0.01 * float(vax_rate_percent)]
    params_dict['I0'] = [int(I0)]
    params_dict['R0'] = float(R0)
    params_dict['incubation_period'] = float(latent_period)
    params_dict['infectious_period'] = float(infectious_period)
    params_dict['threshold_values'] = [int(outbreak_threshold)]
    params_dict['vaccine_efficacy'] = float(0.01 * vaccine_efficacy)
    params_dict['relative_infectiousness_vaccinated'] = float(1 - 0.01 * vaccinated_infectiousness)

    # Bug I got stuck on for awhile -- dcc.State can certainly handle dictionaries
    # HOWEVER -- callbacks always expect the return type to be a list or tuple
    #   if there are multiple values -- so we wrap the dictionary in a list,
    #   but we do not have to modify anything else -- dash just knows how to
    #   parse this output :)
    return [params_dict]


@callback(
    [Output('inputs_are_valid', 'data'),
     Output('warning_str', 'children')],
    [Input('dashboard_params', 'data')]
)
def check_inputs_validity(params_dict: dict) -> str:
    """
    IMPORTANT: vax_proportion must be between [0,1] --

    TODO: fix inconsistencies with variable naming for vax percent --
        I can see this being a cause of a bug/misunderstanding in the future
        -- sometimes it's in percent form (so like an int, like 95)
        and sometimes it's in decimal form
    """
    # Assuming single population -- again, single population / multiple population stuff
    #   is confusing here -- and the hardcoding could accidentally lead to mistakes in future

    if not 0 <= params_dict["vax_prop"][0] <= 1:
        warning_str = "Invalid inputs: vaccination rate must be between 0-100%."
        return False, warning_str
    elif params_dict["I0"][0] < 0 or params_dict["population"][0] < 0:
        warning_str = "Invalid inputs: school enrollment and students initially " \
                      "infected must be positive whole numbers. Please adjust."
        return False, warning_str
    elif params_dict["I0"][0] > int((1 - params_dict["vax_prop"][0]) * params_dict["population"][0]):
        warning_str = "Invalid inputs: The number of initially infected students " \
                      "cannot exceed the number of unvaccinated students. Please adjust."
        return False, warning_str
    else:
        return True, ""


# Counterfactual vaccination scenarios and costs
################################################

COUNTERFACTUAL_INPUT_IDS = [inp.id for inp in counterfactual_rate_inputs]
COST_INPUT_IDS = {key: f"cost_{key}" for key in COST_PARAM_SPECS}


@callback(
    [Output(input_id, 'value') for input_id in COUNTERFACTUAL_INPUT_IDS],
    [Input('vax_rate_selector', 'value')],
)
def reset_counterfactual_rates(vax_rate_pct):
    """When the baseline vaccination rate changes, reset the counterfactual
    rates to 1/3, 2/3 ... of the way to 100% (users can then edit them)."""
    if vax_rate_pct is None or not 0 <= vax_rate_pct <= 100:
        return [dash.no_update] * len(COUNTERFACTUAL_INPUT_IDS)
    return counterfactual_vax_rates(vax_rate_pct, len(COUNTERFACTUAL_INPUT_IDS))


@callback(
    [Output('cost_table', 'children'),
     Output('cost_bar_chart', 'figure'),
     Output('cost_cols_rates', 'options'),
     Output('cost_cols_quarantine', 'options'),
     Output('cost_cols_quarantine_box', 'style')],
    [Input('baseline_sim_results', 'data'),
     Input('counterfactual_sim_results', 'data'),
     Input('cost_cols_rates', 'value'),
     Input('cost_cols_quarantine', 'value')] +
    [Input(input_id, 'value') for input_id in COST_INPUT_IDS.values()],
)
def update_cost_results(baseline, others, show_rates, show_quarantine, *cost_values):
    """Arithmetic only (no simulation): changing a cost parameter or the
    shown columns is instant.

    Columns: the current rate with the selected quarantine (always shown), the
    counterfactual vaccination rates (same quarantine), then (ABM engines) the
    current rate with each other quarantine approach."""
    if baseline is None or others is None:
        return "", EMPTY_COST_BAR_FIG, [], [], {"display": "none"}

    # Empty or out-of-range inputs fall back to the configured default
    cost_params = {
        key: COST_DEFAULTS[key] if value is None else float(value)
        for key, value in zip(COST_INPUT_IDS, cost_values)
    }
    I0 = baseline["I0"]

    def q_label(s):
        return QUARANTINE_LABELS[s["quarantine"]] if s.get("quarantine") \
            else "No quarantine (not modelled)"

    current = f"Current ({baseline['vax_rate_pct']:g}%)"
    rate_cols = [s for s in others if s.get("role", "rate") == "rate"]
    q_cols = [s for s in others if s.get("role") == "quarantine"]
    columns = [(baseline, f"{current}\n{q_label(baseline)}")]
    rate_options, q_options = [], []
    for k, s in enumerate(rate_cols, start=1):
        key = f"rate{k}"
        rate_options.append({"label": f" {s['vax_rate_pct']:g}%", "value": key})
        if show_rates is None or key in show_rates:
            columns.append((s, f"{s['vax_rate_pct']:g}% vaccinated\n{q_label(s)}"))
    for s in q_cols:
        key = f"q:{s['quarantine']}"
        q_options.append({"label": f" {QUARANTINE_LABELS[s['quarantine']]}", "value": key})
        if show_quarantine is None or key in show_quarantine:
            columns.append((s, f"{current}\n{q_label(s)}"))

    def quarantine_info(s):
        if s.get("quarantine_days") is not None:      # simulated (ABM)
            return {"quarantine_days": s["quarantine_days"], "n_quarantined": s["n_quarantined"]}
        return None                                   # compartmental: not modelled

    summaries = [
        summarize_scenario(s["new_unvax"], s["new_vax"], I0, baseline["threshold"],
                           cost_params, quarantine=quarantine_info(s))
        for s, _ in columns
    ]
    labels = [lbl for _, lbl in columns]
    if baseline.get("quarantine_days") is not None:
        note = ("Quarantine simulated by the agent-based model: unvaccinated contacts of a detected "
                "case (classmates, grade or whole school) stay home 21 days after each exposure, "
                "which also reduces transmission. “No quarantine” has no quarantine costs.")
    else:
        note = ("No quarantine costs: the compartmental model does not simulate quarantine. "
                "Use an ABM engine to compare quarantine options.")
    return cost_results_table(summaries, labels, baseline["threshold"], note), \
        cost_bar_fig(summaries, labels), rate_options, q_options, \
        {} if q_options else {"display": "none"}


@callback(
    [Output("county_selector", "options"),
     Output("county_selector", "value")],
    [Input("state_selector", "value")],
    prevent_initial_call=True
)
def update_county_selector(state):
    new_county_options = sorted(state_to_df_map[state]["County"].unique())
    default_county_displayed = new_county_options[0]

    return new_county_options, default_county_displayed


@callback(
    [Output('school_selector', 'options'),
     Output('school_selector', 'value')],
    [State('state_selector', 'value'),
     Input('county_selector', 'value')],
    prevent_initial_call=True
)
def update_school_selector(state, county):
    df = get_county_subset_df(state, county)
    df_sort = pd.merge(
        df, df_sort_grades,
        left_on='Age Group', right_on='Grade', how='left')
    df_sort.sort_values(
        by=['School District or Name', 'grade_sort_order'], 
        ascending=[True, True], inplace=True)
    new_school_options = [
        f"{name} ({age_group})" if pd.notna(age_group) and age_group != "" else f"{name}"
        for name, age_group in zip(df_sort["School District or Name"], df_sort["Age Group"])
    ]
    # new_school_options = sorted(
    #     f"{name} ({age_group})" if pd.notna(age_group) and age_group != "" else f"{name}"
    #     for name, age_group in zip(df["School District or Name"], df["Age Group"])
    # )
    default_school_displayed = new_school_options[0]

    return new_school_options, default_school_displayed


@callback(
    [Output('vax_rate_selector', 'value')],
    [State('state_selector', 'value'),
     State('county_selector', 'value'),
     Input('school_selector', 'value')],
    prevent_initial_call=True
)
def get_school_vax_rate(state_str,
                        county_str,
                        school_with_age_str) -> float:
    if school_with_age_str:

        county_subset_df = get_county_subset_df(state_str, county_str)

        # Handle cases where age_group is present or ""
        if ' (' in school_with_age_str:
            school, age_group = school_with_age_str.split(' (')
            age_group = age_group.rstrip(")")
        else:
            school = school_with_age_str
            age_group = ""

        # Filter DataFrame based on the presence or absence of age_group
        df_school = county_subset_df.loc[
            (county_subset_df['School District or Name'] == school) &
            ((county_subset_df['Age Group'] == age_group) | (
                    pd.isna(county_subset_df['Age Group']) & (age_group == "")))
            ]

        school_vax_rate_pct = df_school['MMR Vaccination Rate'].values[0]

        return [school_vax_rate_pct]

    else:

        return [SELECTOR_DEFAULTS["vax_rate"]]


result = subprocess.run("git symbolic-ref -q --short HEAD || git describe --tags --exact-match",
                        shell=True, capture_output=True)
version = result.stdout.decode("utf-8").strip() if result.stdout else "Unknown"

app = Dash(
    prevent_initial_callbacks='initial_duplicate')
server = app.server  # Do we need this?
app.title = f"epiENGAGE Measles Outbreak Simulator v-{version}"

# Google Analytics removed in this test copy (would report to the production property)

app.layout = dbc.Container(
    [
        dcc.Store(id="inputs_are_valid", data=True),
        dcc.Store(id="dashboard_params", data=copy.deepcopy(msp.DEFAULT_MSP_PARAMS)),
        dcc.Store(id="baseline_sim_results", data=None),
        dcc.Store(id="counterfactual_sim_results", data=None),
        *engine_stores(DASHBOARD_CONFIG["simulation_seed"]),

        dbc.Row([navbar], className="my-2"),
        html.Br(),
        html.Br(),

        # Main Layout with Left and Right Sections
        dbc.Row([
            # Left section
            html.Br(),

            dashboard_input_panel(testing_accordion=engine_testing_accordion(),
                                  quarantine_input=quarantine_input()),

            dbc.Col([
                timing_section(),

                html.Br(),

                school_outbreak_projections_header(),

                html.Br(),

                results_header(),

                html.Br(),

                spaghetti_plot_section(),

                # Costs and counterfactual vaccination scenarios, directly below
                # the curves (a separate row started below the tall input panel)
                cost_results_section()], className="col-xl-9")
        ]),

        html.Br(),

        bottom_info_section(),

        bottom_credits()
    ], fluid=True, style={"min-height": "100vh", "display": "flex", "flex-direction": "column"})

print(f"Numba kernels loaded/compiled in {warm_up():.1f} s")

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
