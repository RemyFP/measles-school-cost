"""
Model-engine testing panel: same interface, different simulation engines
(compartmental vs ABM, Python on the server vs JavaScript in the browser).

Data flow (all engines produce the same `sim_results` structure):

  inputs ──build_request (browser)──> sim_request ─┬─ run_server_simulations (Python) ─┐
                                                   └─ engines.run_browser (JavaScript) ─┴─> sim_results
  sim_results ──display_sim_results (Python)──> figure, statistics, cost stores, timing
  spaghetti figure ──engines.end_to_end (browser)──> end-to-end time
"""
import numpy as np
from dash import html, dcc, callback, clientside_callback, ClientsideFunction, \
    Output, Input, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

import measles_single_population as msp
from app_styles import BASE_FONT_FAMILY_STR, BASE_FONT_STYLE, SELECTOR_NOTE_STYLE, \
    SELECTOR_LABEL_STYLE
from app_computation_functions import dashboard_results_fig, \
    EMPTY_SPAGHETTI_PLOT_INFECTED_MA, dashboard_exceedance_prob_str, \
    dashboard_percentiles_str
from app_selectors import counterfactual_rate_inputs
from model_engines import ENGINES, DEFAULT_ENGINE, ABM_STRUCTURES, ABM_PERIODS, \
    ABM_SCHOOL_BETA_MULTIPLIER, ABM_QUARANTINE, ABM_SETTINGS_DEFAULT, QUARANTINE_LABELS, \
    run_server_engine

N_REPS_OPTIONS = [100, 500, 1000, 2000, 5000, 10000]
N_CURVES = 20

LABEL_STYLE = {"font-family": BASE_FONT_FAMILY_STR, "font-size": "13pt",
               "margin-bottom": "0.2em", "margin-top": "0.8em", "font-weight": "600"}
SUBLABEL_STYLE = {"font-family": BASE_FONT_FAMILY_STR, "font-size": "12pt", "textAlign": "left",
                  "margin-bottom": "0.2em", "margin-top": "0.7em", "font-weight": "600"}


def engine_config(seed: int) -> dict:
    """Static data passed to the browser (single source of truth: model_engines)."""
    return {
        "seed": seed,
        "n_curves": N_CURVES,
        "browser_engines": [k for k, v in ENGINES.items() if v[2] == "browser"],
        "engine_model": {k: v[1] for k, v in ENGINES.items()},
        "abm_structures": ABM_STRUCTURES,
        "abm_periods": ABM_PERIODS,
        "abm_school_beta": ABM_SCHOOL_BETA_MULTIPLIER,
        "abm_quarantine": ABM_QUARANTINE,
    }


def engine_stores(seed: int) -> list:
    return [
        dcc.Store(id="engine_config", data=engine_config(seed)),
        dcc.Store(id="sim_request", data=None),
        dcc.Store(id="sim_results", data=None),
    ]


def engine_testing_accordion() -> html.Div:
    d = ABM_SETTINGS_DEFAULT
    body = dbc.Col([
        html.Div("Model engine", style=LABEL_STYLE),
        dcc.Dropdown(id="engine_selector", value=DEFAULT_ENGINE, clearable=False,
                     options=[{"label": v[0], "value": k} for k, v in ENGINES.items()],
                     style={"textAlign": "left", "font-size": "11pt"}),
        html.Div("Simulations per scenario", style=LABEL_STYLE),
        dcc.Dropdown(id="n_reps_selector", value=1000, clearable=False,
                     options=N_REPS_OPTIONS, style={"textAlign": "left"}),
        html.Div("ABM settings (ignored by compartmental engines)", style=LABEL_STYLE),
        html.H3("Well-mixed school + exponential periods = same model as the compartmental one.",
                style={**SELECTOR_NOTE_STYLE, "margin-top": "0.2em"}),
        html.Div("1. School structure (choose one)", style={**SUBLABEL_STYLE}),
        dcc.RadioItems(id="abm_structure", value=d["structure"], inline=False,
                       options=[{"label": " Well-mixed school (anyone can infect anyone)",
                                 "value": "well_mixed"},
                                {"label": " Classes / grades / school: 50% of contacts in class, "
                                          "20% in grade, 30% school-wide (β calibrated)",
                                 "value": "school"}],
                       style={"textAlign": "left"}),
        html.Div("2. Incubation and infectious periods (choose one)", style={**SUBLABEL_STYLE}),
        dcc.RadioItems(id="abm_periods", value=d["periods"], inline=False,
                       options=[{"label": " Exponential periods (compartmental: 5 infectious days, "
                                          "isolation implicit)", "value": "exponential"},
                                {"label": " Measles natural history: incubation 10.5 d (5–18), "
                                          "infectious 8 d (6–10), rash at mid-point",
                                 "value": "natural_history"}],
                       style={"textAlign": "left"}),
        html.Div("3. Interventions", style={**SUBLABEL_STYLE}),
        dcc.Checklist(id="abm_isolation", value=["on"] if d["isolation"] else [],
                      options=[{"label": " Isolate cases 1 day after rash onset", "value": "on"}],
                      style={"textAlign": "left"}),
        dbc.Button("Re-run (same inputs)", id="rerun_button", n_clicks=0, size="sm",
                   color="secondary", className="mt-3"),
    ])
    return html.Div(dbc.Accordion(
        [dbc.AccordionItem(body, title="Model Engine (testing) ▾", item_id="engine",
                           style={"textAlign": "center"})],
        flush=True, always_open=True, active_item=[]))      # collapsed by default


def quarantine_input() -> html.Div:
    """Quarantine selector, shown with the model inputs (school enrollment,
    vaccination rate...). Only the agent-based engines simulate quarantine."""
    return html.Div([
        html.H4("Quarantine of unvaccinated contacts", style={**SELECTOR_LABEL_STYLE}),
        html.H3("21 days after each exposure (agent-based model only).",
                style={**SELECTOR_NOTE_STYLE, "margin-bottom": "0.4em"}),
        dcc.Dropdown(id="abm_quarantine", value=ABM_SETTINGS_DEFAULT["quarantine"], clearable=False,
                     searchable=False,
                     options=[{"label": "No quarantine", "value": "none"},
                              {"label": "Classmates", "value": "class"},
                              {"label": "Grade", "value": "grade"},
                              {"label": "Whole school", "value": "school"}],
                     style={**BASE_FONT_STYLE, "width": "14em", "textAlign": "left"}),
    ], className="d-flex flex-column align-items-center")


def timing_section() -> html.Div:
    """Simulation timing, collapsed by default."""
    body = html.Div([
        html.Div(id="timing_summary"),
        html.Div(id="timing_end_to_end", style={"font-weight": "600", "margin-top": "0.4em"}),
    ], style={"font-family": BASE_FONT_FAMILY_STR, "font-size": "12pt", "textAlign": "left"})
    return html.Div(dbc.Accordion(
        [dbc.AccordionItem(body, title="Simulation time ▾", item_id="timing")],
        flush=True, always_open=True, active_item=[]),
        className="mt-2", style={"background-color": "#f4f8fb"})


# Callbacks
###########

clientside_callback(
    ClientsideFunction(namespace="engines", function_name="build_request"),
    Output("sim_request", "data"),
    [Input("rerun_button", "n_clicks"),
     Input("dashboard_params", "data"),
     Input("inputs_are_valid", "data"),
     Input("engine_selector", "value"),
     Input("abm_structure", "value"),
     Input("abm_periods", "value"),
     Input("abm_isolation", "value"),
     Input("abm_quarantine", "value"),
     Input("n_reps_selector", "value")] +
    [Input(inp.id, "value") for inp in counterfactual_rate_inputs] +
    [State("engine_config", "data")],
)

@callback(Output("abm_isolation", "options"), Input("abm_periods", "value"))
def isolation_option(periods):
    """With exponential periods isolation is already part of the 5-day
    infectious period (compartmental model), so the option does not apply."""
    disabled = periods == "exponential"
    label = " Isolate cases 1 day after rash onset" + \
        (" (natural history only)" if disabled else "")
    return [{"label": label, "value": "on", "disabled": disabled}]


clientside_callback(
    ClientsideFunction(namespace="engines", function_name="run_browser"),
    Output("sim_results", "data", allow_duplicate=True),
    [Input("sim_request", "data"), State("engine_config", "data")],
    prevent_initial_call=True,
)

clientside_callback(
    ClientsideFunction(namespace="engines", function_name="end_to_end"),
    Output("timing_end_to_end", "children"),
    [Input("spaghetti_plot", "figure"), State("sim_results", "data")],
    prevent_initial_call=True,
)


@callback(Output("sim_results", "data"), Input("sim_request", "data"))
def run_server_simulations(req):
    if req is None or (req["valid"] and req["is_browser"]):
        raise PreventUpdate
    if not req["valid"]:
        return {"valid": False}
    p = req["params"]
    result = run_server_engine(req["engine"], p, req["vax_rates_pct"], req["abm_settings"],
                               req["n_reps"], req["n_curves"], req["seed"], rerun=req.get("rerun", 0))
    result.update(valid=True, I0=p["I0"][0], threshold=int(p["threshold_values"][0]),
                  population=int(p["population"][0]),
                  quarantine_scope=req["abm_settings"]["quarantine"])
    return result


@callback(
    [Output("spaghetti_plot", "figure"),
     Output("prob_threshold_plus_new_str", "children"),
     Output("cases_expected_over_threshold_str", "children"),
     Output("cases_expected_over_threshold_vaccinated_str", "children"),
     Output("outbreak_title", "children"),
     Output("cases_condition", "children"),
     Output("baseline_sim_results", "data"),
     Output("counterfactual_sim_results", "data"),
     Output("timing_summary", "children")],
    Input("sim_results", "data"),
)
def display_sim_results(res):
    """Same statistics and figure as the production dashboard, whatever the engine."""
    if not res or not res.get("valid"):
        return EMPTY_SPAGHETTI_PLOT_INFECTED_MA, "", "", "", "", "", None, None, ""

    I0, threshold = res["I0"], res["threshold"]
    base = res["scenarios"][0]
    unvax, vax = np.asarray(base["new_unvax"]), np.asarray(base["new_vax"])
    total = unvax + vax
    over = total >= threshold

    prob_str = dashboard_exceedance_prob_str(over.mean())
    if not over.any():
        unvax_str, vax_str = f"Fewer than {threshold} new infections", ""
    else:
        u_lb, u_ub = np.percentile(unvax[over], [2.5, 97.5])
        v_lb, v_ub = np.percentile(vax[over], [2.5, 97.5])
        unvax_str = "Unvaccinated cases: " + dashboard_percentiles_str(I0, u_lb, u_ub)
        vax_str = "Vaccinated cases: " + dashboard_percentiles_str(0, v_lb, v_ub)

    curves = np.asarray(res["curves"], dtype=float)
    if not res["curves_are_ma7"]:
        curves = msp.calculate_np_moving_average(curves, 7)
    n_curves, n_days = curves.shape
    df_spaghetti = msp.transform_matrix_to_long_df(
        curves.T, colnames=list(range(n_curves)), id_col="day", var_name="simulation_idx",
        id_values=np.arange(1, n_days + 1), value_name="result")
    ix_median = int(np.argmin(np.abs(np.asarray(res["curve_new_total"]) - np.median(total))))
    fig = dashboard_results_fig(df_spaghetti=df_spaghetti, index_sim_closest_median=ix_median,
                                nb_curves_displayed=N_CURVES, curve_selection_seed=12345)

    scen = res["scenarios"]
    baseline = {**scen[0], "I0": I0, "threshold": threshold, "population": res["population"],
                "quarantine_scope": res.get("quarantine_scope")}

    return fig, prob_str, unvax_str, vax_str, \
        f"Chance of exceeding {threshold} new infections", \
        f"*if exceeds {threshold} new infections*", \
        baseline, scen[1:], timing_table(res)


def timing_table(res) -> html.Div:
    t = res["timing"]
    label, _, where = ENGINES[res["engine"]]
    rows = []
    if t.get("curves_ms") is not None:
        rows.append(("Outbreak curves (20 simulations, separate model)", t["curves_ms"]))
    reused = t.get("reused") or [False] * len(t["scenario_ms"])
    for i, (s, ms, hit) in enumerate(zip(res["scenarios"], t["scenario_ms"], reused)):
        if s.get("role") == "quarantine":
            name = f"Current rate, {QUARANTINE_LABELS[s['quarantine']].lower()}"
        else:
            name = f"Current rate ({s['vax_rate_pct']:g}%)" if i == 0 else f"Scenario {s['vax_rate_pct']:g}%"
        rows.append((name + (" — reused (inputs unchanged)" if hit else ""), ms))
    if where == "server":
        threads = f"{t['threads']} thread{'s' if t['threads'] > 1 else ''} on the server"
    else:
        threads = f"{t['threads']} Web Worker{'s' if t['threads'] > 1 else ''}" \
            if res["engine"] == "abm_js_workers" else "main browser thread"
    note = " — includes starting the workers (first run)" if t.get("workers_started") else ""
    return html.Div([
        html.Div(f"Engine: {label}", style={"font-weight": "600"}),
        html.Div(f"{t['n_reps']} simulations per scenario, {threads}", style={"font-size": "11pt"}),
        dbc.Table([html.Tbody(
            [html.Tr([html.Td(name), html.Td(f"{ms:,.0f} ms", style={"text-align": "right",
                                                                       "color": "#999" if "reused" in name else None})])
             for name, ms in rows] +
            # total of all scenarios (reused ones count with their original
            # time), then what this update actually simulated
            [html.Tr([html.Td(html.B("Total simulation time" + note)),
                      html.Td(html.B(f"{sum(ms for _, ms in rows):,.0f} ms"),
                              style={"text-align": "right"})])] +
            ([html.Tr([html.Td("Simulated in this update (other scenarios reused)"),
                       html.Td(f"{t['total_ms']:,.0f} ms", style={"text-align": "right"})])]
             if any(reused) else [])
        )], size="sm", className="mb-0 mt-1", style={"max-width": "40em"}),
    ])
