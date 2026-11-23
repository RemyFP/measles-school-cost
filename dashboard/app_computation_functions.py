import measles_single_population as msp
import pandas as pd
import numpy as np
import plotly.express as px

from randomgen import PCG64

from app_styles import SPAGHETTI_PLOT_AXIS_CONFIG


def dashboard_exceedance_prob_str(exceedance_prob: int):
    """
    Returns string to populate the written text portion of the dashboard
    corresponding to probability of exceeding X new total infections,
    where X is the chosen outbreak threshold_value
    """

    if exceedance_prob < 0.01:
        exceedance_prob_str = "< 1%"
    # VERY IMPORTANT: HTML NEEDS AN ESCAPE BEFORE >
    # Otherwise literally the whole dashboard breaks, it's crazy
    elif exceedance_prob > 0.99:
        exceedance_prob_str = "\> 99%"
    else:
        exceedance_prob_str = '{:.0%}'.format(exceedance_prob)

    return exceedance_prob_str


def dashboard_percentiles_str(
        init_infected,
        lb_new,
        ub_new):
    """
    THIS INCLUDES INITIAL INFECTED!!!! :)
    """

    lb_total, ub_total = init_infected + lb_new, init_infected + ub_new

    all_cases_cond_percentiles_str = str(int(lb_total)) + ' - ' + str(int(ub_total))

    return all_cases_cond_percentiles_str


def dashboard_spaghetti(df_plot: pd.DataFrame,
                        spaghetti_color_map: dict):
    """
    Returns plotly.graph_objects.Figure
    """

    # Infuriating trying to get an empty lineplot NOT to display negative numbers
    #   -- still does this even with repeated attempts at overriding by adjusting
    #   "range"-related parameters :(
    # To fix... don't show ticks if plot is empty

    if df_plot.empty:
        is_nonempty_df = False
    else:
        is_nonempty_df = True

    fig = px.line(
        df_plot,
        x='day',
        y='result',
        color='simulation_idx',
        color_discrete_map=spaghetti_color_map)

    fig.update_traces(hovertemplate="Day %{x}<br>%{y:.1f} infected<extra></extra>")
    fig.update_traces(line=dict(width=2))  # Reduce line thickness

    fig.update_layout(showlegend=False,
                      plot_bgcolor='white',
                      margin=dict(l=0, r=0, t=0, b=0),
                      xaxis=dict(
                          title="Day",
                          **SPAGHETTI_PLOT_AXIS_CONFIG,
                          linecolor="grey",
                          showticklabels=is_nonempty_df),
                      yaxis=dict(
                          title="Number of infected students",
                          **SPAGHETTI_PLOT_AXIS_CONFIG,
                          linecolor="black",
                          showticklabels=is_nonempty_df
                      ))

    return fig


def dashboard_results_fig(df_spaghetti: pd.DataFrame,
                          index_sim_closest_median: int,
                          nb_curves_displayed: int,
                          curve_selection_seed: int):
    # df_spaghetti must have columns "day", "result", "simulation_idx"

    light_grey = 'rgb(220, 220, 220)'

    color_map = {
        x: light_grey
        for x in df_spaghetti['simulation_idx']
    }
    color_map[index_sim_closest_median] = 'rgb(0, 153, 204)'  # blue

    possible_idx = df_spaghetti['simulation_idx']
    possible_idx = possible_idx[possible_idx != index_sim_closest_median]

    sample_idx = np.random.Generator(PCG64(curve_selection_seed)).choice(possible_idx,
                                                                         nb_curves_displayed,
                                                                         replace=False)

    sim_plot_df = pd.concat([
        df_spaghetti.loc[df_spaghetti['simulation_idx'].isin(sample_idx)],
        df_spaghetti.loc[df_spaghetti['simulation_idx'] == index_sim_closest_median]
    ])

    return dashboard_spaghetti(sim_plot_df, color_map)


# Lauren wanted the default plot to have the same axes as the generated plot
#   from simulations
EMPTY_SPAGHETTI_PLOT_INFECTED_MA = dashboard_spaghetti(
    pd.DataFrame(columns=['day', 'result', 'simulation_idx']), {})


# Cost results
##############
import plotly.graph_objects as go
from dash import html
import dash_bootstrap_components as dbc

from cost_model import COST_COMPONENTS, format_money

COST_BAR_COLORS = ["#bf5700", "#f2a65a", "#0099cc", "#102c41", "#8c8c8c"]


def _prob_str(p: float) -> str:
    # Plain-text version of dashboard_exceedance_prob_str (no markdown escape)
    if p < 0.01:
        return "< 1%"
    if p > 0.99:
        return "> 99%"
    return '{:.0%}'.format(p)


def _range_str(rng, fmt) -> str:
    return "—" if rng is None else f"{fmt(rng[0])} – {fmt(rng[1])}"


def cost_results_table(summaries: list[dict], scenario_labels: list[str],
                       threshold: int, note: str = "") -> html.Div:
    """
    summaries: output of cost_model.summarize_scenario, one per scenario,
        the first one being the current (baseline) vaccination rate.
    Expected values are followed by the 95% range over all simulations.
    """
    base = summaries[0]
    n0 = lambda x: f"{x:,.0f}"
    n1 = lambda x: f"{x:,.1f}"

    def mr(key, fmt_mean, fmt_range=None):
        """'mean (lb – ub)' for every scenario."""
        fr = fmt_range or fmt_mean
        return [f"{fmt_mean(s[f'mean_{key}'])} ({_range_str(s[f'range_{key}'], fr)})"
                for s in summaries]

    rows = [
        (f"Chance of exceeding {threshold} new infections", [_prob_str(s["prob_outbreak"]) for s in summaries]),
        ("Expected cases", mr("cases", n1, n0)),
        (f"Likely outbreak size if exceeds {threshold} new infections",
         [_range_str(s["cases_range_if_outbreak"], n0) for s in summaries]),
        ("Expected hospitalizations", mr("hospitalizations", n1)),
        ("Expected outpatient visits", mr("outpatient_visits", n1, n0)),
        ("Isolation: school days missed by cases", mr("school_days_missed", n0)),
        ("Isolation: parent work days lost", mr("caregiver_work_days_lost", n0)),
        ("Students quarantined", mr("students_quarantined", n0)),
        ("Quarantine: school days missed", mr("quarantine_school_days_missed", n0)),
        ("Quarantine: parent work days lost", mr("quarantine_caregiver_work_days_lost", n0)),
        ("Total school days missed (all students, isolation + quarantine)",
         mr("total_school_days_missed", n0)),
    ]
    for key, name in COST_COMPONENTS.items():
        if any(s[f"mean_{key}"] > 0 for s in summaries):
            rows.append((f"Expected cost: {name.lower()}", mr(key, format_money)))
    rows += [
        ("Expected total cost", mr("cost_total", format_money)),
        (f"Total cost if exceeds {threshold} new infections",
         [_range_str(s["cost_total_range_if_outbreak"], format_money) for s in summaries]),
        ("Cases averted vs. current", ["—"] + [n1(base["mean_cases"] - s["mean_cases"]) for s in summaries[1:]]),
        ("Cost averted vs. current", ["—"] + [format_money(base["mean_cost_total"] - s["mean_cost_total"])
                                               for s in summaries[1:]]),
    ]

    bold_rows = {"Expected total cost", "Cost averted vs. current"}
    def th(lbl):   # "line 1\nline 2" -> two lines; second line (quarantine) lighter
        first, _, second = lbl.partition("\n")
        return html.Th([first] + ([html.Br(), html.Span(second, style={"fontWeight": "400",
                                                                        "fontSize": "11pt"})]
                                  if second else []), style={"textAlign": "center"})

    header = html.Thead(html.Tr([html.Th("")] + [th(lbl) for lbl in scenario_labels]))
    body = html.Tbody([
        html.Tr([html.Td(name)] + [html.Td(v, style={"textAlign": "center"}) for v in values],
                style={"fontWeight": "700"} if name in bold_rows else {})
        for name, values in rows
    ])
    table = dbc.Table([header, body], bordered=False, hover=True, striped=True, size="sm",
                      responsive=True,   # scrolls horizontally on small screens
                      style={"fontSize": "13pt", "fontFamily": "Sans-serif"})
    return html.Div([table, html.Div(note, style={"fontSize": "11pt", "fontStyle": "italic",
                                                  "textAlign": "left"})])


def cost_bar_fig(summaries: list[dict], scenario_labels: list[str]) -> go.Figure:
    """Stacked bars: expected cost per introduction, by component and scenario."""
    fig = go.Figure()
    scenario_labels = [lbl.replace("\n", "<br>") for lbl in scenario_labels]
    for (key, name), color in zip(COST_COMPONENTS.items(), COST_BAR_COLORS):
        values = [s[f"mean_{key}"] for s in summaries]
        if any(v > 0 for v in values):
            fig.add_trace(go.Bar(name=name, x=scenario_labels, y=values, marker_color=color,
                                 hovertemplate="%{x}<br>" + name + ": $%{y:,.0f}<extra></extra>"))
    return _style_cost_fig(fig)


def _style_cost_fig(fig: go.Figure) -> go.Figure:
    fig.update_layout(barmode="stack", plot_bgcolor="white", height=380,
                      margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="h", y=-0.15, font=dict(size=14)),
                      yaxis=dict(title="Expected cost per introduction ($)", tickprefix="$",
                                 tickformat=",.0f", gridcolor="rgb(242,242,242)",
                                 title_font=dict(size=16), tickfont=dict(size=14)),
                      xaxis=dict(tickfont=dict(size=13)))
    return fig


EMPTY_COST_BAR_FIG = _style_cost_fig(go.Figure())
