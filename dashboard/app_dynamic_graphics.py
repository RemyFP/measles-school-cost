# These functions/components CHANGE based on callbacks
#   -- contains results from simulations based on user inputs

from dash import Dash, html, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc
from cost_model import N_COUNTERFACTUALS

from app_static_graphics import \
    vaccination_rate_label, school_size_label, I0_label, R0_label, \
    latent_period_label, infectious_period_label, threshold_selector_label, \
    vaccine_efficacy_selector_label, vaccinated_infectiousness_selector_label
from app_styles import BASE_FONT_FAMILY_STR, RESULTS_HEADER_STYLE, \
    SELECTOR_NOTE_STYLE
from app_computation_functions import EMPTY_SPAGHETTI_PLOT_INFECTED_MA
from app_selectors import school_size_selector, \
    I0_selector, vaccination_rate_selector, state_selector, \
    county_selector, school_selector, R0_selector, latent_period_selector, \
    infectious_period_selector, threshold_selector, \
    vaccine_efficacy_selector, vaccinated_infectiousness_selector, \
    cost_param_inputs, counterfactual_rate_inputs
from app_computation_functions import EMPTY_COST_BAR_FIG
from cost_model import COST_PARAM_SPECS


def results_header():
    return dbc.Row(
        [
            # Chance of an Outbreak
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div(
                                [
                                    dcc.Markdown(id='outbreak_title',
                                                 children='Chance of exceeding 10 new infections',
                                                 style={**RESULTS_HEADER_STYLE, 'fontWeight': '500'}
                                                 ),
                                    dcc.Markdown(id='blank_space_for_alignment_with_next_col',
                                                 children="*if exceeds 10 new infections*",
                                                 style={'font-size': '14pt', "visibility": "hidden", "margin": "none"}),
                                    dcc.Markdown(id='prob_threshold_plus_new_str',
                                                 style={**RESULTS_HEADER_STYLE, 'color': '#bf5700',
                                                        "font-size": "22pt", "font-weight": "800", 'margin-top': '0.5em'}
                                                 ),
                                ],
                                style={
                                    'textAlign': 'center',
                                    'fontFamily': BASE_FONT_FAMILY_STR,
                                    'fontSize': '18pt',
                                    'border': 'none'
                                }
                            )
                        ]
                    ),
                    style={'border': 'none'}
                ),
            ),

            # Expected Outbreak Size
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div(
                                [
                                    dcc.Markdown(id='cases',
                                                 children='Likely outbreak size',
                                                 style={'color': '#black', 'fontWeight': '500',
                                                        'font-size': '20pt', 'margin': 'none'}
                                                 ),
                                    dcc.Markdown(id='cases_condition',
                                                 children="*if exceeds 10 new infections*",
                                                 style={'font-size': '14pt', "margin": "none"}),
                                    dcc.Markdown(id='cases_expected_over_threshold_str',
                                                 style={'color': '#bf5700', 'fontWeight': '800',
                                                        'font-size': '22pt', 'margin-top': '0.5em'}
                                                 ),
                                    dcc.Markdown(id='cases_expected_over_threshold_vaccinated_str',
                                                 style={'color': '#bf5700', 'fontWeight': '800',
                                                        'font-size': '22pt', 'margin-top': '0.5em'}
                                                 ),
                                ],
                                style={
                                    'textAlign': 'center',
                                    'fontFamily': BASE_FONT_FAMILY_STR,
                                    'fontSize': '18pt',
                                    'border': 'none'
                                }
                            )
                        ]
                    ),
                    style={'border': 'none'}
                ),
                style={'borderLeft': '3px solid #bf5700'}
            ),
        ],
    )


def spaghetti_plot_section():
    return dbc.Row([
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H3("This graph shows 20 plausible school outbreak curves.",
                            style={"text-align": "center", "margin-top": "1em", "margin-bottom": "1em",
                                   "margin-left": "1.8em",
                                   "font-family": BASE_FONT_FAMILY_STR,
                                   "font-size": "14pt", "font-weight": "400", "font-style": "italic"}),
                    dcc.Graph(id="spaghetti_plot", figure=EMPTY_SPAGHETTI_PLOT_INFECTED_MA),
                ]),
                style={'border': 'none', 'padding': '0'},
            ),
        ),
    ], style={"border-top": "2px solid black", "border-left": "1em", "padding": "none",
              "width": "100%", "margin-top": "1em"})


# Define the accordion separately
epi_params_accordion = html.Div(
    dbc.Accordion(
        [
            dbc.AccordionItem(
                dbc.Col(
                    [
                        html.Br(),
                        dbc.Col(html.Div(R0_label), className="mb-2"),
                        dbc.Col(html.Div(R0_selector), className="mb-2"),
                        dbc.Col(html.Div(latent_period_label), className="mb-2"),
                        dbc.Col(html.Div(latent_period_selector), className="mb-2"),
                        dbc.Col(html.Div(infectious_period_label), className="mb-2"),
                        dbc.Col(html.Div(infectious_period_selector), className="mb-2"),
                        dbc.Col(html.Div(threshold_selector_label), className="mb-2"),
                        dbc.Col(html.Div(threshold_selector), className="mb-2"),
                        dbc.Col(html.Div(vaccine_efficacy_selector_label), className="mb-2"),
                        dbc.Col(html.Div(vaccine_efficacy_selector), className="mb-2"),
                        dbc.Col(html.Div(vaccinated_infectiousness_selector_label), className="mb-2"),
                        dbc.Col(html.Div(vaccinated_infectiousness_selector), className="mb-2"),
                    ]
                ),
                title="Change Parameters ▾",
                style={"textAlign": "center"}  # Add this line to center the text
            ),
        ],
        flush=True,
        always_open=False,  # Ensures sections can be toggled independently
        active_item=[],  # Empty list means all sections are closed by default
    )
)

def cost_params_accordion() -> html.Div:
    """Dropdown with counterfactual vaccination rates and all cost
    parameters (generated from cost_parameters.toml)."""
    label_style = {"font-family": BASE_FONT_FAMILY_STR, "font-size": "13pt",
                   "margin-bottom": "0.2em", "margin-top": "0.6em"}
    group_style = {"font-family": BASE_FONT_FAMILY_STR, "font-size": "15pt",
                   "font-weight": "600", "margin-top": "1.2em"}

    children = [
        html.Br(),
        html.H3("Default values are placeholders pending literature review.",
                style={**SELECTOR_NOTE_STYLE}),
        html.Div("Counterfactual vaccination rates (%)", style=group_style),
        html.H3("Default: 1/3 and 2/3 of the way from the current rate to 100%.",
                style={**SELECTOR_NOTE_STYLE, "margin-top": "0.4em"}),
        html.Div(
            [html.Div(inp, style={"margin": "0 0.5em"}) for inp in counterfactual_rate_inputs],
            className="d-flex justify-content-center"),
    ]
    current_group = None
    for key, spec in COST_PARAM_SPECS.items():
        if spec["group"] != current_group:
            current_group = spec["group"]
            children.append(html.Div(current_group, style=group_style))
        children.append(html.Div(spec["label"], style=label_style))
        children.append(html.Div(cost_param_inputs[key],
                                 className="d-flex justify-content-center"))

    return html.Div(
        dbc.Accordion(
            [dbc.AccordionItem(dbc.Col(children), title="Cost Parameters ▾",
                               style={"textAlign": "center"})],
            flush=True,
            always_open=False,
            active_item=[],
        )
    )


def cost_column_selector() -> html.Div:
    """Checkboxes choosing the cost-table columns (options filled by the cost
    callback). The current vaccination rate column is always shown."""
    label_style = {"font-weight": "600", "margin-right": "0.8em"}
    box = {"display": "flex", "flex-wrap": "wrap", "align-items": "center", "gap": "0 1.2em"}
    check = dict(inline=True, inputStyle={"margin-right": "0.3em"},
                 labelStyle={"margin-right": "1.2em"})
    return html.Div([
        html.Div([html.Span("Show vaccination rates:", style=label_style),
                  dcc.Checklist(id="cost_cols_rates", options=[], **check,
                                value=[f"rate{k}" for k in range(1, N_COUNTERFACTUALS + 1)])],
                 style=box),
        html.Div([html.Span("Show other quarantine approaches (current rate):", style=label_style),
                  dcc.Checklist(id="cost_cols_quarantine", options=[], **check,
                                value=["q:none", "q:class", "q:grade", "q:school"])],
                 id="cost_cols_quarantine_box", style=box),
    ], style={"font-family": BASE_FONT_FAMILY_STR, "font-size": "12pt", "text-align": "left",
              "margin-bottom": "0.8em"})


def cost_results_section():
    return dbc.Row([
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H3("Outbreak burden, costs, and the impact of higher vaccination",
                            style={"text-align": "center", "margin-top": "1em",
                                   "font-family": BASE_FONT_FAMILY_STR,
                                   "font-size": "20pt", "font-weight": "500"}),
                    html.H3("Expected values are averaged over all simulated introductions, "
                            "including those that do not lead to an outbreak. "
                            "In parentheses: 95% range over all simulated introductions. "
                            "Rows ending in “if exceeds …” are 95% ranges among simulations exceeding the outbreak threshold.",
                            style={**SELECTOR_NOTE_STYLE, "margin-bottom": "1em",
                                   "line-height": "1.3"}),
                    cost_column_selector(),
                    html.Div(id="cost_table"),
                    dcc.Graph(id="cost_bar_chart", figure=EMPTY_COST_BAR_FIG),
                ]),
                style={'border': 'none', 'padding': '0'},
            ),
        ),
    ], style={"border-top": "2px solid black", "width": "100%", "margin-top": "1em"})


# Define the accordion separately
school_district_accordion = dbc.Accordion(
    [
        dbc.AccordionItem(
            html.Div(
                [
                    html.H3("ISD rates are district averages.", style={**SELECTOR_NOTE_STYLE}),
                    html.H3("Rates at individual schools may be higher or lower.", style={**SELECTOR_NOTE_STYLE}),
                    html.H3("School enrollment doesn't change automatically.",
                            style={**SELECTOR_NOTE_STYLE, "font-weight": "bold"}),
                    html.H3("Please update the value manually.", style={**SELECTOR_NOTE_STYLE, "font-weight": "bold"}),
                    dbc.Col(html.Div(state_selector), className="mb-2 p-0"),
                    dbc.Col(html.Div(county_selector), className="mb-2 p-0"),
                    dbc.Col(html.Div(school_selector), className="mb-2 p-0"),
                ]
            ),
            title="School/District Lookup ▾ ",
            style={"font-size": "18pt", "width": "100%", "margin": "none"},
            className="m-0"
        ),
    ],
    flush=True,
    always_open=False,  # Ensures sections can be toggled independently
    active_item=[],  # Empty list means all sections are closed by default
)


def input_panel_builder(school_size_header: html.H4,
                        school_size_input: dcc.Input,
                        I0_header: html.H4,
                        I0_input: dcc.Input,
                        vaccination_rate_header: html.H4,
                        vaccination_rate_input: dcc.Input,
                        top_accordion: dbc.Accordion,
                        bottom_accordion: dbc.Accordion,
                        cost_accordion=None,
                        testing_accordion=None,
                        quarantine_input=None) -> dbc.Col:
    return dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                    *([] if testing_accordion is None else [
                        dbc.Row(dbc.Col(testing_accordion, style={"width": "100%"}),
                                style={"border": "2px solid #0099cc", "border-radius": "6px",
                                       "margin": "0 0.2em 1em 0.2em"}),
                    ]),
                    html.H3("Model Inputs", style={"margin-left": "0.2em", "margin-top": "0.5em",
                                                   "font-family": BASE_FONT_FAMILY_STR,
                                                   "font-size": "24pt", "font-weight": "500",
                                                   "textAlign": "center"}, className="mt-2"),
                    dbc.Row([
                        dbc.Col([
                            html.Div(id='warning_str',
                                     style={"color": "red", "font-size": "12", "text-align": "center"},
                                     className="d-flex flex-column align-items-center")
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div(school_size_header),
                            html.Div(school_size_input),
                        ], className="d-flex flex-column align-items-center"),
                    ], className="d-flex flex-column align-items-center mb-2"),

                    dbc.Row([
                        dbc.Col([
                            html.Div(I0_header),
                            html.Div(I0_input)
                        ], className="d-flex flex-column align-items-center"),
                    ], className="d-flex flex-column align-items-center mb-2"),

                    dbc.Row([
                        dbc.Col(html.Div(vaccination_rate_header),
                                className="d-flex flex-column align-items-center"),
                    ]),

                    dbc.Row([
                        dbc.Col([
                            html.H3("Enter value or select from Lookup.", style={**SELECTOR_NOTE_STYLE})
                        ]),
                    ]),

                    dbc.Row([
                        dbc.Col([
                            html.Div(vaccination_rate_input), html.Div(" OR ", style={"font-size": "16pt",
                                                                                      "margin-top": "0.5em",
                                                                                      "margin-bottom": "0.5em"}),
                            html.Div(top_accordion, style={"width": "100%", "textAlign": "center"}),
                        ], className="d-flex flex-column align-items-center"),
                    ], style={} if quarantine_input is not None
                       else {"border-bottom": "2px solid black", "margin-right": "0.2em"}),

                    *([] if quarantine_input is None else [
                        dbc.Row(dbc.Col(quarantine_input, className="mt-3 mb-3"),
                                style={"border-bottom": "2px solid black", "margin-right": "0.2em"}),
                    ]),

                    html.Br(),
                    html.H3("Epidemic Parameters", style={"margin-left": "0.2em", "margin-top": "0.5em",
                                                          "font-family": BASE_FONT_FAMILY_STR,
                                                          "font-size": "24pt", "font-weight": "500",
                                                          "textAlign": "center"}),
                    html.Br(),
                    dbc.Row(dbc.Col(html.I(
                        "Caution – Default values reflect published estimates. Significant changes may result in inaccurate projections."),
                        className="mb-2 align-items-center",
                        style={"font-size": "14pt", "textAlign": "center"})),

                    dbc.Row([
                        # same gap above the separator line as after the quarantine input
                        dbc.Col(bottom_accordion, style={"width": "100%", "textAlign": "center",
                                                         "margin-bottom": "8px"}),
                    ], style={} if cost_accordion is None
                       else {"border-bottom": "2px solid black", "margin-right": "0.2em"}),

                    *([] if cost_accordion is None else [
                        html.Br(),
                        html.H3("Cost Parameters", style={"margin-left": "0.2em", "margin-top": "0.5em",
                                                          "font-family": BASE_FONT_FAMILY_STR,
                                                          "font-size": "24pt", "font-weight": "500",
                                                          "textAlign": "center"}),
                        dbc.Row([
                            dbc.Col(cost_accordion, className="mb-2",
                                    style={"width": "100%", "textAlign": "center"}),
                        ]),
                    ]),
                ]
            ),
            style={'border': 'none'}
        ),
        width=3, xs=12, sm=12, md=12, lg=12, xl=3,
        style={"border-right": "2px solid black", "padding": "10px"},
    )


def dashboard_input_panel(testing_accordion=None, quarantine_input=None):
    return input_panel_builder(school_size_header=school_size_label,
                               school_size_input=school_size_selector,
                               I0_header=I0_label,
                               I0_input=I0_selector,
                               vaccination_rate_header=vaccination_rate_label,
                               vaccination_rate_input=vaccination_rate_selector,
                               top_accordion=school_district_accordion,
                               bottom_accordion=epi_params_accordion,
                               cost_accordion=cost_params_accordion(),
                               testing_accordion=testing_accordion,
                               quarantine_input=quarantine_input)
