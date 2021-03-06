# -*- coding: utf-8 -*-
import collections
import numpy as np

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from . import app
from .doe import build_lhs_grid, build_sukharev_grid, build_maxmin_grid, _get_sampling_df, build_random_grid, build_kmeans_grid, run_mds, measure_sampling_quality

VARIABLES = collections.OrderedDict([
    ('temperature',
     dict(label='temperature / C', range=[100.0, 200.0])),
    ('h2o', dict(label='water / mL', range=[1.0, 6.0])),
    ('power', dict(label='microwave power / W', range=[
     150.0, 250.0])),
])
NVARS_DEFAULT = len(VARIABLES)

# Intialize all variables for callbacks
NVARS_MAX = 10
for i in range(len(VARIABLES), NVARS_MAX):
    k = 'variable_{}'.format(i + 1)
    VARIABLES[k] = dict(label=k, range=[0, 1])

VAR_IDS = list(VARIABLES.keys())
VAR_LABELS = [v['label'] for v in list(VARIABLES.values())]

WEIGHT_RANGE = [0, 1.0]
NGRID = NVARS_MAX


def get_controls(id, description: str, range: tuple):

    label = dcc.Input(
        id=id + "_label", type='text', value=description, className="label")
    range_low = dcc.Input(
        id=id + "_low", type='number', value=range[0], className="range")
    range_high = dcc.Input(
        id=id + "_high", type='number', value=range[1], className="range")

    return html.Tr([
        html.Td(label),
        html.Td([range_low, html.Span('to'), range_high]),
    ],
        id=id + "_tr")


controls_dict = collections.OrderedDict()
for k, v in list(VARIABLES.items()):
    controls = get_controls(k, v['label'], v['range'])
    controls_dict[k] = controls

head_row = html.Tr([
    html.Th('Variable'),
    html.Th('Range')
])

controls_html = html.Table(
    [head_row] + list(controls_dict.values()), id='controls')
label_states = [
    dash.dependencies.State(k + "_label", 'value') for k in VAR_IDS
]

low_states = [dash.dependencies.State(k + "_low", 'value') for k in VAR_IDS]
high_states = [dash.dependencies.State(k + "_high", 'value') for k in VAR_IDS]

inp_nvars = html.Tr([
    html.Td('Number of variables: '),
    html.Td(
        dcc.Input(
            id='inp_nvars',
            type='number',
            value=NVARS_DEFAULT,
            max=NVARS_MAX,
            min=1,
            className="nvars range"))
])

inp_nsamples = html.Tr([
    html.Td('Number of samples: '),
    html.Td(
        dcc.Input(
            id='nsamples', type='number', value=10,
            className="nsamples range"))
])


btn_compute = html.Div([
    html.Button('compute', id='btn_compute', className='action-button'),

    html.Div([
        dcc.Tabs(id="tabs-sampling", value='table', children=[
            dcc.Tab(label='Experimental design table', value='table'),
            dcc.Tab(label='Low-dimensional visualization of sampling and metrics',
                    value='sampling_vis'),
        ]),
        html.Div(id='tabs-content')
    ])
])


# Dropdown to select the DoE method
dropdown_method = dcc.Dropdown(
    id='dropdown_method',
    options=[
        {'label': 'Latin hypercube (simple)', 'value': 'lhs_simple'},
        {'label': 'Sukarev', 'value': 'sukarev'},
        {'label': 'Maximin', 'value': 'maximin'},
        {'label': 'random', 'value': 'random'},
        {'label': 'k-mean', 'value': 'k_means'},
    ],
    value='lhs_simple',
    multi=False
)


# Delete the importance columns if lhs is selected
# ToDo: maybe also remove it completely
@app.callback(
    dash.dependencies.Output("dropdown_output", "children"),
    [dash.dependencies.Input("dropdown_method", "value")]
)
def print(value):
    return value


# Callbacks to hide unselected variables
for i in range(NVARS_MAX):

    @app.callback(
        dash.dependencies.Output(VAR_IDS[i] + "_tr", 'style'),
        [dash.dependencies.Input('inp_nvars', 'value')])
    def toggle_visibility(nvars, i=i):
        """Callback for setting variable visibility"""
        style = {}

        if i + 1 > nvars:
            style['display'] = 'none'

        return style


states = label_states + low_states + high_states
states += [dash.dependencies.State('inp_nvars', 'value')]
states += [dash.dependencies.State('nsamples', 'value')]

variables_div = html.Div(
    [
        html.H3("Define your design space"),
        html.Table([inp_nvars, inp_nsamples]),
        controls_html,
        html.H3("Select the DOE method"),
        dropdown_method,
        html.Div('', id='dropdown_output', style={'display': 'none'}),
        html.H3("Run the DOE"),
        btn_compute,
    ],
    id="container_select",
    # tag for iframe resizer
    **{'data-iframe-height': ''},
)


ninps = len(label_states + low_states + high_states) + 3


@app.callback(
    dash.dependencies.Output('tabs-content', 'children'),
    [dash.dependencies.Input('btn_compute', 'n_clicks'), dash.dependencies.Input('tabs-sampling', 'value')],  states + [dash.dependencies.State('dropdown_output', 'value')])
def on_compute(n_clicks, tab, *args):
    """Callback for clicking compute button"""
    if n_clicks is None:
        return ''

    if len(args) != ninps:
        raise ValueError("Expected {} arguments".format(ninps))

    # parse arguments
    nvars = args[-3]
    nsamples = args[-2]
    mode = args[-1]

    app.logger.info("nsamples: {}, nvars: {}, mode: {}, tab: {}".format(
        nsamples, nvars, mode, tab))

    labels = args[:nvars]
    low_vals = np.array([args[i + NVARS_MAX] for i in range(nvars)])
    high_vals = np.array([args[i + 2 * NVARS_MAX] for i in range(nvars)])
    app.logger.info('low {}, high {}'.format(low_vals, high_vals))

    df = _get_sampling_df(labels, low_vals, high_vals, nsamples)

    if mode == 'lhs_simple':
        df_output = build_lhs_grid(df, nsamples)
    elif mode == 'lhs_spacefilling':
        df_output = build_lhs_grid(df, nsamples, spacefilling=True)
    elif mode == 'sukarev':
        df_output = build_sukharev_grid(df, nsamples)
    elif mode == 'maximin':
        df_output = build_maxmin_grid(df, nsamples)
    elif mode == 'random':
        df_output = build_random_grid(df, nsamples)
    elif mode == 'kmean':
        df_output = build_kmeans_grid(df, nsamples)
    else:
        df_output = build_lhs_grid(df, nsamples)
        # raise ValueError("Unknown mode '{}'".format(mode))

    from .common import generate_table

    table = generate_table(df_output, download_link=True)

    projected = run_mds(df_output.values)

    fig = go.Figure(
        data=[go.Scatter(x=projected[:, 0], y=projected[:, 1], mode='markers')])

    fig.update_layout(go.Layout(paper_bgcolor='rgba(0,0,0,0)',
                                xaxis_showgrid=True, yaxis_showgrid=True))
    graph = dcc.Graph(
        id='sampling',
        figure=fig
    )
    sampling_table = measure_sampling_quality(df_output)

    if tab == 'table':
        return html.Div(table)
    elif tab == 'sampling_vis':
        return html.Div([
            html.Div([graph]),
            html.Div([sampling_table])
        ])
