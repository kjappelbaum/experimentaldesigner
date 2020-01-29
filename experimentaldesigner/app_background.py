# -*- coding: utf-8 -*-
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import dash_html_components as html
import dash_core_components as dcc
from pyDOE import lhs
from diversipy import hycusampling
import numpy as np
np.random.seed(821196)


FULL_FACTORIAL = '''
## Full Factorial Design
The most naive approach to select a sampling is to simply take all possible combinations, i.e. all factors (like temperatures and pressures) at all possible levels (all possible and relevant temperatures and pressure). This covers the design space completely, but this would also require the most experiments. To improve upon that, enhanecd DoE techniqeus have been developed.
'''


LHS = '''
## Latin Hypercube Sampling
In [latin hypercube sampling](https://en.wikipedia.org/wiki/Latin_hypercube_sampling), we bin the space along each dimension (experimental factor like the temperature) and then randomly sample from each bin. This gives us a near random sampling, which is good for statistical purposes.
Additionally, this has the advantage that the sampling along each dimension is uniform. By default, this sampling does not necessarily is best in filling the space.  
'''

MAXMIN = '''
## Maximin sampling 
In maximum sampling we try select points such that they have the maximum pairwise distances from each other. Doing this exactly is too expensive for larger problems, wherefore one typically uses an approximation in which one starts by randomly selecting one point and then adds the point that is farthest away from it to this set. One then continues adding the point that has the maximimum minimum distance from the already selected ploints. 
'''


full_factorial = np.array(hycusampling.grid(9, 2))
fig_full_factorial = go.Figure(
    data=[go.Scatter(x=full_factorial[:, 0], y=full_factorial[:, 1], mode='markers')])

fig_full_factorial.update_layout(go.Layout(paper_bgcolor='rgba(0,0,0,0)',
                                           xaxis_showgrid=True, yaxis_showgrid=True, title='Full factorial sampling'))
graph_full_factorial = dcc.Graph(
    id='full_factorial',
    figure=fig_full_factorial
)


latin_hypercube = np.array(lhs(2, 5))
lhs = pd.DataFrame({'x': latin_hypercube[:, 0], 'y': latin_hypercube[:, 1]})
fig_latin_hypercube = px.scatter(
    lhs, x='x', y='y', marginal_x='histogram', marginal_y='histogram')
fig_latin_hypercube.update_layout(go.Layout(paper_bgcolor='rgba(0,0,0,0)',
                                            xaxis_showgrid=True, yaxis_showgrid=True, title='Latin hypercube sampling'))


graph_latin_hypercube = dcc.Graph(
    id='full_factorial',
    figure=fig_latin_hypercube
)


layout = [
    html.Div(
        [
            html.H1('Some notes about design of experiments (DoE)'),
            html.P('All example figures show how the experimental sampling would look like in two dimensions which both have a range from 0 to 1.'),
            html.Div(
                dcc.Markdown(FULL_FACTORIAL),
            ),
            html.Div(
                graph_full_factorial
            ),
            html.Div(
                dcc.Markdown(LHS),
            ),
            html.Div(
                graph_latin_hypercube
            ),
            html.Div(
                dcc.Markdown(MAXMIN),
            ),
        ],
        id="container",
        # tag for iframe resizer
        **{'data-iframe-height': ''},
    )
]
