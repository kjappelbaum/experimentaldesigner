# -*- coding: utf-8 -*-
import dash_html_components as html
import dash_core_components as dcc

FULL_FACTORIAL = '''
## Full Factorial Design
The most naive approach to select a sampling is to simply take all possible combinations, i.e. all factors (like temperatures and pressures) at all possible levels (all possible and relevant temperatures and pressure). This covers the design space completely, but this would also require the most experiments. To improve upon that, enhanecd DoE techniqeus have been developed.
'''


LHS = '''
## Latin Hypercube Sampling
In latin hypercube sampling, we bin the space along each dimension (experimental factor like the temperature) and then randomly sample from each bin. 

This has the advantage that the sampling along each dimension is uniform. By default, this sampling does not necessarily is best in filling the space.  
'''

layout = [
    html.Div(
        [
            html.Div(html.H1(app.title), id="maintitle"),
            html.Div(
                dcc.Markdown(FULL_FACTORIAL),
            ),
            html.Div(
                dcc.Markdown(LHS),
            )
        ],
        id="container",
        # tag for iframe resizer
        **{'data-iframe-height': ''},
    )
]
