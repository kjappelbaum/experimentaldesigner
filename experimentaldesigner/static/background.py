# -*- coding: utf-8 -*-
import dash_html_components as html

layout = [
    html.Div(
        [
            html.Div(html.H1(app.title), id="maintitle"),
        ],
        id="container",
        # tag for iframe resizer
        **{'data-iframe-height': ''},
    )
]
