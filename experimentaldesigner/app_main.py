# -*- coding: utf-8 -*-
import dash_html_components as html
from .variables import variables_div
from . import app

ABOUT = """
Design of experiments is the study of how to choose experiments---in the most efficient way---to understand the influence of some factors on the experimental outcome.

There are myriads of different techniques that allow this. To faciliate their use, especially in experimental chemistry, this app allows to create experimental design for arbitrary number of factors.
"""

ABOUT_HTML = [html.P(i) for i in ABOUT.split("\n\n")]

layout = [
    html.Div(
        [
            html.Div(html.H1(app.title), id="maintitle"),
            html.H2("About"),
            html.P(['We recommend that you visit ',        html.A(
                    "the page with background information",
                    href="background/"), "."]),
            html.Div(
                ABOUT_HTML,
                className="info-container"),
            html.H2("Create your experimental design"),
            variables_div,
            html.Div([html.P("\n\n")]),
            html.P([
                "Find the code ",
                html.A(
                    "on github",
                    href="https://github.com/kjappelbaum/experimentaldesigner",
                    target='_blank'), "."
            ]),
        ],
        id="container",
        # tag for iframe resizer
        **{'data-iframe-height': ''},
    )
]
