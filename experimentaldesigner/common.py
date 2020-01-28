# -*- coding: utf-8 -*-
import dash_html_components as html
import dash_table_experiments as dt
import base64
import urllib
import io
import pandas as pd


def render_df(df):
    return html.Div([
        # Use the DataTable prototype component:
        # github.com/plotly/dash-table-experiments
        dt.DataTable(rows=df.to_dict('records')),
    ])


def generate_table(dataframe, max_rows=100, download_link=False):

    components = []
    if download_link:
        csv_string = dataframe.to_csv(
            index=False, encoding='utf-8', float_format='%.2f')
        link = html.A(
            'Download CSV',
            download="experimental_conditions.csv",
            href="data:text/csv;charset=utf-8," +
            urllib.parse.quote(csv_string),
            target="_blank",
            className='button')

    components.append(
        html.Table(
            # Header
            [html.Tr([html.Th(col) for col in dataframe.columns])] +

            # Body
            [
                html.Tr([
                    html.Td(cell_format(dataframe.iloc[i][col]))
                    for col in dataframe.columns
                ]) for i in range(min(len(dataframe), max_rows))
            ]))

    components.append(html.Div([html.P("\n\n")]))
    components.append(link)
    components.append(html.Div([html.P("\n\n")]))

    return components


def cell_format(value):
    if isinstance(value, float):
        return "{:.2f}".format(value)
    return value
