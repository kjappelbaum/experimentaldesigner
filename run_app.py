# -*- coding: utf-8 -*-
from experimentaldesigner import app, app_main
import dash.dependencies as dep


@app.callback(
    dep.Output('page-content', 'children'), [dep.Input('url', 'pathname')])
def display_page(pathname):
    # pylint: disable=no-else-return
    if pathname is None:
        return app_main.layout

    return app_main.layout


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
