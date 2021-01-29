import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy
import webbrowser
import os
import logging

# TODO: need to explain how to close flask server
def main():
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)  # Only error messages will be printed

    df = df_setup()
    app = dash_setup(df)

    # The reloader has not yet run - open the browser
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        webbrowser.open_new('http://127.0.0.1:8050/')

    # Otherwise, continue as normal
    app.run_server(host="127.0.0.1", port=8050)

    return app


def df_setup():
    df = pd.read_json(f"Summarised_results/combined_results.json")

    for key in df.keys():
        if isinstance(df[key][0], numpy.float64):
            df[key] = df[key].map("{:,.3f}".format)  # Number of significant figures to use for floats
        elif isinstance(df[key][0], numpy.int64):
            df[key] = df[key].map("{:,d}".format)  # Convert to integer object so number searching works

    column_order = [0, 4, 8, 6, 9, 1, 7, 5, 10, 2, 3]
    for i in range(11, len(df.columns)):
        column_order.insert(3, i)  # Used to insert additional columns if more than 2 parameters used in paramValues.csv

    df = df[df.columns[column_order]]  # Reorganise columns

    return df


def dash_setup(df):
    colors = {
        'background': '#f8f8f8',
        'text': '#000000'
    }
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
    app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
        dcc.Markdown('# fRAT Report \n Interactive table', style={'textAlign': 'center', 'color': colors['text']}),
        dash_table.DataTable(
            id='datatable-interactivity',
            columns=[{"name": i, "id": i, "deletable": True, "selectable": True} for i in df.columns],
            data=df.to_dict('records'),
            filter_action="native",
            sort_action="native",
            sort_mode="multi",
            row_deletable=True,
            page_action="native",
            page_current=0,
            page_size=len(df['index'].unique()),
            style_cell={'textAlign': 'left'},
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ],
        )
    ])

    return app


if __name__ == '__main__':
    main()
