import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash_table.Format import Format, Scheme
import fnmatch
import pandas as pd
import numpy
import webbrowser
from flask import Flask
import signal
import threading
from gevent.pywsgi import WSGIServer

from utils import *

server_address = ("localhost", 8050)
server = None
appserver = None


class WebServer(threading.Thread):
    def __init__(self):
        super().__init__()

    def run(self):
        global server
        server = WSGIServer(server_address, appserver, log=None)
        server.serve_forever()


def shutdown(num, info):
    print(f'Shutting down server on port {server_address[1]}.\n')
    server.stop()
    server.close()


def main():
    df = df_setup()
    dash_setup(df)

    WebServer().start()
    webbrowser.open_new('http://127.0.0.1:8050/')
    signal.signal(signal.SIGINT, shutdown)

    print('Use CTRL+C to close the server.')


def df_setup():
    print(f'--- Creating interactive report on server localhost:{server_address[1]} ---')
    print('Select the results directory created by fRAT.')
    folder = Utils.file_browser(title='Select the directory output by the fRAT')

    df = pd.read_json(f"{folder}/Summarised_results/combined_results.json")

    column_order = [0, 4, 8, 6, 9, 1, 7, 5, 10, 2, 3]
    for i in range(11, len(df.columns)):
        column_order.insert(3, i)  # Used to insert additional columns if more than 2 parameters used in paramValues.csv

    df = df[df.columns[column_order]]  # Reorganise columns

    df["id"] = df.index

    return df


def dash_setup(df):
    global appserver

    colors = {'background': '#f8f8f8', 'text': '#000000'}
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    appserver = Flask(__name__)
    app = dash.Dash(__name__, server=appserver, external_stylesheets=external_stylesheets)

    columns = []
    for i, key in enumerate(df.columns):
        # Column format setup
        columns.append({"name": key, "id": key, "deletable": True, "selectable": True})

        if isinstance(df[key][0], numpy.float64):
            columns[i].update({"type": "numeric", "format": Format(precision=2, scheme=Scheme.fixed)})
        elif isinstance(df[key][0], numpy.int64):
            columns[i].update({"type": "numeric", "format": Format(precision=2, scheme=Scheme.decimal_integer)})

    app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
        dcc.Markdown('# fRAT Report \n Interactive table', style={'textAlign': 'center', 'color': colors['text']}),
        dash_table.DataTable(
            id='datatable',
            columns=columns,
            data=df.to_dict('records'),
            filter_action="native",
            sort_action="native",
            sort_mode="multi",
            row_deletable=True,
            row_selectable='multi',
            selected_rows=[],
            page_action="native",
            page_current=0,
            page_size=len(df['index'].unique()),
            style_cell={'textAlign': 'left'},
        ),

        html.Div(dcc.Dropdown(id='barchart_dropdown',
                              options=[
                                  {'label': 'Mean', 'value': 'Mean'},
                                  {'label': 'Minimum', 'value': 'Min'},
                                  {'label': 'Maximum', 'value': 'Max'},
                                  {'label': 'Voxels', 'value': 'Voxels'},
                                  {'label': 'Excluded voxels (percentage)', 'value': 'Excluded_Voxels'}
                              ],
                              value='Mean')),

        html.Div(dcc.Graph(id='barchart', figure={"layout": {
            "height": 500,
            }}))
    ])

    @app.callback(Output("datatable", "style_data_conditional"),
                  Input("datatable", "derived_viewport_selected_row_ids"))
    def style_selected_rows(selected_rows):
        if selected_rows is None:
            return dash.no_update

        style = [{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}]
        [style.append({"if": {"filter_query": f"{{id}} ={i}"}, "backgroundColor": "#D2F3FF"}) for i in selected_rows]

        return style

    @app.callback(
        Output('barchart', 'figure'),
        [Input('datatable', 'derived_viewport_selected_row_ids'),
         Input('barchart_dropdown', 'value')])
    def update_graph(selected_rows, dropdown_value):
        if selected_rows is None:
            return dash.no_update

        dff = df.loc[df['id'].isin(selected_rows)]

        ConfInts = []
        if dropdown_value == 'Mean':
            ConfInts = dff[fnmatch.filter(df.columns, 'Conf_Int*')[0]]
        elif dropdown_value == 'Excluded_Voxels':
            dff['Excluded_Voxels'] = dff['Excluded_Voxels'] / (dff['Excluded_Voxels'] + dff['Voxels']) * 100

        return {
            'data': [dict(x=dff['index'],
                          y=dff[dropdown_value],
                          error_y=dict(type='data', array=ConfInts,
                                       thickness=2, width=50),
                          barmode="group",
                          type='bar',
                          )]
        }


if __name__ == '__main__':
    main()
