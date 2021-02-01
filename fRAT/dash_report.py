import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
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
    print(f'Shutting down website server on port {server_address[1]}.\n')
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
    global appserver

    colors = {'background': '#f8f8f8', 'text': '#000000'}
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    appserver = Flask(__name__)

    app = dash.Dash(__name__, server=appserver, external_stylesheets=external_stylesheets)

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
            row_selectable='multi',
            selected_rows=[],
            page_action="native",
            page_current=0,
            page_size=len(df['index'].unique()),
            style_cell={'textAlign': 'left'},
            # style_data_conditional=[
            #     {
            #         'if': {'row_index': 'odd'},
            #         'backgroundColor': 'rgb(248, 248, 248)'
            #     }
            # ],
        ),
        html.Div(id='figure-container'),
    ])

    @app.callback(Output('datatable-interactivity', 'style_data_conditional'),
                  [Input('datatable-interactivity', 'selected_rows'), Input('datatable-interactivity', 'derived_virtual_indices')])
    def update_styles(selected_rows, derived_virtual_indices):
        print(derived_virtual_indices)
        return [{'if': {'derived_virtual_indices': i}, 'background_color': '#D2F3FF'} for i in selected_rows]


    @app.callback(Output("figure_container", "children"),
                 [Input('datatable', 'rows'),
                  Input('datatable', 'selected_row_indices')])
    def update_barchart(rows, selected_row_indices):
        print(rows, selected_row_indices)
        if selected_row_indices is None:
            selected_row_indices = []

        dff = df if rows is None else pd.DataFrame(rows)

        fig = None
        if rows is not None:
            selected_rows = [rows[i] for i in selected_row_indices]
            print(selected_rows)
            fig = px.bar(df, x="Voxels", y="Mean", barmode="group")

            return [
                dcc.Graph(
                    id='test',
                    figure={
                        "data": [
                            {
                                "x": dff["index"],
                                "y": dff["Voxels"],
                                "type": "bar",
                                "marker": {"color": colors},
                            }
                        ],
                        "layout": {
                            "xaxis": {"automargin": True},
                            "yaxis": {
                                "automargin": True,
                                "title": {"text": 'test'}
                            },
                            "height": 250,
                            "margin": {"t": 10, "l": 10, "r": 10},
                        },
                    },
                )
                # # check if column exists - user may have deleted it
                # # If `column.deletable=False`, then you don't
                # # need to do this check.
                # for column in ["pop", "lifeExp", "gdpPercap"] if column in dff
            ]


if __name__ == '__main__':
    main()
