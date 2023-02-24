import fnmatch
import signal
import threading
import webbrowser

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import numpy
from dash.dependencies import Input, Output
from dash_table.Format import Format, Scheme
from flask import Flask
from gevent.pywsgi import WSGIServer

from .utils import *

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


def main(config_file):
    df = df_setup(config_file)
    dash_setup(df)

    WebServer().start()
    webbrowser.open_new('http://127.0.0.1:8050/')
    signal.signal(signal.SIGINT, shutdown)

    print('Use CTRL+C to close the server.')


def df_setup(config_file):
    print(f'--- Creating interactive report on server localhost:{server_address[1]} ---')
    print('Select the results directory created by fRAT.')
    folder = Utils.file_browser(title='Select the directory output by the fRAT')

    config = Utils.load_config(f'{Path(os.path.abspath(__file__)).parents[1]}/configuration_profiles/', config_file)

    if config.averaging_type == 'Session averaged':
        subfolder = 'Session_averaged_results'
        print(f"Showing {subfolder.replace('_', ' ')}.")

    else:
        subfolder = 'Participant_averaged_results'
        print(f"Showing {subfolder.replace('_', ' ')}.")

    df = pd.read_json(f"{folder}/Overall/Summarised_results/{subfolder}/combined_results.json")

    r = re.compile("Conf_Int_*")
    Conf_Int = list(filter(r.match, df.columns))[0]  # Find confidence interval level

    if config.averaging_type == 'Session averaged':
        column_order = ['index', 'Mean', Conf_Int, 'Median', 'Std_dev',
                        'Minimum', 'Maximum', 'Total voxels', 'Excluded voxels', 'Average voxels per session',
                        'Sessions', 'File_name']
    else:
        column_order = ['index', 'Mean', Conf_Int, 'Median', 'Std_dev',
                        'Minimum', 'Maximum', 'Total voxels',
                        'Excluded voxels', 'Average voxels per session',
                        'Participants', 'Sessions', 'File_name']

    if 'Percentage change from baseline' in df.columns:
        column_order.insert(3, 'Percentage change from baseline')
        column_order.insert(4, 'Baseline')

    crit_params = list(set(df.columns) ^ set(column_order))  # Find elements not in both lists

    for crit_param in crit_params:
        column_order.insert(1, crit_param)  # Insert critical parameters after index

    df = df[column_order]  # Reorganise columns

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

        html.Div([html.Label(['Statistic to display:',

                              dcc.Dropdown(id='barchart_dropdown',
                                           options=[
                                               {'label': 'Mean', 'value': 'Mean'},
                                               {'label': 'Standard deviation', 'value': 'Std_dev'},
                                               {'label': 'Median', 'value': 'Median'},
                                               {'label': 'Minimum', 'value': 'Minimum'},
                                               {'label': 'Maximum', 'value': 'Maximum'},
                                               {'label': 'Total voxels', 'value': 'Total voxels'},
                                               {'label': 'Average voxels per session', 'value': 'Average voxels per session'},
                                               {'label': 'Excluded voxels (percentage)', 'value': 'Excluded voxels'}
                                           ],
                                           value='Mean',
                                           style={"width": "40%"},
                                           clearable=False
                                           )]),
                  html.Br(),
                  dcc.Graph(id='barchart', figure={"layout": {"height": 500}})
                  ],
                 style={"width": "70%", 'marginLeft': 40, 'marginRight': 10, 'marginTop': 10, 'marginBottom': 10,
                        'padding': '10px 0px 0px 10px'},
                 )
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
        [Input('datatable', 'selected_row_ids'),
         Input('barchart_dropdown', 'value'),
         Input('barchart_dropdown', 'options')])
    def update_graph(selected_rows, dropdown_value, dropdown_options):
        if selected_rows is None:
            return dash.no_update

        dff = df.loc[df['id'].isin(selected_rows)].copy()

        dropdown_label = [i['label'] for i in dropdown_options if i['value'] == dropdown_value][0]

        ConfInts = []
        if dropdown_value == 'Mean':
            ConfInts = dff[fnmatch.filter(df.columns, 'Conf_Int*')[0]]

        elif dropdown_value == 'Excluded voxels':
            dff['Excluded voxels'] = dff['Excluded voxels'] / (dff['Excluded voxels'] + dff['Total voxels']) * 100
            dropdown_label = 'Percentage of voxels excluded'

        intwidth = 35
        barwidth = 0.09 * (len(selected_rows))

        return {
            'data': [{"x": dff['index'],
                     "y": dff[dropdown_value],
                     "width": barwidth,
                     "error_y": {"type": 'data', "array": ConfInts, "thickness": 2, "width": intwidth},
                     "type": 'bar'}],

            "layout": {
                'title': 'Interactive barchart',
                "xaxis": {"title": "Region of interest"},
                "yaxis": {"title": dropdown_label},
                'plot_bgcolor': colors['background'],
                'paper_bgcolor': colors['background']
            },
        }
