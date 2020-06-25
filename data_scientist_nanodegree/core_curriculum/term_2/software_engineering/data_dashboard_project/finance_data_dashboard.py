# Data dashboard dash application
# The applicaation can be run stand alone or within a django web site using django_plotly_dash
# Based on https://github.com/justdjango/My_Dashboard/tree/master/finance
# python finance_data_dashboard.py

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
import colorlover as cl
import datetime as dt
import numpy as np
import pandas as pd
import quandl
from django_plotly_dash import DjangoDash

APP_DJANGO_PLOTLY_DASH_NAME = 'finance_data_dashboard'
TICKER_FILENAME = 'data/tickers.csv'
INDICATOR_FILENAME = 'data/indicators.csv'
DEFAULT_WINDOW_SIZE_BOLLINGER_BANDS = 10
DEFAULT_NUM_OF_STD_BOLLINGER_BANDS = 5
DEFAULT_AVAILABLE_YEARS = 5
DEFAULT_TICKERS = ['AAPL', 'MSFT']
DEFAULT_INDICATORS = ['tot_revnu', 'gross_profit', 'ebitda']
QUANDL_PERSONAL_KEY = 'YOUR_QUANDL_PERSONAL_KEY'
EXTERNAL_STYLESHEETS = [
    'https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css',
    {
        'href': 'https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css',
        'rel': 'stylesheet',
        'integrity': 'sha384-BVYiiSIFeK1dGmJRAkycuHAHRg32OmUcww7on3RYdg4Va+PmSTsz/K68vbdEjh4u',
        'crossorigin': 'anonymous'
    }
]

colorscale = cl.scales['9']['qual']['Paired']
time_dictionary = {'1W' : 7, '1M' : 30, '1Y' : 365, '5Y' : 1825} 

quandl.ApiConfig.api_key = QUANDL_PERSONAL_KEY


def _create_app(django_plotly_dash = False, ticker_filename = TICKER_FILENAME, indicator_filename = INDICATOR_FILENAME):
    '''
    Creates dash application

    Args:
        django_plotly_dash (boolean): django_plotly_dash or not. Default value False
        ticker_filename (str): ticker filename. Default value TICKER_FILENAME
        indicator_filename (str):: indicator filename. Default value INDICATOR_FILENAME

    Returns:
        app (dash.Dash or DjangoDash): Dash or DjangoDash application
    '''

    if django_plotly_dash == False:
        app = dash.Dash(__name__, external_stylesheets = EXTERNAL_STYLESHEETS)
    else:
        app = DjangoDash(APP_DJANGO_PLOTLY_DASH_NAME, add_bootstrap_links = True)

    df_ticker = pd.read_csv(ticker_filename)
    df_indicator = pd.read_csv(indicator_filename)
    data_end_time = dt.datetime.strptime('2018-03-27', '%Y-%m-%d') # dt.datetime.now() quandl does not provide data updated
    # data_start_time = data_end_time - dt.timedelta(days = 365)
    window_size_bollinger_bands = DEFAULT_WINDOW_SIZE_BOLLINGER_BANDS
    num_of_std_bollinger_bands = DEFAULT_NUM_OF_STD_BOLLINGER_BANDS
    list_year = np.arange(data_end_time.date().year, data_end_time.date().year - DEFAULT_AVAILABLE_YEARS, -1)

    app.layout = html.Div(
        [
            dbc.Nav(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.A('Data dashboard', href='/', className = 'navbar-brand' )
                                ], className = 'navbar-header')
                            , html.Div(
                                [
                                    html.Ul(
                                        [
                                            html.Li(html.A('Made with Udacity', href='https://www.udacity.com/'))
                                            , html.Li(html.A('Github', href='https://github.com/simonerigoni/udacity/tree/master/data_scientist_nanodegree/core_curriculum/term_2/software_engineering/data_dashboard_project'))
                                        ], className = 'nav navbar-nav')
                                ], className = 'collapse navbar-collapse')
                        ], className = 'container')
                ], className = 'navbar navbar-inverse navbar-fixed-top')
            , html.Div(
                [
                    html.Div(
                        [
                            html.H1('Quandle Finance Explorer', className = 'text-center'),
                            html.H4('Data available only to {}'.format(data_end_time.date())),
                            html.H3('Compare Stocks'),
                            dcc.Dropdown(id = 'dropdown-stock-tickers', options = [{'label': s[0], 'value': s[1]} for s in zip(df_ticker.Company_Name, df_ticker.Ticker)], value = DEFAULT_TICKERS, multi = True),
                            html.H3('Timescale'),
                            dcc.RadioItems(id = 'radioitems-timescale', options = [{'label': t, 'value': t} for t in time_dictionary], value = '1Y'),
                            html.H3('Bollinger bands parameters'),
                            dcc.Checklist(id = 'checklist-enable-bollinger-bands', options = [{'label': 'Enable', 'value': 'enable'}], value = ['enable']),
                            html.H4('Window size'),
                            dcc.Input(id = 'input-window-size-bollinger-bands', type = 'number', value = window_size_bollinger_bands),
                            html.H4('Number of standard deviation'),
                            dcc.Input(id = 'input-num-of-std-bollinger-bands', type = 'number', value = num_of_std_bollinger_bands),
                            html.H3('Graphs'),
                            html.Div(id = 'graphs'),
                            html.H3('Indicators'),
                            html.H4('Years'),
                            dcc.Dropdown(id = 'dropdown-years', options = [{'label': year, 'value': year} for year in list_year], value = [str(data_end_time.date().year), str(data_end_time.date().year - 1)], multi = True),
                            html.H4('Indicators'),
                            dcc.Dropdown(id = 'dropdown-indicators', options = [{'label': s[0], 'value': s[1]} for s in zip(df_indicator.Name, df_indicator.Column_Code)], value = DEFAULT_INDICATORS, multi = True),
                            html.Div(id = 'tables')         
                        ], className = 'container')
                ], className = 'jumbotron')
        ], className = 'container')

    @app.callback(dash.dependencies.Output('graphs','children'), [dash.dependencies.Input('dropdown-stock-tickers', 'value'), 
                                                                    dash.dependencies.Input('radioitems-timescale', 'value'),
                                                                    dash.dependencies.Input('checklist-enable-bollinger-bands', 'value'), 
                                                                    dash.dependencies.Input('input-window-size-bollinger-bands', 'value'),  
                                                                    dash.dependencies.Input('input-num-of-std-bollinger-bands', 'value')                                                                    
                                                                ])

    def update_graph(stock_tickers, timescale, enable_bollinger_bands, window_size_bollinger_bands, num_of_std_bollinger_bands):
        '''
        Update the graphs

        Args:
            stock_tickers (list): selected tickers
            timescale (list): selected timescale
            enable_bollinger_bands (str): enable or disable bollinger bands
            window_size_bollinger_bands (int): window size for bollinger bands
            num_of_std_bollinger_bands (int): number of standar deviation for bollinger bands

        Returns:
            graphs (list): list of graphs
        '''
        data_start_time = (data_end_time - dt.timedelta(days = time_dictionary[timescale])).date()
        enable_bollinger_bands = True if len(enable_bollinger_bands) > 0 and enable_bollinger_bands[0] == 'enable' else False
        graphs = []

        for i, ticker in enumerate(stock_tickers):
            graphs.append(html.H4(ticker))
            try:
                df = quandl.get('WIKI/' + ticker, start_date = data_start_time, end_date = data_end_time)
            except:
                #graphs.append(html.H3('Data is not available for {}'.format(ticker))#, className = {'marginTop': 20, 'marginBottom': 20}))
                graphs.append(html.H5('Data is not available'))
                continue

            candlestick = {
                'x': df.index,
                'open': df['Open'],
                'high': df['High'],
                'low': df['Low'],
                'close': df['Close'],
                'type': 'candlestick',
                'name': ticker,
                'legendgroup': ticker,
                'increasing': {'line': {'color': colorscale[0]}},
                'decreasing': {'line': {'color': colorscale[1]}}
            }
                  
            if enable_bollinger_bands == True:
                bb_bands = bollinger_bands(df.Close, window_size_bollinger_bands, num_of_std_bollinger_bands)

                bollinger_traces = [{
                    'x': df.index, 'y': y,
                    'type': 'scatter', 'mode': 'lines',
                    'line': {'width': 1, 'color': colorscale[(i * 2) % len(colorscale)]},
                    'hoverinfo': 'none',
                    'legendgroup': ticker,
                    'showlegend': True if i == 0 else False,
                    'name': '{} - bollinger bands'.format(ticker)
                } for i, y in enumerate(bb_bands)]

            #graphs.append(html.H4(ticker))
            graphs.append(dcc.Graph(
                id = ticker,
                figure = {
                    'data': [candlestick] + bollinger_traces if enable_bollinger_bands == True else [candlestick],
                    'layout': {
                        'margin': {'b': 0, 'r': 10, 'l': 60, 't': 0},
                        'legend': {'x': 0}
                    }
                }
            ))
        return graphs

    @app.callback(dash.dependencies.Output('tables','children'), [dash.dependencies.Input('dropdown-stock-tickers', 'value'), 
                                                                    dash.dependencies.Input('dropdown-years', 'value'),
                                                                    dash.dependencies.Input('dropdown-indicators', 'value')                                                         
                                                                ])
    def update_table(stock_tickers, years, indicators):
        '''
        Update the tables

        Args:
            stock_tickers (list): selected tickers
            years (int): selected years
            indicators (list): selected indicators

        Returns:
            tables (list): list of tables
        '''
        tables = []
        for i, ticker in enumerate(stock_tickers):
            tables.append(html.H4(ticker))
            try:
                colonne = ['m_ticker', 'per_end_date', 'per_type', 'per_cal_year'] + indicators
                df = quandl.get_table('ZACKS/FC', paginate = False, ticker = ticker, qopts={'columns': colonne})
                #cd ..print(df)
            except:
                tables.append(html.H5('Data is not available'))
                continue

            if df.empty == True:
                tables.append(html.H5('Data is not available'))
                continue

            df = df[df['per_type'] == 'A'] 
            df = df[['per_cal_year'] + indicators]         
            anni = df['per_cal_year'].to_list()
            #print(anni)
            df = df.set_index('per_cal_year')
            df = df.transpose()
            df = df.reset_index()
            df.columns = ['Indicator'] + [str(y) for y in anni]
            #print(df)
            tables.append(dash_table.DataTable(id = ticker, columns=[{"name": column, "id": column} for column in df.columns], data = df.to_dict('records')))
        return tables

    return app


def bollinger_bands(price, window_size = DEFAULT_WINDOW_SIZE_BOLLINGER_BANDS, num_of_std = DEFAULT_NUM_OF_STD_BOLLINGER_BANDS):
    '''
    Compute bollinger bands

    Args:
        price (float): ticker price
        window_size (int): window size
        num_of_std (int): number of standar deviation
    Returns:
        rolling_mean (float): rolling mean value
        upper_band (float): upper band
        lower_band (float): lower band
    '''
    rolling_mean = price.rolling(window = window_size).mean()
    rolling_std  = price.rolling(window = window_size).std()
    upper_band = rolling_mean + (rolling_std * num_of_std)
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return rolling_mean, upper_band, lower_band


if __name__ == '__main__':
    app = _create_app()
    app.run_server(debug = True)
else:
    app = _create_app(django_plotly_dash = True, ticker_filename = APP_DJANGO_PLOTLY_DASH_NAME + '/' + TICKER_FILENAME, indicator_filename = APP_DJANGO_PLOTLY_DASH_NAME + '/' + INDICATOR_FILENAME)
