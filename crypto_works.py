import os
import numpy as np
import pandas as pd
import pickle
import quandl
import time
import urllib
import urllib.request
import hashlib
import codecs
import hmac
import json
from poloniex import Poloniex

from datetime import datetime
import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
py.init_notebook_mode(connected=False)
import matplotlib.pyplot as plt

def get_quandl_data(quandl_id):
    '''Download and cache Quandl dataseries'''
    cache_path = '{}.pkl'.format(quandl_id).replace('/', '-')
    try:
        f = open(cache_path, 'rb')
        df = pickle.load(f)
        print('Loaded {} from cache'.format(quandl_id))
    except (OSError, IOError) as e:
        print('Downloading {} from Quandl'.format(quandl_id))
        df = quandl.get(quandl_id, returns="pandas")
        df.to_pickle(cache_path)
        print('Cached {} at {}'.format(quandl_id, cache_path))
    return df


# In[5]: # Pull Kraken BTC price exchange data
btc_usd_price_kraken = get_quandl_data('BCHARTS/KRAKENUSD')

btc_trace = go.Scatter (x=btc_usd_price_kraken.index, y=btc_usd_price_kraken[ 'Weighted Price' ])
py.plot([btc_trace])

# Pull pricing data for 3 more BTC exchanges
exchanges = ['COINBASE', 'BITSTAMP', 'ITBIT']
exchange_data = {}
exchange_data['KRAKEN'] = btc_usd_price_kraken

for exchange in exchanges:
    exchange_code = 'BCHARTS/{}USD'.format(exchange)
    btc_exchange_df = get_quandl_data(exchange_code)
    exchange_data[exchange] = btc_exchange_df


def merge_dfs_on_column(dataframes, labels, col):
    '''Merge a single column of each dataframe into a new combined dataframe'''
    series_dict = {}
    for index in range(len(dataframes)):
        series_dict[labels[index]] = dataframes[index][col]

    return pd.DataFrame(series_dict)
btc_usd_datasets = merge_dfs_on_column(list(exchange_data.values()), list(exchange_data.keys()), 'Weighted Price')

print(btc_usd_datasets.tail(10))


def df_scatter(df, title, seperate_y_axis=False, y_axis_label='', scale='linear', initial_hide=False):
    '''Generate a scatter plot of the entire dataframe'''
    label_arr = list(df)
    series_arr = list(map(lambda col: df[col], label_arr))

    layout = go.Layout(
        title=title,
        legend=dict(orientation="h"),
        xaxis=dict(type='date'),
        yaxis=dict(
            title=y_axis_label,
            showticklabels=not seperate_y_axis,
            type=scale
        )
    )

    y_axis_config = dict(
        overlaying='y',
        showticklabels=False,
        type=scale)

    visibility = 'visible'
    if initial_hide:
        visibility = 'legendonly'

    # Form Trace For Each Series
    trace_arr = []
    for index, series in enumerate(series_arr):
        trace = go.Scatter(
            x=series.index,
            y=series,
            name=label_arr[index],
            visible=visibility
        )

        # Add seperate axis for the series
        if seperate_y_axis:
            trace['yaxis'] = 'y{}'.format(index + 1)
            layout['yaxis{}'.format(index + 1)] = y_axis_config
        trace_arr.append(trace)
    fig = go.Figure(data=trace_arr, layout=layout)
    py.plot(fig)

df_scatter(btc_usd_datasets, 'Bitcoin Price (USD) By Exchange')

btc_usd_datasets.replace(0, np.nan, inplace=True)
df_scatter(btc_usd_datasets, 'Bitcoin Price (USD) By Exchange')

btc_usd_datasets['avg_btc_price_usd'] = btc_usd_datasets.mean(axis=1)
btc_trace = go.Scatter(x=btc_usd_datasets.index, y=btc_usd_datasets['avg_btc_price_usd'])
py.iplot([btc_trace])

def get_json_data(json_url, cache_path):
    '''Download and cache JSON data, return as a dataframe.'''
    try:
        f = open(cache_path, 'rb')
        df = pickle.load(f)
        print('Loaded {} from cache'.format(json_url))
    except (OSError, IOError) as e:
        print('Downloading {}'.format(json_url))
        df = pd.read_json(json_url)
        df.to_pickle(cache_path)
        print('Cached {} at {}'.format(json_url, cache_path))
    return df
base_polo_url = 'https://poloniex.com/public?command=returnChartData&currencyPair={}&start={}&end={}&period={}'
start_date = datetime.strptime('2015-01-01', '%Y-%m-%d') # get data from the start of 2015
end_date = datetime.now() # up until today
pediod = 86400 # pull daily data (86,400 seconds per day)

Key = '1JR3MYNK-SL953LC1-ZQPYJJ1K-M3X81JXT'
Sign = 'dca1be2c43acf5ab9fa9d8f841058524775c35399ec667a36ad22008608a2a1b1ea5f304deb1902756fcdba8daac13f4c3fb45e735c8b690a5243a2168f94de4'

def api_call(command):
    nonce = int(round(time.time()-599900000)*10)
    parms = {"command":command, "nonce":nonce}
    parms = urllib.parse.urlencode(parms).encode()

    hashed = hmac.new(Sign.encode(), parms, digestmod=hashlib.sha512)
    signature = hashed.hexdigest()
    headers = {"Key": Key, "Sign": signature}

    req = urllib.request.Request("https://poloniex.com/tradingApi", headers=headers)
    try:
        conn = urllib.request.urlopen(req, data=parms)
    except urllib.error.HTTPError as e:
        conn = e
    print(conn.status,conn.reason)
    return json.loads(conn.read().decode())

balances = api_call("closeMarginPosition")

def get_crypto_data(poloniex_pair):
    '''Retrieve cryptocurrency data from poloniex'''
    json_url = base_polo_url.format(poloniex_pair, start_date.timestamp(), end_date.timestamp(), pediod)
    data_df = get_json_data(json_url, poloniex_pair)
    data_df = data_df.set_index('date')
    return data_df
altcoins = ['ETH','LTC','XRP','ETC','STR','DASH','SC','XMR','XEM']

altcoin_data = {}
for altcoin in altcoins:
    coinpair = 'BTC_{}'.format(altcoin)
    crypto_price_df = get_crypto_data(coinpair)
    altcoin_data[altcoin] = crypto_price_df4
    altcoin_data[ 'ETH' ].tail ()
