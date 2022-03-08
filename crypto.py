from cProfile import label
import datetime
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
from datetime import timedelta


class Crypto:
    """
    This class enables data loading, plotting and statistical analysis of a given stock,
     upon initialization load a sample of data to check if stock exists.

    """

    def __init__(self, symbol="BTC", model = 'FB_PROPHET'):
        """
        Object initialized with a currency, default='BTCUSD'
        """
        self.end = datetime.datetime.today()
        self.start = self.end - datetime.timedelta(days=4)
        self.symbol = symbol
        self.model = model
        self.data = None
        # self.data = self.load_data(self.start, self.end)

    def plot_raw_data(self, fig):
        """
        Plot time-serie line chart of closing price on a given plotly.graph_objects.Figure object
        """
        # Test Change to display graph on streamlit, needs to be changed
        X = np.arange(1,100)
        y = np.random.rand(100,)
        fig = fig.add_trace(
            go.Scatter(
                x=X,
                y=y,
                mode="lines",
                name=self.symbol,
            )
        )
        return fig


    @staticmethod
    def nearest_business_day(DATE: datetime.date):
        """
        Takes a date and transform it to the nearest business day,
        static because we would like to use it without a stock object.
        """
        if DATE.weekday() == 5:
            DATE = DATE - datetime.timedelta(days=1)

        if DATE.weekday() == 6:
            DATE = DATE + datetime.timedelta(days=1)
        return DATE

    def show_fb_proph(self, data, pred):
        '''
        Display FB_Prophet
        '''
        data = pd.DataFrame(data)
        pred = pd.DataFrame(pred)
        fig1 = go.Figure(data=[go.Candlestick(x=data['time'],
                                        open=data['open'],
                                        high=data['high'],
                                        low=data['low'],
                                        close=data['close'],
                                        name = 'Historical Data'),
                go.Scatter(x=pred.ds.iloc[-14:],
                            y=pred.yhat.iloc[-14:],
                            mode='lines',
                            name = 'Prediction')
                ])
        return fig1

    def show_sarimax(self, data, pred):
        '''
        Display SARIMAX Model
        '''
        pass

    def show_lstm(self, data, pred):
        """ Plot the final results with actual prices and predicted prices (from test generator) for given crypto """
        # plt.plot(data['actual_price'], color = 'g', label = f'Actual prices of {self.symbol}')
        # plt.plot(data['pred_future_price'], color = 'b', label = f'Predicted prices of {self.symbol}')
        # plt.legend(loc='best')
        # plt.xlabel('Date')
        # plt.ylabel('Price in USD')
        # plt.title(f'Actual and predicted prices of {self.symbol} in USD')
        # plt.show()

    def all_grapher(self, data, pred):
        pred_x = pred.index[-14:]
        end_date = pd.to_datetime(pred.index[-1])
        start_date = end_date-timedelta(days=30)

        fig = go.Figure(data=[
                go.Candlestick(x = data.index,
                            open = data['open'],
                            high = data['high'],
                            low = data['low'],
                            close = data['close'],
                            name = 'Historical Data'),
                go.Scatter( x = pred_x,
                            y = pred['Predicted Price'],
                            mode = 'lines',
                            name = 'Prediction'),
                go.Scatter( x=pred_x, # +pred_x[::-1], # x, then x reversed,
                            y=pred['MAX Price'] , # upper, then lower reversed
                            mode = 'lines',
                            line=dict(width=0),
                            hoverinfo="skip",
                            showlegend=False),
                go.Scatter( x=pred_x, # +pred_x[::-1], # x, then x reversed,
                            y=pred['MIN Price'], # supper, then lower reversed
                            fill='tonexty',
                            mode = 'lines',
                            line=dict(width=0),
                            fillcolor='rgba(68, 68, 68, 0.3)',
                            hoverinfo="skip",
                            showlegend=False)])
        fig.update_xaxes(type="date", range=[start_date, end_date])
        return fig

    def test_api(self):
        url = 'http://127.0.0.1:8000/fbprophet_predict'
        # url = 'https://cryptopreddeployment-h7pfeyag5q-ew.a.run.app/fbprophet_predict'
        params = {'selected_crypto':self.symbol, 'format':'json'}
        response = requests.get(url, params=params).json()
        self.data = pd.DataFrame(response['data']).reset_index()
        pred = pd.DataFrame(response['predict'])
        self.data['index'] = pd.to_datetime(self.data['index'])
        return self.all_grapher(self.data,pred)

    def predict_model(self):

        url = 'http://127.0.0.1:8000/fbprophet_predict' # Update with url from GCP
        params = {'model':self.model,'selected_crypto':self.symbol,'format':'json'}
        response = requests.get(url, params=params).json()
        self.data = pd.DataFrame(response['data']).reset_index()
        pred = pd.DataFrame(response['predict'])

        return self.all_grapher(self.data, pred)
