import datetime
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests
from datetime import timedelta
import warnings

warnings.filterwarnings('ignore')


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
        self.pred = None
        self.api_available = False
        # self.data = self.load_data(self.start, self.end)

    def _generate_demo_data(self):
        """Generate demo data when API is unavailable"""
        # Generate 180 days of historical data
        end_date = datetime.datetime.now()
        dates = pd.date_range(end=end_date, periods=180, freq='D')
        
        # Base price depends on symbol
        base_prices = {'BTC': 45000, 'ETH': 3000, 'LTC': 150}
        base = base_prices.get(self.symbol, 100)
        
        # Generate realistic price movements
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.03, 180)
        prices = base * np.exp(np.cumsum(returns))
        
        # Create OHLC data
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.005, 180)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.02, 180))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.02, 180))),
            'close': prices
        }, index=dates)
        
        # Generate 14 days of predictions
        pred_dates = pd.date_range(start=dates[-1] + timedelta(days=1), periods=14, freq='D')
        last_price = prices[-1]
        pred_returns = np.random.normal(0.001, 0.02, 14)
        pred_prices = last_price * np.exp(np.cumsum(pred_returns))
        
        pred = pd.DataFrame({
            'Predicted Price': pred_prices,
            'MIN Price': pred_prices * 0.95,
            'MAX Price': pred_prices * 1.05
        }, index=pred_dates)
        pred.index = pred.index.strftime('%Y-%m-%d')
        
        return data, pred

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
        start_date = end_date-timedelta(days=180)

        fig = go.Figure(data=[
                go.Candlestick(x = data.index,
                            open = data['open'],
                            high = data['high'],
                            low = data['low'],
                            close = data['close'],
                            # name = '<b>Historical Data</b>'),
                            name = 'Historical Data'),
                go.Scatter( x = pred_x,
                            y = pred['Predicted Price'],
                            mode = 'lines',
                            # name = '<b>Prediction</b>',
                            name = 'Prediction',
                            line=dict(color='rgba(51, 153, 255, 1)'),
                            ),
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
                            fillcolor='rgba(51, 153, 255, 0.2)',
                            hoverinfo="skip",
                            showlegend=False),
                ]
                        )
        fig.update_xaxes(type="date", range=[start_date, end_date])
        return fig

    def test_api(self):
        url = 'http://127.0.0.1:8000/fbprophet_predict'
        params = {'selected_crypto':self.symbol, 'format':'json'}
        try:
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.data = pd.DataFrame(data['data']).reset_index()
                pred = pd.DataFrame(data['predict'])
                self.data['index'] = pd.to_datetime(self.data['index'])
                self.api_available = True
                return self.all_grapher(self.data, pred)
        except Exception:
            pass
        # Fallback to demo data
        self.api_available = False
        self.data, self.pred = self._generate_demo_data()
        return self.all_grapher(self.data, self.pred)

    def predict_model(self):
        # Try to fetch from API first
        url = 'https://cryptopredict-h7pfeyag5q-du.a.run.app/predict_model'
        params = {'model':self.model,'selected_crypto':self.symbol,'format':'json'}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.data = pd.DataFrame(data['data'])
                self.pred = pd.DataFrame(data['predict'])
                self.pred.index = self.pred.index.str.replace('T00:00:00','')
                self.api_available = True
                return self.all_grapher(self.data, self.pred)
        except Exception:
            pass
        
        # Fallback to demo data if API fails
        self.api_available = False
        self.data, self.pred = self._generate_demo_data()
        return self.all_grapher(self.data, self.pred)
