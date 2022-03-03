import datetime
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests


import yfinance as yf


class Stock:
    """
    This class enables data loading, plotting and statistical analysis of a given stock,
     upon initialization load a sample of data to check if stock exists.

    """

    def __init__(self, symbol="BTCUSD"):
        """
        Object initialized with a currency, default='BTCUSD'
        """
        self.end = datetime.datetime.today()
        self.start = self.end - datetime.timedelta(days=4)
        self.symbol = symbol
        self.data = self.load_data(self.start, self.end)

    def plot_raw_data(self, fig):
        """
        Plot time-serie line chart of closing price on a given plotly.graph_objects.Figure object
        """
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

    def show_delta(self):
        """
        Visualize a summary of the stock change over the specified time period
        """
        epsilon = 1e-6
        i = self.start
        j = self.end
        s = self.data.query("date==@i")['Close'].values[0]
        e = self.data.query("date==@j")['Close'].values[0]

        difference = round(e - s, 2)
        change = round(difference / (s + epsilon) * 100, 2)
        e = round(e, 2)
        cols = st.columns(2)
        (color, marker) = ("green", "+") if difference >= 0 else ("red", "-")

        cols[0].markdown(
            f"""<p style="font-size: 90%;margin-left:5px">{self.symbol} \t {e}</p>""",
            unsafe_allow_html=True)
        cols[1].markdown(
            f"""<p style="color:{color};font-size:90%;margin-right:5px">{marker} \t {difference} {marker} {change} % </p>""",
            unsafe_allow_html=True)
