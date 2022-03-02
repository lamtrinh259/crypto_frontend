import datetime
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np


import yfinance as yf


class Stock:
    """
    This class enables data loading, plotting and statistical analysis of a given stock,
     upon initialization load a sample of data to check if stock exists.

    """

    def __init__(self, symbol="GOOG"):

        self.end = datetime.datetime.today()
        self.start = self.end - datetime.timedelta(days=4)
        self.symbol = symbol
        self.data = self.load_data(self.start, self.end)

    @st.cache(show_spinner=False) #Using st.cache allows st to load the data once and cache it.
    def load_data(self, start, end, inplace=False):
        """
        takes a start and end dates, download data do some processing and returns dataframe
        """

        data = yf.download(self.symbol, start, end + datetime.timedelta(days=1))
        #Check if there is data
        try:
            assert len(data) > 0
        except AssertionError:
            print("Cannot fetch data, check spelling or time window")
        data.reset_index(inplace=True)
        data.rename(columns={"Date": "datetime"}, inplace=True)
        data["date"] = data.apply(lambda raw: raw["datetime"].date(), axis=1)

        data = data[["date", 'Close']]
        if inplace:
            self.data = data
            self.start = start
            self.end = end
            return True
        return data

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
