# simple_streamlit_app.py
"""
A simple streamlit app
"""

import datetime
from turtle import width
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from crypto import Crypto


st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.title('Crypto forecast dashboard')


def nearest_business_day(DATE: datetime.date):
    """
    Takes a date and transform it to the nearest business day
    """
    if DATE.weekday() == 5:
        DATE = DATE - datetime.timedelta(days=1)

    if DATE.weekday() == 6:
        DATE = DATE + datetime.timedelta(days=1)
    return DATE


# ------ layout setting---------------------------
window_selection_c = st.sidebar.container() # create an empty container in the sidebar
window_selection_c.markdown("## Insights") # add a title to the sidebar container
# sub_columns = window_selection_c.columns(2) #Split the container into two columns for start and end date

# ----------Time window selection-----------------
YESTERDAY=datetime.date.today()-datetime.timedelta(days=1)
YESTERDAY = nearest_business_day(YESTERDAY) #Round to business day

DEFAULT_START=YESTERDAY - datetime.timedelta(days=700)
DEFAULT_START = nearest_business_day(DEFAULT_START)

# START = sub_columns[0].date_input("From", value=DEFAULT_START, max_value=YESTERDAY - datetime.timedelta(days=1))
# END = sub_columns[1].date_input("To", value=YESTERDAY, max_value=YESTERDAY, min_value=START)


# ---------------stock selection------------------
CRYPTO = np.array([ "BTC/USD", "ETH/USD", "LTC/USD"])
SYMB = window_selection_c.selectbox("select currency", CRYPTO)

# chart_width = st.expander(label="chart width").slider("", 1000, 2800, 1400) #arguments: label, min, default, max

# # # ------------------------Plot stock linechart--------------------

fig=go.Figure()
# crypto = Stock(symbol=SYMB)
crypto = Crypto(SYMB.split('/')[0])
# crypto.load_data(START, END, inplace=True)
st.write(crypto.test_api())
fig = crypto.plot_raw_data(fig)

#---------------styling for plotly-------------------------
fig.update_layout(
            width=950,  # Change Chart width to Constant
            margin=dict(l=0, r=0, t=0, b=0, pad=0),
            legend=dict(
                x=0,
                y=0.99,
                traceorder="normal",
                font=dict(size=12),
            ),
            autosize=False,
            template="plotly_dark",
)
col1, col2 = st.columns([4,1])
with col1:
    st.write(fig)
with col2:
    st.image('https://alternative.me/images/fng/crypto-fear-and-greed-index-2022-3-3.png', width=300)

change_c = st.sidebar.container()
with change_c:
    crypto.test_api()
