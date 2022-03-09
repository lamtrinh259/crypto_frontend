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

test_df = pd.DataFrame({
     'first column': [1, 2, 3, 4],
     'second column': [10, 20, 30, 40],
    })

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

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
window_selection_c.markdown("## Currency Selection") # add a title to the sidebar container
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
MODEL_LI = np.array(['FB_PROPHET', 'SARIMAX', 'LSTM'])
SYMB = window_selection_c.selectbox("Currency", CRYPTO)
MODEL = window_selection_c.selectbox("Model", MODEL_LI)
window_selection_c.write('### Instructions')
window_selection_c.write('1. Select Currency to Predict')
window_selection_c.write('2. Select Model to use to Predict')
window_selection_c.write('3. Invest According to the Predictions')
# chart_width = st.expander(label="chart width").slider("", 1000, 2800, 1400) #arguments: label, min, default, max

# # # ------------------------Plot stock linechart--------------------

# crypto = Stock(symbol=SYMB)

change_c = st.sidebar.container()
with change_c:
    fig=go.Figure()
    crypto = Crypto(SYMB.split('/')[0], MODEL)
    # crypto.load_data(START, END, inplace=True)
    # fig = crypto.plot_raw_data(fig)
    fig = crypto.predict_model()
    crypto.predict_model()


#---------------styling for plotly-------------------------
fig.update_layout(
            xaxis_title='Date',
            yaxis_title=crypto.symbol,
            width=1250,  # Change Chart width to Constant
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
st.write(fig)


csv = convert_df(crypto.data)


st.download_button(
    label="Download dataset as CSV",
    data=csv,
    file_name='dataset.csv',
    mime='text/csv',
)


col1, col2, col3= st.columns(3)
with col1:
    # Change the fear and grid to the last day of the dataset
    st.write(' ')
    st.image('https://alternative.me/images/fng/crypto-fear-and-greed-index-2022-3-3.png', width=390)
with col2:
    # Historical Data for the fear and greed
    st.write('### Why Measure Fear and Greed?')
    st.write('The crypto market behaviour is very emotional. People tend to get greedy when the market is rising which results in **FOMO** (Fear of missing out). Also, people often sell their coins in irrational reaction of seeing red numbers. With our Fear and Greed Index, we try to save you from your own emotional overreactions. There are two simple assumptions:')
    st.write('`Extreme fear` can be a sign that investors are too worried. That could be a buying opportunity.')
    st.write('##### When Investors are getting too greedy, that means the market is due for a correction.')
    st.write('Therefore, we analyze the current sentiment of the Bitcoin market and crunch the numbers into a simple meter from 0 to 100. Zero means `Extreme Fear`, while 100 means `Extreme Greed`. See below for further information on our data sources.')
with col3:
    # Description for the fear and greed
    st.write('### {} Predictions'.format(MODEL))
    st.write(crypto.pred)

    csv = convert_df(crypto.pred)

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='predictions.csv',
        mime='text/csv',
    )
