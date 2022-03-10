# simple_streamlit_app.py
"""
A simple streamlit app
"""

import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from crypto_frontend.crypto import Crypto

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

def predict():
    # st.title('Crypto forecast dashboard')
    st.markdown("<h1 style='text-align: center; color: white;'>Crypto forecast dashboard</h1>", unsafe_allow_html=True)
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



    # # # ------------------------Plot stock linechart--------------------

    # crypto = Stock(symbol=SYMB)

    # change_c = st.sidebar.container()
    # with change_c:
    fig=go.Figure()
    crypto = Crypto(SYMB.split('/')[0], MODEL)
    # crypto.load_data(START, END, inplace=True)
    # fig = crypto.plot_raw_data(fig)
    fig = crypto.predict_model()
    # crypto.predict_model()


    #---------------styling for plotly-------------------------
    fig.update_layout(
                xaxis_title='Date',
                yaxis_title='USD',
                yaxis={'side': 'right'} ,
                title= '{} Prediction for {}/USD'.format(crypto.model, crypto.symbol),
                title_x=0.5,
                # width=1450,  # Change Chart width to Constant
                # height=800, # leave commented out for default column size
                margin=dict(l=0, r=0, t=50, b=30, pad=0),
                legend=dict(
                    x=0.45,
                    y=0.01,
                    # traceorder="normal",
                    # font=dict(size=15),
                    traceorder="reversed",
                    font=dict(
                        family="Courier",
                        # size=20,
                        color="black"
                    ),
                    bgcolor="LightSteelBlue",
                    bordercolor="White",
                    borderwidth=1
                ),
                # autosize=False,
                template="plotly_dark"

    )
    fig.add_vrect(x0=crypto.data.index[-1], x1=crypto.data.index[-1], \
        annotation_text='End of historical date')

    # column_graph = st.columns(1)
    # with column_graph:
    st.write(fig)


    csv = convert_df(crypto.data)


    st.download_button(
        label="Download Historical Dataset as CSV",
        data=csv,
        file_name='dataset.csv',
        mime='text/csv',
    )


    col1, col2, col3= st.columns(3)
    with col1:
        # Change the fear and grid to the last day of the dataset
        st.write(' ')
        last_date = pd.to_datetime(crypto.data.index[-1].replace('T00:00:00',''))
        last_date = '{dt.year}-{dt.month}-{dt.day}'.format(dt = last_date)
        st.image('https://alternative.me/images/fng/crypto-fear-and-greed-index-{}.png'.format(last_date))
    with col2:
        # Historical Data for the fear and greed
        st.write('### Why Measure Fear and Greed?')
        st.write('The fear and greed index is a composite score number that is calculated based on these 6 factors surrounding Bitcoin (corresponding weights inside parentheses): ')
        st.markdown('- volatility and maximum drawdown of Bitcoin (25%)\n- market momentum/volume (25%)\n- social media (15%)\n- surveys (15%)\n- dominance (10%)\n- trends (10%)')
        st.write('A zero score means `Extreme Fear`, while 100 means `Extreme Greed`. If the score is between 46 and 54, then the market is feeling neutral. In general, while the market is in `extreme fear` it signifies a good opportunity to **buy**. On the other hand, if the market is feeling extremely greedy (close to 100), then it signifies that there may be a correction (prices will come down) in the near future. Credits go to alternative.me for providing this index.')
        # st.write('The crypto market behaviour is very emotional. People tend to get greedy when the market is rising which results in **FOMO** (Fear of missing out). Also, people often sell their coins in irrational reaction of seeing red numbers. With our Fear and Greed Index, we try to save you from your own emotional overreactions. There are two simple assumptions:')
        # st.write('`Extreme fear` can be a sign that investors are too worried. That could be a buying opportunity.')
        # st.write('##### When Investors are getting too greedy, that means the market is due for a correction.')
        # st.write('Therefore, we analyze the current sentiment of the Bitcoin market and crunch the numbers into a simple meter from 0 to 100. Zero means `Extreme Fear`, while 100 means `Extreme Greed`. See below for further information on our data sources.')
    with col3:
        # Description for the fear and greed
        st.write('### {} Predictions'.format(MODEL))

        pred_df = crypto.pred.copy()
        pred_df = pred_df.applymap('{:,.2f}'.format)
        pred_df.columns = ['Predict','Min','Max']
        st.dataframe(pred_df)

        csv = convert_df(crypto.pred)

        st.download_button(
            label="Download Prediction data as CSV",
            data=csv,
            file_name='predictions.csv',
            mime='text/csv',
        )


def index_page():
    st.markdown("<h1 style='text-align: center; color: white;'>Welcome To Crypto Predict</h1>", unsafe_allow_html=True)
    # Welcome to the world of cryptocurrency prices forecasting. We are a group of crypto enthusiasts who just happen to also know about data science and machine learning. We created this app for users like you that are interested in cryptocurrency, and want to gather more information about some of the crypto assets in the top 100 (by market capitalization) so that you can make your own informed decisions. You can begin by selecting the crypto asset of interest and the corresponding forecast method in the left panel.
    st.markdown('Welcome to the world of cryptocurrency prices forecasting.\n We are a group of crypto enthusiasts who just happen to also know about data science and machine learning. We created this app for users like you that are interested in cryptocurrency, and want to gather more information about some of the crypto assets in the top 100 (by market capitalization) so that you can make your own informed decisions. You can begin by selecting the crypto asset of interest and the corresponding forecast method in the left panel.')
    st.markdown('')
    st.write('<iframe src="https://giphy.com/embed/7FBY7h5Psqd20" width="418" height="480" frameBorder="0" class="giphy-embed" allowFullScreen></iframe>',unsafe_allow_html=True)
    st.write('## Disclaimer: Not Financial Advice')
    st.write('Investment into cryptocurrency is inherently risky. You may lose all of your money. Do conduct your own due diligence and consult your financial advisor before making any investment decisions.')
    st.write('We are just providing informational and educational content and none of this material should be considered financial advice, investment advice, nor trading advice.')
    # Disclaimer: Not Financial Advice
    # Investment into cryptocurrency is inherently risky. You may lose all of your money. Do conduct your own due diligence and consult your financial advisor before making any investment decisions.
    # We are just providing informational and educational content and none of this material should be considered financial advice, investment advice, nor trading advice.
    pass

# ------ layout setting---------------------------
window_selection_c = st.sidebar.container() # create an empty container in the sidebar
# st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
state = window_selection_c.radio('',('Home', 'Predict'))

window_selection_c.markdown("## Currency Selection") # add a title to the sidebar container
# sub_columns = window_selection_c.columns(2) #Split the container into two columns for start and end date

# ----------Time window selection-----------------
YESTERDAY=datetime.date.today()-datetime.timedelta(days=1)

DEFAULT_START=YESTERDAY - datetime.timedelta(days=700)

# START = sub_columns[0].date_input("From", value=DEFAULT_START, max_value=YESTERDAY - datetime.timedelta(days=1))
# END = sub_columns[1].date_input("To", value=YESTERDAY, max_value=YESTERDAY, min_value=START)


# ---------------stock selection------------------

CRYPTO = np.array([ "BTC/USD", "ETH/USD", "LTC/USD"])
MODEL_LI = np.array(['FB_PROPHET', 'SARIMAX', 'LSTM'])
SYMB = window_selection_c.selectbox("Currency", CRYPTO)
MODEL = window_selection_c.selectbox("Model", MODEL_LI)
# window_selection_c.radio('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
# clicked = window_selection_c.button('Predict')


if state == 'Home':
    index_page()
    window_selection_c.write('### Instructions')
    window_selection_c.write('1. Select Currency to Predict')
    window_selection_c.write('2. Select Model to use to Predict')
    window_selection_c.write('3. Invest According to the Predictions')

else:
    predict()
    window_selection_c.write('### Instructions')
    window_selection_c.write('1. Select Currency to Predict')
    window_selection_c.write('2. Select Model to use to Predict')
    window_selection_c.write('3. Invest According to the Predictions')

# chart_width = st.expander(label="chart width").slider("", 1000, 2800, 1400) #arguments: label, min, default, max
