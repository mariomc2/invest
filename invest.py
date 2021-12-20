import copy
import pandas as pd
import numpy as np
import pandas_datareader as pdr
from pandas_datareader._utils import RemoteDataError
import streamlit as st
from streamlit_tags import st_tags, st_tags_sidebar
from datetime import date, timedelta
import seaborn as sns
import matplotlib.pyplot as plt

DAYS_YEAR = 250

@st.cache(allow_output_mutation=True, show_spinner=False)
def get_price(ticker, start, end):
    '''
    Get data from Yahoo!
    '''
    try:
        data = pdr.DataReader(ticker, "yahoo", start=start, end=end)
        data.columns.names = (None,None)
    except RemoteDataError:
        return None

    return data["Adj Close"]

def load_prices(start, end):
    '''
    Load prices sequnetially and construct a dataset
    '''
    data = pd.DataFrame()
    for i, ticker in enumerate(st.session_state.tickers):
        with st.spinner("Fetching " + ticker):
            price = copy.deepcopy(get_price([ticker], start, end))
            # Check if response is a DataFrame
            if isinstance(price, pd.DataFrame):
                if data.empty:
                    data = price
                else:
                    data[ticker] = price
    # Normalize if more than 1 ticker
    if data.shape[1] >1:
            data *= 100 / data.iloc[0]

    return data

def plot_prices(prices, returns):
    st.write("""
    ## Historical Prices
    """)
    st.line_chart(prices)

    if st.session_state.show_correl:
        st.write("""
            ## Historical Correlations
            """)
        with st.spinner("Plotting Correlations"):
            st.pyplot(sns.pairplot(returns))


def get_tickers():
    '''
    Show the tag widget
    '''
    st_tags_sidebar(label="Tickers:",
                    text="Press enter to add more",
                    value=["EWJ", "GLD", "IEI", "IGOV", "LQD", "QQQ", "SPDW", "SPEM", "TIP", "VOO"],
                    suggestions=["EWJ", "GLD", "IEI", "IGOV", "LQD", "QQQ", "SPDW", "SPEM", "TIP", "VOO"],
                    maxtags=20,
                    key="tickers") 

def sim_portfolios(returns, weights_dict, normalise):
    '''
    Simulate portfolios
    '''
    # Calculate number of days to simulate
    days = round(DAYS_YEAR * st.session_state.years)
    num_sims = st.session_state.num_sims

    # Extract the weights
    weights = np.array(list(map(lambda x: np.float64(weights_dict[x]), returns.columns)))
    if normalise:
        weights /= weights.sum()
    else:
        weights /= 100.0
    weights = np.tile(weights, (num_sims, days,1))

    # Select random days to simulate returns of the same distribution and correlation
    random_returns_idx = np.random.RandomState(0).randint(low=0, high=returns.shape[0], size=(num_sims, days))
    returns_sims = returns.values[random_returns_idx]

    # Calculate portfolio returns
    returns_portfolios = np.zeros((num_sims, days))
    returns_portfolios = (returns_sims * weights).sum(axis=2)
    returns_portfolios = np.insert(returns_portfolios, 0, 0, axis=1)

    # Calculate portfolio levels
    portfolios = 100 * (1+returns_portfolios).cumprod(axis=1)

    return portfolios


st.sidebar.number_input("Annualised target return as percetange",
                        min_value=1, max_value=50, value=5, step=1, format="%i", key="target")
st.sidebar.number_input("Number of years for the investment",
                        min_value=1, max_value=20, value=10, step=1, format="%i", key="years")
st.sidebar.number_input("Number of simulations",
                        min_value=1, max_value=10000, value=1000, step=1, format="%i", key="num_sims")

# Dates
today = date.today()
look_back = timedelta(days=365*5)
start = st.sidebar.date_input("Start Date",value=today-look_back, max_value=today, key="start")
end = st.sidebar.date_input("End Date", value=today, max_value=today, key="end")


get_tickers()

prices = load_prices(start=start, end=end)
returns = prices.pct_change().dropna()

no_data = np.array(st.session_state.tickers)[np.isin(st.session_state.tickers, prices.columns, invert=True)]
if no_data.size>0:
    text = """
    ## Tickers with no price data

    """
    for item in no_data:
        text+= "\n- "+item
    st.sidebar.warning(text)

st.sidebar.checkbox("Visualise Correlations", key="show_correl")

# if "tickers" not in st.session_state:
#     st.session_state.tickers = ["EWJ", "GLD", "IEI", "IGOV", "LQD", "QQQ", "SPDW", "SPEM", "TIP", "VOO"]
    
st.write("""
# Investment Simulation App
""")

plot_prices(prices, returns)

# Weights
portfolios = np.empty((0, 0))
with st.form("weights"):
    st.write("## Select the weights for each asset")
    weights_dict = dict()
    for ticker in prices.columns:
        weights_dict[ticker] = st.slider(ticker, min_value=0, max_value=100, value=10, step=1, format="%i")

    normalise = st.checkbox("Do you want to normalise the weights to add up to 100%?", value=True)

    submitted = st.form_submit_button("Submit")

    if submitted:
        st.write("Sum of the weights", sum(weights_dict.values()), "Normalise", normalise)
        portfolios = sim_portfolios(returns, weights_dict, normalise)

if portfolios.size > 0:
    p_num = 20
    p_idx = np.random.RandomState(1).randint(low=0, high=portfolios.shape[0], size=p_num)

    plot_df = pd.DataFrame(portfolios[p_idx].T, 
                index=np.arange(portfolios.shape[1])/DAYS_YEAR,
                columns=list(map(lambda x: "sim {:02d}".format(int(x)+1), np.arange(p_num))))

    st.write("""
            ## Simulations
            Note: This assumes that the distributions of returns is going to be the same as seen
            in the range of selected dates.
            
            Note: **Past performance is no guarantee of future results.**
            """)
    st.line_chart(plot_df)

    target_df = pd.DataFrame(100 * np.insert(np.zeros(portfolios.shape[1]-1) + 
                                    np.power(1 + st.session_state.target/100,
                                            1/DAYS_YEAR), 0, 1).cumprod(),
                                index= plot_df.index, columns=["Target"])
    
    st.write("""
            ### Target Path
            """)
    st.line_chart(target_df)

    prob_target = pd.DataFrame((portfolios >= target_df.values.T).sum(axis=0)[1:]/portfolios.shape[0],
                                index=plot_df.index[1:], columns=["Probability"])

    st.write("""
            ### Probability of Being Above the Target
            """)
    st.line_chart(prob_target)
