import streamlit as st
import pandas as pd
from utils import get_midprice_mm, get_tradehistory, get_mytrades


# prices = pd.read_csv("data/round-1/log1-prices.txt", delimiter=';')
prices = pd.read_csv("data/round-2/prices_round_2_day_1.csv", delimiter=';')
prices = prices.loc[prices["day"]==0]
df = prices.loc[prices["product"]=="PICNIC_BASKET2"]
df.loc[:, "mid_price"] = get_midprice_mm(df)


def get_trades(infile, prod, mp=None):
    hist = get_tradehistory(infile)
    myhist = get_mytrades(hist, prod)
    if mp is not None:
        myhist = myhist.merge(mp[["timestamp", "mid_price"]], on="timestamp", how="left")

    return myhist


# st.title("RESIN Trade History (OWN)")
# st.dataframe(get_trades("data/round-1/round1_final.log", "RAINFOREST_RESIN"))

# st.title("KELP Trade History (OWN)")
# st.dataframe(get_trades("data/round-1/current.log", "KELP", kelp))

st.title("PRODUCT Trade History (OWN)")
st.dataframe(get_trades("data/round-2/basket.log", "PICNIC_BASKET2", df))


