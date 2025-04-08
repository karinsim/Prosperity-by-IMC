import streamlit as st
import pandas as pd
from utils import get_midprice_mm, get_tradehistory, get_mytrades


prices = pd.read_csv("data/round-1/log1-prices.txt", delimiter=';')
kelp = prices.loc[prices["product"]=="KELP"]
kelp["mid_price"] = get_midprice_mm(kelp)
# squid = prices.loc[prices["product"]=="SQUID_INK"]
# squid["mid_price"] = get_midprice_mm(squid)


def get_trades(infile, prod, mp=None):
    hist = get_tradehistory(infile)
    myhist = get_mytrades(hist, prod)
    if mp is not None:
        myhist = myhist.merge(mp[["timestamp", "mid_price"]], on="timestamp", how="left")

    return myhist


# st.title("RESIN Trade History (OWN)")
# st.dataframe(get_trades("data/round-1/log1.log", "RAINFOREST_RESIN"))
# st.title("RESIN Trade History (ERIC)")
# st.dataframe(get_trades("data/round-1/ericliu.log", "RAINFOREST_RESIN"))

st.title("KELP Trade History (OWN)")
st.dataframe(get_trades("data/round-1/kelp1.log", "KELP", kelp))
st.title("KELP Trade History (ERIC)")
st.dataframe(get_trades("data/round-1/ericliu.log", "KELP", kelp))


