import streamlit as st
import pandas as pd
from utils import get_midprice_mm, get_tradehistory, get_mytrades


# prices = pd.read_csv("data/round-1/log1-prices.txt", delimiter=';')
# prices = pd.read_csv("data/round-3/backtest.csv", delimiter=';')
# prices = prices.loc[prices["day"]==2]
# df = prices.loc[prices["product"]=="PICNIC_BASKET2"]
# df.loc[:, "mid_price"] = get_midprice_mm(df)


def get_trades(infile, prod, mp=None):
    hist = get_tradehistory(infile)
    myhist = get_mytrades(hist, prod)
    if mp is not None:
        myhist = myhist.merge(mp[["timestamp", "mid_price"]], on="timestamp", how="left")

    return myhist

# df1 = get_trades("data/round-3/current.log", "PICNIC_BASKET2")
# df2 = get_trades("data/round-3/current.log", "CROISSANTS")
# df3 = get_trades("data/round-3/current.log", "JAMS")
# df4 = get_trades("data/round-3/current.log", "DJEMBES")
# df = pd.concat((df1, df2, df3)).sort_values("timestamp")

df = get_trades("data/round-4/current.log", "MAGNIFICENT_MACARONS")
st.title("PRODUCT Trade History")
st.dataframe(df)

pnl_current = pd.read_csv("data/round-4/current.csv", sep=";")
df = pnl_current.loc[pnl_current["product"]=="MAGNIFICENT_MACARONS"][["timestamp", "mid_price", "profit_and_loss"]]
st.dataframe(df)



