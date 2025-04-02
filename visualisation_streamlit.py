import streamlit as st
import pandas as pd
from utils import get_midprice_mm, get_orderbook, get_tradehistory, get_mytrades


price = pd.read_csv("data/ericliu.csv", delimiter=";")
kelp = price.loc[price["product"]=="KELP"]
mp = get_midprice_mm(kelp)
kelp["mid_price"] = mp
# get order book
orderbook = get_orderbook("data/orderbook.log")
# update true fairprice based on live orderbook
for time in [57900, 82600, 130200]:
    selldict = orderbook[time]["sell"]
    buydict = orderbook[time]["buy"]
    trueask = min(selldict, key=selldict.get)
    truebid = max(buydict, key=buydict.get)
    kelp.loc[kelp["timestamp"] == time, "mid_price"] = (trueask+truebid)/2

# the entire trade history (not just own trades)
infile = "data/ericliu.log"
hist = get_tradehistory(infile)
# resin_hist = get_mytrades(hist)
# resin_hist = resin_hist.merge(kelp[["timestamp", "mid_price"]], on="timestamp", how="left")
kelp_hist = get_mytrades(hist, "KELP")
kelp_hist = kelp_hist.merge(kelp[["timestamp", "mid_price"]], on="timestamp", how="left")

infile2 = "data/current.log"
hist2 = get_tradehistory(infile2)
kelp_hist2 = get_mytrades(hist2, "KELP")
kelp_hist2 = kelp_hist2.merge(kelp[["timestamp", "mid_price"]], on="timestamp", how="left")


# calculate open positions
kelp_hist_copy = kelp_hist.copy()
arr = kelp_hist_copy.loc[kelp_hist_copy["seller"]=="SUBMISSION"]["quantity"].array * -1
kelp_hist_copy.loc[kelp_hist_copy["seller"]=="SUBMISSION", "quantity"] = arr
kelp_hist["position"] = kelp_hist_copy["quantity"].cumsum()

st.title("Kelp Data")
st.dataframe(kelp[["timestamp", "mid_price", "profit_and_loss"]])

st.title("Kelp Trade History (ERIC)")
st.dataframe(kelp_hist)
st.title("Kelp Trade History (OWN)")
st.dataframe(kelp_hist2)


