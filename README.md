# Prosperity3


utils.py
Contains utility functions to analyse the log output:

1. get_tradehistory: extracts all trades from the log file (including trades by other market participants) and returns a dataframe
2. get_orderbook: returns a dictionary of dictionaries; each dictionary corresponds to the buy and sell orders the algorithm sees at the corresponding timestamp (the price csv only includes top 3 orders, but the live order book can include more than that)
3. get_mytrades: returns a dataframe of own trades only
4. get_pnl: calculate the pnl given the trade history; reproduces the Prosperity pnl closely
5. get_midprice_mm: calculates the midprice corresponding to the market-maker quotes (relevant for KELP, SQUID)



