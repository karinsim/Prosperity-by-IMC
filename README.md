# Prosperity3# 


###utils.py###
Contains utility functions to analyse the log output:
\begin{enumerate}
\item get_tradehistory: extracts all trades from the log file (including trades by other market participants) and returns a dataframe
\item get_orderbook: returns a dictionary of dictionaries; each dictionary corresponds to the buy and sell orders the algorithm sees at the corresponding timestamp (the price csv only includes top 3 orders, but the live order book can include more than that)
\item get_mytrades: returns a dataframe of own trades only
\item get_pnl: calculate the pnl given the trade history; reproduces the Prosperity pnl closely
\item get_midprice_mm: calculates the midprice corresponding to the market-maker quotes (relevant for KELP)
\end{enumerate}



