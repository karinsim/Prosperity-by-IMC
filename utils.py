import pandas as pd
import numpy as np
from datamodel import Trade
import re


def get_prices_log(infile, outfile):
    # extract price & pnl data from (end-of-round) log files

    fp = open(infile)
    start = False
    start_signal = "Activities log"
    end_signal = "Trade History"
    
    with open(outfile, "w") as f:
        for line in fp:

            if not start:
                if start_signal in line:
                    start = True

            elif start:
                if end_signal in line:
                    break
                
                f.write(line)
                
    f.close()
    fp.close()


def get_tradehistory(file):
    # extract trade history from raw log output
    fp = open(file)
    start = False
    cont = ["[", "]", "{"]
    hist = []
    empty = {}

    for line in fp:
        flag = False
        if start:
            for c in cont:
                if c in line:
                    flag = True
            
            if not flag:
                if "}" in line:
                    hist.append(empty)
                    empty = {}
                    continue
                
                item1 = line.split()[0][1:-2]
                item2 = line.split()[1].replace(",", "").replace('"', "")
                if item2.isnumeric():
                    item2 = int(item2)
                
                toadd = {item1: item2}
                empty.update(toadd)

        elif "Trade History" in line:
            start = True

    fp.close()
    
    return pd.DataFrame(hist)


def get_orderbook(file):
    orderbook = {}
    
    with open(file, "r") as f:
        log_entry = ""
        inside_braces = False
        
        for line in f:
            if "Activities log" in line:
                break  # Stop processing when reaching "Activities log"
            
            if "{" in line:
                inside_braces = True
                log_entry = line.strip()
            elif inside_braces:
                log_entry += " " + line.strip()
                if "}" in line:
                    inside_braces = False
                    
                    timestamp_match = re.search(r'"timestamp": (\d+)', log_entry)
                    lambda_log_match = re.search(r'"lambdaLog": "(.*?)"', log_entry)
                    
                    if timestamp_match and lambda_log_match:
                        timestamp = int(timestamp_match.group(1))
                        lambda_log = lambda_log_match.group(1)
                        
                        sell_match = re.search(r'sell \{(.*?)\}', lambda_log)
                        buy_match = re.search(r'buy \{(.*?)\}', lambda_log)
                        
                        sell_dict = {}
                        if sell_match:
                            sell_entries = sell_match.group(1).split(', ')
                            for entry in sell_entries:
                                if ':' in entry:
                                    k, v = entry.split(': ')
                                    sell_dict[int(k)] = int(v)
                        
                        buy_dict = {}
                        if buy_match:
                            buy_entries = buy_match.group(1).split(', ')
                            for entry in buy_entries:
                                if ':' in entry:
                                    k, v = entry.split(': ')
                                    buy_dict[int(k)] = int(v)
                        
                        orderbook[timestamp] = {"sell": sell_dict, "buy": buy_dict}
    
    return orderbook


def get_mytrades(hist, prod="RAINFOREST_RESIN"):
    """ extract own trades from a record of all market trades
    hist: dataframe returned by get_tradehistory()
    """ 
    mytrades = []
    
    for _, h in hist.iterrows():
        if h["symbol"] == prod:
            if h["seller"] == "SUBMISSION" or h["buyer"] == "SUBMISSION":
                mytrades.append(h)
    
    return pd.DataFrame(mytrades)


def get_pnl(mytrades, timestamps, market_price):
    # calculate pnl as a sum of realised and unrealised pnl; for one type of product only
    # takes as input a dataframe of own trades (obtained from Prosperity log files)

    open_buy = []       # tuple (price, quantity, unrealised pnl per quantity)
    open_sell = []
    pnl_realised = 0.
    pnls = []
    dt = timestamps[1] - timestamps[0]

    for timestamp in timestamps:

        if timestamp > 0:
            # check if there was a new trade in the previous timestep
            latest_trades = mytrades.loc[mytrades["timestamp"]== timestamp - dt]
            current_price = market_price.loc[market_price["timestamp"]==timestamp]["mid_price"].unique()[0]

            for _, trade in latest_trades.iterrows():
                if trade["buyer"] == "SUBMISSION":
                    open_buy.append({
                        "price": trade["price"],
                        "quantity": trade["quantity"],
                        "unrealised": current_price - trade["price"]
                    })
                elif trade["seller"] == "SUBMISSION":
                    open_sell.append({
                        "price": trade["price"],
                        "quantity": trade["quantity"],
                        "unrealised": trade["price"] - current_price
                    })

        # Realized PnL
        while len(open_buy) != 0 and len(open_sell) != 0:
            buy = open_buy[0]
            sell = open_sell[0]
            close_qty = min(buy["quantity"], sell["quantity"])
            pnl_realised += (sell["price"] - buy["price"]) * close_qty
            buy["quantity"] -= close_qty
            sell["quantity"] -= close_qty

            if buy["quantity"] == 0:
                open_buy = open_buy[1:]
            if sell["quantity"] == 0:
                open_sell = open_sell[1:]
        
        pnl_unrealised = 0.
        if len(open_buy) == 0 and len(open_sell) != 0:
            for sell in open_sell:
                pnl_unrealised += sell["unrealised"] * sell["quantity"]
        elif len(open_sell) == 0 and len(open_buy) != 0:
            for buy in open_buy:
                pnl_unrealised += buy["unrealised"] * buy["quantity"]
              
        pnls.append(pnl_realised + pnl_unrealised)
    
    return pnls


def aggregate_trades(trades):
    """
    Utility function to aggregate all trades with the same nature (buy/sell) and the same price into one entry.
    """

    buys, sells = [], []
    buyprices, sellprices = [], []

    for trade in trades:
        if trade.buyer == "SUBMISSION":
            if trade.price in buyprices:
                ind = np.where(np.array(buyprices)==trade.price)[0][0]
                updated_vol = trade.quantity + buys[ind].quantity
                buys[ind] = Trade(
                            symbol=trade.symbol,
                            price=trade.price,
                            quantity=updated_vol,
                            buyer="SUBMISSION",
                            seller="",
                            timestamp=trade.timestamp)
            else:
                buys.append(trade)
                buyprices.append(trade.price)
        
        elif trade.seller == "SUBMISSION":
            if trade.price in sellprices:
                ind = np.where(np.array(sellprices)==trade.price)[0][0]
                updated_vol = trade.quantity + sells[ind].quantity
                sells[ind] = Trade(
                            symbol=trade.symbol,
                            price=trade.price,
                            quantity=updated_vol,
                            buyer="",
                            seller="SUBMISSION",
                            timestamp=trade.timestamp)
            else:
                sells.append(trade)
                sellprices.append(trade.price)
       
    return buys + sells


def get_midprice_mm(df1):
    """
    Extract the true midprice based on a market maker placing larger orders on both sides (e.g. KELP)
    """
    fp = []
    for _, df in df1.iterrows():
        bid_vol = list(df[["bid_volume_1", "bid_volume_2", "bid_volume_3"]])
        where = np.nanargmax(bid_vol)
        bid = list(df[["bid_price_1", "bid_price_2", "bid_price_3"]])[where]
        ask_vol = list(df[["ask_volume_1", "ask_volume_2", "ask_volume_3"]])
        where = np.nanargmax(ask_vol)
        ask = list(df[["ask_price_1", "ask_price_2", "ask_price_3"]])[where]
        fp.append((bid+ask)/2)
    
    return np.array(fp)


