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


def get_mytrades(hist, prod="RAINFOREST_RESIN"):
    """ extract own trades from a record of all market trades
    hist: dataframe returned by get_tradehistory()
    """ 
    mytrades = []
    pos = 0
    
    for _, h in hist.iterrows():
        df = h.copy()
        if df["symbol"] == prod:
            if df["seller"] == "SUBMISSION":
                pos -= df["quantity"]
                df["position"] = pos
                mytrades.append(df)
            elif df["buyer"] == "SUBMISSION":
                pos += df["quantity"]
                df["position"] = pos
                mytrades.append(df)

    mytrades = pd.DataFrame(mytrades) 
    # mytrades.drop("symbol", axis=1, inplace=True)
    mytrades.drop("currency", axis=1, inplace=True)        
    
    return mytrades


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


def get_midprice_mm(df1):
    """
    Extract the true midprice based on a market maker placing larger orders on both sides (e.g. KELP)
    """
    fp = []
    for _, df in df1.iterrows():       
        bid_vol = list(df[["bid_volume_1", "bid_volume_2", "bid_volume_3"]])
        if np.isnan(bid_vol).all():
            fp.append(df["mid_price"])
            continue
        where = np.nanargmax(bid_vol)
        bid = list(df[["bid_price_1", "bid_price_2", "bid_price_3"]])[where]
        ask_vol = list(df[["ask_volume_1", "ask_volume_2", "ask_volume_3"]])
        if np.isnan(ask_vol).all():
            fp.append(df["mid_price"])
            continue
        where = np.nanargmax(ask_vol)
        ask = list(df[["ask_price_1", "ask_price_2", "ask_price_3"]])[where]
        fp.append((bid+ask)/2)
    
    return np.array(fp)


def find_signal_zscore(history, window, mult, mult_strong, hits, strong_hits):
        
    """
    Find trading signals using mean-reversion by tracking Z scores. 
    Hits: number of consecutive hits on the upper band needed to constitute a signal
    Strong hits: number of consecutive signals to constitute a strong signal
    """

    sells, buys, strong_sells, strong_buys = [], [], [], []

    if len(history) < window:
        return sells, buys, strong_sells, strong_buys

    # Mean reversion strategy
    sma = np.convolve(history, np.ones(window)/window, mode='valid')
    std = np.array([np.std(history[i:i+window]) for i in range(len(history) - window + 1)])
    upper, lower, signal = 0, 0, 0

    for i in range(window-1, len(history)):
        
        z = ((history[i] - sma[i-(window-1)]) / std[i-(window-1)] if std[i-(window-1)] > 0 else None)

        if z is None:
            continue
        
        if np.abs(z) >= mult_strong:
            if z > 0:
                strong_sells.append(i)
            else:
                strong_buys.append(i)
            continue

        if z < -mult:
            lower += 1
            upper = 0
        elif z > mult:
            upper += 1
            lower = 0
        else:
            upper = lower = 0
        
        if upper >= hits:      # sell signal
            if signal >= 0:
                signal = -1
                sells.append(i)
            else:
                signal -= 1
            # upper = lower = 0

        elif lower >= hits:        # buy signal
            if signal > 0:
                signal += 1
            else:
                signal = 1
                buys.append(i)
            # upper = lower = 0

        if signal == -strong_hits:      # strong sell; reset signal and counter
            strong_sells.append(i)
            signal = upper = lower = 0
            
        elif signal == strong_hits:        # strong buy; reset signal and counter
            strong_buys.append(i)
            signal = upper = lower = 0

    return sells, buys, strong_sells, strong_buys


def find_signal_momentum(prices, lookback=5, threshold=0.001, strong_threshold=0.004,
                         hits=1, strong_hits=10):

    upper, lower, signal = 0, 0, 0

    buys, sells, strongbuys, strongsells = [], [], [], []

    for i in range(lookback, len(prices)):
        returns = (prices[i] - prices[i-lookback]) / prices[i]

        if returns < -strong_threshold:
            strongbuys.append(i)
            upper, lower, signal = 0, 0, 0
            continue

        elif returns > strong_threshold:
            strongsells.append(i)
            upper, lower, signal = 0, 0, 0
            continue
        
        else:
            if returns < -threshold:
                lower += 1
                upper = 0
            elif returns > threshold:
                upper += 1
                lower = 0
            else:
                upper = lower = 0
    
        if upper >= hits:      # sell signal
            if signal >= 0:
                signal = -1
                sells.append(i)
            else:
                signal -= 1
            upper = lower = 0

        elif lower >= hits:        # buy signal
            if signal > 0:
                signal += 1
            elif signal < 0:
                signal = 1
                buys.append(i)
            upper = lower = 0

        if signal == -strong_hits:      # strong sell; reset signal and counter
            strongsells.append(i)
            signal = upper = lower = 0
            
        elif signal == strong_hits:        # strong buy; reset signal and counter
            strongbuys.append(i)
            signal = upper = lower = 0

    return sells, buys, strongsells, strongbuys


def find_signal_breakout(prices, lookback=20, lookback_strong=100, hits=1, strong_hits=10):
    upper, lower, signal = 0, 0, 0
    buys, sells, strongbuys, strongsells = [], [], [], []

    for i in range(lookback, len(prices)):

        if i >= lookback_strong:
            high = np.max(prices[i - lookback_strong:i])
            low = np.min(prices[i - lookback_strong:i])
            if prices[i] > high:
                strongsells.append(i)
                continue
            elif prices[i] < low:
                strongbuys.append(i)
                continue

        high = np.max(prices[i - lookback:i])
        low = np.min(prices[i - lookback:i])

        if prices[i] > high:
            upper += 1
            lower = 0
        elif prices[i] < low:
            lower += 1
            upper = 0
        else:
            upper = lower = 0
        
        if upper >= hits:      # sell signal
            if signal >= 0:
                signal = -1
                sells.append(i)
            else:
                signal -= 1
            upper = lower = 0

        elif lower >= hits:        # buy signal
            if signal > 0:
                signal += 1
            elif signal < 0:
                signal = 1
                buys.append(i)
            upper = lower = 0

        if signal == -strong_hits:      # strong sell; reset signal and counter
            strongsells.append(i)
            signal = upper = lower = 0
            
        elif signal == strong_hits:        # strong buy; reset signal and counter
            strongbuys.append(i)
            signal = upper = lower = 0

    return sells, buys, strongsells, strongbuys


def find_spread(PB, contents, basket=1):
    spread = []

    if basket == 1:
        CROSS, JAM, DJ = contents
        for (_, row1), (_, row2), (_, row3), (_, row4) in zip(CROSS.iterrows(), JAM.iterrows(), DJ.iterrows(), PB.iterrows()):

            implied_bid = 6 * row1["bid_price_1"] + 3 * row2["bid_price_1"] + row3["bid_price_1"]
            implied_ask = 6 * row1["ask_price_1"] + 3 * row2["ask_price_1"] + row3["ask_price_1"]
            implied_ask_vol = min([row1["ask_volume_1"] // 6, row2["ask_volume_1"] // 3, row3["ask_volume_1"]])
            implied_bid_vol = min([row1["bid_volume_1"] // 6, row2["bid_volume_1"] // 3, row3["bid_volume_1"]])

            basket_bid = row4["bid_price_1"]
            basket_ask = row4["ask_price_1"]
            basket_ask_vol = row4["ask_volume_1"]
            basket_bid_vol = row4["bid_volume_1"]

            spread.append((basket_bid * basket_ask_vol + basket_ask * basket_bid_vol) / (basket_bid_vol + basket_ask_vol) -
                                (implied_bid * implied_ask_vol + implied_ask * implied_bid_vol) / (implied_bid_vol + implied_ask_vol))
    elif basket == 2:
        CROSS, JAM = contents
        for (_, row1), (_, row2), (_, row3) in zip(CROSS.iterrows(), JAM.iterrows(), PB.iterrows()):

            implied_bid = 4 * row1["bid_price_1"] + 2 * row2["bid_price_1"] 
            implied_ask = 4 * row1["ask_price_1"] + 2 * row2["ask_price_1"]
            implied_ask_vol = min([row1["ask_volume_1"] // 4, row2["ask_volume_1"] // 2])
            implied_bid_vol = min([row1["bid_volume_1"] // 4, row2["bid_volume_1"] // 2])

            basket_bid = row3["bid_price_1"]
            basket_ask = row3["ask_price_1"]
            basket_ask_vol = row3["ask_volume_1"]
            basket_bid_vol = row3["bid_volume_1"]

            spread.append((basket_bid * basket_ask_vol + basket_ask * basket_bid_vol) / (basket_bid_vol + basket_ask_vol) -
                                (implied_bid * implied_ask_vol + implied_ask * implied_bid_vol) / (implied_bid_vol + implied_ask_vol))
    else:
        return None

    return np.array(spread)



