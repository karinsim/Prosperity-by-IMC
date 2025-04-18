from datamodel import OrderDepth, UserId, TradingState, Order, Trade, Observation
import jsonpickle
from typing import List
import string
import numpy as np
import json
import math


def black_scholes_call_price(S, K, T, sigma,r):
        """
        Computes the Black-Scholes price for a call option.
        S: Underlying price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free rate
        sigma: Volatility
        """
        d1 = (math.log(S/K) + (r+ 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        
        # Using the error function to get the standard normal CDF
        N_d1 = 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
        N_d2 = 0.5 * (1 + math.erf(d2 / math.sqrt(2)))
        return S * N_d1 - K * N_d2*math.exp(-r*T)


 # Regression-based volatility estimator
def estimated_volatility(S, K, a=0.150195656, b=-1.15870319e-06, c=1.23214223e-07):
    """
    Computes volatility from the linear regression equation.
    x is defined as the difference between the underlying price and the strike price.
    """
    x = S - K
    return a + b * x + c * x**2


class Trader:
    def __init__(self):

        print("Trader initialised!")

        # set lower limits to reduce exposure?
        self.POS_LIM = {"RAINFOREST_RESIN": 50, "KELP": 50, "SQUID_INK": 50,
                        "CROISSANTS": 250, "JAMS": 350, "DJEMBES": 60,
                        "PICNIC_BASKET1": 60, "PICNIC_BASKET2": 100,
                        "VOLCANIC_ROCK": 400,
                        "VOLCANIC_ROCK_VOUCHER_9500": 200,
                        "VOLCANIC_ROCK_VOUCHER_9750": 200,
                        "VOLCANIC_ROCK_VOUCHER_10000": 200,
                        "VOLCANIC_ROCK_VOUCHER_10250": 200,
                        "VOLCANIC_ROCK_VOUCHER_10500": 200,
                        "MAGNIFICENT_MACARONS": 75}
        self.prods = list(self.POS_LIM.keys())
        self.synthetic = {"PICNIC_BASKET1": "SPREAD1", "PICNIC_BASKET2": "SPREAD2"}
        self.prods += list(self.synthetic.values())
        self.basket_contents={"PICNIC_BASKET1": {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1},
                              "PICNIC_BASKET2": {"CROISSANTS": 4, "JAMS": 2}}

        # for tracking open positions / updating memory
        self.timer = 0
        self.open_buys = {prod: {} for prod in self.prods}
        self.open_sells = {prod: {} for prod in self.prods}
        # self.recorded_time = {prod: -1 for prod in self.prods}    # last recorded time of own_trades

        # historical data and past signals for Z-score trading signals
        self.history = {prod: [] for prod in self.prods}
        self.upper = {prod: 0 for prod in self.prods}
        self.lower = {prod: 0 for prod in self.prods}
        self.signal = {prod: 0 for prod in self.prods}
        self.strong_signal = {prod: 0 for prod in self.prods}


    def get_synthetic_trades(self, state: TradingState):
        for basket in self.basket_contents:
            if basket not in state.own_trades:
                continue
            spread = self.synthetic[basket]
            basket_trades = [trade for trade in state.own_trades[basket] 
                        if trade.timestamp == state.timestamp - 100]
            basket_bought = [trade for trade in basket_trades if trade.buyer=="SUBMISSION"]
            basket_sold = [trade for trade in basket_trades if trade.seller=="SUBMISSION"]
            weights = self.basket_contents[basket]
            bought_qty = {basket: sum(trade.quantity for trade in basket_bought)}
            sold_qty = {basket: sum(trade.quantity for trade in basket_sold)}
            contents_bought = {}
            contents_sold = {}
            for prod in weights:
                if prod not in state.own_trades:
                    continue
                contents_bought[prod] = [trade for trade in state.own_trades[prod] 
                                          if (trade.timestamp == state.timestamp - 100) 
                                          and (trade.buyer == "SUBMISSION")]
                contents_sold[prod] = [trade for trade in state.own_trades[prod] 
                                          if (trade.timestamp == state.timestamp - 100) 
                                          and (trade.seller == "SUBMISSION")]
                bought_qty[prod] = sum(trade.quantity for trade in contents_bought[prod])
                sold_qty[prod] = sum(trade.quantity for trade in contents_sold[prod])
            
            full_basket_bought = 0
            max_possible = bought_qty.get(basket, 0)
            for _ in range(max_possible):
                can_fill = True
                for prod, weight in weights.items():
                    if sold_qty.get(prod, 0) < weight:
                        can_fill = False
                        break
                if can_fill:
                    full_basket_bought += 1
                    for prod, weight in weights.items():
                        sold_qty[prod] -= weight
                    bought_qty[basket] -= 1
                else:
                    break
            full_basket_sold = 0
            max_possible = sold_qty.get(basket, 0)
            for _ in range(max_possible):
                can_fill = True
                for prod, weight in weights.items():
                    if bought_qty.get(prod, 0) < weight:
                        can_fill = False
                        break
                if can_fill:
                    full_basket_sold += 1
                    for prod, weight in weights.items():
                        bought_qty[prod] -= weight
                    sold_qty[basket] -= 1
                else:
                    break
            
            state.own_trades[spread] = []
            used_contents = {prod: {"index": 0, "remaining": 0} for prod in weights}
            basket_idx = 0
            basket_remaining = 0
            for _ in range(full_basket_bought):
                while basket_idx < len(basket_bought) and basket_remaining == 0:
                    basket_remaining = basket_bought[basket_idx].quantity
                if basket_idx >= len(basket_bought):
                    break 
                basket_trade = basket_bought[basket_idx]
                basket_remaining -= 1
                if basket_remaining == 0:
                    basket_idx += 1
                synthetic_price = 0
                valid = True
                for prod, weight in weights.items():
                    trades = contents_sold.get(prod, [])
                    needed_qty = weight
                    track = used_contents[prod]
                    j = track["index"]
                    rem = track["remaining"]
                    while j < len(trades) and needed_qty > 0:
                        trade = trades[j]
                        available = rem if rem > 0 else trade.quantity
                        use_qty = min(needed_qty, available)
                        synthetic_price += trade.price * use_qty
                        needed_qty -= use_qty
                        available -= use_qty
                        if available == 0:
                            j += 1
                            rem = 0
                        else:
                            rem = available
                    if needed_qty > 0:
                        valid = False
                        break
                    used_contents[prod] = {"index": j, "remaining": rem}
                if valid:
                    spread_price = basket_trade.price - synthetic_price
                    state.own_trades[spread].append(
                        Trade(spread, spread_price, 1, "SUBMISSION", None, basket_trade.timestamp)
                    )
            basket_left = []
            contents_left = {}
            if basket_idx < len(basket_bought):
                if basket_remaining > 0:
                    leftover_trade = basket_bought[basket_idx]
                    basket_left.append(Trade(basket, leftover_trade.price, basket_remaining,
                                            leftover_trade.buyer, leftover_trade.seller, leftover_trade.timestamp))
                    basket_idx += 1
                if basket_idx < len(basket_bought):
                    basket_left.extend(basket_bought[basket_idx:])
            for prod, track in used_contents.items():
                trades = contents_sold.get(prod, [])
                prod_left = []
                j = track["index"]
                rem = track["remaining"]
                if j < len(trades):
                    if rem > 0:
                        leftover_trade = trades[j]
                        prod_left.append(Trade(prod, leftover_trade.price, rem,
                                            leftover_trade.buyer, leftover_trade.seller, leftover_trade.timestamp))
                        j += 1
                    prod_left.extend(trades[j:])
                contents_left[prod] = prod_left

            used_contents = {prod: {"index": 0, "remaining": 0} for prod in weights}
            basket_idx = 0
            basket_remaining = 0
            for _ in range(full_basket_sold):
                while basket_idx < len(basket_sold) and basket_remaining == 0:
                    basket_remaining = basket_sold[basket_idx].quantity
                if basket_idx >= len(basket_sold):
                    break 
                basket_trade = basket_sold[basket_idx]
                basket_remaining -= 1
                if basket_remaining == 0:
                    basket_idx += 1
                synthetic_price = 0
                valid = True
                for prod, weight in weights.items():
                    trades = contents_bought.get(prod, [])
                    needed_qty = weight
                    track = used_contents[prod]
                    j = track["index"]
                    rem = track["remaining"]
                    while j < len(trades) and needed_qty > 0:
                        trade = trades[j]
                        available = rem if rem > 0 else trade.quantity
                        use_qty = min(needed_qty, available)
                        synthetic_price += trade.price * use_qty
                        needed_qty -= use_qty
                        available -= use_qty
                        if available == 0:
                            j += 1
                            rem = 0
                        else:
                            rem = available
                    if needed_qty > 0:
                        valid = False
                        break
                    used_contents[prod] = {"index": j, "remaining": rem}
                if valid:
                    spread_price = basket_trade.price - synthetic_price
                    state.own_trades[spread].append(
                        Trade(spread, spread_price, 1, None, "SUBMISSION", basket_trade.timestamp)
                    )
            if basket_idx < len(basket_sold):
                if basket_remaining > 0:
                    leftover_trade = basket_sold[basket_idx]
                    basket_left.append(Trade(basket, leftover_trade.price, basket_remaining,
                                            leftover_trade.buyer, leftover_trade.seller, leftover_trade.timestamp))
                    basket_idx += 1
                if basket_idx < len(basket_sold):
                    basket_left.extend(basket_sold[basket_idx:])
            for prod, track in used_contents.items():
                trades = contents_bought.get(prod, [])
                prod_left = []
                j = track["index"]
                rem = track["remaining"]
                if j < len(trades):
                    if rem > 0:
                        leftover_trade = trades[j]
                        prod_left.append(Trade(prod, leftover_trade.price, rem,
                                            leftover_trade.buyer, leftover_trade.seller, leftover_trade.timestamp))
                        j += 1
                    prod_left.extend(trades[j:])
                if prod in contents_left:
                    contents_left[prod] += prod_left
                else:
                    contents_left[prod] = prod_left

            if len(basket_left) > 0:
                state.own_trades[basket] = basket_left
            for prod in weights:
                if len(contents_left[prod]) > 0:
                    state.own_trades[prod] = contents_left[prod]


    def correct_macaron_conversion(self, state: TradingState):
        # assume that conversion is always in the opposite direction
        # currently only selling in the local exchange, no buying locally
        prod = "MAGNIFICENT_MACARONS"
        current_pos = state.position.get(prod, 0)

        if len(self.open_sells[prod]) == 0:
            return None

        if len(self.open_sells[prod]) > 0 and current_pos <= 0:
            sold_qty = sum(self.open_sells[prod].values())
            if current_pos == -sold_qty:    # nothing converted
                return None     
            # -sold_qty < current_pos <= 0
            converted = sold_qty + current_pos
            sorted_items = sorted(self.open_sells[prod].items())
            for price, volume in sorted_items:
                if converted >= volume:
                    converted -= volume
                    del self.open_sells[prod][price]
                else:
                    self.open_sells[prod][price] -= converted
                    break
    
        return None


    def update_open_pos(self, state: TradingState):
        """
        Update open positions according to updated own trades
        Later try to buy/sell lower/higher than open trades
        """

        self.get_synthetic_trades(state)
        mac = "MAGNIFICENT_MACARONS"

        for prod in state.own_trades:
            trades = state.own_trades[prod]
            trades = [trade for trade in trades if trade.timestamp == state.timestamp - 100]
            for trade in trades:
                remaining_quantity = trade.quantity
                if trade.buyer == "SUBMISSION":
                    sold_price = sorted(list(self.open_sells[prod].keys()), reverse=True)
                    # match with currently open positions
                    for price in sold_price:  
                        if trade.price >= price: 
                            break  
                        if remaining_quantity <= 0:
                            break  
                        
                        if price in self.open_sells[prod]:
                            available_quantity = self.open_sells[prod][price]
                            if remaining_quantity >= available_quantity:  
                                remaining_quantity -= available_quantity  
                                del self.open_sells[prod][price]  
                            else:  
                                self.open_sells[prod][price] -= remaining_quantity  
                                remaining_quantity = 0
                        else:
                            continue
                    if remaining_quantity > 0:
                        if trade.price in self.open_buys[prod]:
                            self.open_buys[prod][trade.price] += remaining_quantity
                        else:
                            self.open_buys[prod][trade.price] = remaining_quantity
                        
                else:
                    bought_price = sorted(list(self.open_sells[prod].keys()))
                    for price in bought_price:  
                        if trade.price <= price: 
                            break  
                        if remaining_quantity <= 0:
                            break 
                        if price in self.open_buys[prod]:
                            available_quantity = self.open_buys[prod][price]
                            if remaining_quantity >= available_quantity:  
                                remaining_quantity -= available_quantity  
                                del self.open_buys[prod][price]  
                            else:  
                                self.open_buys[prod][price] -= remaining_quantity  
                                remaining_quantity = 0
                        else:
                            continue
                    if remaining_quantity > 0:
                        if trade.price in self.open_sells[prod]:
                            self.open_sells[prod][trade.price] += remaining_quantity
                        else:
                            self.open_sells[prod][trade.price] = remaining_quantity
            
            if prod in self.synthetic.values():
                state.position[prod] = sum(self.open_buys[prod].values()) - sum(self.open_sells[prod].values())
            # sanity check: position
            if prod not in self.basket_contents["PICNIC_BASKET1"] and prod != mac:
                pos = state.position.get(prod, 0)
                if sum(self.open_buys[prod].values()) - sum(self.open_sells[prod].values()) != pos:
                    print("BEEP! Open positions incorrectly tracked! ", prod)
                # assert sum(self.open_buys[prod].values()) - sum(self.open_sells[prod].values()) == 0, \
                #     "BEEP! Open positions incorrectly tracked!"
        
        if mac in state.own_trades:
            self.correct_macaron_conversion(state)
            if sum(self.open_buys[mac].values()) - sum(self.open_sells[mac].values()) != \
                state.position.get("MAGNIFICENT_MACARONS", 0):
                    print("BEEP! Open positions incorrectly tracked!", mac)

        return None


    def check_orders(self, state, orders, prod):
        ### SANITY CHECK BECAUSE I'M SCARRED FROM THE TYPO IN ROUND 1 ###
        total_buy, total_sell = 0, 0
        pos = state.position.get(prod, 0)
        remove = []
        for i, order in enumerate(orders):
            if order.quantity > 0:
                total_buy += order.quantity
                if total_buy > self.POS_LIM[prod] - pos:
                    remove.append(i)
                    total_buy -= order.quantity
            elif order.quantity < 0:
                total_sell += order.quantity
                if total_sell > pos + self.POS_LIM[prod]:
                    remove.append(i)
                    total_sell -= order.quantity

        if len(remove) > 0:
            print("BEEP BEEP BEEP BEEP BEEP")
            return [order for i, order in enumerate(orders) if i not in set(remove)]
        else:
            return orders


    def find_signal_zscore(self, prod, window, mult, mult_strong):
        
        """
        Find trading signals using mean-reversion by tracking Z scores. 
        Hits: number of consecutive hits on the upper band needed to constitute a signal
        Strong hits: number of consecutive signals to constitute a strong signal
        """

        history = self.history[prod]

        self.strong_signal[prod] = 0

        if len(history) < window:
            return None

        # Mean reversion strategy
        if len(history) >= window:
            sma = np.mean(np.array(history[-window:]))
            std = np.std(np.array(history[-window:]), ddof=1)
            if std > 0:
                z = (history[-1] - sma) / std
            else:
                return None

        if np.abs(z) >= mult_strong:
            if z > 0:
                self.strong_signal[prod] = -1
            else:
                self.strong_signal[prod] = 1
            return None

        if z < -mult:
            self.lower[prod] += 1
            self.upper[prod] = 0
        elif z > mult:
            self.lower[prod] = 0
            self.upper[prod] += 1
        else:
            self.upper[prod] = self.lower[prod] = 0

        return None


    def find_signal_momentum(self, prod, lookback, threshold, strong_threshold):

        self.strong_signal[prod] = 0

        if len(self.history[prod]) < lookback + 1:
            return None

        returns = (self.history[prod][-1] - self.history[prod][-(1+lookback)]) / self.history[prod][-1]

        if returns < -strong_threshold:
            self.strong_signal[prod] = 1
            self.upper[prod] = self.lower[prod] = self.signal[prod] = 0

        elif returns > strong_threshold:
            self.strong_signal[prod] = -1
            self.upper[prod] = self.lower[prod] = self.signal[prod] = 0
        
        if np.abs(self.strong_signal[prod]) == 1:
            return None

        if returns < -threshold:
            self.upper[prod] = 0
            self.lower[prod] += 1
        elif returns > threshold:
            self.upper[prod] += 1
            self.lower[prod] = 0
        else:
            self.upper[prod] = self.lower[prod] = 0
        return None

    
    def find_signal_breakout(self, prod, lookback, lookback_strong, reset=False):
        self.strong_signal[prod] = 0

        if len(self.history[prod]) < lookback + 1:
            return None
        
        self.history[prod] = self.history[prod][-(lookback_strong+1):]

        if len(self.history[prod]) == lookback_strong + 1:
            high = np.max(self.history[prod][:-1])
            low = np.min(self.history[prod][:-1])
            if self.history[prod][-1] > high:
                self.strong_signal[prod] = -1
                return None
            if self.history[prod][-1] < low:
                self.strong_signal[prod] = 1
                return None
        
        high = np.max(self.history[prod][-(lookback+1):-1])
        low = np.min(self.history[prod][-(lookback+1):-1])

        if self.history[prod][-1] > high:
            self.upper[prod] += 1
            self.lower[prod] = 0
        elif self.history[prod][-1] < low:
            self.lower[prod] += 1
            self.upper[prod] = 0
        else:
            if reset:
                self.upper[prod] = self.lower[prod] = 0
        return None
    

    def process_signal(self, prod, hits, strong_hits):
        if np.abs(self.strong_signal[prod]) == 1:
            return None
        
        if self.upper[prod] >= hits:      # sell signal
            if self.signal[prod] > 0:
                self.signal[prod] = -1
            else:
                self.signal[prod] -= 1
            self.upper[prod] = self.lower[prod] = 0     # reset counter

        elif self.lower[prod] >= hits:        # buy signal
            if self.signal[prod] > 0:
                self.signal[prod] += 1
            else:
                self.signal[prod] = 1
            self.upper[prod] = self.lower[prod] = 0     # reset counter

        if self.signal[prod] == -strong_hits:      # strong sell; reset signal and counter
            self.strong_signal[prod] = -1
            self.signal[prod] = self.lower[prod] = self.upper[prod] = 0
            
        elif self.signal[prod] == strong_hits:        # strong buy; reset signal and counter
            self.strong_signal[prod] = 1
            self.signal[prod] = self.lower[prod] = self.upper[prod] = 0
        return None
    

    def order_resin(self, state: TradingState):
        orders: list[Order] = []
        prod = "RAINFOREST_RESIN"
        pos_lim = self.POS_LIM[prod]
        # free parameters
        fairprice = 10000
        clear_lim = pos_lim - 2
        # end of parameters

        # track long and short separately to prevent cancelling out
        current_short, current_long, current_pos = 0, 0, 0
        if prod in state.position:
            current_pos = state.position[prod]
            if current_pos > 0:
                current_long += current_pos
            else:
                current_short += current_pos

        order_depth = state.order_depths[prod]
        sellorders = sorted(list(order_depth.sell_orders.items()))
        buyorders = sorted(list(order_depth.buy_orders.items()), reverse=True)

        # market taking
        asks = []
        bids = []
        for sellorder in sellorders:
            ask, ask_amount = sellorder
            if ask > fairprice:
                break
            asks.append(ask) 
            if current_long < pos_lim:
                if ask < fairprice:
                    mybuyvol = min(-ask_amount, pos_lim-current_long)
                    if mybuyvol > 0:
                        orders.append(Order(prod, ask, mybuyvol))
                        current_long += mybuyvol

        for buyorder in buyorders:
            bid, bid_amount = buyorder
            if bid < fairprice:
                break
            bids.append(bid) 
            if current_short > -pos_lim:
                if bid > fairprice:
                    mysellvol = min(bid_amount, pos_lim+current_short)
                    mysellvol *= -1
                    if mysellvol < 0:
                        orders.append(Order(prod, bid, mysellvol))
                        current_short += mysellvol

        # clear open positions if approaching position limit (buy/sell at fairprice to increase chance of order acceptance)
        if current_long > clear_lim:
            mysellvol = -min(current_short+pos_lim, current_long-clear_lim)
            if mysellvol < 0:
                orders.append(Order(prod, fairprice, mysellvol))
                current_short += mysellvol
        if current_short < -clear_lim:
            mybuyvol = max(pos_lim-current_long, -(clear_lim+current_short))
            if mybuyvol > 0:
                orders.append(Order(prod, fairprice, mybuyvol))
                current_long += mybuyvol

        # market making: fill the remaining orders up to position limit
        bestask, bestbid = sellorders[0][0], buyorders[0][0]
        mmask = min(order_depth.sell_orders, key=order_depth.sell_orders.get) 
        mmbid = max(order_depth.buy_orders, key=order_depth.buy_orders.get)

        if current_long < pos_lim:
            qty = pos_lim - current_long
            if bestbid < fairprice:
                price = min(bestbid + 1, fairprice - 2)
            else:
                price = min(mmbid + 1, fairprice - 1)
            orders.append(Order(prod, price, qty))
        if current_short > -pos_lim:
            qty = pos_lim + current_short
            if bestask > fairprice:
                price = max(bestask - 1, fairprice + 2)
            else:
                price = max(mmask - 1, fairprice + 1)
            orders.append(Order(prod, price, -qty))

        return orders


    def order_kelp(self, state: TradingState, sigma, m, c):
        prod = "KELP"
        orders: list[Order] = []
        order_depth = state.order_depths[prod]
        pos_lim = self.POS_LIM[prod]
        signal = None

        long_lim, short_lim = 15, 5
        mult = 0.75

        # # to log live order book
        # print("sell", order_depth.sell_orders)
        # print("buy", order_depth.buy_orders)

        # calculate fairprice based on market-making bots
        fairprice = (min(order_depth.sell_orders, key=order_depth.sell_orders.get) 
              + max(order_depth.buy_orders, key=order_depth.buy_orders.get)) / 2
        round_bid = round(fairprice - 1)
        round_ask = round(fairprice + 0.5)

        # adaptive lower limit
        regressed_mean = m * state.timestamp + c
        if fairprice - regressed_mean < -mult * sigma:      # buy signal
            long_lim = 30
            short_lim = 1
            signal = "BUY"
        elif fairprice - regressed_mean > mult * sigma:     # sell signal
            short_lim = 15
            long_lim = 0
            signal = "SELL"

        # track long and short separately to prevent cancelling out
        current_short, current_long = 0, 0
        if prod in state.position:
            current_pos = state.position[prod]
            if current_pos > 0:
                current_long += current_pos
            else:
                current_short += current_pos
        
        else:
            current_pos = 0
        
        sellorders = sorted(list(order_depth.sell_orders.items()))
        buyorders = sorted(list(order_depth.buy_orders.items()), reverse=True)
        
        # market taking
        asks, bids = [], []
        for sellorder in sellorders:
            ask, ask_amount = sellorder
            if ask > fairprice:
                break
            asks.append(ask)
            if current_long < pos_lim:
                if ask < fairprice:
                    mybuyvol = min(-ask_amount, pos_lim-current_long)
                    if mybuyvol > 0:
                        orders.append(Order(prod, ask, mybuyvol))
                        current_long += mybuyvol

        for buyorder in buyorders:
            bid, bid_amount = buyorder
            if bid < fairprice:
                break
            bids.append(bid)
            if current_short > -pos_lim:
                if bid > fairprice:
                    mysellvol = min(bid_amount, pos_lim+current_short)
                    mysellvol *= -1
                    orders.append(Order(prod, bid, mysellvol))
                    current_short += mysellvol
        
        # clear open positions if approaching position limit
        if current_long > long_lim and current_short > -pos_lim:
            mysellvol = min(current_short+pos_lim, current_long-long_lim)
            if len(self.open_buys[prod]) > 0:
                bought_prices = sorted(list(self.open_buys[prod].keys()), reverse=True)
                i = 0
                while mysellvol > 0 and i < len(bought_prices) and current_short > -pos_lim:
                    qty = min(mysellvol, self.open_buys[prod][bought_prices[i]], current_short+pos_lim)
                    orders.append(Order(prod, int(bought_prices[i]+1), -qty))
                    i += 1
                    mysellvol -= qty
                    current_short -= qty
            else:
                if len(bids) > 0:
                    orders.append(Order(prod, max(bids[0], round_ask), -mysellvol))
                else:
                    orders.append(Order(prod, round_ask, -mysellvol))
                current_short -= mysellvol
        if current_short < -short_lim and current_long < pos_lim:
            mybuyvol = min(pos_lim-current_long, -(short_lim+current_short))
            if len(self.open_sells[prod]) > 0:
                sold_prices = sorted(list(self.open_sells[prod].keys()))
                i = 0
                while mybuyvol > 0 and i < len(sold_prices) and current_long < pos_lim:
                    qty = min(mybuyvol, self.open_sells[prod][sold_prices[i]], pos_lim-current_long)
                    orders.append(Order(prod, int(sold_prices[i]-1), qty))
                    i += 1
                    mybuyvol -= qty
                    current_long += qty
            else:
                if len(asks) > 0:
                    orders.append(Order(prod, min(asks[0], round_bid), mybuyvol))
                else:
                    orders.append(Order(prod, round_bid, mybuyvol))
                current_long += mybuyvol

        # market making: fill the remaining orders up to position limit
        bestask, bestbid = sellorders[0][0], buyorders[0][0]
        mmask = min(order_depth.sell_orders, key=order_depth.sell_orders.get) 
        mmbid = max(order_depth.buy_orders, key=order_depth.buy_orders.get)
    
        if current_long < pos_lim:
            qty = pos_lim - current_long
            if signal == "BUY":     # want to buy more at the cost of profit
                if bestbid < fairprice:
                    price = min(bestbid+1, int(fairprice+0.5))
                else:
                    price = min(mmbid+1, int(fairprice+0.5))  
            else:       # prioritise profit
                price = min(round_bid, bestbid+1, bestask)
            orders.append(Order(prod, price, qty))
        if current_short > -pos_lim:
            qty = pos_lim + current_short
            if signal == "SELL":        # want to sell more at the cost of profit
                if bestask > fairprice:
                    price = max(bestask-1, int(fairprice))
                else:
                    price = max(mmask-1, int(fairprice))
            else:       # prioritise profit
                price = max(round_ask, bestask-1, bestbid)

            orders.append(Order(prod, price, -qty))
    
        ### SANITY CHECK BECAUSE I'M SCARRED FROM THE TYPO IN ROUND 1 ###
        orders = self.check_orders(state, orders, prod)

        # print("orders: ", orders)
        return orders


    def order_squid(self, state: TradingState):
        prod = "SQUID_INK"
        orders: list[Order] = []
        order_depth = state.order_depths[prod]
        pos_lim = self.POS_LIM[prod]

        # free parameters #
        soft_lim = 35
        clear_lim = 20
        maxtake = 5
        maxmake = 10
        lookback = 10
        lookback_strong = 500
        hits = 1
        strong_hits = 10
        # end of parameters #

        # calculate fairprice based on market-making bots
        fairprice = (min(order_depth.sell_orders, key=order_depth.sell_orders.get) 
              + max(order_depth.buy_orders, key=order_depth.buy_orders.get)) / 2

        self.history[prod].append(fairprice)
        self.history[prod] = self.history[prod][-(lookback+1):]

        # self.find_signal_momentum(prod, lookback, threshold, strong_threshold)
        self.find_signal_breakout(prod, lookback, lookback_strong)
        self.process_signal(prod, hits, strong_hits)
        
        # track long and short separately to prevent cancelling out
        current_short, current_long = 0, 0
        if prod in state.position:
            current_pos = state.position[prod]
            if current_pos > 0:
                current_long += current_pos
            else:
                current_short += current_pos
        else:
            current_pos = 0
            
        sellorders = sorted(list(order_depth.sell_orders.items()))
        buyorders = sorted(list(order_depth.buy_orders.items()), reverse=True)
        asks = [item[0] for item in sellorders]
        bids = [item[0] for item in buyorders]
        bought_prices = list(self.open_buys[prod].keys())
        sold_prices = list(self.open_sells[prod].keys())

        if len(bought_prices) > 0:
            maxbought = max(bought_prices)
            minbought = min(bought_prices)
        else:
            maxbought = None
            minbought = None
        if len(sold_prices) > 0:
            minsold = min(sold_prices)
            maxsold = max(sold_prices)
        else:
            minsold = None
            maxsold = None

        # market taking
        if self.strong_signal[prod] == 1 and current_long < pos_lim:
            mybuyvol = min(-sellorders[0][1], pos_lim-current_long)
            mybuyvol = min(mybuyvol, maxtake)
            condition = (sellorders[0][0] <= minbought if minbought is not None else True)
            if mybuyvol > 0 and condition:
                # update minbought to avoid buying higher in the same timestep
                minbought = sellorders[0][0]
                orders.append(Order(prod, sellorders[0][0], mybuyvol))
                current_long += mybuyvol
        elif current_long < soft_lim and (minbought is not None or minsold is not None):
            for sellorder in sellorders:
                ask, ask_amount = sellorder
                condition = ((minbought is not None and ask < minbought) or
                            (minsold is not None and ask < minsold))
                if self.signal[prod] == 1 and condition:
                    mybuyvol = min(-ask_amount, soft_lim-current_long)
                    mybuyvol = min(mybuyvol, maxtake)
                    if mybuyvol > 0:
                        # update minbought to avoid buying higher in the same timestep
                        if minbought is None:
                            minbought = ask
                        else:
                            if ask < minbought:
                                minbought = ask
                        orders.append(Order(prod, ask, mybuyvol))
                        current_long += mybuyvol
                    else:
                        break

        if (self.strong_signal[prod] == -1 or self.signal[prod] == -1) and current_short > -pos_lim:
            bid, bid_amount = buyorders[0]
            mysellvol = min(bid_amount, pos_lim+current_short)
            mysellvol = min(mysellvol, maxtake)
            mysellvol *= -1
            condition = (bid > maxsold if maxsold is not None else True)
            if mysellvol < 0 and condition:
                # update maxsold to avoid selling lower in the same timestep
                maxsold = bid
                orders.append(Order(prod, bid, mysellvol))
                current_short += mysellvol
        elif current_short > -soft_lim and (maxsold is not None or maxbought is not None):
            for buyorder in buyorders:
                bid, bid_amount = buyorder
                condition = ((maxbought is not None and bid > maxbought) or
                            (maxsold is not None and bid > maxsold))
                if self.signal[prod] == -1 and condition:
                    mysellvol = min(bid_amount, soft_lim+current_short)
                    mysellvol = min(mysellvol, maxtake)
                    mysellvol *= -1
                    if mysellvol < 0:
                        # update minbought to avoid buying higher in the same timestep
                        if maxsold is None:
                            maxsold = bid
                        else:
                            if bid > maxsold:
                                maxsold = bid
                        orders.append(Order(prod, bid, mysellvol))
                        current_short += mysellvol
                
        # clear open positions if approaching position limit
        if current_long > clear_lim:
            for price in bought_prices:
                qty = min(self.open_buys[prod][price], pos_lim+current_short)
                if qty > 0:
                    if bids[0] - price > 1:
                        orders.append(Order(prod, bids[0], -qty))
                    elif asks[0] > price + 1:
                        orders.append(Order(prod, asks[0]-1, -qty))
                    else:
                        orders.append(Order(prod, max(int(price), int(fairprice+0.5)), -qty))
                    current_short -= qty
        
        if current_short < -clear_lim:
            for price in sold_prices:
                qty = min(self.open_sells[prod][price], pos_lim-current_long)
                if qty > 0:
                    if price - asks[0] > 1:
                        orders.append(Order(prod, asks[0], qty))
                    elif price - bids[0] > 1:
                        orders.append(Order(prod, bids[0]+1, qty))
                    else:
                        orders.append(Order(prod, min(int(price), int(fairprice)), qty))
                    current_long += qty
            
        # market making: strong signal (ignore soft limits)
        bestbid = buyorders[0][0]
        bestask = sellorders[0][0]
        if self.strong_signal[prod] == 1 and current_long < pos_lim:
            if maxbought is None:
                price = bestbid  + 1
            else:
                price = min(bestbid+1, int(maxbought))
            qty = min(pos_lim-current_long, maxmake)
            orders.append(Order(prod, price, qty))

        if self.strong_signal[prod] == -1 and current_short > -pos_lim:
            if minsold is None:
                price = bestask - 1
            else:
                price = max(bestask-1, int(minsold))
            qty = min(current_short+pos_lim, maxmake)
            orders.append(Order(prod, price, -qty))

        return self.check_orders(state, orders, prod)
    

    def order_basket(self, state: TradingState, current_positions, basket, adverse_lim=0, clear_lim=40):

        maxbuy = 10

        if basket not in state.order_depths:
            return None
        
        contents = self.basket_contents[basket]
        basket_position = state.position.get(basket, 0)
        basket_order_depth = state.order_depths[basket]
        basket_best_bid = max(basket_order_depth.buy_orders.keys())
        basket_best_ask = min(basket_order_depth.sell_orders.keys())
        basket_bid_vol = basket_order_depth.buy_orders[basket_best_bid]
        basket_ask_vol = -basket_order_depth.sell_orders[basket_best_ask]
        basket_swmp = (basket_best_bid * basket_ask_vol + 
                       basket_best_ask * basket_bid_vol) / (basket_bid_vol + basket_ask_vol)

        # get synthetic (implied) midprice
        bid_vols, ask_vols = [], []
        synthetic_swmp = 0
        implied_bid, implied_ask = 0, 0
        for prod in contents:
            if state.order_depths[prod].buy_orders:
                best_bid = max(state.order_depths[prod].buy_orders.keys())
            else:
                best_bid = 0
            if state.order_depths[prod].sell_orders:
                best_ask = min(state.order_depths[prod].sell_orders.keys())
            else:
                best_ask = float("inf")

            implied_bid += contents[prod] * best_bid
            implied_ask += contents[prod] * best_ask

            if best_bid > 0:
                bid_vols.append(state.order_depths[prod].buy_orders[best_bid] // contents[prod])
            if best_ask < float("inf"):
                ask_vols.append(-state.order_depths[prod].sell_orders[best_ask] // contents[prod])

            synthetic_swmp += (best_ask + best_bid) / 2 * contents[prod]
        
        if implied_bid > 0 and implied_ask < float("inf"):
            implied_ask_vol = min(ask_vols)
            implied_bid_vol = min(bid_vols)
            synthetic_swmp = (implied_bid * implied_ask_vol + 
                              implied_ask * implied_bid_vol) / (implied_bid_vol + implied_ask_vol)
        else:
            return None         # contents midprice unavailable

        spread = basket_swmp - synthetic_swmp
        self.history[basket].append(spread)
        self.history[basket] = self.history[basket][-(100+1):]
        
        if basket == "PICNIC_BASKET1":
            self.find_signal_breakout(basket, 20, 100)
            self.process_signal(basket, 2, 10)
        elif basket == "PICNIC_BASKET2":
            self.find_signal_zscore(basket, 20, 2.5, 3.5)
            self.process_signal(basket, 2, 10)
        
        content_orders = {}

        if basket_position < -clear_lim:      # BUY BASKET, SELL CONTENTS
            print("clearing")
            basket_lim = self.POS_LIM[basket] - basket_position
            for spread_sold in self.open_sells[self.synthetic[basket]]:
                if spread > spread_sold or basket_position >= self.POS_LIM[basket]:    # want to buy lower than sold
                    continue
                qty_basket = [basket_ask_vol, implied_bid_vol, basket_lim, 
                              -(adverse_lim+basket_position), self.open_sells[self.synthetic[basket]][spread_sold]]
                # check positions of individual content
                for prod in contents:
                    available = self.POS_LIM[prod] + current_positions[prod]
                    qty_basket.append(available // contents[prod])
                qty_basket = min(qty_basket)
                if qty_basket > 0:
                    for prod in contents:
                        content_orders[prod] = Order(prod, best_bid, -qty_basket * contents[prod])
                    basket_order = Order(basket, basket_best_ask, qty_basket)

                    return basket_order, content_orders

        if basket_position > clear_lim:        # SELL BASKET, BUY CONTENTS
            print("clearing")
            basket_lim = self.POS_LIM[basket] + basket_position
            for spread_bought in self.open_buys[self.synthetic[basket]]:
                if spread < spread_bought or basket_position <= -self.POS_LIM[basket]:    # want to sell higher than bought
                    continue
                qty_basket = [basket_bid_vol, implied_ask_vol, basket_lim,
                              basket_position-adverse_lim, self.open_buys[self.synthetic[basket]][spread_bought]]
                # check positions of individual content
                for prod in contents:
                    available = self.POS_LIM[prod] - current_positions[prod]
                    qty_basket.append(available // contents[prod])
                qty_basket = min(qty_basket)
                if qty_basket > 0:
                    for prod in contents:
                        content_orders[prod] = Order(prod, best_ask, qty_basket * contents[prod])
                    basket_order = Order(basket, basket_best_bid, -qty_basket)

                    return basket_order, content_orders

        # STRONG SIGNAL
        if self.strong_signal[basket] == 1 and basket_position <= adverse_lim:       # BUY BASKET, SELL CONTENTS
            print("strong signal")
            basket_lim = self.POS_LIM[basket] - basket_position
            qty_basket = [basket_ask_vol, implied_bid_vol, basket_lim, maxbuy]

            # check positions of individual content
            for prod in contents:
                available = self.POS_LIM[prod] + current_positions[prod]
                qty_basket.append(available // contents[prod])
            
            qty_basket = min(qty_basket)
            if qty_basket > 0:
                basket_order = Order(basket, basket_best_ask, qty_basket)

                for prod in contents:
                    best_bid = max(state.order_depths[prod].buy_orders.keys())
                    content_orders[prod] = Order(prod, best_bid, -qty_basket * contents[prod])
                
                return basket_order, content_orders
        
        if self.strong_signal[basket] == -1 and basket_position >= -adverse_lim:     # SELL BASKET, BUY CONTENTS
            print("strong signal")
            basket_lim = self.POS_LIM[basket] + basket_position
            qty_basket = [basket_bid_vol, implied_ask_vol, basket_lim, maxbuy]

            # check positions of individual content
            for prod in contents:
                available = self.POS_LIM[prod] - current_positions[prod]
                qty_basket.append(available // contents[prod])
            
            qty_basket = min(qty_basket)
            if qty_basket > 0:
                basket_order = Order(basket, basket_best_bid, -qty_basket)

                for prod in contents:
                    best_ask = min(state.order_depths[prod].sell_orders.keys())
                    content_orders[prod] = Order(prod, best_ask, qty_basket * contents[prod])
                
                return basket_order, content_orders
        
        # Normal signal: check validity (buy lower than previously bought/sold)
        if np.abs(self.signal[basket]) == 1:
            print("normal signal")
            if len(self.open_buys[self.synthetic[basket]]) > 0:
                my_dict = self.open_buys[self.synthetic[basket]]
                min_bought = min(list(my_dict.keys()))
                avg_bought = sum(price * vol for price, vol in my_dict.items()) / sum(my_dict.values())
            else:
                avg_bought = min_bought = None
            if len(self.open_sells[self.synthetic[basket]]) > 0:
                my_dict = self.open_sells[self.synthetic[basket]]
                max_sold = max(list(my_dict.keys()))
                avg_sold = sum(price * vol for price, vol in my_dict.items()) / sum(my_dict.values())
            else:
                avg_sold = None
                max_sold = None

        if self.signal[basket] == 1 and basket_position <= adverse_lim:       # BUY BASKET, SELL CONTENTS
            print("buy signal")
            if (min_bought is None or spread <= min_bought) and (avg_sold is None or spread < avg_sold):
                basket_lim = adverse_lim - basket_position
                qty_basket = [basket_ask_vol, implied_bid_vol, basket_lim, maxbuy]
                for prod in contents:
                    available = self.POS_LIM[prod] + current_positions[prod]
                    qty_basket.append(available // contents[prod])
                qty_basket = min(qty_basket)
                if qty_basket > 0:
                    basket_order = Order(basket, basket_best_ask, qty_basket)
                    for prod in contents:
                        best_bid = max(state.order_depths[prod].buy_orders.keys())
                        content_orders[prod] = Order(prod, best_bid, -qty_basket * contents[prod])
                    return basket_order, content_orders

        if self.signal[basket] == -1 and basket_position >= -adverse_lim:       # SELL BASKET, BUY CONTENTS
           print("sell signal")
           if (avg_bought is None or spread > avg_bought) and (max_sold is None or spread >= max_sold):
                basket_lim = adverse_lim + basket_position
                qty_basket = [basket_bid_vol, implied_ask_vol, basket_lim, maxbuy]
                for prod in contents:
                    available = self.POS_LIM[prod] - current_positions[prod]
                    qty_basket.append(available // contents[prod])
                qty_basket = min(qty_basket)
                if qty_basket > 0:
                    basket_order = Order(basket, basket_best_bid, -qty_basket)
                    for prod in contents:
                        best_ask = min(state.order_depths[prod].sell_orders.keys())
                        content_orders[prod] = Order(prod, best_ask, qty_basket * contents[prod])
                    return basket_order, content_orders
        return None


    def order_volcanic_rock_option(self, state: TradingState, option: str):
        """
        Trades the call options on VOLCANIC_ROCK using Black-Scholes pricing.
        """
        orders: list[Order] = []
        if option not in state.order_depths:
            return orders
        order_depth = state.order_depths[option]
        pos_lim = self.POS_LIM[option]

        # Get the underlying mid-price for VOLCANIC_ROCK.
        underlying = "VOLCANIC_ROCK"
        if underlying in state.order_depths and state.order_depths[underlying].buy_orders and state.order_depths[underlying].sell_orders:
            best_bid_und = max(
                state.order_depths[underlying].buy_orders.keys())
            best_ask_und = min(
                state.order_depths[underlying].sell_orders.keys())
            S = (best_bid_und + best_ask_und) / 2
        else:
            return orders

        # Extract the strike price from the option name.
        try:
            strike = float(option.split("_")[-1])
        except Exception:
            return orders

        remaining_time = 6e6 - state.timestamp
        T = remaining_time / 365e6
        r = 0
        
        sigma_est = estimated_volatility(S, strike)
        theoretical_price = black_scholes_call_price(S, strike, T, sigma_est, r)
        # Correct theoretical price using the average deviation (#numerics stuff)
        if strike == 9500:
            avg_diff = -0.0332
        elif strike == 9750:
            avg_diff = 0.0652
        elif strike == 10000:
            avg_diff = 0.3740
        elif strike == 10250:
            avg_diff = 0.7489
        elif strike == 10500:
            avg_diff = 0.4404
        else:
            avg_diff = 0.0
        corrected_price = theoretical_price - avg_diff
        theoretical_price=corrected_price

        # Determine the current market mid-price for the option.
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            market_mid = (best_bid + best_ask) / 2
        else:
            market_mid = 0

        current_pos = state.position.get(option, 0)
        
        # sd = standard deviation of estimate to actual
        if strike == 9500:
            sd = 0.2973
        elif strike == 9750:
            sd = 0.4681
        elif strike == 10000:
            sd = 1.6689
        elif strike == 10250:
            sd = 0.3257
        elif strike == 10500:
            sd = 0.2947
        else:
            sd = 0.4
            
        print(f"Current pos: {current_pos}")
        
        
        # buy
        orders.append(
            Order(option, round(theoretical_price-sd*0.4*np.exp((current_pos)/200)), pos_lim-current_pos))

        # sell
        orders.append(
            Order(option, round(theoretical_price+sd*0.4*np.exp((-current_pos)/200)), -pos_lim-current_pos))

        # sd*0.4*np.exp((current_pos)/200) is a safety margin on top of the
        # price estimate to ensure profitability
        # Is dependent on current position in relation to the limit to increase
        # the propensity to sell and decrease the propensity to buy, when the upper
        # limit is approached and vice verca.
        
        return orders
    

    def order_VR(self, state: TradingState):
        prod = "VOLCANIC_ROCK"
        orders: list[Order] = []
        if prod not in state.order_depths:
            return []
        order_depth = state.order_depths[prod]
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return []
        pos_lim = self.POS_LIM[prod]

        # free parameters #
        soft_lim = 100
        maxtake = 15
        window = 100
        mult = 2.
        mult_strong = 2.5
        hits = 2
        strong_hits = 10
        # end of parameters #

        # calculate fairprice based on market-making bots
        fairprice = (min(order_depth.sell_orders, key=order_depth.sell_orders.get) 
              + max(order_depth.buy_orders, key=order_depth.buy_orders.get)) / 2
        
        self.history[prod].append(fairprice)
        self.history[prod] = self.history[prod][-window:]

        self.find_signal_zscore(prod, window, mult, mult_strong)
        self.process_signal(prod, hits, strong_hits)

        # track long and short separately to prevent cancelling out
        current_short, current_long = 0, 0
        if prod in state.position:
            current_pos = state.position[prod]
            if current_pos > 0:
                current_long += current_pos
            else:
                current_short += current_pos
        
        else:
            current_pos = 0
            
        sellorders = sorted(list(order_depth.sell_orders.items()))
        buyorders = sorted(list(order_depth.buy_orders.items()), reverse=True)
        bought_prices = list(self.open_buys[prod].keys())
        sold_prices = list(self.open_sells[prod].keys())

        if len(bought_prices) > 0:
            maxbought = max(bought_prices)
            minbought = min(bought_prices)
        else:
            maxbought = None
            minbought = None
        if len(sold_prices) > 0:
            minsold = min(sold_prices)
            maxsold = max(sold_prices)
        else:
            minsold = None
            maxsold = None

        # market taking
        if self.strong_signal[prod] == 1 and current_long < soft_lim:
            
            if minbought is None and minsold is None:
                condition = True
            elif minbought is not None:
                condition = sellorders[0][0] < minbought - 5
            elif minsold is not None:
                condition = sellorders[0][0] < minsold - 3

            mybuyvol = min(-sellorders[0][0], soft_lim-current_long)
            mybuyvol = min(mybuyvol, maxtake)
            if mybuyvol > 0 and condition:
                # update minbought to avoid buying higher in the same timestep
                minbought = sellorders[0][0]
                orders.append(Order(prod, sellorders[0][0], mybuyvol))
                current_long += mybuyvol
        elif current_long < soft_lim and (minbought is not None or minsold is not None):
            for sellorder in sellorders:
                ask, ask_amount = sellorder
                condition = ((minbought is not None and ask < minbought - 8) or
                            (minsold is not None and ask < minsold - 5))
                minbought = ask
                if self.signal[prod] == 1 and condition:
                    mybuyvol = min(-ask_amount, soft_lim-current_long)
                    mybuyvol = min(mybuyvol, maxtake)
                    if mybuyvol > 0:
                        orders.append(Order(prod, ask, mybuyvol))
                        current_long += mybuyvol
                    else:
                        break

        if (self.strong_signal[prod] == -1 or self.signal[prod] == -1) and current_short > -soft_lim:

            if maxsold is None and maxbought is None:
                condition = True
            elif maxsold is not None:
                condition = buyorders[0][0] > maxsold + 5
            elif maxbought is not None:
                condition = buyorders[0][0] > maxbought + 3
        
            bid, bid_amount = buyorders[0]
            mysellvol = min(bid_amount, soft_lim+current_short)
            mysellvol = min(mysellvol, maxtake)
            mysellvol *= -1
            if mysellvol < 0 and condition:
                # update maxsold to avoid selling lower in the same timestep
                maxsold = bid
                orders.append(Order(prod, bid, mysellvol))
                current_short += mysellvol
        elif current_short > -soft_lim and (maxsold is not None or maxbought is not None):
            for buyorder in buyorders:
                bid, bid_amount = buyorder
                condition = ((maxbought is not None and bid > maxbought + 5) or
                            (maxsold is not None and bid > maxsold + 8))
                if self.signal[prod] == -1 and condition:
                    mysellvol = min(bid_amount, soft_lim+current_short)
                    mysellvol = min(mysellvol, maxtake)
                    mysellvol *= -1
                    if mysellvol < 0:
                        # update minbought to avoid buying higher in the same timestep
                        if maxsold is None:
                            maxsold = bid
                        else:
                            if bid > maxsold:
                                maxsold = bid
                        orders.append(Order(prod, bid, mysellvol))
                        current_short += mysellvol

        return self.check_orders(state, orders, prod)


    def order_macarons(self, state: TradingState):
        # try to import from Pristine at a lower price
        # it seems like short-selling is heavily rewarded
        prod = "MAGNIFICENT_MACARONS"
        if prod not in state.order_depths:
            return [], 0
        conv_lim = 10
        conversion = 0
        pos_lim = self.POS_LIM[prod]
        pos = state.position.get(prod, 0)
        orders = []
        order_depth = state.order_depths[prod]
        # bestbid = max(order_depth.buy_orders.keys())
        bestask = min(order_depth.sell_orders.keys())
        min_profit = 2

        fairprice = (min(order_depth.sell_orders, key=order_depth.sell_orders.get) 
              + max(order_depth.buy_orders, key=order_depth.buy_orders.get)) / 2
        self.history[prod].append(fairprice)
        self.history[prod] = self.history[prod][-10:]

        self.find_signal_zscore(prod, 10, 2.5, 5.)
        self.process_signal(prod, 1, 50)

        # if self.signal[prod] <= -1 or self.strong_signal[prod] == -1:     # mean reversion
        if self.signal[prod] >= 1 or self.strong_signal[prod] == 1:     # bollinger band
            maxsell = pos + pos_lim
            soldqty = 0
            prices = [bestask-2, bestask-1]
            for price in prices:
                if soldqty < maxsell:
                    orders.append(Order(prod, price, -min(maxsell // len(prices), maxsell-soldqty)))
                    soldqty += maxsell // len(prices)

        obs = state.observations.conversionObservations[prod]
        implied_ask = obs.askPrice + obs.importTariff + obs.transportFees

        if pos < 0:
            sorted_items = sorted(self.open_sells[prod].items(), reverse=True)
            for price, volume in sorted_items:
                if price - implied_ask < min_profit:
                    break
                if conversion < conv_lim:
                    conversion += max(min(volume, conv_lim-conversion), 0)
        
        return orders, conversion
    

    def run(self, state: TradingState):
        
        self.timer += 100   
        conversions = 0

        ### UPDATE WITH DAY N DATA ###
        linreg = {"KELP":  (15.11018100844262, 9.768257499187243e-06, 2057.517022160843)}

        if self.timer != state.timestamp:
            if state.traderData != None and state.traderData != "":
                traderObject = jsonpickle.decode(state.traderData)
                self.open_buys = {item: {float(k): v for k, v in inner.items()}
                                for item, inner in traderObject["open buys"].items()}
                self.open_sells = {item: {float(k): v for k, v in inner.items()}
                                for item, inner in traderObject["open sells"].items()}
                self.history = traderObject["history"]
                self.upper = traderObject["upper"]
                self.lower = traderObject["lower"]
                self.signal = traderObject["signal"]
                self.strong_signal = traderObject["strong signal"]
        
        self.update_open_pos(state)

        result = {}

        # result["RAINFOREST_RESIN"] = self.order_resin(state)
        # result["KELP"] = self.order_kelp(state, *linreg["KELP"])
        # result["SQUID_INK"] = self.order_squid(state)

        # result["CROISSANTS"] = []
        # result["JAMS"] = []
        # result["DJEMBES"] = []

        # # adverse_lim, clear_lim, lookback, lookback_strong, hits, strong_hits, maxbuy
        # # adv lim: 0, clear lim: 30 basket1, 10 basket2
        # basket_params = {"PICNIC_BASKET1": (30, 60), "PICNIC_BASKET2":  (20, 100)}  
        # current_positions = {prod: (state.position[prod] if prod in state.position else 0) 
        #                      for prod in self.basket_contents["PICNIC_BASKET1"]} 

        # for basket in self.synthetic:
        #     args = basket_params[basket]
        #     spread_order = self.order_basket(state, current_positions, basket, *args)
        #     if spread_order is not None:
        #         basket_order, content_orders = spread_order
        #         basket_order = self.check_orders(state, [basket_order], basket)
        #         result[basket] = basket_order
        #         for prod in content_orders:
        #             result[prod].append(content_orders[prod])
        #             current_positions[prod] += content_orders[prod].quantity
        
        # for prod in self.basket_contents[basket]:
        #     result[prod] = self.check_orders(state, result[prod], prod)

        # result["VOLCANIC_ROCK"] = self.order_VR(state)

        # options = [
        #     "VOLCANIC_ROCK_VOUCHER_9500",
        #     "VOLCANIC_ROCK_VOUCHER_9750",
        #     "VOLCANIC_ROCK_VOUCHER_10000",
        #     "VOLCANIC_ROCK_VOUCHER_10250",
        #     "VOLCANIC_ROCK_VOUCHER_10500"
        # ]
        # for option in options:
        #     result[option] = self.order_volcanic_rock_option(state, option)

        mac = "MAGNIFICENT_MACARONS"
        if mac in state.position:
            print("position: ", state.position["MAGNIFICENT_MACARONS"])

        mac_orders, conversions = self.order_macarons(state)
        result["MAGNIFICENT_MACARONS"] = mac_orders

        traderObject = {"open buys": self.open_buys, 
                        "open sells": self.open_sells, 
                        "history": self.history,
                        "upper": self.upper,
                        "lower": self.lower,
                        "signal": self.signal,
                        "strong signal": self.strong_signal}
        
        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData

