from datamodel import OrderDepth, UserId, TradingState, Order, Trade, Observation
import jsonpickle
from typing import List
import string
import numpy as np
import json
import math
import statistics as st

### CHANGE TO OLIVIA!!
SIGNAL_PERSON = "Olivia"


TIME_TO_EXPIRY = 4e6


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


def black_scholes_call_delta(S, K, T, sigma, r):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / \
        (sigma * math.sqrt(T))
    return 0.5 * (1 + math.erf(d1 / math.sqrt(2)))   # Φ(d1)


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

        # for tracking open positions / updating memory
        self.timer = 0
        self.open_buys = {prod: {} for prod in self.prods}
        self.open_sells = {prod: {} for prod in self.prods}
        # self.recorded_time = {prod: -1 for prod in self.prods}    # last recorded time of own_trades

        # historical data and past signals for trading signals; store global extremum for assets relying on Olivia
        self.global_min = {prod: None for prod in self.prods}
        self.global_max = {prod: None for prod in self.prods}
        self.history = {prod: [] for prod in self.prods}
        self.sunlight = []
        self.previous_sunchange = None
        self.macaron_panic = False
        self.last_sold_macaron = None
        self.upper = {prod: 0 for prod in self.prods}
        self.lower = {prod: 0 for prod in self.prods}
        self.signal = {prod: 0 for prod in self.prods}
        self.strong_signal = {prod: 0 for prod in self.prods}

        # paremeters for OPTIONS
        self.linreg = {"a": 0.150195656,
                       "b": -1.15870319e-06,
                       "c": 1.23214223e-07}


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

            # sanity check: position
            if prod != mac:
                pos = state.position.get(prod, 0)
                if sum(self.open_buys[prod].values()) - sum(self.open_sells[prod].values()) != pos:
                    print("BEEP! Open positions incorrectly tracked! ", prod)
        
        if mac in state.own_trades:
            self.correct_macaron_conversion(state)
            if sum(self.open_buys[mac].values()) - sum(self.open_sells[mac].values()) != \
                state.position.get("MAGNIFICENT_MACARONS", 0):
                    print("BEEP! Open positions incorrectly tracked!", mac)
            if len(self.open_sells[mac]) > 0:
                self.last_sold_macaron = state.own_trades[mac][0].timestamp
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


    def order_kelp(self, state: TradingState):
        prod = "KELP"
        orders: list[Order] = []
        order_depth = state.order_depths[prod]
        pos_lim = self.POS_LIM[prod]
        clear_lim = 15
        adverse_lim = 50        # leave some room for global max/min

        # calculate fairprice based on market-making bots
        fairprice = (min(order_depth.sell_orders, key=order_depth.sell_orders.get) 
              + max(order_depth.buy_orders, key=order_depth.buy_orders.get)) / 2
        self.history[prod].append(fairprice)
        round_bid = round(fairprice - 1)
        round_ask = round(fairprice + 0.5)

        # track long and short separately to prevent cancelling out
        current_short, current_long = 0, 0
        current_pos = state.position.get(prod, 0)
        if current_pos > 0:
            current_long += current_pos
        else:
            current_short += current_pos
    
        sellorders = sorted(list(order_depth.sell_orders.items()))
        buyorders = sorted(list(order_depth.buy_orders.items()), reverse=True)
        asks, bids = [], []
        for sellorder in sellorders:
            ask, ask_amount = sellorder
            if ask > fairprice:
                break
            asks.append(ask)
            if current_long < adverse_lim:
                if ask < fairprice:
                    mybuyvol = min(-ask_amount, adverse_lim-current_long)
                    assert(mybuyvol >= 0), "Buy volume negative"
                    orders.append(Order(prod, ask, mybuyvol))
                    current_long += mybuyvol

        for buyorder in buyorders:
            bid, bid_amount = buyorder
            if bid < fairprice:
                break
            bids.append(bid)
            if current_short > -adverse_lim:
                if bid > fairprice:
                    mysellvol = min(bid_amount, adverse_lim+current_short)
                    mysellvol *= -1
                    assert(mysellvol <= 0), "Sell volume positive"
                    orders.append(Order(prod, bid, mysellvol))
                    current_short += mysellvol
        
        # clear open positions if approaching position limit (only when global min/max is not reached)
        if current_long > clear_lim:
            mysellvol = min(current_short+pos_lim, current_long-clear_lim)
            assert mysellvol >= 0, "Sell volume positive!"
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
        if current_short < -clear_lim:
            mybuyvol = max(pos_lim-current_long, -(clear_lim+current_short))
            assert mybuyvol >= 0, "Buy volume negative!"
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

        # market making: fill the remaining orders up to adverse limit
        bestask, bestbid = sellorders[0][0], buyorders[0][0]
        mmask = min(order_depth.sell_orders, key=order_depth.sell_orders.get) 
        mmbid = max(order_depth.buy_orders, key=order_depth.buy_orders.get)

        if current_long < adverse_lim:
            qty = adverse_lim - current_long
            if bestbid < fairprice:
                price = min(bestbid+1, round_bid)
            else:
                price = min(mmbid+1, round_bid)
            orders.append(Order(prod, price, qty))
        if current_short > -adverse_lim:
            qty = adverse_lim + current_short
            if bestask > fairprice:
                price = max(bestask-1, round_ask)
            else:
                price = max(mmask-1, round_ask)
            orders.append(Order(prod, price, -qty))

        # print("orders: ", orders)
        return self.check_orders(state, orders, prod)


    def order_squid(self, state: TradingState):
        # signal was not working well, so rely solely on Lady Olivia
        
        prod = "SQUID_INK"
        orders: list[Order] = []
        order_depth = state.order_depths[prod]
        pos_lim = self.POS_LIM[prod]
        pos = state.position.get(prod, 0)
        current_long, current_short = 0, 0
        if pos > 0:
            current_long = pos
        elif pos < 0:
            current_short = pos
        sell, buy = False, False

        # calculate fairprice based on market-making bots
        fairprice = (min(order_depth.sell_orders, key=order_depth.sell_orders.get) 
              + max(order_depth.buy_orders, key=order_depth.buy_orders.get)) / 2

        self.history[prod].append(fairprice)

        # check for Olivia's signal
        buy, sell = False, False
        market_trades = state.market_trades.get(prod, [])
        olivia_bought = []
        olivia_sold = []
        for trade in market_trades:
            if trade.buyer == SIGNAL_PERSON:         
                olivia_bought.append(trade.price)
            elif trade.seller == SIGNAL_PERSON:
                olivia_sold.append(trade.price)
        
        if len(olivia_bought) > 0 and self.history[prod][-2] < np.mean(self.history[prod][:-1]):
            self.global_min[prod] = self.history[prod][-2]
        elif len(olivia_sold) > 0 and self.history[prod][-2] > np.mean(self.history[prod][:-1]):
            self.global_max[prod] = self.history[prod][-2]
        
        if self.global_max[prod] is not None:
            # check if price is within a tolerable range form the global maximum
            dev = (fairprice - self.global_max[prod]) / self.global_max[prod]
            if dev > -0.005:
                sell = True
        elif self.global_min[prod] is not None:
            # check if price is within a tolerable range form the global maximum
            dev = (fairprice - self.global_min[prod]) / self.global_min[prod]
            if dev < 0.005:
                buy = True
        
        sellorders = sorted(list(order_depth.sell_orders.items()))
        buyorders = sorted(list(order_depth.buy_orders.items()), reverse=True)

        # market taking: if there's global min/max
        # squid max/min tends to be plateaued, so can take one higher/lower too
        if current_long < pos_lim and buy:
            mybuyvol = min(-sellorders[0][1], pos_lim-current_long)
            if mybuyvol > 0:
                orders.append(Order(prod, sellorders[0][0], mybuyvol))
                current_long += mybuyvol
            if current_long < pos_lim:
                orders.append(Order(prod, sellorders[0][0]+1, pos_lim-current_long))    
            return self.check_orders(state, orders, prod)
            
        if current_short > -pos_lim and sell:
            mysellvol = min(buyorders[0][1], pos_lim+current_short)
            if mysellvol > 0:
                orders.append(Order(prod, buyorders[0][0], -mysellvol))
                current_short -= mysellvol
            if current_short > -pos_lim:
                orders.append(Order(prod, buyorders[0][0]-1, pos_lim+current_short)) 
            return self.check_orders(state, orders, prod)
        
        return []
    

    def order_croissants(self, state: TradingState):
        prod = "CROISSANTS"
        orders = []

        sell, buy = False, False

        # calculate fairprice based on market-making bots
        order_depth = state.order_depths[prod]
        fairprice = (min(order_depth.sell_orders, key=order_depth.sell_orders.get) 
              + max(order_depth.buy_orders, key=order_depth.buy_orders.get)) / 2

        self.history[prod].append(fairprice)

        # check for Olivia's signal
        buy, sell = False, False
        market_trades = state.market_trades.get(prod, [])
        olivia_bought = []
        olivia_sold = []
        for trade in market_trades:
            if trade.buyer == SIGNAL_PERSON:         
                olivia_bought.append(trade.price)
            elif trade.seller == SIGNAL_PERSON:
                olivia_sold.append(trade.price)
        
        if len(olivia_bought) > 0 and self.history[prod][-2] < np.mean(self.history[prod][:-1]):
            self.global_min[prod] = self.history[prod][-2]
        elif len(olivia_sold) > 0 and self.history[prod][-2] > np.mean(self.history[prod][:-1]):
            self.global_max[prod] = self.history[prod][-2]
        
        if self.global_max[prod] is not None:
            # check if price is within a tolerable range form the global maximum
            dev = (fairprice - self.global_max[prod]) / self.global_max[prod]
            if dev > -0.002:
                sell = True
        elif self.global_min[prod] is not None:
            # check if price is within a tolerable range form the global maximum
            dev = (fairprice - self.global_min[prod]) / self.global_min[prod]
            if dev < 0.002:
                buy = True
        
        sellorders = sorted(list(order_depth.sell_orders.items()))
        buyorders = sorted(list(order_depth.buy_orders.items()), reverse=True)

        # market taking: if there's global min/max
        pos = state.position.get(prod, 0)
        pos_lim = self.POS_LIM[prod]
        if pos < pos_lim and buy:
            mybuyvol = min(-sellorders[0][1], pos_lim-pos)
            if mybuyvol > 0:
                orders.append(Order(prod, sellorders[0][0], mybuyvol))
            orders.append(Order(prod, sellorders[0][0]+1, min(pos_lim-(pos+mybuyvol), pos_lim-mybuyvol)))
            return orders
            
        if pos > -pos_lim and sell:
            mysellvol = min(buyorders[0][1], pos_lim+pos)
            if mysellvol > 0:
                orders.append(Order(prod, buyorders[0][0], -mysellvol))
                if pos < 0:
                    pos -= mysellvol
                else:
                    pos = -mysellvol
            orders.append(Order(prod, buyorders[0][0]-1, -(pos+pos_lim)))
            return orders

        return []


    def order_jams_djembes(self, state: TradingState):
        orders = {"JAMS": [], "DJEMBES": []}
        MAX_ORDER_SIZE = {
        "JAMS": 5,
        "DJEMBES": 3
        }

        for product in orders:
            if product not in state.order_depths:
                return orders

            order_depth = state.order_depths[product]
            if not order_depth.buy_orders or not order_depth.sell_orders:
                return orders

            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = (min(order_depth.sell_orders, key=order_depth.sell_orders.get) 
              + max(order_depth.buy_orders, key=order_depth.buy_orders.get)) / 2
            spread = best_ask - best_bid
            position = state.position.get(product, 0)
            pos_limit = self.POS_LIM[product]

            if product == "JAMS":
                # Passive Market Making
                buy_price = int(mid_price - 1)
                buy_qty = min(pos_limit - position, MAX_ORDER_SIZE["JAMS"])
                if buy_qty > 0:
                    orders[product].append(Order(product, buy_price, buy_qty))

                sell_price = int(mid_price + 1)
                sell_qty = min(pos_limit + position, MAX_ORDER_SIZE["JAMS"])
                if sell_qty > 0:
                    orders[product].append(Order(product, sell_price, -sell_qty))

            elif product == "DJEMBES":
                # Event-based Sniper Strategy with higher threshold
                if spread >= 6:
                    buy_price = best_bid + 1
                    buy_qty = min(pos_limit - position, MAX_ORDER_SIZE["DJEMBES"])
                    if buy_qty > 0:
                        orders[product].append(Order(product, buy_price, buy_qty))

                    sell_price = best_ask - 1
                    sell_qty = min(pos_limit + position, MAX_ORDER_SIZE["DJEMBES"])
                    if sell_qty > 0:
                        orders[product].append(Order(product, sell_price, -sell_qty))

        return orders


    def order_baskets(self, state: TradingState):
        baskets = ["PICNIC_BASKET1", "PICNIC_BASKET2"]
        pos_lim = [self.POS_LIM[basket] for basket in baskets]
        adverse_lim = [60, 30]
        orders = {basket: [] for basket in baskets}
        hedge_ratio = 2.16
        sgn = [1, -hedge_ratio]
        window = 1000
        threshold = 3.
        exit = 0.5
        pos = [state.position.get(baskets[i], 0) for i in range(2)]

        if baskets[0] not in state.order_depths or baskets[1] not in state.order_depths:
            return orders

        # PB1 - 2.16 PB2: hedge ratio 2.16
        spread = 0
        for prod, sign in zip(baskets, sgn):
            order_depth = state.order_depths[prod]
            fairprice = (min(order_depth.sell_orders, key=order_depth.sell_orders.get) 
                + max(order_depth.buy_orders, key=order_depth.buy_orders.get)) / 2
            spread += sign * fairprice
        self.history[baskets[0]].append(spread)
        self.history[baskets[0]] = self.history[baskets[0]][-window:]

        if len(self.history[baskets[0]]) < window:
            return orders
            
        hist = self.history[baskets[0]]
        ### CHANGE MEAN LATER
        mean = -6629.809185333337
        std = np.std(hist[-window:])
        z = (hist[-1] - mean) / std if std != 0 else 0

        if z > threshold and pos[0] >= -adverse_lim[0] and pos[1] <= adverse_lim[1]:       # sell PB1; buy PB2
            max_b2 = pos_lim[1] - pos[1]
            max_b1 = pos_lim[0] + pos[0]
            bestbid1, bid_vol1 = sorted(state.order_depths[baskets[0]].buy_orders.items(), reverse=True)[0]
            bestask2, ask_vol2 = sorted(state.order_depths[baskets[1]].sell_orders.items())[0]
            ask_vol2 *= -1
            # max_trade = min(max_b2/hedge_ratio, max_b1, bid_vol1/hedge_ratio, ask_vol2)
            max_trade = min(max_b2, max_b1, bid_vol1, ask_vol2)
            max_trade = int(max_trade)

            if max_trade > 0:
                orders[baskets[0]].append(Order(baskets[0], bestbid1, -max_trade))
                orders[baskets[1]].append(Order(baskets[1], bestask2, int(hedge_ratio * max_trade)))
            return orders
        
        if z < -threshold and pos[0] <= adverse_lim[0] and pos[1] >= -adverse_lim[1]:       # buy PB1; sell PB2
            max_b2 = pos_lim[1] + pos[1]
            max_b1 = pos_lim[0] - pos[0]
            bestbid2, bid_vol2 = sorted(state.order_depths[baskets[1]].buy_orders.items(), reverse=True)[0]
            bestask1, ask_vol1 = sorted(state.order_depths[baskets[0]].sell_orders.items())[0]
            ask_vol1 *= -1
            max_trade = min(max_b2/hedge_ratio, max_b1, bid_vol2/hedge_ratio, ask_vol1)
            max_trade = int(max_trade)

            if max_trade > 0:
                orders[baskets[0]].append(Order(baskets[0], bestask1, max_trade))
                orders[baskets[1]].append(Order(baskets[1], bestbid2, -int(hedge_ratio * max_trade)))
            return orders
        
        if z > -exit and z < exit:      # exit position 
            bestbid2, bid_vol2 = sorted(state.order_depths[baskets[1]].buy_orders.items(), reverse=True)[0]
            bestask1, ask_vol1 = sorted(state.order_depths[baskets[0]].sell_orders.items())[0]
            if  pos[0] < 0 and pos[1] > 0:        # buy PB1; sell PB2
                max_b2 = pos[1]
                max_b1 = -pos[0]
                orders[baskets[0]].append(Order(baskets[0], bestask1, -pos[0]))
                orders[baskets[1]].append(Order(baskets[1], bestbid2, -pos[1]))
                return orders
            if  pos[0] > 0 and pos[1] < 0:        # sell PB1; buy PB2
                max_b2 = -pos[1]
                max_b1 = pos[0]
                orders[baskets[0]].append(Order(baskets[0], bestask1, -pos[0]))
                orders[baskets[1]].append(Order(baskets[1], bestbid2, -pos[1]))
                return orders
            
        return orders


    def adapt_IV_inference(self, state: TradingState):
        """
        Numerically invert Black–Scholes for the five call options,
        fit a second-degree curve to IV vs (S-K),
        EWMA-blend into stored coefficients, and keep for next round.
        """
        alpha = 19/20
        beta = 1-alpha
        # 1) Get underlying mid-price S
        und = "VOLCANIC_ROCK"
        od_und = state.order_depths.get(und)
        if not od_und or not od_und.buy_orders or not od_und.sell_orders:
            return
        best_bid_und = max(od_und.buy_orders.keys())
        best_ask_und = min(od_und.sell_orders.keys())
        S = 0.5 * (best_bid_und + best_ask_und)

        # 2) Time to expiry
        T = (TIME_TO_EXPIRY - state.timestamp) / 365e6
        r = 0.0

        # 3) Collect market mid-prices for the five strikes
        strikes = [9500, 9750, 10000, 10250, 10500]
        mid_by_strike: dict[float, float] = {}
        for K in strikes:
            opt = f"VOLCANIC_ROCK_VOUCHER_{K}"
            od = state.order_depths.get(opt)
            if not od or not od.buy_orders or not od.sell_orders:
                return  # require all five
            bid, ask = max(od.buy_orders.keys()), min(od.sell_orders.keys())
            mid_by_strike[K] = 0.5 * (bid + ask)
        
        # 4) Invert to IV via bisection
        def iv_from_mid(mid_price: float, K0: float) -> float:
            lo, hi = 1e-6, 5.0
            for _ in range(50):
                vol = 0.5 * (lo + hi)
                price = black_scholes_call_price(S, K0, T, vol, r)
                if price > mid_price:
                    hi = vol
                else:
                    lo = vol
            return 0.5 * (lo + hi)

        xs, ys = [], []
        for K0, mid in mid_by_strike.items():
            xs.append(S - K0)
            ys.append(iv_from_mid(mid, K0))

        # 5) Fit new quadratic: returns [c2, c1, c0]
        c2, c1, c0 = np.polyfit(xs, ys, 2)

        # 6) EWMA update stored coefficients
        self.linreg["c"] = alpha * self.linreg["c"] + beta * c2
        self.linreg["b"] = alpha * self.linreg["b"] + beta * c1
        self.linreg["a"] = alpha * self.linreg["a"] + beta * c0
    

    def estimated_volatility(self, S, K):
        """
        Computes volatility from the linear regression equation.
        x is defined as the difference between the underlying price and the strike price.
        """
        x = S - K

        return self.linreg["a"] + self.linreg["b"] * x + self.linreg["c"] * x**2


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

        remaining_time = TIME_TO_EXPIRY - state.timestamp
        T = remaining_time / 365e6
        r = 0
        sigma_est = self.estimated_volatility(S, strike)
        theoretical_price = black_scholes_call_price(
            S, strike, T, sigma_est, r)
        opt_delta = black_scholes_call_delta(S, strike, T, sigma_est, r)

        print(f"theoretical price: {theoretical_price}")
        # print(f"sigma_est: {sigma_est}")
        # Determine the current market mid-price for the option.
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            market_mid = (best_bid + best_ask) / 2
        else:
            market_mid = 0

        print(f"Market mid: {market_mid}")
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
        buy_limit = theoretical_price - sd*0.4*math.exp(current_pos/200)
        sell_limit = theoretical_price + sd*0.4*math.exp(-current_pos/200)

        # how many we *may* add
        desired_buy = max(0,  self.POS_LIM[option] - current_pos)
        # how many we *may* remove
        desired_sell = max(0, -self.POS_LIM[option] - current_pos)

        # -------- cum volumes already resting in the book ---------------
        buyable = sum(-v for p, v in order_depth.sell_orders.items()
                      if p <= buy_limit)                     # asks ≤ limit
        sellable = sum(v for p, v in order_depth.buy_orders.items()
                       if p >= sell_limit)                    # bids ≥ limit

        qty_buy = min(buyable,  desired_buy)
        qty_sell = min(sellable, desired_sell)

        orders = []
        if qty_buy:
            orders.append(Order(option, round(buy_limit),  desired_buy))
        if qty_sell:
            orders.append(Order(option, round(sell_limit), -desired_sell))
        # ----------------------------------------------------------------

        # expected instant position after those aggressively priced lots
        new_pos = current_pos + qty_buy - qty_sell
        delta_exposure = new_pos * opt_delta

        return orders, delta_exposure


    def order_volcanic_rock(self, state: TradingState, hedge_delta: float):
        """Neutralise delta by sending ONE order at an extreme price (1 000 000)."""
        prod   = "VOLCANIC_ROCK"
        pos_lim = self.POS_LIM[prod]
        current_pos = state.position.get(prod, 0)
    
        target_qty = round(-hedge_delta) - current_pos
        target_qty = max(-pos_lim - current_pos,
                     min(pos_lim - current_pos, target_qty))   # obey limit
        if target_qty == 0:
            return [], 0
    
        orders = [Order(prod, 1_000_000, target_qty)]          # always at 1e6
        return orders, target_qty


    def order_macarons(self, state: TradingState):
        # try to import from Pristine at a lower price (never long as you only buy back what you sold)
        prod = "MAGNIFICENT_MACARONS"
        END_TIME = 999900
        conv_lim = 10
        conversion = 0
        pos_lim = self.POS_LIM[prod]
        pos = state.position.get(prod, 0)
        orders = []
        sell_all = False
        
        # free parameters
        adverse_lim = 15    # leave room for panic in the market
        CSI = 35
        stable_sunlight = 45
        min_profit = 10
        window = 150
        window_small = 20
        threshold = 2.
        # end of parameters
        
        # sunlight dips/peaks are signals
        if prod not in state.observations.conversionObservations:
            return [], 0

        obs = state.observations.conversionObservations[prod]
        implied_ask = obs.askPrice + obs.importTariff + obs.transportFees
        self.sunlight.append(obs.sunlightIndex)
        self.sunlight = self.sunlight[-100:]

        if self.macaron_panic and obs.sunlightIndex < stable_sunlight:      # hold
            return [], 0

        if len(self.sunlight) < 2:
            current = self.sunlight[-1]
        else:
            current = self.sunlight[-1] - self.sunlight[-2]
        if obs.sunlightIndex < CSI and self.previous_sunchange is not None:     # panic in the market
            if current * self.previous_sunchange < 0 and obs.sunlightIndex < np.mean(self.sunlight[-100:]):
                sell_all = True
                self.macaron_panic = True
        if current != 0:
            self.previous_sunchange = current

        if self.macaron_panic and obs.sunlightIndex > stable_sunlight:      # exit short position, start market taking
            if pos > -adverse_lim:
                self.macaron_panic = False
            order_depth = state.order_depths[prod]
            bestask = min(order_depth.sell_orders.keys())
            bestbid = max(order_depth.buy_orders.keys())
            sorted_sold = sorted(self.open_sells[prod].items(), reverse=True)
            for sold_price, volume in sorted_sold:
                if sold_price - implied_ask < min_profit:
                    break
                conv_vol = max(min(volume, conv_lim-conversion), 0)
                conversion += conv_vol
                volume -= conv_vol
                if conversion >= conv_lim and volume > 0 and sold_price - bestask > min_profit:
                    orders.append(Order(prod, bestask, volume))

            return orders, conversion

        # strong sell signal
        if sell_all:
            qty1 = (pos_lim+pos) // 2
            qty2 = (pos_lim+pos) - qty1
            if prod not in state.order_depths:
                return [Order(prod, implied_ask+min_profit, -qty1), Order(prod, implied_ask+min_profit-1, -qty2)], 0
            else:
                order_depth = state.order_depths[prod]
                bestbid = max(order_depth.buy_orders.keys())
                mysellvol = min(order_depth.buy_orders[bestbid], pos+pos_lim)
                if mysellvol > 0:
                    orders.append(Order(prod, bestbid, -mysellvol))
                    pos -= mysellvol
                if pos > -pos_lim:
                    orders.append(Order(prod, bestbid+1, -(pos+pos_lim)))
                print("panic selling: ", orders)
                return orders, 0

        if prod not in state.order_depths:
            return orders, conversion
        
        # normal market taking around fairvalue
        order_depth = state.order_depths[prod]
        bestbid = max(order_depth.buy_orders.keys())
        bestask = min(order_depth.sell_orders.keys())
        fairprice = (min(order_depth.sell_orders, key=order_depth.sell_orders.get) 
              + max(order_depth.buy_orders, key=order_depth.buy_orders.get)) / 2
        
        self.history[prod].append(fairprice)
        self.history[prod] = self.history[prod][-window:]

        if len(self.history[prod]) < window_small:
            return orders, conversion

        elif len(self.history[prod]) > window_small and len(self.history[prod]) < window:
            mean = np.mean(self.history[prod])
            std = np.std(self.history[prod], ddof=1)
            z = (self.history[prod][-1] - mean) / std if std != 0 else 0
        else:
            mean = np.mean(self.history[prod])
            std = np.std(self.history[prod], ddof=1)
            z = (self.history[prod][-1] - mean) / std if std != 0 else 0
        
        if z > threshold and pos > -adverse_lim:
            if len(self.open_sells[prod]) == 0:
                orders.append(Order(prod, bestbid, -min(20, adverse_lim+pos)))
            else:
                maxsold = max(self.open_sells[prod].keys())
                if bestbid >= maxsold:
                    orders.append(Order(prod, bestbid, -min(20, adverse_lim+pos)))
        elif z < -threshold and pos < 0:
            sorted_sold = sorted(self.open_sells[prod].items(), reverse=True)
            for sold_price, volume in sorted_sold:
                if sold_price - implied_ask < min_profit:
                    break
                conv_vol = max(min(volume, conv_lim-conversion), 0)
                conversion += conv_vol
                volume -= conv_vol
                if conversion >= conv_lim and volume > 0 and sold_price - bestask > min_profit:
                    orders.append(Order(prod, bestask, volume))
        elif self.last_sold_macaron is not None:
            if state.timestamp - self.last_sold_macaron > 100000:       # exit the position if held for too long
                print("BEEP! Overstayed your welcome!")
                min_profit = -5
                sorted_sold = sorted(self.open_sells[prod].items(), reverse=True)
                for sold_price, volume in sorted_sold:
                    if sold_price - implied_ask < min_profit:
                        break
                    conv_vol = max(min(volume, conv_lim-conversion), 0)
                    conversion += conv_vol
                    volume -= conv_vol
                    if conversion >= conv_lim and volume > 0 and sold_price - bestask > min_profit:
                        orders.append(Order(prod, bestask, volume))

        print("final orders: ", orders, ", conversion: ", conversion)
                    
        return orders, conversion
    

    def run(self, state: TradingState):
        
        self.timer += 100   
        conversions = 0

        # if self.timer != state.timestamp:
        #     if state.traderData != None and state.traderData != "":
        #         traderObject = jsonpickle.decode(state.traderData)
        #         self.open_buys = {item: {float(k): v for k, v in inner.items()}
        #                         for item, inner in traderObject["open buys"].items()}
        #         self.open_sells = {item: {float(k): v for k, v in inner.items()}
        #                         for item, inner in traderObject["open sells"].items()}
        #         self.history = traderObject["history"]
        #         self.sunlight = traderObject["sunlight"]
        #         self.previous_sunchange = traderObject["sun change"]
        #         self.macaron_panic = traderObject["panic"]
        #         self.last_sold_macaron = traderObject["last sold macaron"]
        #         self.upper = traderObject["upper"]
        #         self.lower = traderObject["lower"]
        #         self.signal = traderObject["signal"]
        #         self.strong_signal = traderObject["strong signal"]
        #         self.linreg = traderObject["linreg"]
        #         self.global_min = traderObject["global min"]
        #         self.global_max = traderObject["global max"]
        
        self.update_open_pos(state)

        result = {}

        # result["RAINFOREST_RESIN"] = self.order_resin(state)
        # result["KELP"] = self.order_kelp(state)
        # result["SQUID_INK"] = self.order_squid(state)
        # result["CROISSANTS"] = self.order_croissants(state)

        jams_djembes_orders = self.order_jams_djembes(state)
        for prod in ["JAMS", "DJEMBES"]:
            result[prod] = jams_djembes_orders[prod]

        # baskets_result = self.order_baskets(state)
        # for basket in baskets_result:
        #     result[basket] = baskets_result[basket]

            
        options = [
            "VOLCANIC_ROCK_VOUCHER_9500",
            "VOLCANIC_ROCK_VOUCHER_9750",
            "VOLCANIC_ROCK_VOUCHER_10000",
            "VOLCANIC_ROCK_VOUCHER_10250",
            "VOLCANIC_ROCK_VOUCHER_10500"
        ]
        
        # self.adapt_IV_inference(state)

        # delta = 0.0
        # for opt in options:
        #     opt_orders, opt_delta = self.order_volcanic_rock_option(state, opt)
        #     result[opt]   = opt_orders
        #     delta += opt_delta
                
        # print(f"unhedged delta: {delta}")
        # under_orders, hedge_qty = self.order_volcanic_rock(state, delta)
        # result["VOLCANIC_ROCK"] = under_orders
        
        # delta_hedged = delta + hedge_qty   # shares carry delta = 1
        # print(f"hedged delta: {delta_hedged}")

        # mac_orders, conversions = self.order_macarons(state)
        # result["MAGNIFICENT_MACARONS"] = mac_orders

        # traderObject = {"open buys": self.open_buys, 
        #                 "open sells": self.open_sells, 
        #                 "history": self.history,
        #                 "sunlight": self.sunlight,
        #                 "sun change": self.previous_sunchange,
        #                 "panic": self.macaron_panic,
        #                 "last sold macaron": self.last_sold_macaron,
        #                 "upper": self.upper,
        #                 "lower": self.lower,
        #                 "signal": self.signal,
        #                 "strong signal": self.strong_signal,
        #                 "linreg": self.linreg,
        #                 "global min": self.global_min,
        #                 "global max": self.global_max}
        
        # traderData = jsonpickle.encode(traderObject)

        return result, conversions, ""     # CHANGE THISS

