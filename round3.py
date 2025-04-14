from datamodel import OrderDepth, UserId, TradingState, Order
import jsonpickle
from typing import List
import string
import numpy as np
import json


class Trader:
    ### UPDATE SQUID HISTORY WITH NEW DATA (OR REMOVE IT ALTOGETHER) ###
    def __init__(self):

        print("Trader initialised!")

        # set lower limits to reduce exposure?
        self.POS_LIM = {"RAINFOREST_RESIN": 50, "KELP": 50, "SQUID_INK": 50,
                        "CROISSANTS": 250, "JAMS": 350, "DJEMBES": 60,
                        "PICNIC_BASKET1": 60, "PICNIC_BASKET2": 100}
        self.prods = list(self.POS_LIM.keys())
        self.basket_contents={"PICNIC_BASKET1": {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1},
                              "PICNIC_BASKET2": {"CROISSANTS": 4, "JAMS": 2}}

        # for tracking open positions / updating memory
        self.timer = 0
        self.open_buys = {prod: {} for prod in self.prods}
        self.open_sells = {prod: {} for prod in self.prods}
        self.recorded_time = {prod: -1 for prod in self.prods}    # last recorded time of own_trades

        # historical data and past signals for Z-score trading signals
        self.history = {prod: [] for prod in self.prods}
        self.upper = {prod: 0 for prod in self.prods}
        self.lower = {prod: 0 for prod in self.prods}
        self.signal = {prod: 0 for prod in self.prods}
        self.strong_signal = {prod: 0 for prod in self.prods}


    def update_open_pos(self, state: TradingState):
            """
            Update open positions according to updated own trades
            Later try to buy/sell lower/higher than open trades
            """
            for prod in state.own_trades:
                trades = state.own_trades[prod]
                trades = [trade for trade in trades if trade.timestamp > self.recorded_time[prod]]
                if len(trades) > 0:
                    self.recorded_time[prod] = trades[0].timestamp
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
                if prod not in state.position:
                    if sum(self.open_buys[prod].values()) - sum(self.open_sells[prod].values()) != 0:
                        print("BEEP! Open positions incorrectly tracked!")
                    # assert sum(self.open_buys[prod].values()) - sum(self.open_sells[prod].values()) == 0, \
                    #     "BEEP! Open positions incorrectly tracked!"
                else:
                    if sum(self.open_buys[prod].values()) - sum(self.open_sells[prod].values()) != state.position[prod]:
                        print("BEEP! Open positions incorrectly tracked!")
                    # assert sum(self.open_buys[prod].values()) - sum(self.open_sells[prod].values()) == state.position[prod],\
                    #     "BEEP! Open positions incorrectly tracked!"


    def check_orders(self, orders, prod):
        ### SANITY CHECK BECAUSE I'M SCARRED FROM THE TYPO IN ROUND 1 ###
        total_buy, total_sell = 0, 0
        remove = []
        for i, order in enumerate(orders):
            if order.quantity > 0:
                total_buy += order.quantity
                if total_buy > self.POS_LIM[prod]:
                    remove.append(i)
                    total_buy -= order.quantity
            elif order.quantity < 0:
                total_sell += order.quantity
                if total_sell < -self.POS_LIM[prod]:
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
            if state.position[prod] == 50:
                print("LONG LIM REACHED (RESIN)")
            elif state.position[prod] == -50:
                print("SHORT LIM REACHED (RESIN)")
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
        orders = self.check_orders(orders, prod)

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
            price = min(bestbid+1, int(maxbought))
            qty = min(pos_lim-current_long, maxmake)
            orders.append(Order(prod, price, qty))

        if self.strong_signal[prod] == -1 and current_short > -pos_lim:
            price = max(bestask-1, int(minsold))
            qty = min(current_short+pos_lim, maxmake)
            orders.append(Order(prod, price, -qty))

        return self.check_orders(orders, prod)


    def order_croissants(self, state: TradingState):
        prod = "CROISSANTS"
        orders: list[Order] = []
        order_depth = state.order_depths[prod]
        pos_lim = self.POS_LIM[prod]

        # free parameters #
        soft_lim = 50
        clear_lim = 100
        maxtake = 10
        maxmake = 20
        lookback = 10
        lookback_strong = 500
        hits = 1
        strong_hits = 10
        # end of parameters #

        # calculate fairprice based on market-making bots
        fairprice = (min(order_depth.sell_orders, key=order_depth.sell_orders.get) 
              + max(order_depth.buy_orders, key=order_depth.buy_orders.get)) / 2

        self.history[prod].append(fairprice)
        self.history[prod] = self.history[prod][-(lookback_strong+1):]

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
            maxbought = 1e4
            minbought = -1
        if len(sold_prices) > 0:
            minsold = min(sold_prices)
            maxsold = max(sold_prices)
        else:
            minsold = -1
            maxsold = 1e4

        # market taking
        for sellorder in sellorders:
            ask, ask_amount = sellorder
            if (self.strong_signal[prod] == 1 or self.signal[prod] == 1) and (current_long < soft_lim):
                if ask < maxbought:
                    mybuyvol = min(-ask_amount, soft_lim-current_long)
                    mybuyvol = min(mybuyvol, maxtake)
                    if mybuyvol > 0:
                        # update minbought to avoid buying higher in the same timestep
                        if minbought == -1 or ask < minbought:
                            minbought = ask
                        orders.append(Order(prod, ask, mybuyvol))
                        current_long += mybuyvol

        for buyorder in buyorders:
            bid, bid_amount = buyorder
            if (self.strong_signal[prod] == -1 or self.signal[prod] == -1) and current_short > -soft_lim:
                if bid > minsold:
                    mysellvol = min(bid_amount, soft_lim+current_short)
                    mysellvol = min(mysellvol, maxtake)
                    mysellvol *= -1
                    if mysellvol < 0:
                        # update maxsold to avoid selling lower in the same timestep
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
            price = min(bestbid+1, int(maxbought))
            qty = min(pos_lim-current_long, maxmake)
            orders.append(Order(prod, price, qty))

        if self.strong_signal[prod] == -1 and current_short > -pos_lim:
            price = max(bestask-1, int(minsold))
            qty = min(current_short+pos_lim, maxmake)
            orders.append(Order(prod, price, -qty))

        return self.check_orders(orders, prod)
    

    def order_basket(self, state: TradingState, current_positions, basket, lookback=10, lookback_strong=100, hits=5, strong_hits=50, maxbuy=10):
        
        # atm only try to either market take or clear positions, not both together

        if basket not in state.order_depths:
            return None
        if not state.order_depths[basket].buy_orders or not state.order_depths[basket].sell_orders:
            return None

        pos_lim = self.POS_LIM[basket]
        contents = self.basket_contents[basket]
        basket_position = 0
        if basket in state.position:
            basket_position = state.position[basket]
        
        basket_order_depth = state.order_depths[basket]
        basket_best_bid = max(basket_order_depth.buy_orders.keys())
        basket_best_ask = min(basket_order_depth.sell_orders.keys())
        basket_bid_vol = basket_order_depth.buy_orders[basket_best_bid]
        basket_ask_vol = -basket_order_depth.sell_orders[basket_best_ask]
        basket_swmp = (basket_best_bid * basket_ask_vol + 
                       basket_best_ask * basket_bid_vol) / (basket_bid_vol + basket_ask_vol)
        
        print(basket_order_depth.sell_orders, basket_order_depth.buy_orders)

        # get synthetic (implied) midprice
        bid_vols, ask_vols = [], []
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
        
        if implied_bid > 0 and implied_ask < float("inf"):
            implied_ask_vol = min(ask_vols)
            implied_bid_vol = min(bid_vols)
            synthetic_swmp = (implied_bid * implied_ask_vol + 
                              implied_ask * implied_bid_vol) / (implied_bid_vol + implied_ask_vol)
        
        else:
            return None         # contents midprice unavailable

        spread = basket_swmp - synthetic_swmp
        self.history[basket].append(spread)
        self.history[basket] = self.history[basket][-lookback_strong:]

        if len(self.history[basket]) < lookback + 1:
            return None     # not enough memory
        
        self.find_signal_breakout(basket, lookback, lookback_strong, True)
        self.process_signal(basket, hits, strong_hits)
        
        content_orders = {}

        # # clear orders
        # clear_lim = int(0.75 * adverse_lim)
        # if basket_position <= -clear_lim:
        #     if spread_hist[-1] - sma <= - mult2 * std:
        #         print("CLEAR BUY BASKET ", basket)
        #         basket_lim = adverse_lim - basket_position
        #         mybuyvol = [-(basket_position+adverse_lim), implied_bid_vol, basket_lim]

        #         # check positions of individual content (to sell)
        #         for prod in contents:
        #             available = self.POS_LIM[prod] + current_positions[prod]
        #             mybuyvol.append(available // contents[prod])
                
        #         mybuyvol = min(mybuyvol)
        #         if mybuyvol > 0:
        #             basket_order = Order(basket, basket_best_ask, mybuyvol)

        #             for prod in contents:
        #                 best_bid = max(state.order_depths[prod].buy_orders.keys())
        #                 content_orders[prod] = Order(prod, best_bid, -mybuyvol * contents[prod])
                    
        #             return basket_order, content_orders
        
        # if basket_position >= clear_lim:
        #     if spread_hist[-1] - sma >= mult2 * std:
        #         print("CLEAR SELL BASKET ", basket)
        #         basket_lim = basket_position + adverse_lim
        #         mysellvol = [basket_position-adverse_lim, implied_ask_vol, basket_lim]

        #         # check positions of individual content
        #         for prod in contents:
        #             available = self.POS_LIM[prod] - current_positions[prod]
        #             mysellvol.append(available // contents[prod])
        #         mysellvol = min(mysellvol)
        #         if mysellvol > 0:
        #             basket_order = Order(basket, basket_best_bid, -mysellvol)
        #             for prod in contents:
        #                 best_ask = min(state.order_depths[prod].buy_orders.keys())
        #                 content_orders[prod] = Order(prod, best_ask, mysellvol * contents[prod])
        #             return basket_order, content_orders

        if self.strong_signal[basket] == 1 and basket_position < pos_lim:
            print("STRONG SIGNAL")
            basket_lim = pos_lim - basket_position
            qty_basket = [basket_ask_vol, implied_bid_vol, basket_lim, maxbuy]

            # check positions of individual content
            for prod in contents:
                available = self.POS_LIM[prod] + current_positions[prod]
                if available <= 1:
                    print("blocked by ", prod)
                qty_basket.append(available // contents[prod])
            
            qty_basket = min(qty_basket)
            if qty_basket > 0:
                basket_order = Order(basket, basket_best_ask, qty_basket)

                for prod in contents:
                    best_bid = max(state.order_depths[prod].buy_orders.keys())
                    content_orders[prod] = Order(prod, best_bid, -qty_basket * contents[prod])
                
                return basket_order, content_orders
        
        if self.strong_signal[basket] == -1 and basket_position > -pos_lim:
            print("STRONG SIGNAL")
            basket_lim = pos_lim + basket_position
            qty_basket = [basket_bid_vol, implied_ask_vol, basket_lim, maxbuy]

            # check positions of individual content
            for prod in contents:
                available = self.POS_LIM[prod] - current_positions[prod]
                if available <= 1:
                    print("blocked by ", prod)
                qty_basket.append(available // contents[prod])
            
            qty_basket = min(qty_basket)
            if qty_basket > 0:
                basket_order = Order(basket, basket_best_bid, -qty_basket)

                for prod in contents:
                    best_ask = min(state.order_depths[prod].buy_orders.keys())
                    content_orders[prod] = Order(prod, best_ask, qty_basket * contents[prod])
                
                return basket_order, content_orders
            
        return None


    def run(self, state: TradingState):
        self.timer += 100   

        ### UPDATE WITH DAY N DATA ###
        linreg = {"KELP":  (15.11018100844262, 9.768257499187243e-06, 2057.517022160843)}

        if self.timer != state.timestamp:
            if state.traderData != None and state.traderData != "":
                traderObject = jsonpickle.decode(state.traderData)
                self.open_buys = {item: {float(k): v for k, v in inner.items()}
                                for item, inner in traderObject["open buys"].items()}
                self.open_sells = {item: {float(k): v for k, v in inner.items()}
                                for item, inner in traderObject["open sells"].items()}
                self.recorded_time = traderObject["recorded time"]
                self.history = traderObject["history"]
                self.upper = traderObject["upper"]
                self.lower = traderObject["lower"]
                self.signal = traderObject["signal"]
                self.strong_signal = traderObject["strong signal"]
        
        self.update_open_pos(state)

        result = {}
        # debug mode
        # if state.timestamp < 100:

        # result["RAINFOREST_RESIN"] = self.order_resin(state)
        # result["KELP"] = self.order_kelp(state, *linreg["KELP"])
        # result["SQUID_INK"] = self.order_squid(state)

        # result["CROISSANTS"] = []
        # result["JAMS"] = []
        # result["DJEMBES"] = []
        # baskets = ["PICNIC_BASKET1", "PICNIC_BASKET2"]

        # # lookback, lookback_strong, hits, strong_hits, maxbuy
        # basket_params = {"PICNIC_BASKET1": (20, 100, 2, 10, 5), "PICNIC_BASKET2":  (10, 100, 5, 50, 5)}  
        # current_positions = {prod: (state.position[prod] if prod in state.position else 0) 
        #                      for prod in self.basket_contents[baskets[0]]} 
        # for basket in baskets[:1]:
        #     args = basket_params[basket]
        #     spread_order = self.order_basket(state, current_positions, basket, *args)
        #     if spread_order is not None:
        #         basket_order, content_orders = spread_order
        #         result[basket] = [basket_order]
        #         for prod in content_orders:
        #             result[prod].append(content_orders[prod])
        #             current_positions[prod] += content_orders[prod].quantity
        
        # print(self.history["PICNIC_BASKET1"])
        
        # # only order the individual contents if not ordering baskets
        # if "PICNIC_BASKET1" not in result:
        #     result["DJEMBES"] = self.order_component(state, "DJEMBES", logic="breakout", pos_lim=10)
        #     if "PICNIC_BASKET2" not in result:
        #         result["CROISSANTS"] = self.order_component(state, "CROISSANTS", logic="trend", pos_lim=15)
        #         result["JAMS"] = self.order_component(state, "JAMS", logic="momentum", pos_lim=20)

        result["CROISSANTS"] = self.order_croissants(state)

        traderObject = {"open buys": self.open_buys, 
                        "open sells": self.open_sells, 
                        "recorded time": self.recorded_time,
                        "history": self.history,
                        "upper": self.upper,
                        "lower": self.lower,
                        "signal": self.signal,
                        "strong signal": self.strong_signal}
        
        traderData = jsonpickle.encode(traderObject)
        conversions = 1
        return result, conversions, traderData

