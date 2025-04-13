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

        # set lower limit for squid to reduce exposure
        self.POS_LIM = {"RAINFOREST_RESIN": 50, "KELP": 50, "SQUID_INK": 50,
                         "CROISSANTS": 50, "JAMS": 50, "DJEMBES": 50,
                        "PICNIC_BASKET1": 50, "PICNIC_BASKET2": 50}
        
        self.prods = list(self.POS_LIM.keys())

        self.history = {p: [] for p in self.prods}
        self.recorded_time = {p: -1 for p in self.prods}
        self.open_buys = {p: {} for p in self.prods}
        self.open_sells = {p: {} for p in self.prods}
        # SQUID (replace with day 1 data)
        self.squid_hist = []
        self.upper = 0
        self.lower = 0
        self.slow_upper = 0
        self.slow_lower = 0


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
                    assert(mybuyvol >= 0), "Buy volume negative"
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
                    assert(mysellvol <= 0), "Sell volume positive"
                    orders.append(Order(prod, bid, mysellvol))
                    current_short += mysellvol

        # clear open positions if approaching position limit (buy/sell at fairprice to increase chance of order acceptance)
        if current_long > clear_lim:
            mysellvol = -min(current_short+pos_lim, current_long-clear_lim)
            assert mysellvol <= 0, "Sell volume positive!"
            orders.append(Order(prod, fairprice, mysellvol))
            current_short += mysellvol
        if current_short < -clear_lim:
            mybuyvol = max(pos_lim-current_long, -(clear_lim+current_short))
            assert mybuyvol >= 0, "Buy volume negative!"
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
            if state.position[prod] == 50:
                print("LONG LIM REACHED "+prod)
            elif state.position[prod] == -50:
                print("SHORT LIM REACHED"+prod)
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
                    assert(mybuyvol >= 0), "Buy volume negative"
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
        soft_lim = 5
        med_lim = 10
        clear_lim = 10
        maxqty = 5
        swing = 3
        maxmake = 10
        window_fast = 10
        window_slow = 50
        mult = 1.
        mult_strong = 3.
        hits = 3       # number of times mp hits upper/lower to be considered signal
        strong_hits = 20  # number of consecutive hits to be considered strong signal
        strong_signal = None
        action = None
        # end of parameters #

        # # to log live order book
        # print("sell", order_depth.sell_orders)
        # print("buy", order_depth.buy_orders)

        # calculate fairprice based on market-making bots
        fairprice = (min(order_depth.sell_orders, key=order_depth.sell_orders.get) 
              + max(order_depth.buy_orders, key=order_depth.buy_orders.get)) / 2
        
        round_bid = round(fairprice - 1)
        round_ask = round(fairprice + 0.5)

        self.squid_hist.append(fairprice)
        self.squid_hist = self.squid_hist[-window_slow:]
        
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

        z_slow, z_fast = None, None
        # Mean reversion strategy
        if len(self.squid_hist) >= window_fast:
            sma_fast = np.mean(np.array(self.squid_hist[-window_fast:]))
            std_fast = np.std(np.array(self.squid_hist[-window_fast:]), ddof=1)
            z_fast= (self.squid_hist[-1] - sma_fast) / std_fast

        if len(self.squid_hist) == window_slow:
            sma_slow = np.mean(np.array(self.squid_hist))
            std_slow = np.std(np.array(self.squid_hist), ddof=1)
            z_slow = (self.squid_hist[-1] - sma_slow) / std_slow

        if z_fast is not None:
            if z_fast < -mult:
                self.lower += 1
                self.upper = 0
            elif z_fast > mult:
                self.lower = 0
                self.upper += 1
            else:
                self.lower = 0
                self.upper = 0
            if self.upper >= hits:      # sell signal
                action = "SELL"
                self.upper = self.lower = 0
            elif self.lower >= hits:        # buy
                action = "BUY"
                self.upper = self.lower = 0

        if z_slow is not None:
            if z_slow <= -mult_strong or z_fast <= -mult_strong:
                strong_signal = "BUY"
                self.lower = self.upper = 0
            elif z_slow >= mult_strong or z_fast >= mult_strong:
                strong_signal = "SELL"
                self.lower = self.upper = 0
            else:
                if z_slow < -mult:
                    self.slow_lower += 1
                    self.slow_upper = 0
                elif z_slow > mult:
                    self.slow_upper += 1
                    self.slow_lower = 0  

            if self.slow_upper >= strong_hits:      # strong sell
                strong_signal = "SELL"
                self.slow_lower = self.slow_upper = 0
            elif self.slow_lower >= strong_hits:        # strong buy
                strong_signal = "BUY"
                self.slow_lower = self.slow_upper = 0
            
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
            # large price change
            if ask < minbought - swing or (ask < minsold - swing and ask < minbought) and current_long < soft_lim:
                mybuyvol = min(maxqty, pos_lim-current_long, -ask_amount)
                minbought = ask
                # if mybuyvol > 0:
                #     orders.append(Order(prod, ask, mybuyvol))
                #     current_long += mybuyvol
                print("LARGE SWING (BUY), min bought: ", minbought)
            
            else:
                if (strong_signal == "BUY" or action == "BUY") and (current_long < soft_lim):
                    if (ask <= fairprice and ask < maxbought) or (ask > fairprice and ask < minbought):
                        mybuyvol = min(-ask_amount, soft_lim-current_long)
                        mybuyvol = min(mybuyvol, maxqty)
                        assert(mybuyvol >= 0), "Buy volume negative"
                        # update minbought to avoid buying higher in the same timestep
                        if minbought == -1 or ask < minbought:
                            minbought = ask
                        orders.append(Order(prod, ask, mybuyvol))
                        current_long += mybuyvol
                        print("BUY SIGNAL")

        for buyorder in buyorders:
            bid, bid_amount = buyorder
            # large price change
            if bid > maxsold + swing or (bid > maxbought + swing and bid > maxsold) and current_short > -soft_lim:
                mysellvol = min(maxqty, current_short+pos_lim)
                mysellvol *= -1
                maxsold = bid
                # if mysellvol < 0:
                #     orders.append(Order(prod, bid, mysellvol))
                #     current_short += mysellvol
                print("LARGE SWING (SELL), maxsold: ", maxsold, bid)
            
            else:
                if (strong_signal == "SELL" or action == "SELL") and current_short > -soft_lim:
                    if (bid >= fairprice and bid > minsold) or (bid < fairprice and bid > maxsold):
                        mysellvol = min(bid_amount, soft_lim+current_short)
                        mysellvol = min(mysellvol, maxqty)
                        mysellvol *= -1
                        assert(mysellvol <= 0), "Sell volume positive"
                        # update maxsold to avoid selling lower in the same timestep
                        if bid > maxsold:
                            maxsold = bid
                        orders.append(Order(prod, bid, mysellvol))
                        current_short += mysellvol
                        print("SELL SIGNAL")

        # clear open positions if approaching position limit
        if current_long > clear_lim:
            for price in bought_prices:
                qty = min(self.open_buys[prod][price], soft_lim+current_short)
                if qty > 0:
                    if bids[0] > price:
                        orders.append(Order(prod, bids[0], -qty))
                    elif asks[0] > price:
                        orders.append(Order(prod, asks[0]-1, -qty))
                    else:
                        orders.append(Order(prod, max(int(price), int(fairprice+0.5)), -qty))
                    current_short -= qty
        
        if current_short < -clear_lim:
            for price in sold_prices:
                qty = min(self.open_sells[prod][price], soft_lim-current_long)
                if qty > 0:
                    if asks[0] < price:
                        orders.append(Order(prod, asks[0], qty))
                    elif bids[0] < price:
                        orders.append(Order(prod, bids[0]+1, qty))
                    else:
                        orders.append(Order(prod, min(int(price), int(fairprice)), qty))
                    current_long += qty
            
        # market making: strong signal (ignore soft limits)
        bestbid = buyorders[0][0]
        bestask = sellorders[0][0]
        if strong_signal == "BUY" and current_long < med_lim:
            price = min(bestbid+1, round_bid, int(maxbought))
            qty = min(med_lim-current_long, maxmake)
            orders.append(Order(prod, bestbid+1, qty))
            print("STRONG BUY")

        if strong_signal == "SELL" and current_short > -med_lim:
            price = max(bestask-1, round_ask, int(minsold))
            qty = min(current_short+med_lim, maxmake)
            orders.append(Order(prod, price, -qty))
            print("STRONG SELL")

        return self.check_orders(orders, prod)
            
    
    def order_component(self, state: TradingState, prod: str, logic: str):
        orders: List[Order] = []
        order_depth = state.order_depths[prod]
        pos = state.position.get(prod, 0)
        mid_price = None
        if order_depth.buy_orders and order_depth.sell_orders:
            mid_price = (max(order_depth.buy_orders) + min(order_depth.sell_orders)) / 2
        else:
            return []

        self.history[prod].append(mid_price)

        if logic == "trend" and len(self.history[prod]) > 30:
            sma = np.mean(self.history[prod][-30:])
            if mid_price > sma + 2 and pos > -self.POS_LIM[prod]:
                orders.append(Order(prod, int(mid_price - 1), -5))
            elif mid_price < sma - 2 and pos < self.POS_LIM[prod]:
                orders.append(Order(prod, int(mid_price + 1), 5))

        elif logic == "momentum" and len(self.history[prod]) > 20:
            delta = mid_price - self.history[prod][-20]
            if delta > 3 and pos > -self.POS_LIM[prod]:
                orders.append(Order(prod, int(mid_price - 1), -5))
            elif delta < -3 and pos < self.POS_LIM[prod]:
                orders.append(Order(prod, int(mid_price + 1), 5))

        elif logic == "breakout" and len(self.history[prod]) > 10:
            recent = self.history[prod][-10:]
            if mid_price > max(recent) and pos > -self.POS_LIM[prod]:
                orders.append(Order(prod, int(mid_price - 1), -5))
            elif mid_price < min(recent) and pos < self.POS_LIM[prod]:
                orders.append(Order(prod, int(mid_price + 1), 5))

        return orders

    def order_basket(self, state: TradingState, basket: str):
        orders: List[Order] = []
        prices = {}
        for p in ["CROISSANTS", "JAMS", "DJEMBES", "PICNIC_BASKET1", "PICNIC_BASKET2"]:
            od = state.order_depths.get(p, None)
            if od and od.sell_orders and od.buy_orders:
                prices[p] = (min(od.sell_orders) + max(od.buy_orders)) // 2

        pos = state.position.get(basket, 0)
        margin = 2

        if basket == "PICNIC_BASKET1":
            components = 6 * prices.get("CROISSANTS", 0) + 3 * prices.get("JAMS", 0) + prices.get("DJEMBES", 0)
        else:
            components = 4 * prices.get("CROISSANTS", 0) + 2 * prices.get("JAMS", 0)

        basket_price = prices.get(basket)
        if basket_price is None:
            return []

        if components - basket_price > margin and pos < self.POS_LIM[basket]:
            orders.append(Order(basket, basket_price + 1, 1))
        elif basket_price - components > margin and pos > -self.POS_LIM[basket]:
            orders.append(Order(basket, basket_price - 1, -1))

        return orders


    def run(self, state: TradingState):

        ### UPDATE WITH DAY N DATA ###
        linreg = linreg = {"KELP":  (14.422964389386117, 1.173227183514493e-05, 2052.6409802838834)}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
            self.open_buys = {item: {float(k): v for k, v in inner.items()}
                            for item, inner in traderObject["open buys"].items()}
            self.open_sells = {item: {float(k): v for k, v in inner.items()}
                            for item, inner in traderObject["open sells"].items()}
            self.recorded_time = traderObject["recorded time"]
            self.squid_hist = traderObject["squid history"]
            self.upper = traderObject["squid upper"]
            self.lower = traderObject["squid lower"]
            self.slow_upper = traderObject["squid slow upper"]
            self.slow_lower = traderObject["squid slow lower"]
            self.history = traderObject["history"]
    
        self.update_open_pos(state)

        result = {}
        # debug mode
        # if state.timestamp < 100:

        result["RAINFOREST_RESIN"] = self.order_resin(state)
        result["KELP"] = self.order_kelp(state, *linreg["KELP"])
        result["SQUID_INK"] = self.order_squid(state)

        result["CROISSANTS"] = self.order_component(state, "CROISSANTS", logic="trend")
        result["JAMS"] = self.order_component(state, "JAMS", logic="momentum")
        result["DJEMBES"] = self.order_component(state, "DJEMBES", logic="breakout")
        result["PICNIC_BASKET1"] = self.order_basket(state, "PICNIC_BASKET1")
        result["PICNIC_BASKET2"] = self.order_basket(state, "PICNIC_BASKET2")
    
        traderObject = {"open buys": self.open_buys, 
                        "open sells": self.open_sells, 
                        "recorded time": self.recorded_time,
                        "squid history": self.squid_hist,
                        "squid upper": self.upper,
                        "squid lower": self.lower,
                        "squid slow upper": self.slow_upper,
                        "squid slow lower": self.slow_lower,
                        "history": self.history}

        traderData = jsonpickle.encode(traderObject)

        conversions = 1
        return result, conversions, traderData

