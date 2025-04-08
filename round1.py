from datamodel import OrderDepth, UserId, TradingState, Order
import jsonpickle
from typing import List
import string
import numpy as np
import json


class Trader:
    def __init__(self):

        self.POS_LIM = {"RAINFOREST_RESIN": 50, "KELP": 50, "SQUID_INK": 50}
        self.prods = ["RAINFOREST_RESIN", "KELP", "SQUID_INK"]

        # if tracking open position / relying on memory
        self.open_buys = {prod: {} for prod in self.prods}
        self.open_sells = {prod: {} for prod in self.prods}
        self.recorded_time = {prod: -1 for prod in self.prods}    # last recorded time of own_trades

        # SQUID
        slow, fast, signal = 26, 12, 9
        self.squid_hist = []
        self.squid_flag = False
        self.squid_upper = 0
        self.squid_lower = 0
        self.alpha_fast = 2 / (fast + 1)
        self.alpha_slow = 2 / (slow + 1)
        self.alpha_signal = 2 / (signal + 1)
        self.ema_fast = None
        self.ema_slow = None
        self.macd = None


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
                    assert sum(self.open_buys[prod].values()) - sum(self.open_sells[prod].values()) == 0, \
                        "Open positions incorrectly tracked!"
                else:
                    assert sum(self.open_buys[prod].values()) - sum(self.open_sells[prod].values()) == state.position[prod],\
                        "Open positions incorrectly tracked!"


    def order_resin(self, state: TradingState):
        orders: list[Order] = []
        prod = "RAINFOREST_RESIN"
        pos_lim = self.POS_LIM[prod]
        # free parameters
        fairprice = 10000
        clear_lim = pos_lim - 1
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

        # clear open positions if approaching position limit
        if current_long > clear_lim:
            mysellvol = -min(current_short+pos_lim, current_long-clear_lim)
            assert mysellvol <= 0, "Sell volume positive!"
            if len(bids) > 0:
                orders.append(Order(prod, max(bids[0], fairprice), mysellvol))
            else:
                orders.append(Order(prod, fairprice, mysellvol))
            current_short += mysellvol
        if current_short < -clear_lim:
            mybuyvol = max(pos_lim-current_long, -(clear_lim+current_short))
            assert mybuyvol >= 0, "Buy volume negative!"
            if len(asks) > 0:
                orders.append(Order(prod, min(asks[0], fairprice), mybuyvol))
            else:
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

        # # to log live order book
        # print("sell", order_depth.sell_orders)
        # print("buy", order_depth.buy_orders)

        # calculate fairprice based on market-making bots
        fairprice = (min(order_depth.sell_orders, key=order_depth.sell_orders.get) 
              + max(order_depth.buy_orders, key=order_depth.buy_orders.get)) / 2
        round_bid = round(fairprice - 1)
        round_ask = round(fairprice + 0.5)

        # track long and short separately to prevent cancelling out
        current_short, current_long = 0, 0
        if prod in state.position:
            if state.position[prod] == 50:
                print("LONG LIM REACHED (KELP)")
            elif state.position[prod] == -50:
                print("SHORT LIM REACHED (KELP)")
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
                    assert(mysellvol <= 0), "Sell volume positive"
                    orders.append(Order(prod, bid, mysellvol))
                    current_short += mysellvol
        
        # clear open positions if approaching position limit
        if current_long > clear_lim:
            mysellvol = min(current_short+pos_lim, current_long-clear_lim)
            assert mysellvol >= 0, "Sell volume positive!"
            if len(self.open_buys[prod]) > 0:
                bought_prices = sorted(list(self.open_buys[prod].keys()), reverse=True)
                i = 0
                while mysellvol > 0 and i < len(bought_prices):
                    qty = min(mysellvol, self.open_buys[prod][bought_prices[i]])
                    orders.append(Order(prod, int(bought_prices[i]+1), -qty))
                    i += 1
                    mysellvol -= qty
                    current_short -= qty
            else:
                if len(bids) > 0:
                    orders.append(Order(prod, max(bids[0], round_ask), mysellvol))
                else:
                    orders.append(Order(prod, round_ask, mysellvol))
                current_short += mysellvol
        if current_short < -clear_lim:
            mybuyvol = max(pos_lim-current_long, -(clear_lim+current_short))
            assert mybuyvol >= 0, "Buy volume negative!"
            if len(self.open_sells[prod]) > 0:
                sold_prices = sorted(list(self.open_sells[prod].keys()))
                i = 0
                while mybuyvol > 0 and i < len(sold_prices):
                    qty = min(mybuyvol, self.open_sells[prod][sold_prices[i]])
                    orders.append(Order(prod, int(sold_prices[i]-1), qty))
                    i += 1
                    mybuyvol += qty
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
            if bestbid < fairprice:
                price = min(bestbid+1, round_bid)
            else:
                price = min(mmbid+1, round_bid)
            orders.append(Order(prod, price, qty))
        if current_short > -pos_lim:
            qty = pos_lim + current_short
            if bestask > fairprice:
                price = max(bestask-1, round_ask)
            else:
                price = max(mmask-1, round_ask)
            orders.append(Order(prod, price, -qty))

        # print("orders: ", orders)
        return orders


    def order_squid(self, state: TradingState):
        prod = "SQUID_INK"
        orders: list[Order] = []
        order_depth = state.order_depths[prod]
        pos_lim = self.POS_LIM[prod]

        ### params ###
        window = 20
        mult = 2.
        threshold = 1.
        maxqty = 10
        hits = 3
        ### end of parameters ###

        # # to log live order book
        # print("sell", order_depth.sell_orders)
        # print("buy", order_depth.buy_orders)

        # calculate fairprice based on market-making bots
        fairprice = (min(order_depth.sell_orders, key=order_depth.sell_orders.get) 
              + max(order_depth.buy_orders, key=order_depth.buy_orders.get)) / 2
        self.squid_hist.append(fairprice)

        if not self.squid_flag:
            if len(self.squid_hist) >= window:
                self.squid_flag = True
        
        sellorders = sorted(list(order_depth.sell_orders.items()))
        buyorders = sorted(list(order_depth.buy_orders.items()), reverse=True)

        # track long and short separately to prevent cancelling out
        current_short, current_long = 0, 0
        if prod in state.position:
            if state.position[prod] == 50:
                print("LONG LIM REACHED (SQUID)")
            elif state.position[prod] == -50:
                print("SHORT LIM REACHED (SQUID)")
            current_pos = state.position[prod]
            if current_pos > 0:
                current_long += current_pos
            else:
                current_short += current_pos
        
        else:
            current_pos = 0
        
         # MACD
        if self.ema_fast is None:
            self.ema_fast = fairprice
            self.ema_slow = fairprice
            self.macd = self.ema_signal = 0 
        else:
            self.ema_fast = self.alpha_fast * fairprice + (1 - self.alpha_fast) * self.ema_fast
            self.ema_slow = self.alpha_slow * fairprice + (1 - self.alpha_slow) * self.ema_slow
            self.ema_signal = self.alpha_signal * self.macd + (1 - self.alpha_signal) * self.ema_signal
            self.macd = self.ema_fast - self.ema_slow

        # check if you have enough historical data
        if not self.squid_flag:
            if len(self.squid_hist) == window:
                self.squid_flag = True

        if self.squid_flag:
            sma = np.mean(np.array(self.squid_hist))
            upper = sma + mult * np.std(np.array(self.squid_hist))
            lower = sma - mult * np.std(np.array(self.squid_hist))
            price_move =  (upper - lower) * threshold
            if fairprice >= upper + price_move:
                self.squid_upper += 1
                self.squid_lower = 0
            if fairprice <= lower - price_move:
                self.squid_lower += 1
                self.squid_upper = 0
            else:
                self.squid_upper = 0
                self.squid_lower = 0
                
            # buy/sell when midprice touches upper/lower band N=hits number of times consecutively
            if current_long < pos_lim and self.squid_lower >= hits and self.macd > self.ema_signal:
                ask, ask_amount = sellorders[0]     # only consider lowest ask for now
                mybuyvol = min(-ask_amount, pos_lim-current_long, maxqty)
                assert(mybuyvol >= 0), "Buy volume negative"
                current_long += mybuyvol
                orders.append(Order(prod, ask, mybuyvol))
            
            if current_short > -pos_lim and self.squid_upper >= hits and self.macd < self.ema_signal:
                bid, bid_amount = buyorders[0]
                mysellvol = min(bid_amount, pos_lim+current_short, maxqty)
                mysellvol *= -1
                assert(mysellvol <= 0), "Sell volume positive"
                current_short += mysellvol
                orders.append(Order(prod, bid, mysellvol))

            self.squid_hist = self.squid_hist[1:]

        return orders


    def run(self, state: TradingState):
        
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

            self.open_buys = {item: {float(k): v for k, v in inner.items()}
                            for item, inner in traderObject["open buys"].items()}
            self.open_sells = {item: {float(k): v for k, v in inner.items()}
                            for item, inner in traderObject["open sells"].items()}
            self.recorded_time = traderObject["recorded time"]
        
        self.update_open_pos(state)

        if any(bool(inner) for inner in self.open_buys.values()) or any(bool(inner) for inner in self.open_sells.values()):
            traderObject = {"open buys": self.open_buys, 
                            "open sells": self.open_sells, 
                            "recorded time": self.recorded_time}
            traderData = jsonpickle.encode(traderObject)
        else:
            traderData = ""

        result = {}
        # # debug mode
        # if state.timestamp < 900:
        # result["RAINFOREST_RESIN"] = self.order_resin(state)
        # result["KELP"] = self.order_kelp(state)
        result["SQUID_INK"] = self.order_squid(state)
        
        conversions = 1
        return result, conversions, traderData
    
