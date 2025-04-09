from datamodel import OrderDepth, UserId, TradingState, Order
import jsonpickle
from typing import List
import string
import numpy as np
import json


class Trader:
    def __init__(self):

        print("Trader initialised!")

        self.POS_LIM = {"RAINFOREST_RESIN": 50, "KELP": 50, "SQUID_INK": 50}
        self.prods = ["RAINFOREST_RESIN", "KELP", "SQUID_INK"]

        # if tracking open position / relying on memory
        self.open_buys = {prod: {} for prod in self.prods}
        self.open_sells = {prod: {} for prod in self.prods}
        self.recorded_time = {prod: -1 for prod in self.prods}    # last recorded time of own_trades
        self.signal = 0

        # SQUID
        self.squid_hist = [1842.5, 1844.5, 1843.5, 1842.5, 1842.0, 1841.5, 1841.0, 1839.5, 1833.0, 1833.5,
                           1832.5, 1831.5, 1832.5, 1832.5, 1830.5, 1831.5, 1833.0, 1833.5, 1838.5, 1839.5]
        self.upper = 0
        self.lower = 0

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
                        print("Open positions incorrectly tracked!")
                else:
                    if sum(self.open_buys[prod].values()) - sum(self.open_sells[prod].values()) != state.position[prod]:
                        print("Open positions incorrectly tracked!")
                #     assert sum(self.open_buys[prod].values()) - sum(self.open_sells[prod].values()) == 0, \
                #         "Open positions incorrectly tracked!"
                # else:
                #     assert sum(self.open_buys[prod].values()) - sum(self.open_sells[prod].values()) == state.position[prod],\
                #         "Open positions incorrectly tracked!"


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

        # mean reversion parameters #
        pos_lim = 35
        clear_lim = 20
        maxqty = 8
        maxmake = 20
        window = 20
        mult = 1.25
        mult_exit = 0.2
        hits = 1
        strong_hits = 8
        consecutive_hits = 1
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
        self.squid_hist = self.squid_hist[-window:]
        
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

        # Mean reversion strategy
        sma = np.mean(np.array(self.squid_hist))
        std = np.std(np.array(self.squid_hist))
        z = (self.squid_hist[-1] - sma) / std

        if z < -mult:
            self.lower += 1
            self.upper = 0
        elif z > mult:
            self.lower = 0
            self.upper += 1
        elif np.abs(z) < mult_exit:
            action = "EXIT"
            self.lower = 0
            self.upper = 0
        else:
            self.lower = 0
            self.upper = 0
    
        if self.upper >= hits:
            action = "SELL"
            self.signal -= 1
            if self.upper >= strong_hits:
                strong_signal = "SELL"
        elif self.lower >= hits:
            action = "BUY"
            self.signal += 1
            if self.lower >= strong_hits:
                strong_signal = "BUY"
            
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
        if action == "BUY":
            if (self.signal == 1 or self.signal >= consecutive_hits or strong_signal == "BUY") and (current_long < pos_lim):
                for sellorder in sellorders:
                    ask, ask_amount = sellorder
                    if ask <= fairprice and ask < maxbought:
                        mybuyvol = min(-ask_amount, pos_lim-current_long)
                        mybuyvol = min(mybuyvol, maxqty)
                        assert(mybuyvol >= 0), "Buy volume negative"
                        orders.append(Order(prod, ask, mybuyvol))
                        current_long += mybuyvol
                    elif ask > fairprice and ask < minbought:
                        mybuyvol = min(-ask_amount, pos_lim-current_long)
                        mybuyvol = min(mybuyvol, maxqty)
                        assert(mybuyvol >= 0), "Buy volume negative"
                        orders.append(Order(prod, ask, mybuyvol))
                        current_long += mybuyvol

        if action == "SELL":
            if (self.signal == -1 or self.signal <= -consecutive_hits or strong_signal == "SELL") and (current_short > -pos_lim):
                for buyorder in buyorders:
                    bid, bid_amount = buyorder
                    if bid < fairprice:
                        break
                    if bid >= fairprice and bid > minsold:
                        mysellvol = min(bid_amount, pos_lim+current_short)
                        mysellvol = min(mysellvol, maxqty)
                        mysellvol *= -1
                        assert(mysellvol <= 0), "Sell volume positive"
                        orders.append(Order(prod, bid, mysellvol))
                        current_short += mysellvol
                    elif bid < fairprice and bid > maxsold:
                        mysellvol = min(bid_amount, pos_lim+current_short)
                        mysellvol = min(mysellvol, maxqty)
                        mysellvol *= -1
                        assert(mysellvol <= 0), "Sell volume positive"
                        orders.append(Order(prod, bid, mysellvol))
                        current_long += mybuyvol

        # clear open positions if approaching position limit
        if current_long > clear_lim or action == "EXIT":
            for price in bought_prices:
                if fairprice > price:
                    orders.append(Order(prod, round_ask+1, -self.open_buys[prod][price]))
                elif bids[0] > price:
                    orders.append(Order(prod, bids[0], -self.open_buys[prod][price]))
                elif asks[0] > price:
                    orders.append(Order(prod, asks[0]-1, -self.open_buys[prod][price]))
                else:
                    orders.append(Order(prod, int(price), -self.open_buys[prod][price]))
        
        if current_short < -clear_lim or action == "EXIT":
            for price in sold_prices:
                if fairprice < price:
                    orders.append(Order(prod, round_bid-1, self.open_sells[prod][price]))
                elif asks[0] < price:
                    orders.append(Order(prod, asks[0], self.open_sells[prod][price]))
                elif bids[0] < price:
                    orders.append(Order(prod, bids[0]+1, self.open_sells[prod][price]))
                else:
                    orders.append(Order(prod, int(price), self.open_sells[prod][price]))
            
        # market making: strong signal
        if strong_signal == "BUY" and current_long < pos_lim:
            bestbid = buyorders[0][0]
            mmbid = max(order_depth.buy_orders, key=order_depth.buy_orders.get)
            qty = pos_lim - current_long
            if bestbid < fairprice:
                price = min(bestbid+1, round_bid-1, int(maxbought))
            else:
                price = min(mmbid+1, round_bid, int(maxbought))
            qty = min(qty, maxmake)
            orders.append(Order(prod, price, qty))
        if strong_signal == "SELL" and current_short > -pos_lim:
            bestask = sellorders[0][0]
            mmask = max(order_depth.sell_orders, key=order_depth.sell_orders.get)
            qty = current_short + pos_lim
            if bestask > fairprice:
                price = max(bestask-1, round_ask+1, int(minsold))
            else:
                price = max(mmask-1, round_ask, int(minsold))
            qty = min(qty, maxmake)
            orders.append(Order(prod, price, -qty))
            print("Make order: ", orders[-1])

        return orders
    

    def run(self, state: TradingState):
        
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
            self.signal = traderObject["squid signal"]
        
        self.update_open_pos(state)

        result = {}
        # # debug mode
        # if state.timestamp < 100:
        result["RAINFOREST_RESIN"] = self.order_resin(state)
        result["KELP"] = self.order_kelp(state)
        result["SQUID_INK"] = self.order_squid(state)

        traderObject = {"open buys": self.open_buys, 
                        "open sells": self.open_sells, 
                        "recorded time": self.recorded_time,
                        "squid history": self.squid_hist,
                        "squid upper": self.upper,
                        "squid lower": self.lower,
                        "squid signal": self.signal}
        traderData = jsonpickle.encode(traderObject)

        conversions = 1
        return result, conversions, traderData
    
