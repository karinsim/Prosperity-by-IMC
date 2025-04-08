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
        self.last_recorded_time = -100
        self.amnesia = True
        self.open_buys = {prod: {} for prod in self.prods}
        self.open_sells = {prod: {} for prod in self.prods}
        self.recorded_time = {prod: -1 for prod in self.prods}    # last recorded time of own_trades


    def update_open_pos(self, state: TradingState):
            """
            Update open positions according to updated own trades
            Later try to buy/sell lower/higher than open trades
            """

            for prod in state.own_trades:
                trades = state.own_trades[prod]
                trades = [trade for trade in trades if trade.timestamp > self.recorded_time[prod]]
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
                self.recorded_time[prod] = trade.timestamp

                # sanity check: position
                if not self.amnesia:
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


    def order_kelp(self, prod, state: TradingState):
        orders: list[Order] = []
        order_depth = state.order_depths[prod]
        pos_lim = self.POS_LIM[prod]

        # # to log live order book
        # print("sell", order_depth.sell_orders)
        # print("buy", order_depth.buy_orders)

        # calculate fairprice based on market-making bots
        fairprice = (min(order_depth.sell_orders, key=order_depth.sell_orders.get) 
              + max(order_depth.buy_orders, key=order_depth.buy_orders.get)) / 2

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
        for sellorder in sellorders:
            ask, ask_amount = sellorder

            if current_long < pos_lim:
                if ask < fairprice:
                    mybuyvol = min(-ask_amount, pos_lim-current_long)
                    assert(mybuyvol >= 0), "Buy volume negative"
                    orders.append(Order(prod, ask, mybuyvol))
                    current_long += mybuyvol

        for buyorder in buyorders:
            bid, bid_amount = buyorder

            if current_short > -pos_lim:
                if bid > fairprice:
                    mysellvol = min(bid_amount, pos_lim+current_short)
                    mysellvol *= -1
                    assert(mysellvol <= 0), "Sell volume positive"
                    orders.append(Order(prod, bid, mysellvol))
                    current_short += mysellvol

        # market making: fill the remaining orders up to position limit
        make_bid = round(fairprice - 1)
        make_ask = round(fairprice + 0.5)
        if current_long < pos_lim:
            bestbid = buyorders[0][0]
            qty = (pos_lim - current_long)
            price = min(bestbid+1, make_bid)
            orders.append(Order(prod, price, qty))
            current_long += qty
        if current_short > -pos_lim:
            bestask = sellorders[0][0]
            qty = pos_lim + current_short
            price = max(bestask-1, make_ask)
            orders.append(Order(prod, price, -qty))
            current_short -= qty

        # print("orders: ", orders)
        return orders


    def run(self, state: TradingState):

        traderObject = {"SAMPLE"}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        
        if self.last_recorded_time + 100 == state.timestamp:
            self.amnesia = False

        result = {}

        # # debug mode
        # if state.timestamp < 5e4:
        result["RAINFOREST_RESIN"] = self.order_resin(state)
        # result["KELP"] = self.order_kelp("KELP", state)
        # result["SQUID_INK"] = self.order_kelp("SQUID_INK", state)

        traderData = jsonpickle.encode(traderObject)
        
        conversions = 1
        return result, conversions, traderData
    
