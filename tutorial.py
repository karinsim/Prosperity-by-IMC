from datamodel import OrderDepth, UserId, TradingState, Order
import jsonpickle
from typing import List
import string
import numpy as np
import json


# Save history as global variables (Prosperity server seems to randomly reinitialise Trader)
PRODUCTS = ["RAINFOREST_RESIN", "KELP"]


class Trader:
    def __init__(self):

        print("Trader initialised!")

        self.POS_LIM = {"RAINFOREST_RESIN": 50, "KELP": 50}
        self.prods = ["RAINFOREST_RESIN", "KELP"]
        self.positions_correct = False


    def order_resin(self, state: TradingState):
        orders: list[Order] = []
        prod = "RAINFOREST_RESIN"
        pos_lim = self.POS_LIM[prod]
        # free parameters
        fairprice = 10000
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
        bestask, bestbid = sellorders[0][0], buyorders[0][0]
        if current_long < pos_lim:
            qty = pos_lim - current_long
            price = min(bestbid + 1, fairprice - 2)
            orders.append(Order(prod, price, qty))
        if current_short > -pos_lim:
            qty = pos_lim + current_short
            price = max(bestask - 1, fairprice + 2)
            orders.append(Order(prod, price, -qty))

        return orders


    def order_kelp(self, state: TradingState):
        orders: list[Order] = []
        prod = "KELP"
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
                print("LONG LIM REACHED")
            elif state.position[prod] == -50:
                print("SHORT LIM REACHED")
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

        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}

        # debug mode
        if state.timestamp < 1e10:
            result["RAINFOREST_RESIN"] = self.order_resin(state)
            result["KELP"] = self.order_kelp(state)

        # traderData = "SAMPLE"
        traderData = jsonpickle.encode(traderObject)
        print(traderData)
        
        conversions = 1
        return result, conversions, traderData
    
