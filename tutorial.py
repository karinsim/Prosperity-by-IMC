from datamodel import OrderDepth, UserId, TradingState, Order
import jsonpickle
from typing import List
import string
import numpy as np


class Trader:
    def __init__(self):

        print("Trader initialised!")

        self.POS_LIM = {"RAINFOREST_RESIN": 50, "KELP": 50}
        self.prods = ["RAINFOREST_RESIN", "KELP"]
        self.open_buys = {prod: {} for prod in self.prods}
        self.open_sells = {prod: {} for prod in self.prods}
        self.recorded_time = {prod: -1 for prod in self.prods}    # last recorded time of own_trades

        self.kelp_flag = False
        self.kelp_hist = []


    def update_open_pos(self, state: TradingState):
        """
        Update open positions according to updated own trades
        Later try to buy/sell lower/higher than open trades
        """

        # print("before: ", self.open_sells, self.open_buys)
        # print("trades: ", state.own_trades)

        for prod in state.own_trades:
            trades = state.own_trades[prod]
            if trades[0].timestamp > self.recorded_time[prod]:
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

        # print("after: ", self.open_sells, self.open_buys)

        # sanity check: position
        for prod in self.prods:
            # if prod not in state.position:
            #     if sum(self.open_buys[prod].values()) - sum(self.open_sells[prod].values()) != 0:
            #         print("Open positions incorrectly tracked!")
                # assert sum(self.open_buys[prod].values()) - sum(self.open_sells[prod].values()) == 0, "wrong open pos2"
            # else:
            if prod in state.position:
                if sum(self.open_buys[prod].values()) - sum(self.open_sells[prod].values()) != state.position[prod]:
                    print("Open positions incorrectly tracked!")
                    # print("after: ", self.open_sells, self.open_buys)
                # assert sum(self.open_buys[prod].values()) - sum(self.open_sells[prod].values()) == state.position[prod], "wrong open pos"


    def order_resin(self, state: TradingState):
        orders: list[Order] = []
        prod = "RAINFOREST_RESIN"
        # free parameters
        fairprice = 10000
        make_bid = fairprice - 2
        make_ask = fairprice + 2
        atol = 1
        param1 = 1.0
        # end of parameters

        # track long and short separately to prevent cancelling out
        current_short, current_long = 0, 0
        if prod in state.position:
            current_pos = state.position[prod]
            if current_pos > 0:
                current_long += current_pos
            else:
                current_short += current_pos

        pos_lim = self.POS_LIM[prod]
        order_depth = state.order_depths[prod]
        sellorders = sorted(list(order_depth.sell_orders.items()))
        buyorders = sorted(list(order_depth.buy_orders.items()), reverse=True)
        
        # market taking
        for sellorder in sellorders:
            ask, ask_amount = sellorder

            if current_long < pos_lim:
                if ask <= fairprice + atol:
                    mybuyvol = min(-ask_amount, pos_lim-current_long)
                    assert(mybuyvol >= 0), "Buy volume negative"
                    orders.append(Order(prod, ask, mybuyvol))
                    current_long += mybuyvol
                else:
                    # if price is higher than the fp, can still buy if it's lower than the current open sells
                    price_list = sorted(list(self.open_sells[prod].keys()))
                    for price in price_list:
                        if ask < price:
                            mybuyvol = min(ask_amount, self.open_sells[prod][price],
                                            pos_lim-current_long)
                            assert(mybuyvol >= 0), "Buy volume negative"
                            orders.append(Order(prod, ask, mybuyvol))
                            current_long += mybuyvol

        for buyorder in buyorders:
            bid, bid_amount = buyorder

            if current_short > -pos_lim:
                if bid >= fairprice - atol:
                    mysellvol = min(bid_amount, pos_lim+current_short)
                    mysellvol *= -1
                    assert(mysellvol <= 0), "Sell volume positive"
                    orders.append(Order(prod, bid, mysellvol))
                    current_short += mysellvol
                else:
                    price_list = sorted(list(self.open_buys[prod].keys()), reverse=True)
                    for price in price_list:
                        if bid > price:
                            mysellvol = min(bid_amount, self.open_buys[prod][price],
                                            pos_lim+current_short)
                            assert(mysellvol <= 0), "Sell volume positive"
                            orders.append(Order(prod, bid, mysellvol))
                            current_short += mysellvol

        # market making: fill the remaining orders up to position limit
        if current_long < pos_lim:
            qty1 = int((pos_lim - current_long) * param1)
            qty2 = pos_lim - current_long - qty1
            orders.append(Order(prod, make_bid, qty1))
            orders.append(Order(prod, make_bid-1, qty2))   # try to buy even lower
        if current_short > -pos_lim:
            qty1 = int((pos_lim + current_short) * param1)
            qty2 = pos_lim + current_short - qty1
            orders.append(Order(prod, make_ask, -qty1))
            orders.append(Order(prod, make_ask+1, -qty2))   # try to sell even higher

        return orders


    def order_kelp(self, state: TradingState):
        orders: list[Order] = []
        prod = "KELP"
        order_depth = state.order_depths[prod]

        # # to log live order book
        # print("sell", order_depth.sell_orders)
        # print("buy", order_depth.buy_orders)

        # calculate fairprice based on market-making bots
        fairprice = (min(order_depth.sell_orders, key=order_depth.sell_orders.get) 
              + max(order_depth.buy_orders, key=order_depth.buy_orders.get)) / 2
        self.kelp_hist.append(fairprice)

        # free parameters
        pos_lim = self.POS_LIM[prod]
        adverse_lim = pos_lim - 5
        maxqty = 10
        atol = 0
        window = 20
        mult = 1.75
        param1 = 0.75
        make_bid = round(fairprice - 1)
        make_ask = round(fairprice + 0.5)
        # end of parameters

        if not self.kelp_flag:
            if len(self.kelp_hist) == window:
                self.kelp_flag = True

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
                if ask <= fairprice + atol:
                    mybuyvol = min(-ask_amount, pos_lim-current_long, maxqty)
                    assert(mybuyvol >= 0), "Buy volume negative"
                    orders.append(Order(prod, ask, mybuyvol))
                    current_long += mybuyvol
                else:
                    # reduce open position if approaching position limit
                    if current_pos <= -adverse_lim:
                    # if len(self.open_sells[prod]) > 0:
                        min_sell = max(self.open_sells[prod].keys())        # max: easiest; min: more profitable
                        if ask < min_sell + atol:
                                mybuyvol = min(self.open_sells[prod][min_sell], adverse_lim)
                                assert(mybuyvol >= 0), "Buy volume negative"
                                orders.append(Order(prod, ask, mybuyvol))
                                current_long += mybuyvol

        for buyorder in buyorders:
            bid, bid_amount = buyorder

            if current_short > -pos_lim:
                if bid >= fairprice - atol:
                    mysellvol = min(bid_amount, pos_lim+current_short, maxqty)
                    mysellvol *= -1
                    assert(mysellvol <= 0), "Sell volume positive"
                    orders.append(Order(prod, bid, mysellvol))
                    current_short += mysellvol
                
                else:
                    # reduce open position if approaching position limit
                    if current_pos >= adverse_lim:
                    # if len(self.open_buys[prod]) > 0:
                        max_buy = min(self.open_buys[prod].keys())        # min: easiest; max: more profitable
                        if bid > max_buy - atol:
                                mysellvol = max(-self.open_buys[prod][max_buy], -adverse_lim)
                                assert(mysellvol <= 0), "Sell volume positive"
                                orders.append(Order(prod, bid, mysellvol))
                                current_short += mysellvol

        # close open positions with mean reversion strategy
        # if self.kelp_flag:
        #     mean, std = np.mean(self.kelp_hist), np.std(self.kelp_hist)
        #     upper, lower = mean + mult * std, mean - mult * std
        #     if fairprice <= lower:      # exit short position
        #         if len(self.open_sells[prod]) > 0:
        #             min_sell = min(self.open_sells[prod].keys())
        #             for sellorder in sellorders:
        #                 ask, ask_amount = sellorder
        #                 if ask < min_sell:
        #                     mybuyvol = min(self.open_sells[prod][min_sell],
        #                                     pos_lim-current_long)
        #                     assert(mybuyvol >= 0), "Buy volume negative"
        #                     orders.append(Order(prod, ask, mybuyvol))
        #                     current_long += mybuyvol
        #     elif fairprice >= upper:      # exit long position
        #         if len(self.open_buys[prod]) > 0:
        #             max_buy = max(self.open_buys[prod].keys())
        #             for buyorder in buyorders:
        #                 bid, bid_amount = buyorder
        #                 if bid > max_buy:
        #                     mysellvol = min(pos_lim+current_short, 
        #                                     self.open_buys[prod][max_buy])
        #                     mysellvol *= -1
        #                     assert(mysellvol <= 0), "Sell volume positive"
        #                     orders.append(Order(prod, bid, mysellvol))
        #                     current_short += mysellvol

        #     self.kelp_hist = self.kelp_hist[1:]

        # market making: fill the remaining orders up to position limit
        if current_long < adverse_lim:
            qty1 = int((adverse_lim - current_long) * param1)
            qty2 = adverse_lim - current_long - qty1
            orders.append(Order(prod, make_bid, qty1))
            orders.append(Order(prod, make_bid-1, qty2))   # try to buy even lower
            current_long += qty1 + qty2
        if current_short > -adverse_lim:
            qty1 = int((adverse_lim + current_short) * param1)
            qty2 = adverse_lim + current_short - qty1
            orders.append(Order(prod, make_ask, -qty1))
            orders.append(Order(prod, make_ask+1, -qty2))   # try to sell even higher
            current_short -= (qty1 + qty2)

        # print("orders: ", orders)
        return orders


    def run(self, state: TradingState):

        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        print('length of history', len(self.kelp_hist))

        result = {}

        # debug mode
        if state.timestamp < 1e10:

            print("pos: ", state.position)
            print("obs: ", state.observations)

            self.update_open_pos(state)
            result["RAINFOREST_RESIN"] = self.order_resin(state)
            result["KELP"] = self.order_kelp(state)

        # traderData = "SAMPLE"
        traderData = jsonpickle.encode(traderObject)
        
        conversions = 1
        return result, conversions, traderData
