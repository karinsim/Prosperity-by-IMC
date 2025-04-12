
from datamodel import Order, OrderDepth, TradingState
import jsonpickle
from typing import List, Dict
import numpy as np


class Trader:
    def __init__(self):
        print("Round 2 Trader Initialized")

        self.POS_LIM = {
            "CROISSANTS": 50, "JAMS": 50, "DJEMBES": 50,
            "PICNIC_BASKET1": 50, "PICNIC_BASKET2": 50
        }

        self.prods = list(self.POS_LIM.keys())

        self.history = {p: [] for p in self.prods}
        self.recorded_time = {p: -1 for p in self.prods}
        self.open_buys = {p: {} for p in self.prods}
        self.open_sells = {p: {} for p in self.prods}

    def update_open_pos(self, state: TradingState):
        for prod in state.own_trades:
            trades = state.own_trades[prod]
            trades = [trade for trade in trades if trade.timestamp > self.recorded_time[prod]]
            if trades:
                self.recorded_time[prod] = trades[0].timestamp
            for trade in trades:
                remaining_quantity = trade.quantity
                if trade.buyer == "SUBMISSION":
                    for price in sorted(self.open_sells[prod]):
                        if remaining_quantity <= 0:
                            break
                        if price in self.open_sells[prod]:
                            avail = self.open_sells[prod][price]
                            if remaining_quantity >= avail:
                                remaining_quantity -= avail
                                del self.open_sells[prod][price]
                            else:
                                self.open_sells[prod][price] -= remaining_quantity
                                remaining_quantity = 0
                    if remaining_quantity > 0:
                        self.open_buys[prod][trade.price] = self.open_buys[prod].get(trade.price, 0) + remaining_quantity
                else:
                    for price in sorted(self.open_buys[prod]):
                        if remaining_quantity <= 0:
                            break
                        if price in self.open_buys[prod]:
                            avail = self.open_buys[prod][price]
                            if remaining_quantity >= avail:
                                remaining_quantity -= avail
                                del self.open_buys[prod][price]
                            else:
                                self.open_buys[prod][price] -= remaining_quantity
                                remaining_quantity = 0
                    if remaining_quantity > 0:
                        self.open_sells[prod][trade.price] = self.open_sells[prod].get(trade.price, 0) + remaining_quantity

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
        if state.traderData:
            traderObject = jsonpickle.decode(state.traderData)
            self.history = traderObject["history"]
            self.open_buys = {item: {float(k): v for k, v in inner.items()} for item, inner in traderObject["open_buys"].items()}
            self.open_sells = {item: {float(k): v for k, v in inner.items()} for item, inner in traderObject["open_sells"].items()}
            self.recorded_time = traderObject["recorded_time"]

        self.update_open_pos(state)

        result = {
            "CROISSANTS": self.order_component(state, "CROISSANTS", logic="trend"),
            "JAMS": self.order_component(state, "JAMS", logic="momentum"),
            "DJEMBES": self.order_component(state, "DJEMBES", logic="breakout"),
            "PICNIC_BASKET1": self.order_basket(state, "PICNIC_BASKET1"),
            "PICNIC_BASKET2": self.order_basket(state, "PICNIC_BASKET2"),
        }

        traderObject = {
            "history": self.history,
            "open_buys": self.open_buys,
            "open_sells": self.open_sells,
            "recorded_time": self.recorded_time
        }

        traderData = jsonpickle.encode(traderObject)
        return result, 0, traderData
