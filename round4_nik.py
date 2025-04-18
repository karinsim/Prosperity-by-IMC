
from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import numpy as np
import jsonpickle

class Trader:
    def __init__(self):
        self.POS_LIM = {
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100
        }
        self.basket1 = {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1}
        self.basket2 = {"CROISSANTS": 4, "JAMS": 2}
        self.spread_history = []
        self.window = 100
        self.max_ticks = 30  # Holding period limit

    def get_basket_price(self, state: TradingState, basket: dict):
        price = 0
        for prod, qty in basket.items():
            if prod not in state.order_depths:
                return None
            od = state.order_depths[prod]
            if od.buy_orders and od.sell_orders:
                best_bid = max(od.buy_orders.keys())
                best_ask = min(od.sell_orders.keys())
                mid = (best_bid + best_ask) / 2
                price += mid * qty
            else:
                return None
        return price

    def run(self, state: TradingState):
        orders = {}
        conversions = 0

        p1 = self.get_basket_price(state, self.basket1)
        p2 = self.get_basket_price(state, self.basket2)
        if p1 is None or p2 is None:
            return {k: [] for k in self.POS_LIM}, conversions, ""

        spread = p1 - p2
        self.spread_history.append(spread)
        if len(self.spread_history) > self.window:
            self.spread_history.pop(0)

        for basket in self.POS_LIM:
            orders[basket] = []

        # Load persistent trader state
        trader_state = {"tick_in_trade": 0, "in_trade": False}
        if state.traderData:
            try:
                trader_state = jsonpickle.decode(state.traderData)
            except:
                pass

        # Z-score logic
        if len(self.spread_history) >= self.window:
            hist = np.array(self.spread_history[-self.window:])
            mean = np.mean(hist)
            std = np.std(hist)
            x = self.spread_history[-1]

            if std == 0:
                return orders, conversions, jsonpickle.encode(trader_state)

            z_score = (x - mean) / std
            pos1 = state.position.get("PICNIC_BASKET1", 0)
            pos2 = state.position.get("PICNIC_BASKET2", 0)

            # Tiers for pyramiding
            tiers_basket1 = {2.5: 30, 1.5: 20, 1.0: 10}
            tiers_basket2 = {2.5: 50, 1.5: 30, 1.0: 15}

            def tiered_size(z, tiers):
                abs_z = abs(z)
                for threshold, size in sorted(tiers.items(), reverse=True):
                    if abs_z >= threshold:
                        return size
                return 0

            size1 = tiered_size(z_score, tiers_basket1)
            size2 = tiered_size(z_score, tiers_basket2)

            z_exit_threshold = 0.5
            trader_state["tick_in_trade"] += 1 if trader_state.get("in_trade", False) else 0

            if abs(z_score) < z_exit_threshold or trader_state["tick_in_trade"] > self.max_ticks:
                # Exit logic
                if pos1 < 0:
                    bid_price1 = max(state.order_depths["PICNIC_BASKET1"].buy_orders.keys())
                    orders["PICNIC_BASKET1"].append(Order("PICNIC_BASKET1", bid_price1, min(-pos1, 30)))
                if pos2 > 0:
                    ask_price2 = min(state.order_depths["PICNIC_BASKET2"].sell_orders.keys())
                    orders["PICNIC_BASKET2"].append(Order("PICNIC_BASKET2", ask_price2, -min(pos2, 50)))
                if pos1 > 0:
                    ask_price1 = min(state.order_depths["PICNIC_BASKET1"].sell_orders.keys())
                    orders["PICNIC_BASKET1"].append(Order("PICNIC_BASKET1", ask_price1, -min(pos1, 30)))
                if pos2 < 0:
                    bid_price2 = max(state.order_depths["PICNIC_BASKET2"].buy_orders.keys())
                    orders["PICNIC_BASKET2"].append(Order("PICNIC_BASKET2", bid_price2, min(-pos2, 50)))
                trader_state["in_trade"] = False
                trader_state["tick_in_trade"] = 0

            elif z_score > 1.0:
                # Short spread with pyramiding
                vol1 = min(self.POS_LIM["PICNIC_BASKET1"] + pos1, size1)
                vol2 = min(self.POS_LIM["PICNIC_BASKET2"] - pos2, size2)
                if vol1 > 0 and vol2 > 0:
                    ask_price1 = min(state.order_depths["PICNIC_BASKET1"].sell_orders.keys())
                    bid_price2 = max(state.order_depths["PICNIC_BASKET2"].buy_orders.keys())
                    orders["PICNIC_BASKET1"].append(Order("PICNIC_BASKET1", ask_price1, -vol1))
                    orders["PICNIC_BASKET2"].append(Order("PICNIC_BASKET2", bid_price2, vol2))
                    trader_state["in_trade"] = True

            elif z_score < -1.0:
                # Long spread with pyramiding
                vol1 = min(self.POS_LIM["PICNIC_BASKET1"] - pos1, size1)
                vol2 = min(self.POS_LIM["PICNIC_BASKET2"] + pos2, size2)
                if vol1 > 0 and vol2 > 0:
                    bid_price1 = max(state.order_depths["PICNIC_BASKET1"].buy_orders.keys())
                    ask_price2 = min(state.order_depths["PICNIC_BASKET2"].sell_orders.keys())
                    orders["PICNIC_BASKET1"].append(Order("PICNIC_BASKET1", bid_price1, vol1))
                    orders["PICNIC_BASKET2"].append(Order("PICNIC_BASKET2", ask_price2, -vol2))
                    trader_state["in_trade"] = True

        traderData = jsonpickle.encode(trader_state)
        return orders, conversions, traderData
