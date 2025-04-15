from datamodel import OrderDepth, UserId, TradingState, Order
import jsonpickle
from typing import List
import string
import numpy as np
import json
import math
import statistics as st


class Trader:
    ### UPDATE SQUID HISTORY WITH NEW DATA (OR REMOVE IT ALTOGETHER) ###
    def __init__(self):

        print("Trader initialised!")

        # set lower limit for squid to reduce exposure
        self.POS_LIM = {
            "RAINFOREST_RESIN": 50, "KELP": 50, "SQUID_INK": 50,
            "CROISSANTS": 250, "JAMS": 350, "DJEMBES": 60,
            "PICNIC_BASKET1": 60, "PICNIC_BASKET2": 100,
            "VOLCANIC_ROCK": 400,
            "VOLCANIC_ROCK_VOUCHER_9500": 200,
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200,
            "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200
        }
        self.prods = list(self.POS_LIM.keys())
        self.basket_contents = {"PICNIC_BASKET1": {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1},
                                "PICNIC_BASKET2": {"CROISSANTS": 4, "JAMS": 2}}

        # if tracking open position / relying on memory
        self.timer = 0
        self.open_buys = {prod: {} for prod in self.prods}
        self.open_sells = {prod: {} for prod in self.prods}
        # last recorded time of own_trades
        self.recorded_time = {prod: -1 for prod in self.prods}

        # SQUID (replace with day 1 data)
        self.squid_hist = []
        self.upper = 0
        self.lower = 0
        self.slow_upper = 0
        self.slow_lower = 0

        # SPREAD (baskets)
        self.spread_hist = {"PICNIC_BASKET1": [], "PICNIC_BASKET2": []}
        self.history = {prod: [] for prod in self.prods}

    def black_scholes_call_price(S, K, T, sigma,r):
        """
        Computes the Black-Scholes price for a call option.
        S: Underlying price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free rate
        sigma: Volatility
        
        print(f"Underlying price: {S}")
        print(f"Strike price: {K}")
        print(f"Time to expiration (in years): {T}")
        print(f"Risk-free rate: {r}")
        print(f"Volatility: {sigma}")
        """
        d1 = (math.log(S/K) + (r+ 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        
        # Using the error function to get the standard normal CDF
        N_d1 = 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
        N_d2 = 0.5 * (1 + math.erf(d2 / math.sqrt(2)))
        return S * N_d1 - K * N_d2*math.exp(-r*T)
    
    
    # Regression-based volatility estimator
    def estimated_volatility(S, K, a=0.150195656, b=-1.15870319e-06, c=1.23214223e-07):
        """
        Computes volatility from the linear regression equation.
        x is defined as the difference between the underlying price and the strike price.
        """
        x = S - K
        return a + b * x + c * x**2

    def update_open_pos(self, state: TradingState):
        """
        Update open positions according to updated own trades
        Later try to buy/sell lower/higher than open trades
        """
        for prod in state.own_trades:
            trades = state.own_trades[prod]
            trades = [trade for trade in trades if trade.timestamp >
                      self.recorded_time[prod]]
            if len(trades) > 0:
                self.recorded_time[prod] = trades[0].timestamp
            for trade in trades:
                remaining_quantity = trade.quantity
                if trade.buyer == "SUBMISSION":
                    sold_price = sorted(
                        list(self.open_sells[prod].keys()), reverse=True)
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
                    assert (mybuyvol >= 0), "Buy volume negative"
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
                    assert (mysellvol <= 0), "Sell volume positive"
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
                    assert (mybuyvol >= 0), "Buy volume negative"
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
                bought_prices = sorted(
                    list(self.open_buys[prod].keys()), reverse=True)
                i = 0
                while mysellvol > 0 and i < len(bought_prices) and current_short > -pos_lim:
                    qty = min(
                        mysellvol, self.open_buys[prod][bought_prices[i]], current_short+pos_lim)
                    orders.append(Order(prod, int(bought_prices[i]+1), -qty))
                    i += 1
                    mysellvol -= qty
                    current_short -= qty
            else:
                if len(bids) > 0:
                    orders.append(
                        Order(prod, max(bids[0], round_ask), -mysellvol))
                else:
                    orders.append(Order(prod, round_ask, -mysellvol))
                current_short -= mysellvol
        if current_short < -short_lim and current_long < pos_lim:
            mybuyvol = min(pos_lim-current_long, -(short_lim+current_short))
            if len(self.open_sells[prod]) > 0:
                sold_prices = sorted(list(self.open_sells[prod].keys()))
                i = 0
                while mybuyvol > 0 and i < len(sold_prices) and current_long < pos_lim:
                    qty = min(
                        mybuyvol, self.open_sells[prod][sold_prices[i]], pos_lim-current_long)
                    orders.append(Order(prod, int(sold_prices[i]-1), qty))
                    i += 1
                    mybuyvol -= qty
                    current_long += qty
            else:
                if len(asks) > 0:
                    orders.append(
                        Order(prod, min(asks[0], round_bid), mybuyvol))
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
            z_fast = (self.squid_hist[-1] - sma_fast) / std_fast

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
            # # large price change
            # if ask < minbought - swing or (ask < minsold - swing and ask < minbought) and current_long < soft_lim:
            #     mybuyvol = min(maxqty, pos_lim-current_long, -ask_amount)
            #     minbought = ask
            #     # if mybuyvol > 0:
            #     #     orders.append(Order(prod, ask, mybuyvol))
            #     #     current_long += mybuyvol
            #     print("LARGE SWING (BUY), min bought: ", minbought)

            # else:
            if (strong_signal == "BUY" or action == "BUY") and (current_long < soft_lim):
                if (ask <= fairprice and ask < maxbought) or (ask > fairprice and ask < minbought):
                    mybuyvol = min(-ask_amount, soft_lim-current_long)
                    mybuyvol = min(mybuyvol, maxqty)
                    assert (mybuyvol >= 0), "Buy volume negative"
                    # update minbought to avoid buying higher in the same timestep
                    if minbought == -1 or ask < minbought:
                        minbought = ask
                    orders.append(Order(prod, ask, mybuyvol))
                    current_long += mybuyvol
                    print("BUY SIGNAL")

        for buyorder in buyorders:
            bid, bid_amount = buyorder
            # # large price change
            # if bid > maxsold + swing or (bid > maxbought + swing and bid > maxsold) and current_short > -soft_lim:
            #     mysellvol = min(maxqty, current_short+pos_lim)
            #     mysellvol *= -1
            #     maxsold = bid
            #     # if mysellvol < 0:
            #     #     orders.append(Order(prod, bid, mysellvol))
            #     #     current_short += mysellvol
            #     print("LARGE SWING (SELL), maxsold: ", maxsold, bid)

            # else:
            if (strong_signal == "SELL" or action == "SELL") and current_short > -soft_lim:
                if (bid >= fairprice and bid > minsold) or (bid < fairprice and bid > maxsold):
                    mysellvol = min(bid_amount, soft_lim+current_short)
                    mysellvol = min(mysellvol, maxqty)
                    mysellvol *= -1
                    assert (mysellvol <= 0), "Sell volume positive"
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
                        orders.append(
                            Order(prod, max(int(price), int(fairprice+0.5)), -qty))
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
                        orders.append(
                            Order(prod, min(int(price), int(fairprice)), qty))
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

    def order_basket(self, state: TradingState, current_positions,
                     basket="PICNIC_BASKET1", window=20, mult1=2.5, mult2=1.2,
                     clear_lim=20, adverse_lim=40, maxbuy=5):

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

        if np.abs(basket_position) >= pos_lim:
            print("POSITION LIMIT REACHED! "+basket)
            return None

        basket_order_depth = state.order_depths[basket]
        basket_best_bid = max(basket_order_depth.buy_orders.keys())
        basket_best_ask = min(basket_order_depth.sell_orders.keys())
        basket_bid_vol = basket_order_depth.buy_orders[basket_best_bid]
        basket_ask_vol = -basket_order_depth.sell_orders[basket_best_ask]
        basket_swmp = (basket_best_bid * basket_ask_vol +
                       basket_best_ask * basket_bid_vol) / (basket_bid_vol + basket_ask_vol)

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
                bid_vols.append(
                    state.order_depths[prod].buy_orders[best_bid] // contents[prod])
            if best_ask < float("inf"):
                ask_vols.append(
                    -state.order_depths[prod].sell_orders[best_ask] // contents[prod])

        if implied_bid > 0 and implied_ask < float("inf"):
            implied_ask_vol = min(ask_vols)
            implied_bid_vol = min(bid_vols)
            synthetic_swmp = (implied_bid * implied_ask_vol +
                              implied_ask * implied_bid_vol) / (implied_bid_vol + implied_ask_vol)

        else:
            return None         # contents midprice unavailable

        spread = basket_swmp - synthetic_swmp
        self.spread_hist[basket].append(spread)
        self.spread_hist[basket] = self.spread_hist[basket][-window:]
        spread_hist = self.spread_hist[basket].copy()

        if len(spread_hist) < window:
            return None     # not enough memory

        # mean reversion
        sma = np.mean(spread_hist)
        std = np.std(spread_hist, ddof=1)

        content_orders = {}

        # clear orders
        if basket_position <= -clear_lim:
            if spread_hist[-1] - sma <= - mult2 * std:
                print("CLEAR BUY BASKET")
                basket_lim = pos_lim - basket_position
                mybuyvol = [-(basket_position+clear_lim),
                            implied_bid_vol, basket_lim]

                # check positions of individual content (to sell)
                for prod in contents:
                    available = self.POS_LIM[prod] + current_positions[prod]
                    mybuyvol.append(available // contents[prod])

                mybuyvol = min(mybuyvol)
                if mybuyvol > 0:
                    basket_order = Order(basket, basket_best_ask, mybuyvol)

                    for prod in contents:
                        best_bid = max(
                            state.order_depths[prod].buy_orders.keys())
                        content_orders[prod] = Order(
                            prod, best_bid, -mybuyvol * contents[prod])

                    return basket_order, content_orders

        if basket_position >= clear_lim:
            if spread_hist[-1] - sma >= mult2 * std:
                print("CLEAR SELL BASKET")
                basket_lim = basket_position + pos_lim
                mysellvol = [basket_position-clear_lim,
                             implied_ask_vol, basket_lim]

                # check positions of individual content
                for prod in contents:
                    available = self.POS_LIM[prod] - current_positions[prod]
                    mysellvol.append(available // contents[prod])
                mysellvol = min(mysellvol)
                if mysellvol > 0:
                    basket_order = Order(basket, basket_best_bid, -mysellvol)
                    for prod in contents:
                        best_ask = min(
                            state.order_depths[prod].buy_orders.keys())
                        content_orders[prod] = Order(
                            prod, best_ask, mysellvol * contents[prod])
                    return basket_order, content_orders

        # implied_ask_vol, basket_ask_vol both positive
        if spread_hist[-1] - sma >= mult1 * std and basket_position > -adverse_lim:
            print("SELL BASKET SIGNAL")
            # basket price > synthetic price --> buy contents & sell basket
            basket_lim = pos_lim + basket_position
            qty_basket = [basket_bid_vol, implied_ask_vol, basket_lim, maxbuy]

            # check positions of individual content
            for prod in contents:
                available = self.POS_LIM[prod] - current_positions[prod]
                qty_basket.append(available // contents[prod])

            qty_basket = min(qty_basket)
            if qty_basket > 0:
                basket_order = Order(basket, basket_best_bid, -qty_basket)

                for prod in contents:
                    best_ask = min(state.order_depths[prod].buy_orders.keys())
                    content_orders[prod] = Order(
                        prod, best_ask, qty_basket * contents[prod])

                return basket_order, content_orders

        if spread_hist[-1] - sma <= - mult1 * std and basket_position < adverse_lim:
            print("BUY BASKET SIGNAL")
            # basket price < synthetic price --> buy basket & sell contents
            basket_lim = pos_lim - basket_position
            qty_basket = [basket_ask_vol, implied_bid_vol, basket_lim, maxbuy]

            # check positions of individual content
            for prod in contents:
                available = self.POS_LIM[prod] + current_positions[prod]
                qty_basket.append(available // contents[prod])

            qty_basket = min(qty_basket)
            if qty_basket > 0:
                basket_order = Order(basket, basket_best_ask, qty_basket)

                for prod in contents:
                    best_bid = max(state.order_depths[prod].buy_orders.keys())
                    content_orders[prod] = Order(
                        prod, best_bid, -qty_basket * contents[prod])

                return basket_order, content_orders

        return None

    def order_component(self, state: TradingState, prod: str, logic: str, pos_lim: int):
        # written by Nikola
        orders: List[Order] = []
        order_depth = state.order_depths[prod]
        pos = state.position.get(prod, 0)
        mid_price = None
        sellorders = sorted(list(order_depth.sell_orders.items()))
        buyorders = sorted(list(order_depth.buy_orders.items()), reverse=True)
        if order_depth.buy_orders and order_depth.sell_orders:
            mid_price = (min(order_depth.sell_orders, key=order_depth.sell_orders.get)
                         + max(order_depth.buy_orders, key=order_depth.buy_orders.get)) / 2
            round_bid = round(mid_price - 1)
            round_ask = round(mid_price + 0.5)
        else:
            return []

        self.history[prod].append(mid_price)
        bestask, ask_vol = sellorders[0]
        bestbid, bid_vol = buyorders[0]

        if logic == "trend" and len(self.history[prod]) > 30:
            sma = np.mean(self.history[prod][-30:])
            if mid_price > sma + 2 and pos > -pos_lim:
                orders.append(
                    Order(prod, max(round_ask, bestask), -min(pos_lim+pos, -ask_vol)))
            elif mid_price < sma - 2 and pos < pos_lim:
                orders.append(Order(prod, min(round_bid, bestbid),
                              min(bid_vol, pos_lim-pos)))

        elif logic == "momentum" and len(self.history[prod]) > 20:
            delta = mid_price - self.history[prod][-20]
            if delta > 3 and pos > -pos_lim:
                orders.append(
                    Order(prod, max(round_ask, bestask), -min(pos_lim+pos, -ask_vol)))
            elif delta < -3 and pos < pos_lim:
                orders.append(Order(prod, min(round_bid, bestbid),
                              min(bid_vol, pos_lim-pos)))

        elif logic == "breakout" and len(self.history[prod]) > 10:
            recent = self.history[prod][-10:]
            if mid_price > max(recent) and pos > -pos_lim:
                orders.append(
                    Order(prod, max(round_ask, bestask), -min(pos_lim+pos, -ask_vol)))
            elif mid_price < min(recent) and pos < pos_lim:
                orders.append(Order(prod, min(round_bid, bestbid),
                              min(bid_vol, pos_lim-pos)))

        return orders

    def order_volcanic_rock(self, state: TradingState):
        """
        Basic market making for the underlying VOLCANIC_ROCK.
        This method follows a simple structure similar to order_resin.
        """
        orders: list[Order] = []
        prod = "VOLCANIC_ROCK"
        if prod not in state.order_depths:
            return orders
        order_depth = state.order_depths[prod]
        pos_lim = self.POS_LIM[prod]

        # Calculate mid-price from available buy/sell orders
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2
        else:
            if order_depth.buy_orders:
                mid_price = max(order_depth.buy_orders.keys())
            elif order_depth.sell_orders:
                mid_price = min(order_depth.sell_orders.keys())
            else:
                return orders

        self.history["VOLCANIC_ROCK"].append(mid_price)

        current_pos = state.position.get(prod, 0)
        # Market-taking: buy if ask is below mid-price
        return orders
        for ask, ask_vol in sorted(order_depth.sell_orders.items()):
            if ask <= mid_price and current_pos < pos_lim:
                qty = min(-ask_vol, pos_lim - current_pos)
                orders.append(Order(prod, ask, qty))
                current_pos += qty
        # Market-taking: sell if bid is above mid-price
        for bid, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid >= mid_price and current_pos > -pos_lim:
                qty = min(bid_vol, pos_lim + current_pos)
                orders.append(Order(prod, bid, -qty))
                current_pos -= qty
        return orders

    def order_volcanic_rock_option(self, state: TradingState, option: str):
        """
        Trades the call options on VOLCANIC_ROCK using Black-Scholes pricing.
        The option symbol is assumed to be of the form:
          'VOLCANIC_ROCK_VOUCHER_<strike>',
        where <strike> is the strike price.
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

        remaining_time = 6e6 - state.timestamp
        T = remaining_time / 365e6
        """
        if underlying in self.history and len(self.history[underlying]) >= 100:
            # Use the most recent 1000 prices
            prices = np.array(self.history[underlying][-100:])
            log_returns = np.diff(np.log(prices))
            # Annualization: each timestep is 1/1e6 day so there are 365e6 timesteps in a year.
            sigma = np.std(log_returns, ddof=1) * np.sqrt(365e4)
        else:
            sigma = 0.34  # fallback value
        """    
        #r= -0.16 # 40k$ delta-6.433
        #r= 0.02 # 2.5k$, delta 11.9700
        #r= 0.0001 # 18k$, delta 
        #r= -0.06 # 39.4k$, delta  3.7058
        r = 0
        
        
        sigma_est = Trader.estimated_volatility(S, strike)
        theoretical_price = Trader.black_scholes_call_price(S, strike, T, sigma_est, r)
        # Correct theoretical price using the average deviation.
        if strike == 9500:
            avg_diff = -0.0332
        elif strike == 9750:
            avg_diff = 0.0652
        elif strike == 10000:
            avg_diff = 0.3740
        elif strike == 10250:
            avg_diff = 0.7489
        elif strike == 10500:
            avg_diff = 0.4404
        else:
            avg_diff = 0.0
        corrected_price = theoretical_price - avg_diff
        theoretical_price=corrected_price

        print(f"theoretical price: {theoretical_price}")
        #print(f"sigma_est: {sigma_est}")
        # Determine the current market mid-price for the option.
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            market_mid = (best_bid + best_ask) / 2
        else:
            market_mid = 0
            """
            if order_depth.buy_orders:
                market_mid = max(order_depth.buy_orders.keys())
            elif order_depth.sell_orders:
                market_mid = min(order_depth.sell_orders.keys())
            else:
                return orders
            """
        print(f"Market mid: {market_mid}")
        current_pos = state.position.get(option, 0)
        # Underpriced: if market price is below theoretical value, then buy
        """
        if market_mid < theoretical_price and current_pos < pos_lim:
            for ask, ask_vol in sorted(order_depth.sell_orders.items()):
                if ask <= theoretical_price and current_pos < pos_lim:
                    qty = min(-ask_vol, pos_lim - current_pos)
                    orders.append(Order(option, ask, qty))
                    current_pos += qty
        # Overpriced: if market price is above theoretical value, then sell
        elif market_mid > theoretical_price and current_pos > -pos_lim:
            for bid, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
                if bid >= theoretical_price and current_pos > -pos_lim:
                    qty = min(bid_vol, pos_lim + current_pos)
                    orders.append(Order(option, bid, -qty))
                    current_pos -= qty
        """
        # buy for one over fair price, sell for 1 under fair
        orders.append(
            Order(option, round(theoretical_price*0-95), pos_lim-current_pos))
        orders.append(
            Order(option, round(theoretical_price+1.05), -pos_lim-current_pos))

        return orders

    def run(self, state: TradingState):
        self.timer += 100

        ### UPDATE WITH DAY N DATA ###
        linreg = {"KELP":  (14.422964389386117,
                            1.173227183514493e-05, 2052.6409802838834)}

        if self.timer != state.timestamp:
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
                self.spread_hist = traderObject["spread history"]
                self.history = traderObject["history"]

        self.update_open_pos(state)

        result = {}
        # debug mode
        # if state.timestamp < 100:
        if False:
            result["RAINFOREST_RESIN"] = self.order_resin(state)
            result["KELP"] = self.order_kelp(state, *linreg["KELP"])
            result["SQUID_INK"] = self.order_squid(state)
            result["CROISSANTS"] = []
            result["JAMS"] = []
            result["DJEMBES"] = []
            baskets = ["PICNIC_BASKET1", "PICNIC_BASKET2"]

            # window, mult1, mult2, clearlim, adverselim, maxbuy
            MR_params = [(20, 2.5, 1.5, 10, 35, 15),
                         (20, 2.5, 1.5, 15, 60, 15)]

            current_positions = {prod: (state.position[prod] if prod in state.position else 0)
                                 for prod in self.basket_contents[baskets[0]]}
            for basket, args in zip(baskets, MR_params):
                spread_order = self.order_basket(
                    state, current_positions, basket, *args)
                if spread_order is not None:
                    basket_order, content_orders = spread_order
                    result[basket] = [basket_order]
                    for prod in content_orders:
                        result[prod].append(content_orders[prod])
                        current_positions[prod] += content_orders[prod].quantity

            # only order the individual contents if not ordering baskets
            if "PICNIC_BASKET1" not in result:
                result["DJEMBES"] = self.order_component(
                    state, "DJEMBES", logic="breakout", pos_lim=10)
                if "PICNIC_BASKET2" not in result:
                    result["CROISSANTS"] = self.order_component(
                        state, "CROISSANTS", logic="trend", pos_lim=15)
                    result["JAMS"] = self.order_component(
                        state, "JAMS", logic="momentum", pos_lim=20)

            # New asset: VOLCANIC_ROCK and its options
        result["VOLCANIC_ROCK"] = self.order_volcanic_rock(state)
        options = [
            "VOLCANIC_ROCK_VOUCHER_9500",
            "VOLCANIC_ROCK_VOUCHER_9750",
            "VOLCANIC_ROCK_VOUCHER_10000",
            "VOLCANIC_ROCK_VOUCHER_10250",
            "VOLCANIC_ROCK_VOUCHER_10500"
        ]
        for option in options:
            result[option] = self.order_volcanic_rock_option(state, option)

        traderObject = {"open buys": self.open_buys,
                        "open sells": self.open_sells,
                        "recorded time": self.recorded_time,
                        "squid history": self.squid_hist,
                        "squid upper": self.upper,
                        "squid lower": self.lower,
                        "squid slow upper": self.slow_upper,
                        "squid slow lower": self.slow_lower,
                        "spread history": self.spread_hist,
                        "history": self.history}
        traderData = jsonpickle.encode(traderObject)

        conversions = 1
        return result, conversions, traderData
