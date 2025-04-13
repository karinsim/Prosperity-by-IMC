import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# ---------------------------------------------------
# Helper: Proportion of deals at a given first bid
def p_deal(f):
    """
    Returns the overall proportion of deals (cumulative probability)
    at first bid f, computed as weighted contributions:
      - Lower group: reserves uniformly in [160,200] with weight 40/110.
      - Upper group: reserves uniformly in [250,320] with weight 70/110.
    """
    if f < 160:
        return 0.0
    elif f < 200:
        return ((f - 160) / 110)
    elif f < 250:
        return (40/110)
    elif f <= 320:
        return ((f - 210) / 110)
    else:
        return 1.0

# ---------------------------------------------------
# Compute optimal bids and best total profit as before.
avg_bid_range = np.arange(160, 321)     # average bid from 160 to 320 inclusive
first_bid_range = np.arange(160, 321)     # possible first bids

best_total_profit_list = []
best_first_bid_list = []
best_second_bid_list = []

for avg_bid in avg_bid_range:
    best_total_profit = -np.inf
    best_f_for_avg = None
    best_s_for_avg = None
    
    for f in first_bid_range:
        p_f = p_deal(f)
        profit_f = p_f * (320 - f)
        for s in range(f, 320):
            # For s >= avg_bid the effect = 1, otherwise < 1.
            effect = min(1, ((320-avg_bid)/(320-s))**3)
            # Deals in round 2 occur between f and s: difference in cumulative proportion.
            profit_s = (p_deal(s) - p_f) * (320-s) * effect
            total_profit = profit_f + profit_s

            if total_profit > best_total_profit:
                best_total_profit = total_profit
                best_f_for_avg = f
                best_s_for_avg = s

    best_total_profit_list.append(best_total_profit)
    best_first_bid_list.append(best_f_for_avg)
    best_second_bid_list.append(best_s_for_avg)

avg_bid_values = avg_bid_range
best_total_profit_arr = np.array(best_total_profit_list)
best_first_bid_arr = np.array(best_first_bid_list)
best_second_bid_arr = np.array(best_second_bid_list)

# Optional overall view plot
plt.figure(figsize=(10,6))
plt.plot(avg_bid_values, best_total_profit_arr, 'b-', lw=2, label="Best Total Profit")
plt.plot(avg_bid_values, best_first_bid_arr, 'r-', lw=2, label="Optimal First Bid")
plt.plot(avg_bid_values, best_second_bid_arr, 'g-', lw=2, label="Optimal Second Bid")
plt.xlabel("Average Bid (SeaShells)")
plt.ylabel("Value (SeaShells)")
plt.title("Optimal Total Profit & Bids vs. Average Bid")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------------------------------
# Modified profit_s equation and solving for different factors.
#
# The round-2 profit function (for s>=avg_bid so that effect=1) is:
#
#    profit_s(s) = ((s-210)/110 - 4/11)*(320-s)
#
# We want to solve for s the equation:
#
#    ((s-210)/110 - 4/11)*(320-s) = factor*(best_total_profit - (4/11*120))
#
# where factor = 0.95, 0.9, 0.85, or 0.8.
#
# Set a constant threshold = (4/11)*120.
threshold = (4/11) * 120  # constant baseline term

def f_for_s(s, avg_bid, best_total_profit, factor):
    """
    f_for_s(s) = (((s-210)/110 - 4/11)*(320-s)) - T,
    with T = factor*(best_total_profit - threshold).
    This function is to be solved for f_for_s(s)=0.
    """
    T = factor * (best_total_profit - threshold)
    return (((s - 210)/110) - (4/11))*(320-s)*min(1,((320-avg_bid)/(320-s))**3) - T

# We will compute, for each avg_bid, the solutions for s for all four factors.
factors = [0.95, 0.9, 0.85, 0.8]
# Dictionaries to store the lower and upper branch solutions for each factor:
solutions_lower = {}
solutions_upper = {}

for factor in factors:
    sol_lower = []
    sol_upper = []
    # Loop over each avg_bid index.
    for i in range(len(avg_bid_values)):
        avg_bid = avg_bid_values[i]
        best_tot = best_total_profit_arr[i]
        best_s = best_second_bid_arr[i]
        # Use a lower bound for s: at least the max of 210 and the optimal first bid.
        lower_bound = max(210, best_first_bid_list[i])
        upper_bound = 320.0
        # Solve for the lower branch solution in [lower_bound, best_s]
        try:
            s_low = brentq(lambda s: f_for_s(s, avg_bid, best_tot, factor), lower_bound, best_s)
        except ValueError:
            s_low = np.nan
        # Solve for the upper branch solution in [best_s, upper_bound]
        try:
            s_high = brentq(lambda s: f_for_s(s, avg_bid, best_tot, factor), best_s, upper_bound)
        except ValueError:
            s_high = np.nan
        
        sol_lower.append(s_low)
        sol_upper.append(s_high)
        
    solutions_lower[factor] = np.array(sol_lower)
    solutions_upper[factor] = np.array(sol_upper)

# ---------------------------------------------------
# Plot: Original Optimal Second Bid and the bands for factors 0.95, 0.9, 0.85, and 0.8.
plt.figure(figsize=(10, 10))
plt.plot(avg_bid_values, best_second_bid_arr, 'k-', lw=2, label="Optimal Second Bid")

# Define colors and linestyles for each factor.
colors = {0.95: 'g', 0.9: 'b', 0.85: 'm', 0.8: 'c'}
linestyles_lower = {0.95: '--', 0.9: '--', 0.85: '--', 0.8: '--'}
linestyles_upper = {0.95: '-.', 0.9: '-.', 0.85: '-.', 0.8: '-.'}

for factor in factors:
    plt.plot(avg_bid_values, solutions_lower[factor], linestyle=linestyles_lower[factor],
             color=colors[factor], lw=2, label=f"{factor} Lower")
    plt.plot(avg_bid_values, solutions_upper[factor], linestyle=linestyles_upper[factor],
             color=colors[factor], lw=2, label=f"{factor} Upper")

plt.xlabel("Average Bid (SeaShells)")
plt.ylabel("Second Bid (SeaShells)")
plt.title("Optimal Second Bid and Modified Bands for Factors 0.95, 0.9, 0.85, and 0.8")
plt.xlim(260, 320)
plt.ylim(260, 320)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
