# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 15:38:02 2025

@author: kohle
"""

import numpy as np
import pandas as pd

# Define the trading table as a numpy array
goods = ["Snowballs", "Pizza's", "Silicon Nuggets", "SeaShells"]
exchange_rates = np.array([
    [1.0, 1.45, 0.52, 0.72],
    [0.7, 1.0, 0.31, 0.48],
    [1.95, 3.1, 1.0, 1.49],
    [1.34, 1.98, 0.64, 1.0]
])

# Number of trades to simulate
num_trades = 5
num_goods = len(goods)

# Initialize a 2D array for dynamic programming: rows=steps, cols=goods
dp = np.zeros((num_trades + 1, num_goods))
dp[0, 3] = 1  # Start with 1 SeaShell

# Traceback path to reconstruct the sequence
trace = np.full((num_trades + 1, num_goods), -1)

# Fill the dp table
for step in range(1, num_trades + 1):
    for to_good in range(num_goods):
        for from_good in range(num_goods):
            potential = dp[step - 1, from_good] * exchange_rates[from_good, to_good]
            if potential > dp[step, to_good]:
                dp[step, to_good] = potential
                trace[step, to_good] = from_good
    

# Find the maximum amount of SeaShells at the final step
final_amount = dp[num_trades, 3]
path = []
current_good = 3  # SeaShells

# Trace back the path
for step in range(num_trades, 0, -1):
    path.append((step, goods[current_good]))
    current_good = trace[step, current_good]
path.append((0, goods[current_good]))
path.reverse()

# Prepare the result
results = {
    "final_amount": final_amount,
    "path": path
}
print(results)
