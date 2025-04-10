import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define the data for each container as (x_value, inhabitants)
data = [
    (10, 1),
    (80, 6),
    (37, 3),
    (17, 1),
    (90, 10),
    (31, 2),
    (50, 4),
    (20, 2),
    (73, 4),
    (89, 9)
]

# Create a DataFrame from the data
df = pd.DataFrame(data, columns=["x_value", "inhabitants"])

# Set k values from 0 to 35
k_vals = np.arange(0, 36)

# Create the plot
plt.figure(figsize=(12, 8))

# Compute and plot the ratios for each container
for idx, row in df.iterrows():
    x = row["x_value"]
    y = row["inhabitants"]
    ratios = [10000*x / (y + k) for k in k_vals]
    plt.plot(k_vals, ratios, marker="o", label=f"{x}x")  # Label using the multiplicator
plt.yscale('log')
plt.xscale('log')
plt.xlabel("Player teams")
plt.ylabel("Payoff")
plt.title("Line Plot of payoffs")
plt.legend(title="Multiplicators", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()
