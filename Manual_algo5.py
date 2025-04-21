import math, pandas as pd

def optimal_allocation(move_dict, total_capital=1_000_000):
    """
    move_dict      : {'asset': expected % move tomorrow (positive=up, negative=down)}
    total_capital  : bankroll in $, default 1 m

    Returns
    -------
    allocation : {'asset': integer % of capital ( +long / –short )}
    exp_net    : expected $ P&L after fees
    """
    # build every profitable 1‑percentage‑point “chunk”
    chunks = []                              # (marginal_net, asset)
    for asset, move in move_dict.items():
        p = abs(move)                        # size of move
        gross_1 = total_capital * p / 10_000 # profit from a 1 % position
        fee_prev = 0
        for x in range(1, 101):              # x = new total % after adding this chunk
            fee_new      = 120 * x**2
            marginal_net = gross_1 - (fee_new - fee_prev)
            if marginal_net <= 0:            # further chunks would be worse—stop
                break
            chunks.append((marginal_net, asset))
            fee_prev = fee_new

    # sort chunks by marginal edge and grab up to 100 of them (100 % max investable)
    chunks.sort(reverse=True, key=lambda t: t[0])
    selected = chunks[:100]

    # build the final allocation & net P/L
    alloc, net = {}, 0.0
    for m_net, asset in selected:
        alloc[asset] = alloc.get(asset, 0) + 1
        net += m_net

    # flip sign for shorts
    alloc = {a: (v if move_dict[a] > 0 else -v) for a, v in alloc.items()}
    return alloc, net


if __name__ == "__main__":
    # example: the newsletter numbers we just estimated
    moves = {
        "Cacti Needle": -24.5,
        "Solar Panels": -15.6666,
        "Quantum Coffee": -25,
        "Haystacks": 10,
        "Striped Shirts": 6.66666,
        "Ranch Sauce": 7.3333,
        "Red Flags": 9.5,
        "VR Monocle": 11.5,
        "Moonshine": 1.5,
    }

    alloc, exp_net = optimal_allocation(moves)
    print("Optimal allocation (% of capital, +long / –short):")
    for k, v in alloc.items():
        print(f"{k:15s}: {v:+d}%")
    print(f"\nTotal invested : {sum(abs(v) for v in alloc.values())}%")
    print(f"Expected net   : ${exp_net:,.0f}")
