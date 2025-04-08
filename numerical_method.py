# numerical_method.py

import numpy as np
import pandas as pd
from itertools import product

def expected_cycle_profit_and_length(w0, w1, w2, params, n_simulations):
    total_profit, total_length = 0.0, 0.0
    for _ in range(n_simulations):
        length = np.random.randint(5, 15)
        profit = 0.0
        for _ in range(length):
            state = np.random.choice([0, 1, 2], p=params['initial_state_dist'])
            belief = np.array([0.0, 0.0, 0.0])
            belief[state] = 1.0

            if belief[2] >= w2:
                action = 2
            elif belief[1] >= w1:
                action = 1
            elif belief[0] >= w0:
                action = 0
            else:
                action = 0

            reward = params['rewards'][state] * params['p_buy'][state] - params['costs'][action]
            profit += reward

        total_profit += profit
        total_length += length

    return total_profit / n_simulations, total_length / n_simulations

def find_optimal_thresholds(params, grid_size=11, n_simulations=100):
    grid = np.linspace(0, 1, grid_size)
    results = []
    best_profit_rate = -np.inf
    best_thresholds = (0, 0, 0)

    for w0, w1, w2 in product(grid, repeat=3):
        profit, length = expected_cycle_profit_and_length(w0, w1, w2, params, n_simulations)
        profit_rate = profit / length if length > 0 else -np.inf

        results.append({
            'w0': w0, 'w1': w1, 'w2': w2,
            'profit': profit,
            'length': length,
            'profit_rate': profit_rate
        })

        if profit_rate > best_profit_rate:
            best_profit_rate = profit_rate
            best_thresholds = (w0, w1, w2)

    results_df = pd.DataFrame(results)
    return best_thresholds, best_profit_rate, results_df


