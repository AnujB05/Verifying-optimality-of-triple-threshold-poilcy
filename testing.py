# testing.py

import numpy as np
import matplotlib.pyplot as plt
from dataset_creation import create_customer_dataset
from numerical_method import find_optimal_thresholds
from mpl_toolkits.mplot3d import Axes3D
params = {
    'initial_state_dist': [0.4, 0.3, 0.3],
    'rewards': [5, 10, 15],
    'p_buy': [0.2, 0.4, 0.6],
    'p_churn': [0.1, 0.2, 0.3],
    'costs': [1, 2, 3],
    'transition_matrix': [
        [0.6, 0.3, 0.1],
        [0.2, 0.5, 0.3],
        [0.1, 0.3, 0.6]
    ]
}

# 1. Generate dataset
thresholds_for_data = (0.1, 0.1, 0.1)
dataset = create_customer_dataset(thresholds_for_data, params, n_customers=100, time_horizon=20)
print("Dataset created with shape:", dataset.shape)

# 2. Find optimal thresholds
optimal_thresholds, optimal_value, results_df = find_optimal_thresholds(params, grid_size=11, n_simulations=100)
print(f"Optimal thresholds: {optimal_thresholds} with profit rate: {optimal_value:.4f}")

# 3. Visualize results
pivot = results_df[np.isclose(results_df['w2'], optimal_thresholds[2])].pivot(
    index="w0", columns="w1", values="profit_rate"
)

fig, ax = plt.subplots(figsize=(8, 6))
c = ax.imshow(pivot.values, origin='lower', cmap='viridis', 
              extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()],
              aspect='auto')

ax.set_title(f'Profit Rate Heatmap (w2 = {optimal_thresholds[2]})')
ax.set_xlabel('w1')
ax.set_ylabel('w0')

# Mark the optimal point
ax.plot(optimal_thresholds[1], optimal_thresholds[0], 'ro', label='Optimal Threshold')
ax.legend()

fig.colorbar(c, ax=ax, label='Profit Rate')
plt.tight_layout()
plt.show()

def plot_3d_profit_surface(results_df, optimal_thresholds):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    x = results_df['w0']
    y = results_df['w1']
    z = results_df['w2']
    c = results_df['profit_rate']

    sc = ax.scatter(x, y, z, c=c, cmap='viridis', s=40)
    ax.scatter(*optimal_thresholds, 
               color='red', s=100, label='Optimal Thresholds', marker='X')

    ax.set_xlabel('w0')
    ax.set_ylabel('w1')
    ax.set_zlabel('w2')
    ax.set_title('3D Scatter Plot: Profit Rate vs Thresholds')
    fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10, label='Profit Rate')
    ax.legend()
    plt.tight_layout()
    plt.show()

plot_3d_profit_surface(results_df, optimal_thresholds)