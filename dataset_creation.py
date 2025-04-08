# dataset_creation.py

import numpy as np
import pandas as pd

def create_customer_dataset(thresholds, params, n_customers, time_horizon):
    data = []
    for customer_id in range(n_customers):
        state = np.random.choice([0, 1, 2], p=params['initial_state_dist'])
        belief = np.array([0.0, 0.0, 0.0])
        belief[state] = 1.0

        for t in range(time_horizon):
            if belief[2] >= thresholds[2]:
                action = 2
            elif belief[1] >= thresholds[1]:
                action = 1
            elif belief[0] >= thresholds[0]:
                action = 0
            else:
                action = 0  # default action

            reward = params['rewards'][state] * params['p_buy'][state] - params['costs'][action]
            churn_prob = params['p_churn'][state]
            next_state = np.random.choice([0, 1, 2], p=params['transition_matrix'][state])

            data.append({
                "customer_id": customer_id,
                "time": t,
                "state": state,
                "action": action,
                "reward": reward,
                "p_churn_0": params['p_churn'][0],
                "p_churn_1": params['p_churn'][1],
                "p_churn_2": params['p_churn'][2],
                "p_buy_0": params['p_buy'][0],
                "p_buy_1": params['p_buy'][1],
                "p_buy_2": params['p_buy'][2]
            })

            state = next_state

    return pd.DataFrame(data)

