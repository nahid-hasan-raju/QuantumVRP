"""Synthetic VRP data generation with configurable order counts, spatial layouts, and demand patterns.

Generates two CSV files per scenario:
  - locations.csv: node_id, type (Producer/Consumer), x, y coordinates on the grid
  - orders.csv:    order_id, timestamp, producer/consumer assignments with coordinates

Datasets are content-addressed by config label -- identical params produce the same folder,
so re-runs skip generation entirely (cache hit).
"""

import os
import numpy as np
import pandas as pd
from config.config import DataGenConfig


def _generate_order_times(config: DataGenConfig) -> np.ndarray:
    """Generate order timestamps according to the configured demand pattern.

    Three patterns model different real-world demand profiles:
      - uniform:  orders spread evenly across the simulation window
      - unimodal: single peak around hour 6 (e.g., lunch rush)
      - bimodal:  two peaks at hours 4 and 8 (e.g., morning + evening)
    """
    n = config.num_orders
    hours = config.simulation_hours

    if config.demand_pattern == "unimodal":
        # Single Gaussian peak at midpoint of simulation window
        times = np.random.normal(loc=hours * 0.5, scale=hours / 6.0, size=n)
    elif config.demand_pattern == "bimodal":
        # 50/50 mixture of two Gaussian peaks at 1/3 and 2/3 of window
        half = n // 2
        peak1 = np.random.normal(loc=hours / 3.0, scale=hours / 8.0, size=half)
        peak2 = np.random.normal(loc=hours * 2.0 / 3.0, scale=hours / 8.0, size=n - half)
        times = np.concatenate([peak1, peak2])
    else:  # uniform (default)
        times = np.random.uniform(0.01, hours, size=n)

    # Clip to valid simulation range so no orders fall outside [0.01, hours]
    times = np.clip(times, 0.01, hours)
    return np.round(times, 8)


def generate_dataset(config: DataGenConfig, data_root: str = "data_synthetic") -> tuple:
    """Generate locations and orders CSVs for a VRP scenario.

    Returns (locations_df, orders_df). Skips generation if matching
    CSVs already exist in data_root/{label}/ (content-addressed caching).
    """
    dataset_dir = os.path.join(data_root, config.label)
    loc_path = os.path.join(dataset_dir, "locations.csv")
    ord_path = os.path.join(dataset_dir, "orders.csv")

    # Cache hit -- skip regeneration for identical configs
    if os.path.exists(loc_path) and os.path.exists(ord_path):
        print(f"  [CACHE] Loading existing data for {config.label}")
        return pd.read_csv(loc_path), pd.read_csv(ord_path)

    os.makedirs(dataset_dir, exist_ok=True)
    # Fixed seed ensures reproducibility across runs
    np.random.seed(config.seed)

    producer_ids = list(range(config.num_producers))
    consumer_ids = list(range(config.num_producers, config.num_producers + config.num_consumers))

    # Place producers and consumers at random grid positions
    locations = []
    for pid in producer_ids:
        locations.append({
            'node_id': pid, 'type': 'Producer',
            'x': np.random.uniform(0, config.grid_size),
            'y': np.random.uniform(0, config.grid_size),
        })
    for cid in consumer_ids:
        locations.append({
            'node_id': cid, 'type': 'Consumer',
            'x': np.random.uniform(0, config.grid_size),
            'y': np.random.uniform(0, config.grid_size),
        })

    loc_df = pd.DataFrame(locations)

    # O(1) coord lookup instead of O(n) DataFrame filter per order
    coord_lookup = {int(row['node_id']): (row['x'], row['y']) for _, row in loc_df.iterrows()}

    # Pre-generate all order times and assignments vectorized
    order_times = _generate_order_times(config)
    producer_choices = np.random.choice(producer_ids, size=config.num_orders)
    consumer_choices = np.random.choice(consumer_ids, size=config.num_orders)

    orders = []
    for i in range(config.num_orders):
        pid = int(producer_choices[i])
        cid = int(consumer_choices[i])
        x_p, y_p = coord_lookup[pid]
        x_c, y_c = coord_lookup[cid]

        orders.append({
            'order_id': i + 1,
            'time': order_times[i],
            'producer_id': pid,
            'x_prod': x_p, 'y_prod': y_p,
            'consumer_id': cid,
            'x_cons': x_c, 'y_cons': y_c,
            'status': 'Pending',
        })

    # Sort by time so the simulation processes orders chronologically
    orders_df = pd.DataFrame(orders).sort_values('time').reset_index(drop=True)

    loc_df.to_csv(loc_path, index=False)
    orders_df.to_csv(ord_path, index=False)
    print(f"  [GEN] Saved {config.label} ({config.num_orders} orders)")

    return loc_df, orders_df
