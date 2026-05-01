"""Load real city data (road networks, hotspot nodes, producer/consumer locations).

Builds shortest-path distance and time matrices from the city's road graph,
maps real producer/consumer locations to hotspot nodes, and generates synthetic
orders weighted by real location densities.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from config.config import RealDataConfig


def load_city_data(city: str = "albuquerque", data_root: str = "data_real") -> dict:
    """Load and process real city data into simulation-ready structures.

    Steps:
      1. Load hotspot nodes from GeoJSON (census tract centroids)
      2. Map GEOIDs to deterministic integer node_ids (sorted alphabetically)
      3. Count producers/consumers per node from location GeoJSON files
      4. Build locations_df with same schema as synthetic mode
      5. Build shortest-path distance and time matrices from road edges
      6. Cache matrices as .npy files for fast reload

    Returns dict with: locations_df, dist_matrix_km, time_matrix_hours,
                       node_index_map, pickup_counts, dropoff_counts
    """
    city_dir = os.path.join(data_root, city)
    cache_dir = os.path.join(city_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # --- 1. Load hotspot nodes ---
    hotspot_path = os.path.join(city_dir, "hotspot_data.geojson")
    with open(hotspot_path) as f:
        hotspot_data = json.load(f)

    features = hotspot_data["features"]

    # --- 2. Map GEOIDs to integer node_ids (sorted for determinism) ---
    geoids = sorted(feat["properties"]["GEOID"] for feat in features)
    node_index_map = {geoid: idx for idx, geoid in enumerate(geoids)}

    # Build coordinate lookup: geoid -> (lon, lat)
    geoid_coords = {}
    for feat in features:
        geoid = feat["properties"]["GEOID"]
        lon, lat = feat["geometry"]["coordinates"]
        geoid_coords[geoid] = (lon, lat)

    # --- 3. Count producers/consumers per node ---
    pickup_dir = os.path.join(city_dir, "Pickup Point Data")
    dropoff_dir = os.path.join(city_dir, "Drop Point Data")

    pickup_counts = _count_features_by_geoid(pickup_dir)
    dropoff_counts = _count_features_by_geoid(dropoff_dir)

    # --- 4. Build locations_df ---
    locations = []
    for geoid in geoids:
        node_id = node_index_map[geoid]
        lon, lat = geoid_coords[geoid]
        has_pickup = pickup_counts.get(geoid, 0) > 0
        has_dropoff = dropoff_counts.get(geoid, 0) > 0

        if has_pickup and has_dropoff:
            node_type = "Both"
        elif has_pickup:
            node_type = "Producer"
        elif has_dropoff:
            node_type = "Consumer"
        else:
            node_type = "Transit"

        locations.append({
            "node_id": node_id,
            "type": node_type,
            "x": lon,
            "y": lat,
        })

    locations_df = pd.DataFrame(locations)

    # --- 5. Build shortest-path matrices ---
    dist_cache = os.path.join(cache_dir, "dist_matrix_km.npy")
    time_cache = os.path.join(cache_dir, "time_matrix_hours.npy")

    if os.path.exists(dist_cache) and os.path.exists(time_cache):
        print(f"  [CACHE] Loading cached matrices for {city}")
        dist_matrix_km = np.load(dist_cache)
        time_matrix_hours = np.load(time_cache)
    else:
        print(f"  [BUILD] Computing shortest-path matrices for {city}...")
        dist_matrix_km, time_matrix_hours = _build_shortest_path_matrices(
            city_dir, node_index_map
        )
        np.save(dist_cache, dist_matrix_km)
        np.save(time_cache, time_matrix_hours)
        print(f"  [CACHE] Saved matrices to {cache_dir}")

    # Verify graph connectivity
    assert not np.any(np.isinf(dist_matrix_km)), \
        "Distance matrix contains inf -- road graph is not fully connected"
    assert not np.any(np.isinf(time_matrix_hours)), \
        "Time matrix contains inf -- road graph is not fully connected"

    return {
        "locations_df": locations_df,
        "dist_matrix_km": dist_matrix_km,
        "time_matrix_hours": time_matrix_hours,
        "node_index_map": node_index_map,
        "pickup_counts": pickup_counts,
        "dropoff_counts": dropoff_counts,
    }


def _count_features_by_geoid(directory: str) -> dict:
    """Count total features per GEOID across all GeoJSON files in a directory."""
    counts = {}
    if not os.path.isdir(directory):
        return counts

    for fname in os.listdir(directory):
        if not fname.endswith(".geojson"):
            continue
        fpath = os.path.join(directory, fname)
        with open(fpath) as f:
            data = json.load(f)
        for feat in data.get("features", []):
            geoid = feat.get("properties", {}).get("GEOID")
            if geoid:
                counts[geoid] = counts.get(geoid, 0) + 1
    return counts


def _build_shortest_path_matrices(city_dir: str, node_index_map: dict):
    """Parse road edges CSV and compute all-pairs shortest paths.

    Each row in edges_car_bidirectional.csv has forward and reverse distances/times.
    We build two sparse matrices (distance, time) and run Dijkstra on each.
    """
    edges_path = os.path.join(city_dir, "edges_car_bidirectional.csv")
    edges_df = pd.read_csv(edges_path)

    n = len(node_index_map)
    # Build sparse adjacency matrices
    rows_fwd, cols_fwd, dist_fwd, time_fwd = [], [], [], []
    rows_rev, cols_rev, dist_rev, time_rev = [], [], [], []

    for _, row in edges_df.iterrows():
        from_geoid = str(int(row["from_node"]))
        to_geoid = str(int(row["to_node"]))

        if from_geoid not in node_index_map or to_geoid not in node_index_map:
            continue

        i = node_index_map[from_geoid]
        j = node_index_map[to_geoid]

        # Forward direction
        rows_fwd.append(i)
        cols_fwd.append(j)
        dist_fwd.append(row["distance_km"])
        time_fwd.append(row["time_hours"])

        # Reverse direction
        rows_rev.append(j)
        cols_rev.append(i)
        dist_rev.append(row["distance_km_reverse"])
        time_rev.append(row["time_hours_reverse"])

    # Combine forward + reverse into single sparse matrices
    all_rows = rows_fwd + rows_rev
    all_cols = cols_fwd + cols_rev
    all_dist = dist_fwd + dist_rev
    all_time = time_fwd + time_rev

    dist_sparse = csr_matrix((all_dist, (all_rows, all_cols)), shape=(n, n))
    time_sparse = csr_matrix((all_time, (all_rows, all_cols)), shape=(n, n))

    # All-pairs shortest path (Dijkstra is faster for sparse graphs)
    dist_matrix = shortest_path(dist_sparse, method="D", directed=True)
    time_matrix = shortest_path(time_sparse, method="D", directed=True)

    # Zero diagonal (self-loops)
    np.fill_diagonal(dist_matrix, 0.0)
    np.fill_diagonal(time_matrix, 0.0)

    return dist_matrix, time_matrix


def generate_real_orders(config: RealDataConfig, locations_df: pd.DataFrame,
                         pickup_counts: dict, dropoff_counts: dict,
                         node_index_map: dict) -> pd.DataFrame:
    """Generate synthetic orders weighted by real producer/consumer densities.

    For "Both" nodes, role assignment uses a noisy producer_ratio:
      producer_ratio = pickup / (pickup + dropoff) + N(0, noise_std)
    Different seeds produce different role assignments.

    Returns orders_df with standard schema matching synthetic mode.
    """
    np.random.seed(config.seed)

    # Invert node_index_map: node_id -> geoid
    id_to_geoid = {v: k for k, v in node_index_map.items()}

    # Build coordinate lookup from locations_df
    coord_lookup = {int(row["node_id"]): (row["x"], row["y"])
                    for _, row in locations_df.iterrows()}

    # Build weighted producer and consumer pools
    producer_ids = []
    producer_weights = []
    consumer_ids = []
    consumer_weights = []

    for _, row in locations_df.iterrows():
        node_id = int(row["node_id"])
        geoid = id_to_geoid[node_id]
        n_pickup = pickup_counts.get(geoid, 0)
        n_dropoff = dropoff_counts.get(geoid, 0)
        node_type = row["type"]

        if node_type == "Transit":
            continue

        if node_type == "Producer":
            producer_ids.append(node_id)
            producer_weights.append(max(n_pickup, 1))
        elif node_type == "Consumer":
            consumer_ids.append(node_id)
            consumer_weights.append(max(n_dropoff, 1))
        elif node_type == "Both":
            # Noisy role probability
            total = n_pickup + n_dropoff
            prod_ratio = n_pickup / total if total > 0 else 0.5
            prod_ratio += np.random.normal(0, config.noise_std)
            prod_ratio = np.clip(prod_ratio, 0.01, 0.99)

            # Add to both pools with adjusted weights
            producer_ids.append(node_id)
            producer_weights.append(n_pickup * prod_ratio)
            consumer_ids.append(node_id)
            consumer_weights.append(n_dropoff * (1 - prod_ratio))

    # Normalize weights to probabilities
    producer_weights = np.array(producer_weights, dtype=float)
    producer_weights /= producer_weights.sum()
    consumer_weights = np.array(consumer_weights, dtype=float)
    consumer_weights /= consumer_weights.sum()

    producer_ids = np.array(producer_ids)
    consumer_ids = np.array(consumer_ids)

    # Generate order timestamps using same logic as synthetic mode
    order_times = _generate_order_times_real(config)

    # Generate weighted random assignments
    chosen_producers = np.random.choice(producer_ids, size=config.num_orders, p=producer_weights)
    chosen_consumers = np.random.choice(consumer_ids, size=config.num_orders, p=consumer_weights)

    # Ensure producer != consumer for each order
    for i in range(config.num_orders):
        while chosen_consumers[i] == chosen_producers[i]:
            chosen_consumers[i] = np.random.choice(consumer_ids, p=consumer_weights)

    orders = []
    for i in range(config.num_orders):
        pid = int(chosen_producers[i])
        cid = int(chosen_consumers[i])
        x_p, y_p = coord_lookup[pid]
        x_c, y_c = coord_lookup[cid]

        orders.append({
            "order_id": i + 1,
            "time": order_times[i],
            "producer_id": pid,
            "x_prod": x_p, "y_prod": y_p,
            "consumer_id": cid,
            "x_cons": x_c, "y_cons": y_c,
            "status": "Pending",
        })

    orders_df = pd.DataFrame(orders).sort_values("time").reset_index(drop=True)
    return orders_df


def _generate_order_times_real(config: RealDataConfig) -> np.ndarray:
    """Generate order timestamps (same patterns as synthetic mode)."""
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
    else:  # uniform
        times = np.random.uniform(0.01, hours, size=n)

    return np.clip(np.round(times, 8), 0.01, hours)
