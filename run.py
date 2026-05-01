#!/usr/bin/env python3
"""CLI entry point for QuantumVRP parameter sweeps.

Supports two modes:
  - Synthetic: python run.py --mode synthetic (default, random grid data)
  - Real:      python run.py --mode real --city albuquerque (real city road network)
"""

import argparse
import os
import sys
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from config import SweepConfig, RealSweepConfig
from data_generation import generate_dataset, load_city_data, generate_real_orders
from simulation import run_experiment


def parse_args():
    """Parse CLI arguments for sweep configuration and overrides."""
    p = argparse.ArgumentParser(
        description="Run QuantumVRP parameter sweep (synthetic or real city data).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python run.py                                               # synthetic full sweep
  python run.py --orders 2000 --producers 50                  # synthetic focused sweep
  python run.py --mode real --city albuquerque --orders 100   # real city mode
  python run.py --mode real --drivers 10                      # real mode, pin drivers
""",
    )

    # Mode selection
    p.add_argument("--mode", choices=["synthetic", "real"], default="synthetic",
                   help="Data mode: synthetic (random grid) or real (city road network)")
    p.add_argument("--city", default="albuquerque",
                   help="City name for real mode (folder under data_real/)")

    # Sweep control
    p.add_argument("--sweep-name", default=None,
                   help="Name for this sweep run (default: auto-generated timestamp)")
    p.add_argument("--workers", type=int, default=None,
                   help="Max parallel workers (default: min(cpu_count, 8))")
    p.add_argument("--data-root", default=None,
                   help="Root directory for datasets (default: data_synthetic or data_real)")
    p.add_argument("--results-dir", default=None,
                   help="Results directory (default: results/<mode>/<sweep-name>)")
    p.add_argument("--sequential", action="store_true",
                   help="Run experiments sequentially in the main process (no forking)")

    # Parameter overrides -- pin a param to a single value
    p.add_argument("--orders", type=int, default=None,
                   help="Pin num_orders to this value")
    p.add_argument("--producers", type=int, default=None,
                   help="Pin num_producers to this value (synthetic only)")
    p.add_argument("--consumers", type=int, default=None,
                   help="Pin num_consumers to this value (synthetic only)")
    p.add_argument("--drivers", type=int, default=None,
                   help="Pin num_drivers to this value")
    p.add_argument("--window", type=float, default=None,
                   help="Pin window_size to a single value")
    p.add_argument("--windows", type=str, default=None,
                   help="Sweep multiple window sizes (comma-separated, e.g. 0.25,0.5,1.0,2.0,4.0,12.0)")
    p.add_argument("--speed", type=float, default=None,
                   help="Pin average_speed_kph to this value")
    p.add_argument("--grid", type=int, default=None,
                   help="Pin grid_size to this value (synthetic only)")
    p.add_argument("--pattern", type=str, default=None,
                   choices=["uniform", "unimodal", "bimodal"],
                   help="Pin demand_pattern to this value")
    p.add_argument("--seed", type=int, default=None,
                   help="Pin random seed to a single value")
    p.add_argument("--seeds", type=str, default=None,
                   help="Sweep multiple seeds (comma-separated, e.g. 42,123,456)")
    p.add_argument("--hours", type=float, default=None,
                   help="Pin simulation_hours to this value")

    # Solver mode
    p.add_argument("--two-tier", action="store_true",
                   help="Force brute-force on all classical batches (no NN tier). "
                        "Sets brute_force_timeout_s=10 unless --bf-timeout is also given.")
    p.add_argument("--bf-timeout", type=float, default=None,
                   help="Brute-force timeout in seconds (default: 10 with --two-tier, 0 otherwise)")

    return p.parse_args()


def _apply_solver_flags(args, experiments):
    """Apply --two-tier and --bf-timeout flags to all experiment configs."""
    if args.two_tier or args.bf_timeout is not None:
        bf_timeout = args.bf_timeout if args.bf_timeout is not None else (10.0 if args.two_tier else 0.0)
        for exp in experiments:
            if args.two_tier:
                exp.sim.two_tier_mode = True
            exp.sim.brute_force_timeout_s = bf_timeout


def build_synthetic_sweep(args) -> SweepConfig:
    """Build a SweepConfig, replacing param lists with single-element lists for any pinned args."""
    sweep = SweepConfig()

    if args.orders is not None:
        sweep.orders = [args.orders]
    if args.producers is not None:
        sweep.producers = [args.producers]
    if args.consumers is not None:
        sweep.consumers = [args.consumers]
    if args.drivers is not None:
        sweep.num_drivers = [args.drivers]
    if args.windows is not None:
        sweep.window_sizes = [float(w) for w in args.windows.split(",")]
    elif args.window is not None:
        sweep.window_sizes = [args.window]
    if args.speed is not None:
        sweep.average_speeds = [args.speed]
    if args.grid is not None:
        sweep.grid_sizes = [args.grid]
    if args.pattern is not None:
        sweep.demand_patterns = [args.pattern]
    if args.hours is not None:
        sweep.simulation_hours = [args.hours]
    if args.seeds is not None:
        sweep.seeds = [int(s) for s in args.seeds.split(",")]
    elif args.seed is not None:
        sweep.seeds = [args.seed]

    return sweep


def build_real_sweep(args) -> RealSweepConfig:
    """Build a RealSweepConfig with any pinned args."""
    sweep = RealSweepConfig(city=args.city)

    if args.orders is not None:
        sweep.orders = [args.orders]
    if args.drivers is not None:
        sweep.num_drivers = [args.drivers]
    if args.windows is not None:
        sweep.window_sizes = [float(w) for w in args.windows.split(",")]
    elif args.window is not None:
        sweep.window_sizes = [args.window]
    if args.pattern is not None:
        sweep.demand_patterns = [args.pattern]
    if args.hours is not None:
        sweep.simulation_hours = [args.hours]
    if args.seeds is not None:
        sweep.seeds = [int(s) for s in args.seeds.split(",")]
    elif args.seed is not None:
        sweep.seeds = [args.seed]

    return sweep


def run_synthetic_mode(args):
    """Run synthetic mode sweep."""
    if args.sweep_name is None:
        from datetime import datetime
        args.sweep_name = f"sweep_{datetime.now().strftime('%Y%m%d_%H%M')}"

    results_dir = args.results_dir or os.path.join("results", "synthetic", args.sweep_name)
    data_root = args.data_root or "data_synthetic"
    max_workers = args.workers or min(os.cpu_count() or 1, 8)

    sweep = build_synthetic_sweep(args)
    experiments = sweep.generate_experiments()
    _apply_solver_flags(args, experiments)

    os.makedirs(results_dir, exist_ok=True)
    print(f"Mode: synthetic  |  Sweep: {args.sweep_name}")
    print(f"Configs: {len(experiments)}  |  Workers: {max_workers}")
    print(f"Data: {data_root}  |  Results: {results_dir}")

    # --- Generate / cache datasets ---
    print("\n--- Generating datasets ---")
    datasets = {}
    for exp in experiments:
        label = exp.data.label
        if label not in datasets:
            loc_df, ord_df = generate_dataset(exp.data, data_root=data_root)
            datasets[label] = (loc_df, ord_df)
    print(f"Unique datasets: {len(datasets)}")

    # --- Run experiments ---
    _run_experiments(experiments, datasets, results_dir, max_workers,
                     get_meta=lambda exp: {
                         "Num_Producers": exp.data.num_producers,
                         "Num_Consumers": exp.data.num_consumers,
                         "Demand_Pattern": exp.data.demand_pattern,
                         "Grid_Size": exp.data.grid_size,
                         "Seed": exp.data.seed,
                         "Experiment_Label": exp.label,
                     },
                     get_dataset_key=lambda exp: exp.data.label,
                     sequential=args.sequential)


def run_real_mode(args):
    """Run real city mode sweep."""
    if args.sweep_name is None:
        from datetime import datetime
        args.sweep_name = f"sweep_{datetime.now().strftime('%Y%m%d_%H%M')}"

    data_root = args.data_root or "data_real"
    results_dir = args.results_dir or os.path.join("results", "real", args.sweep_name)
    max_workers = args.workers or min(os.cpu_count() or 1, 8)

    # Load city data once (road network, locations, matrices)
    print(f"\n--- Loading city data: {args.city} ---")
    city_data = load_city_data(args.city, data_root=data_root)
    locations_df = city_data["locations_df"]
    dist_matrix_km = city_data["dist_matrix_km"]
    time_matrix_hours = city_data["time_matrix_hours"]
    node_index_map = city_data["node_index_map"]
    pickup_counts = city_data["pickup_counts"]
    dropoff_counts = city_data["dropoff_counts"]

    n_nodes = len(locations_df)
    n_producers = len(locations_df[locations_df["type"].isin(["Producer", "Both"])])
    n_consumers = len(locations_df[locations_df["type"].isin(["Consumer", "Both"])])
    print(f"  Nodes: {n_nodes}  |  Producer nodes: {n_producers}  |  Consumer nodes: {n_consumers}")
    print(f"  Distance range: {dist_matrix_km[dist_matrix_km > 0].min():.2f} - {dist_matrix_km.max():.2f} km")

    sweep = build_real_sweep(args)
    experiments = sweep.generate_experiments()
    _apply_solver_flags(args, experiments)

    os.makedirs(results_dir, exist_ok=True)
    print(f"\nMode: real ({args.city})  |  Sweep: {args.sweep_name}")
    print(f"Configs: {len(experiments)}  |  Workers: {max_workers}")
    print(f"Results: {results_dir}")

    # --- Generate orders per config ---
    print("\n--- Generating orders ---")
    datasets = {}
    for exp in experiments:
        label = exp.real_data.label
        if label not in datasets:
            ord_df = generate_real_orders(
                exp.real_data, locations_df, pickup_counts, dropoff_counts, node_index_map
            )
            datasets[label] = (locations_df, ord_df)
            print(f"  [GEN] {label} ({exp.real_data.num_orders} orders)")
    print(f"Unique datasets: {len(datasets)}")

    # --- Run experiments ---
    _run_experiments(experiments, datasets, results_dir, max_workers,
                     get_meta=lambda exp: {
                         "City": exp.real_data.city,
                         "Demand_Pattern": exp.real_data.demand_pattern,
                         "Seed": exp.real_data.seed,
                         "Experiment_Label": exp.label,
                     },
                     get_dataset_key=lambda exp: exp.real_data.label,
                     time_matrix=time_matrix_hours,
                     dist_matrix=dist_matrix_km,
                     node_index_map=node_index_map,
                     sequential=args.sequential)


def _estimate_cost(exp):
    """Estimate relative experiment cost for scheduling. Higher = more expensive."""
    orders = exp.real_data.num_orders if exp.mode == "real" else exp.data.num_orders
    return orders


def _run_experiments(experiments, datasets, results_dir, max_workers,
                     get_meta, get_dataset_key,
                     time_matrix=None, dist_matrix=None, node_index_map=None,
                     sequential=False):
    """Shared experiment execution logic for both modes."""
    print("\n--- Running experiments ---")
    csv_path = os.path.join(results_dir, "sweep_results.csv")

    completed = set()
    if os.path.exists(csv_path):
        completed = set(pd.read_csv(csv_path)["Experiment_Label"].tolist())

    pending = [e for e in experiments if e.label not in completed]

    # Schedule heaviest experiments first (LPT) to avoid tail-end CPU contention.
    # Order count dominates cost since each order generates quantum solver batches.
    pending.sort(key=lambda e: _estimate_cost(e), reverse=True)

    print(f"Completed: {len(completed)}  |  Pending: {len(pending)}  |  Total: {len(experiments)}")

    if pending:
        header_needed = not os.path.exists(csv_path)

        if sequential:
            for i, exp in enumerate(pending):
                loc_df, ord_df = datasets[get_dataset_key(exp)]
                meta = get_meta(exp)
                print(f"  [{i+1}/{len(pending)}] {exp.label}")
                try:
                    result = run_experiment(
                        exp.sim, loc_df, ord_df, meta,
                        time_matrix=time_matrix, dist_matrix=dist_matrix,
                        node_index_map=node_index_map)
                    row = pd.DataFrame([result])
                    row.to_csv(csv_path, mode="a", header=header_needed, index=False)
                    header_needed = False
                    print(f"    Done.")
                except Exception as e:
                    print(f"    [ERROR] {e}")
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = {}
                for exp in pending:
                    loc_df, ord_df = datasets[get_dataset_key(exp)]
                    meta = get_meta(exp)
                    futures[pool.submit(
                        run_experiment, exp.sim, loc_df, ord_df, meta,
                        time_matrix=time_matrix, dist_matrix=dist_matrix,
                        node_index_map=node_index_map
                    )] = exp.label

                done_count = 0
                for f in as_completed(futures):
                    try:
                        row = pd.DataFrame([f.result()])
                        row.to_csv(csv_path, mode="a", header=header_needed, index=False)
                        header_needed = False
                        done_count += 1
                        if done_count % 10 == 0 or done_count == len(futures):
                            print(f"  Progress: {done_count}/{len(futures)}")
                    except Exception as e:
                        print(f"  [ERROR] {futures[f]}: {e}")

    results_df = pd.read_csv(csv_path)
    print(f"Total results: {len(results_df)}")

    print("\nDone.")


def main():
    args = parse_args()

    if args.mode == "real":
        run_real_mode(args)
    else:
        run_synthetic_mode(args)


if __name__ == "__main__":
    main()
