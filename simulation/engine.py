"""Simulation engine that orchestrates solver runs and collects performance metrics.

Processes orders in fixed-width time windows. Within each window, orders to the same
producer are batched into a single delivery trip. Drivers are assigned by earliest
availability including deadhead travel time. Metrics track delivery times, energy,
and quantum/classical usage.
"""

import math
import time as time_mod
import numpy as np
import pandas as pd
from datetime import datetime

from config.config import SimulationConfig
from .solver import VRPSolver


class SimulationEngine:
    """Runs the time-window VRP simulation."""

    def __init__(self, config: SimulationConfig, locations_df: pd.DataFrame, orders_df: pd.DataFrame,
                 time_matrix: np.ndarray = None, dist_matrix: np.ndarray = None,
                 node_index_map: dict = None):
        self.config = config
        self.locations_df = locations_df
        self.orders_df = orders_df.copy()
        self.time_matrix = time_matrix
        self.dist_matrix = dist_matrix

        # Parse coordinates via column zip (faster than iterrows)
        self.coords = {int(nid): np.array([x, y])
                       for nid, x, y in zip(locations_df['node_id'], locations_df['x'], locations_df['y'])}

        self.solver = VRPSolver(config, self.coords,
                                global_dist_matrix=dist_matrix,
                                node_index_map=node_index_map)
        self.final_routes = []

        # Seeded RNG for reproducible random service times
        self.rng = np.random.RandomState(42)

        # Initialize order tracking columns
        self.orders_df['hop_type'] = 'Pending'
        self.orders_df['method'] = 'Pending'
        self.orders_df['delivered_time'] = np.nan
        self.orders_df['delay'] = np.nan
        self.orders_df['driver_id'] = -1

    def _deadhead_time(self, from_node, to_node):
        """Travel time (hours) from a driver's current location to a producer."""
        if self.time_matrix is not None:
            return self.time_matrix[from_node, to_node]
        else:
            dist = np.linalg.norm(self.coords[from_node] - self.coords[to_node])
            return dist / self.config.average_speed_kph

    def _driver_available_at(self, driver_id, producer_id, t_end,
                             driver_free_time, driver_location):
        """Earliest time this driver can begin serving at the producer."""
        free_at = max(t_end, driver_free_time[driver_id])
        loc = driver_location[driver_id]
        if loc is None:
            return free_at  # first assignment, no deadhead
        return free_at + self._deadhead_time(loc, producer_id)

    def run(self) -> dict:
        """Run the full simulation. Returns a flat metrics dict."""
        # Divide the simulation timeline into fixed-width windows
        sim_duration = self.orders_df['time'].max() + self.config.window_size
        num_windows = math.ceil(sim_duration / self.config.window_size)

        # Track when each driver becomes free and where they are
        driver_free_time = {i: 0.0 for i in range(self.config.num_drivers)}
        driver_location = {i: None for i in range(self.config.num_drivers)}
        hourly_unused_stats = {}
        current_hour_active_drivers = set()

        # Accumulators for new metrics
        total_distance_km = 0.0
        orders_by_method = {'quantum': 0, 'classical': 0, 'nn': 0}
        driver_batch_count = {i: 0 for i in range(self.config.num_drivers)}
        driver_earliest_start = {i: float('inf') for i in range(self.config.num_drivers)}
        driver_latest_finish = {i: 0.0 for i in range(self.config.num_drivers)}

        for i in range(num_windows):
            t_start = i * self.config.window_size
            t_end = (i + 1) * self.config.window_size
            current_hour_int = int(t_start)

            # Reset active-driver tracking at each integer hour boundary
            if t_start == float(current_hour_int):
                current_hour_active_drivers = set()
                for d_id, finish_time in driver_free_time.items():
                    if finish_time > t_start:
                        current_hour_active_drivers.add(d_id)

            # Batch all orders whose timestamp falls in this window
            batch = self.orders_df[
                (self.orders_df['time'] >= t_start) & (self.orders_df['time'] < t_end)
            ]

            if not batch.empty:
                # Group by producer -- orders to the same producer share one delivery trip
                prod_groups = batch.groupby('producer_id')

                for prod_id, group in prod_groups:
                    # Location-aware driver assignment: earliest available at this producer
                    assigned_driver = min(
                        driver_free_time,
                        key=lambda d: self._driver_available_at(
                            d, prod_id, t_end, driver_free_time, driver_location)
                    )
                    current_hour_active_drivers.add(assigned_driver)

                    consumers = group['consumer_id'].tolist()
                    num_orders_in_batch = len(consumers)
                    # Tag as multi-hop or single-hop for downstream analysis
                    h_type = "Multi-Hop" if num_orders_in_batch > 1 else "Single-Hop"
                    self.orders_df.loc[group.index, 'hop_type'] = h_type
                    self.orders_df.loc[group.index, 'driver_id'] = assigned_driver

                    # Solve the TSP for this batch (quantum if small enough)
                    route, method = self.solver.solve_batch(prod_id, consumers)
                    self.orders_df.loc[group.index, 'method'] = method
                    self.final_routes.append((assigned_driver, route))

                    # Track orders per method
                    if "Quantum" in method:
                        orders_by_method['quantum'] += num_orders_in_batch
                    elif "NN" in method:
                        orders_by_method['nn'] += num_orders_in_batch
                    else:
                        orders_by_method['classical'] += num_orders_in_batch

                    # Compute route distance and accumulate total
                    route_dist_km = self.solver.calculate_route_distance(route)
                    total_distance_km += route_dist_km

                    if self.time_matrix is not None:
                        # Real mode: sum precomputed travel times along route
                        trip_time_hours = 0.0
                        for seg_i in range(len(route) - 1):
                            trip_time_hours += self.time_matrix[route[seg_i], route[seg_i + 1]]
                        if route[-1] != route[0]:
                            trip_time_hours += self.time_matrix[route[-1], route[0]]
                    else:
                        trip_time_hours = route_dist_km / self.config.average_speed_kph

                    # Compute deadhead travel time from driver's current location to producer
                    if driver_location[assigned_driver] is not None:
                        deadhead_hours = self._deadhead_time(
                            driver_location[assigned_driver], prod_id)
                    else:
                        deadhead_hours = 0.0

                    # Random service/dwell time at each consumer stop
                    num_consumer_stops = len(route) - 1  # route[0] is producer
                    stop_service_times = self.rng.uniform(
                        self.config.service_time_min_minutes,
                        self.config.service_time_max_minutes,
                        size=num_consumer_stops
                    ) / 60.0  # convert minutes to hours
                    total_service_hours = stop_service_times.sum()

                    # Driver dispatches after window closes (all orders collected)
                    start_service_time = max(t_end, driver_free_time[assigned_driver]) + deadhead_hours
                    finish_service_time = start_service_time + trip_time_hours + total_service_hours
                    driver_free_time[assigned_driver] = finish_service_time

                    # Track per-driver workload
                    driver_batch_count[assigned_driver] += 1
                    driver_earliest_start[assigned_driver] = min(
                        driver_earliest_start[assigned_driver], start_service_time)
                    driver_latest_finish[assigned_driver] = max(
                        driver_latest_finish[assigned_driver], finish_service_time)

                    # Walk the route to record per-order delivery times
                    curr_sim_time = start_service_time
                    curr_node = route[0]
                    for stop_idx, next_node in enumerate(route[1:]):
                        if self.time_matrix is not None:
                            seg_time = self.time_matrix[curr_node, next_node]
                        else:
                            dist_seg = np.linalg.norm(self.coords[curr_node] - self.coords[next_node])
                            seg_time = dist_seg / self.config.average_speed_kph
                        curr_sim_time += seg_time + stop_service_times[stop_idx]
                        done_orders = group[group['consumer_id'] == next_node].index
                        self.orders_df.loc[done_orders, 'delivered_time'] = curr_sim_time
                        self.orders_df.loc[done_orders, 'delay'] = (
                            curr_sim_time - self.orders_df.loc[done_orders, 'time']
                        )
                        curr_node = next_node

                    # Update driver's location to last consumer delivered
                    driver_location[assigned_driver] = route[-1]

            # Record unused drivers at each integer hour boundary
            if t_end == int(t_end):
                hour_recorded = int(t_start)
                active_count = len(current_hour_active_drivers)
                unused_count = self.config.num_drivers - active_count
                hourly_unused_stats[hour_recorded] = unused_count

        # Compute result metrics
        delivered_df = self.orders_df.dropna(subset=['delivered_time'])
        total_delivered = len(delivered_df)
        total_orders = len(self.orders_df)
        avg_delivery_time_min = (delivered_df['delay'].mean() * 60) if not delivered_df.empty else 0
        median_delivery_time_min = (delivered_df['delay'].median() * 60) if not delivered_df.empty else 0
        max_delivery_time_min = (delivered_df['delay'].max() * 60) if not delivered_df.empty else 0
        std_delivery_time_min = (delivered_df['delay'].std() * 60) if not delivered_df.empty else 0

        multi_total = self.orders_df[self.orders_df['hop_type'] == 'Multi-Hop']
        single_total = self.orders_df[self.orders_df['hop_type'] == 'Single-Hop']

        unique_active = delivered_df['driver_id'].unique()
        utilization_ratio = (len(unique_active) / self.config.num_drivers) * 100 if self.config.num_drivers > 0 else 0

        m = self.solver.metrics
        q_batches = m['q_count']
        avg_q_time = (m['q_time'] / q_batches) if q_batches > 0 else 0
        total_batches = m['q_count'] + m['c_count'] + m['nn_count']

        # Driver workload balance
        max_driver_batches = max(driver_batch_count.values()) if driver_batch_count else 0
        max_driver_active_hours = max(
            (driver_latest_finish[d] - driver_earliest_start[d]
             for d in range(self.config.num_drivers)
             if driver_batch_count[d] > 0),
            default=0.0
        )

        return {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Total_Drivers": self.config.num_drivers,
            "Window_Size": self.config.window_size,
            "Average_Speed_KPH": self.config.average_speed_kph,
            "Utilization_Ratio_%": round(utilization_ratio, 2),
            "Classical_Batches": m['c_count'],
            "Classical_Time_s": round(m['c_time'], 4),
            "Quantum_Batches": q_batches,
            "Quantum_Time_s": round(m['q_time'], 4),
            "Avg_Quantum_Time_s": round(avg_q_time, 4),
            "NN_Batches": m['nn_count'],
            "NN_Time_s": round(m['nn_time'], 4),
            "Total_Batches": total_batches,
            "Avg_Batch_Size": round(total_orders / max(total_batches, 1), 2),
            "Peak_RAM_GB": round(m['peak_ram_gb'], 2),
            "Total_Energy_J": round(m['total_energy_joules'], 4),
            "Total_Distance_km": round(total_distance_km, 4),
            "Total_Orders": total_orders,
            "Total_Delivered": total_delivered,
            "Avg_Delivery_Time_Min": round(avg_delivery_time_min, 2),
            "Median_Delivery_Time_Min": round(median_delivery_time_min, 2),
            "Max_Delivery_Time_Min": round(max_delivery_time_min, 2),
            "Std_Delivery_Time_Min": round(std_delivery_time_min, 2),
            "Quantum_Orders": orders_by_method['quantum'],
            "Classical_Orders": orders_by_method['classical'],
            "NN_Orders": orders_by_method['nn'],
            "Quantum_Optimal_Hits": m['quantum_optimal_hits'],
            "MultiHop_Success": int(multi_total['delivered_time'].notna().sum()),
            "MultiHop_Failed": int(len(multi_total) - multi_total['delivered_time'].notna().sum()),
            "SingleHop_Success": int(single_total['delivered_time'].notna().sum()),
            "SingleHop_Failed": int(len(single_total) - single_total['delivered_time'].notna().sum()),
            "Max_Driver_Batches": max_driver_batches,
            "Max_Driver_Active_Hours": round(max_driver_active_hours, 4),
            "Brute_Force_Timeouts": m['brute_force_timeouts'],
        }


def run_experiment(sim_config, loc_df, ord_df, data_meta,
                   time_matrix=None, dist_matrix=None, node_index_map=None):
    """Module-level worker function (picklable for multiprocessing)."""
    t_wall_start = time_mod.time()
    engine = SimulationEngine(sim_config, loc_df, ord_df,
                              time_matrix=time_matrix, dist_matrix=dist_matrix,
                              node_index_map=node_index_map)
    result = engine.run()
    result['Wall_Clock_s'] = round(time_mod.time() - t_wall_start, 4)
    result.update(data_meta)
    return result
