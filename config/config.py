"""Configuration dataclasses for data generation, simulation, and parameter sweeps.

Config layers:
  - DataGenConfig:    controls the synthetic dataset (order count, grid, demand shape)
  - RealDataConfig:   controls real-city dataset (city, order count, demand shape)
  - SimulationConfig: controls the solver and driver fleet (power, speed, thresholds)
  - ExperimentConfig: pairs one data config with one simulation config
  - SweepConfig:      defines parameter ranges for synthetic mode
  - RealSweepConfig:  defines parameter ranges for real city mode
"""

from dataclasses import dataclass, field
from typing import List, Optional
import itertools


@dataclass
class DataGenConfig:
    """Parameters for synthetic dataset generation."""
    num_orders: int = 300            # total delivery orders to generate
    num_producers: int = 5           # pickup locations (warehouses/restaurants)
    num_consumers: int = 100         # delivery destinations (customers)
    simulation_hours: float = 12.0   # time horizon for order timestamps
    grid_size: int = 10              # spatial grid dimension (grid_size x grid_size)
    demand_pattern: str = "uniform"  # temporal distribution: uniform, unimodal, bimodal
    seed: int = 42                   # random seed for reproducibility

    @property
    def label(self) -> str:
        """Content-addressed label used for dataset caching (same params = same folder)."""
        return (f"o{self.num_orders}_p{self.num_producers}_c{self.num_consumers}"
                f"_g{self.grid_size}_{self.demand_pattern}_s{self.seed}")


@dataclass
class RealDataConfig:
    """Parameters for real-city dataset generation.

    The city's road network and location counts come from data_real/{city}/.
    Orders are still synthetically generated but use real spatial weights.
    """
    city: str = "albuquerque"            # folder name under data_real/
    num_orders: int = 300                # delivery orders to generate
    simulation_hours: float = 12.0       # time horizon for order timestamps
    demand_pattern: str = "uniform"      # temporal distribution: uniform, unimodal, bimodal
    seed: int = 42                       # random seed for reproducibility
    noise_std: float = 0.1              # Gaussian noise for producer/consumer role assignment

    @property
    def label(self) -> str:
        """Content-addressed label for real-mode experiments."""
        return (f"real_{self.city}_o{self.num_orders}"
                f"_{self.demand_pattern}_s{self.seed}")


@dataclass
class SimulationConfig:
    """Parameters for the simulation engine and solver."""
    cpu_power_watts: float = 111.0      # assumed CPU power for energy tracking (classical)
    qpu_power_watts: float = 15.0       # assumed QPU power for energy tracking (quantum)
    average_speed_kph: float = 35.0     # driver travel speed for time calculations
    window_size: float = 0.5            # time-window width in hours for order batching
    num_drivers: int = 3                # size of the driver fleet
    service_time_min_minutes: float = 2.0   # min dwell time per consumer stop (minutes)
    service_time_max_minutes: float = 5.0   # max dwell time per consumer stop (minutes)
    quantum_limit: int = 5              # max batch size for quantum solver (n^2 qubits)
    classical_exact_limit: int = 7      # max batch size for brute-force (n! complexity)
    brute_force_timeout_s: float = 0.0  # brute-force timeout in seconds (0 = no timeout)
    quantum_timeout_s: int = 120         # quantum solver timeout in seconds (0 = no timeout)
    two_tier_mode: bool = False          # force brute-force on all classical batches (no NN tier)

    @property
    def label(self) -> str:
        base = f"d{self.num_drivers}_w{self.window_size}_spd{self.average_speed_kph}"
        return f"{base}_2t" if self.two_tier_mode else base


@dataclass
class ExperimentConfig:
    """A single experiment = one dataset config + one simulation config.

    mode="synthetic" uses DataGenConfig (data field).
    mode="real" uses RealDataConfig (real_data field).
    """
    data: DataGenConfig = field(default_factory=DataGenConfig)
    sim: SimulationConfig = field(default_factory=SimulationConfig)
    real_data: Optional[RealDataConfig] = None
    mode: str = "synthetic"

    @property
    def label(self) -> str:
        """Unique identifier used for resume support (skip already-completed experiments)."""
        if self.mode == "real" and self.real_data is not None:
            return f"{self.real_data.label}_{self.sim.label}"
        return f"{self.data.label}_{self.sim.label}"


@dataclass
class SweepConfig:
    """Defines parameter ranges for a grid sweep over experiments.

    Default values represent the standard large sweep. For focused sweeps,
    replace individual lists with single-element lists to pin those params.
    """
    orders: List[int] = field(default_factory=lambda: [2000, 2500]) #, 3000])
    producers: List[int] = field(default_factory=lambda: [100])
    consumers: List[int] = field(default_factory=lambda: [1000])
    num_drivers: List[int] = field(default_factory=lambda: [5, 10, 15, 20, 25, 30])
    window_sizes: List[float] = field(default_factory=lambda: [0.25])
    simulation_hours: List[float] = field(default_factory=lambda: [12.0])
    average_speeds: List[float] = field(default_factory=lambda: [35.0])
    demand_patterns: List[str] = field(default_factory=lambda: ["uniform"] )# , "unimodal", "bimodal"])
    grid_sizes: List[int] = field(default_factory=lambda: [20])
    seeds: List[int] = field(default_factory=lambda: [42])
    curated: bool = False  # when True, skip unrealistic param combos (see filters below)

    def generate_experiments(self) -> List[ExperimentConfig]:
        """Generate the full Cartesian product of param ranges, with optional curation filters."""
        experiments = []
        for orders, prods, cons, drivers, window, hours, speed, pattern, grid, seed in itertools.product(
            self.orders, self.producers, self.consumers,
            self.num_drivers, self.window_sizes,
            self.simulation_hours, self.average_speeds,
            self.demand_patterns, self.grid_sizes, self.seeds,
        ):
            if self.curated:
                # Skip under-resourced configs: need ~1 driver per 60-70 orders
                if orders >= 3000 and drivers < 50:
                    continue
                if orders >= 2500 and drivers < 40:
                    continue
                if orders >= 2000 and drivers < 30:
                    continue
                # Small windows + high orders = too many batches, prohibitively slow
                if orders >= 2000 and window > 0.25:
                    continue
                # Consumer pool should be at least as large as producer count
                if cons < prods:
                    continue

            experiments.append(ExperimentConfig(
                data=DataGenConfig(
                    num_orders=orders,
                    num_producers=prods,
                    num_consumers=cons,
                    simulation_hours=hours,
                    grid_size=grid,
                    demand_pattern=pattern,
                    seed=seed,
                ),
                sim=SimulationConfig(
                    num_drivers=drivers,
                    window_size=window,
                    average_speed_kph=speed,
                ),
            ))
        return experiments


@dataclass
class RealSweepConfig:
    """Defines parameter ranges for a grid sweep over real-city experiments.

    No producers/consumers/grid params -- those come from the real city data.
    """
    city: str = "santa_fe"
    orders: List[int] = field(default_factory=lambda: [500, 1000, 1500, 2000])
    num_drivers: List[int] = field(default_factory=lambda: [5, 10, 15, 20, 30, 40, 50])
    window_sizes: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 1])
    simulation_hours: List[float] = field(default_factory=lambda: [12.0])
    demand_patterns: List[str] = field(default_factory=lambda: ["uniform", "unimodal", "bimodal"])
    seeds: List[int] = field(default_factory=lambda: [123])

    def generate_experiments(self) -> List[ExperimentConfig]:
        """Generate the Cartesian product of real-mode param ranges."""
        experiments = []
        for orders, drivers, window, hours, pattern, seed in itertools.product(
            self.orders, self.num_drivers, self.window_sizes,
            self.simulation_hours, self.demand_patterns, self.seeds,
        ):
            experiments.append(ExperimentConfig(
                mode="real",
                real_data=RealDataConfig(
                    city=self.city,
                    num_orders=orders,
                    simulation_hours=hours,
                    demand_pattern=pattern,
                    seed=seed,
                ),
                sim=SimulationConfig(
                    num_drivers=drivers,
                    window_size=window,
                ),
            ))
        return experiments
