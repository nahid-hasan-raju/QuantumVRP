"""Classical and quantum-inspired VRP solvers with time-window batching.

Routing strategy is chosen by batch size:
  - n <= quantum_limit (default 5):   Qiskit exact eigensolver (quantum-inspired)
  - n <= classical_exact_limit (7):    Brute-force TSP (exact, factorial time)
  - n > classical_exact_limit:         Nearest-neighbor heuristic (greedy, O(n^2))

Energy is tracked per batch: QPU power (15W) for quantum, CPU power (111W) for classical,
multiplied by wall-clock solve time.
"""

import time
import itertools
import os
import signal
import numpy as np
import psutil
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="scipy.sparse")


class _QuantumTimeout(Exception):
    """Raised when a quantum solve exceeds its time limit."""
    pass


def _alarm_handler(signum, frame):
    raise _QuantumTimeout()


from qiskit_optimization.applications import Tsp
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_optimization.algorithms import MinimumEigenOptimizer

from config.config import SimulationConfig


def get_memory_usage_gb() -> float:
    """Current process RSS in GB, used to track peak RAM for quantum batches."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)


class VRPSolver:
    """Hybrid quantum/classical TSP solver for delivery batches."""

    def __init__(self, config: SimulationConfig, coords: dict,
                 global_dist_matrix: np.ndarray = None, node_index_map: dict = None):
        self.config = config
        self.coords = coords
        self.global_dist_matrix = global_dist_matrix
        self.node_index_map = node_index_map
        self.metrics = {
            'q_count': 0, 'c_count': 0, 'nn_count': 0,
            'q_time': 0.0, 'c_time': 0.0, 'nn_time': 0.0,
            'total_energy_joules': 0.0,
            'quantum_optimal_hits': 0,  # how often quantum matches brute-force optimal
            'peak_ram_gb': 0.0,
            'brute_force_timeouts': 0,
        }
        self.efficiency_data = []  # (batch_size, solve_time, method) tuples for plotting

    def calculate_route_distance(self, route, dist_matrix=None) -> float:
        """Total round-trip distance for a route (returns to start).

        Priority: explicit dist_matrix > global_dist_matrix > Euclidean coords.
        """
        if dist_matrix is not None:
            d = 0
            for i in range(len(route) - 1):
                d += dist_matrix[route[i], route[i + 1]]
            if route[-1] != route[0]:
                d += dist_matrix[route[-1], route[0]]
            return d

        # Real mode: look up segments from precomputed global matrix
        if self.global_dist_matrix is not None:
            d = 0
            for i in range(len(route) - 1):
                d += self.global_dist_matrix[route[i], route[i + 1]]
            if len(route) > 0 and route[-1] != route[0]:
                d += self.global_dist_matrix[route[-1], route[0]]
            return d

        # Synthetic mode: Euclidean distance from coordinates
        d = 0
        for i in range(len(route) - 1):
            d += np.linalg.norm(self.coords[route[i]] - self.coords[route[i + 1]])
        if len(route) > 0 and route[-1] != route[0]:
            d += np.linalg.norm(self.coords[route[-1]] - self.coords[route[0]])
        return d

    def _solve_tsp_nearest_neighbor(self, dist_matrix):
        """Greedy nearest-neighbor TSP heuristic starting from node 0.

        O(n^2) -- used when batch is too large for exact methods.
        """
        n = len(dist_matrix)
        visited = np.zeros(n, dtype=bool)
        route = [0]
        visited[0] = True
        total_dist = 0.0

        for _ in range(n - 1):
            current = route[-1]
            # Mask visited nodes with inf, then pick the closest unvisited
            dists = dist_matrix[current].copy()
            dists[visited] = np.inf
            best_next = int(np.argmin(dists))
            route.append(best_next)
            visited[best_next] = True
            total_dist += dists[best_next]

        total_dist += dist_matrix[route[-1], 0]
        return route, total_dist

    def _solve_tsp_classical_brute(self, dist_matrix, timeout_s=0.0):
        """Exact TSP via brute-force permutation (factorial time).

        Only feasible for small batches (n <= classical_exact_limit).
        If timeout_s > 0, returns the best path found so far when time expires.
        Returns (best_path, min_dist, timed_out).
        """
        n = len(dist_matrix)
        nodes = list(range(1, n))
        best_path = []
        min_dist = float('inf')
        deadline = (time.time() + timeout_s) if timeout_s > 0 else 0
        count = 0

        for p in itertools.permutations(nodes):
            current_path = [0] + list(p)
            d = 0
            for i in range(len(current_path) - 1):
                d += dist_matrix[current_path[i], current_path[i + 1]]
            d += dist_matrix[current_path[-1], 0]

            if d < min_dist:
                min_dist = d
                best_path = current_path

            if deadline:
                count += 1
                if count % 1000 == 0 and time.time() > deadline:
                    return best_path, min_dist, True

        return best_path, min_dist, False

    def solve_batch(self, producer_id: int, consumer_ids: list) -> tuple:
        """Solve a delivery batch using quantum or classical methods.

        Returns (route_node_ids, method_name).
        Batch size determines solver: quantum for small, classical for larger.
        """
        t0 = time.time()

        # Build local coordinate array: producer first, then unique consumers
        target_ids = [producer_id] + list(set(consumer_ids))
        local_coords = np.array([self.coords[nid] for nid in target_ids])
        n = len(local_coords)

        # Build local distance matrix: precomputed (real mode) or Euclidean (synthetic)
        if self.global_dist_matrix is not None:
            indices = np.array(target_ids)
            dist_matrix = self.global_dist_matrix[np.ix_(indices, indices)]
        else:
            dist_matrix = np.sqrt(((local_coords[:, None] - local_coords[None, :]) ** 2).sum(axis=-1))

        method = ""
        route_indices = []

        if n > self.config.quantum_limit:
            if self.config.two_tier_mode:
                # 2-TIER MODE: Force brute-force on ALL classical batches
                import math
                print(f"    [CLASSICAL] Batch Size {n} | Brute-force ({n}! = {math.factorial(n)} perms)")
                route_indices, _, timed_out = self._solve_tsp_classical_brute(
                    dist_matrix, timeout_s=self.config.brute_force_timeout_s)
                if timed_out:
                    self.metrics['brute_force_timeouts'] += 1
                    if not route_indices:
                        route_indices, _ = self._solve_tsp_nearest_neighbor(dist_matrix)
                    method = "Classical (Timeout)"
                else:
                    method = "Classical"
                self.metrics['c_count'] += 1
                elapsed = time.time() - t0
                self.metrics['c_time'] += elapsed
            else:
                # 3-TIER MODE: use exact brute-force or NN heuristic based on size
                if n > self.config.classical_exact_limit:
                    print(f"    [CLASSICAL] Batch Size {n} | Nearest-neighbor heuristic")
                    route_indices, _ = self._solve_tsp_nearest_neighbor(dist_matrix)
                    method = "Classical (NN)"
                    self.metrics['nn_count'] += 1
                    elapsed = time.time() - t0
                    self.metrics['nn_time'] += elapsed
                else:
                    import math
                    print(f"    [CLASSICAL] Batch Size {n} | Brute-force ({n}! = {math.factorial(n)} perms)")
                    route_indices, _, timed_out = self._solve_tsp_classical_brute(
                        dist_matrix, timeout_s=self.config.brute_force_timeout_s)
                    if timed_out:
                        self.metrics['brute_force_timeouts'] += 1
                        if not route_indices:
                            route_indices, _ = self._solve_tsp_nearest_neighbor(dist_matrix)
                        method = "Classical (Timeout)"
                    else:
                        method = "Classical"
                    self.metrics['c_count'] += 1
                    elapsed = time.time() - t0
                    self.metrics['c_time'] += elapsed
            # Energy = wall-clock time * CPU power draw
            self.metrics['total_energy_joules'] += elapsed * self.config.cpu_power_watts
        else:
            # Quantum path: use Qiskit exact eigensolver for small batches
            if n > 1:
                print(f"    [QUANTUM] Batch Size {n} ({n ** 2} Qubits) | Optimization running...")

            tsp = Tsp(dist_matrix)
            qp = tsp.to_quadratic_program()

            exact_solver = NumPyMinimumEigensolver()
            solver = MinimumEigenOptimizer(exact_solver)

            # Set up timeout (signal.alarm only works in main thread on Unix)
            use_alarm = (self.config.quantum_timeout_s > 0)
            if use_alarm:
                try:
                    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
                    signal.alarm(self.config.quantum_timeout_s)
                except (ValueError, OSError):
                    use_alarm = False  # not main thread or not Unix

            quantum_end_time = None  # set on timeout/error to split energy accounting

            try:
                result = solver.solve(qp)
                if use_alarm:
                    signal.alarm(0)
                raw = tsp.interpret(result.x)
                route_indices = np.array(raw).flatten().astype(int).tolist()
                method = "Quantum"

                # Verify quantum solution against brute-force optimal
                best_route_c, best_dist_c, _ = self._solve_tsp_classical_brute(dist_matrix)
                quantum_dist = self.calculate_route_distance(route_indices, dist_matrix)

                if quantum_dist <= best_dist_c + 1e-5:
                    self.metrics['quantum_optimal_hits'] += 1

                self.metrics['q_count'] += 1

                current_ram = get_memory_usage_gb()
                if current_ram > self.metrics['peak_ram_gb']:
                    self.metrics['peak_ram_gb'] = current_ram

            except _QuantumTimeout:
                if use_alarm:
                    signal.alarm(0)
                quantum_end_time = time.time()  # mark boundary for energy split
                print(f"      (Quantum timeout after {self.config.quantum_timeout_s}s -> brute-force fallback)")
                route_indices, _, _ = self._solve_tsp_classical_brute(dist_matrix)
                method = "Quantum (Timeout)"
                self.metrics['q_count'] += 1

            except Exception as e:
                if use_alarm:
                    signal.alarm(0)
                quantum_end_time = time.time()  # mark boundary for energy split
                print(f"      (Quantum Error -> Fallback: {e})")
                route_indices, _, _ = self._solve_tsp_classical_brute(dist_matrix)
                method = "Classical (Fallback)"
                self.metrics['c_count'] += 1

            finally:
                if use_alarm:
                    try:
                        signal.signal(signal.SIGALRM, old_handler)
                    except (ValueError, OSError):
                        pass

            elapsed = time.time() - t0
            if quantum_end_time is not None:
                # Split energy: quantum attempt at QPU rate, classical fallback at CPU rate
                q_elapsed = quantum_end_time - t0
                c_elapsed = elapsed - q_elapsed
                self.metrics['q_time'] += q_elapsed
                self.metrics['c_time'] += c_elapsed
                self.metrics['total_energy_joules'] += (
                    q_elapsed * self.config.qpu_power_watts +
                    c_elapsed * self.config.cpu_power_watts
                )
            else:
                self.metrics['q_time'] += elapsed
                self.metrics['total_energy_joules'] += elapsed * self.config.qpu_power_watts

        # Strip duplicate return-to-start if present
        if route_indices and route_indices[-1] == route_indices[0]:
            route_indices = route_indices[:-1]

        # Rotate so driver starts at producer (index 0 in local coords)
        if 0 in route_indices:
            z_idx = route_indices.index(0)
            route_indices = route_indices[z_idx:] + route_indices[:z_idx]

        # Map local indices back to global node IDs
        final_route_ids = [target_ids[i] for i in route_indices]

        if "Quantum" in method:
            self.efficiency_data.append((n, elapsed, 'Quantum'))
        elif "Classical" in method and n > 2:
            self.efficiency_data.append((n, elapsed, 'Classical'))

        return final_route_ids, method
