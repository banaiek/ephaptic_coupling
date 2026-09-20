# Self-generated ephaptic fields improve the neural population code
# Kianoush Banaie Boroujeni and Sabine Kastner
# Code by Kianoush Banaie Boroujeni, September 2026

from dataclasses import dataclass, field
from typing import Dict, Any, Callable
import numpy as np
from config import Config
from geometry import NetworkGeometry
from simulation import simulate_network


@dataclass
class TaskResult:
    name: str
    metrics: Dict[str, float] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    stats: Dict[str, Dict] = field(default_factory=dict)


class Task:

    def __init__(self, config: Config, geometry: NetworkGeometry):
        self.config = config
        self.geometry = geometry
        self.NE = config.network.NE
        self.NI = config.network.NI
        self.N = config.network.N
        self.dt = config.sim.dt

    def run(self, seed: int = 0) -> TaskResult:
        raise NotImplementedError

    def _build_shared_noise(self, T_steps, amplitude, rng, smooth_ms=50.0):
        kernel_len = max(1, int(round(smooth_ms / self.dt)))
        kernel = np.ones(kernel_len) / kernel_len

        if self.config.elif_params.use_geometry:
            K = self.geometry.G.shape[1]
            xi = rng.standard_normal((K, T_steps))
            for k in range(K):
                xi[k] = np.convolve(xi[k], kernel, mode='same')
            noise = amplitude * (self.geometry.G @ xi)
        else:
            raw = rng.standard_normal(T_steps)
            smoothed = np.convolve(raw, kernel, mode='same') * amplitude
            noise = np.outer(np.ones(self.N), smoothed)
        return noise

    def _run_paired_simulation(self, I_ext, T_ms, record_voltage=False):
        result_std = simulate_network(
            self.config, self.geometry, I_ext, T_ms,
            use_elif=False, record_voltage=record_voltage
        )
        result_elif = simulate_network(
            self.config, self.geometry, I_ext, T_ms,
            use_elif=True, record_voltage=record_voltage
        )
        return result_std, result_elif

    def _find_rate_match_delta(self, build_input_fn, T_ms, count_mask,
                               target_rate, n_search_trials=10, seed_base=99999):
        lo, hi = 0.0, 3.0
        for iteration in range(5):
            mid = (lo + hi) / 2
            total_spikes = 0
            for t_idx in range(n_search_trials):
                I_ext = build_input_fn(seed_base + t_idx, mid)
                res = simulate_network(self.config, self.geometry, I_ext, T_ms, use_elif=True)
                total_spikes += res.spikes[:, count_mask].sum()
            est_rate = total_spikes / n_search_trials
            if est_rate > target_rate:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
