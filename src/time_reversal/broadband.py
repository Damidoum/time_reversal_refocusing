import concurrent.futures
import dataclasses
from typing import Tuple

import numpy as np
import tqdm

from time_reversal.config import SimulationConfig
from time_reversal.simulation import run_single_simulation


def setup_frequency_grid(
    w_central: float, delta_omega: float, n_freq: int
) -> np.ndarray:
    """
    Generates the frequency grid.
    """
    return np.linspace(w_central - delta_omega, w_central + delta_omega, n_freq)


def _run_single_freq(
    args: Tuple[SimulationConfig, float, int, bool, bool],
) -> np.ndarray:
    """
    Worker function for parallel frequency simulation.
    Must be top-level for pickling.
    """
    cfg, omega, seed, return_envelope, mixed = args
    cfg_freq = dataclasses.replace(cfg, w=omega)

    # Set seed for reproducible random medium (same map for all frequencies)
    np.random.seed(seed)

    # Run simulation
    res = run_single_simulation(
        cfg_freq, return_history=True, use_fast_solver=True, mixed=mixed
    )

    if res.history is None:
        raise ValueError("Simulation history is None but required for animation.")

    if return_envelope:
        return res.history

    modulation_factor = np.exp(
        1j * cfg_freq.k_const * np.arange(res.history.shape[0]) * cfg_freq.h
    )
    return res.history * modulation_factor[:, np.newaxis]


def simulate_frequencies(
    cfg: SimulationConfig,
    omegas: np.ndarray,
    seed: int = 42,
    return_envelope: bool = False,
    mixed: bool = False,
) -> np.ndarray:
    """
    Runs monochromatic simulations for each frequency in parallel.
    Returns: (n_freq, nz, nx) complex array.
    """
    tasks = [(cfg, w, seed, return_envelope, mixed) for w in omegas]

    final_fields_per_freq = []

    # Use ProcessPoolExecutor for CPU-bound tasks
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # map guarantees order
        results = executor.map(_run_single_freq, tasks)

        # We wrap in list to force execution if lazy, but map yields iterator
        for field in tqdm.tqdm(
            results, total=len(omegas), desc="Simulating Frequencies", leave=False
        ):
            final_fields_per_freq.append(field)

    return np.array(final_fields_per_freq)


def recover_temporal_profile(
    final_fields_per_freq: np.ndarray,
    omegas: np.ndarray,
    t_grid: np.ndarray,
) -> np.ndarray:
    """
    Synthesize time-domain pulse via Fourier transform (summation).

    Field(t) = Integral[ Field(w) * e^{-i w t} dw ]
    Discrete: Sum[ Field(w_j) * e^{-i w_j t} * dw ]
    """
    dw = omegas[1] - omegas[0] if len(omegas) > 1 else 1.0
    n_freq, nz, nx = final_fields_per_freq.shape
    nt = len(t_grid)

    modulation_matrix = np.exp(
        -1j * t_grid[:, np.newaxis] * omegas[np.newaxis, :]
    )  # (nt, n_freq)

    flat_fields = final_fields_per_freq.reshape(n_freq, -1)  # (n_freq, nz*nx)

    temporal_flat = modulation_matrix @ flat_fields  # (nt, nz*nx)

    temporal_signal = temporal_flat.reshape(nt, nz, nx)
    return (dw / (2 * np.pi)) * temporal_signal


def run_broadband_pulse_simulation(
    cfg: SimulationConfig, n_freq: int = 50, delta_omega: float = 2.0, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Main driver for broadband pulse simulation.
    Returns (t_grid, time_field_real).
    """
    omegas = setup_frequency_grid(cfg.w, delta_omega, n_freq)
    complex_fields = simulate_frequencies(cfg, omegas, seed=seed)
    t_max = 2 * cfg.L / cfg.c0 if cfg.t_max is None else cfg.t_max

    nt = 100  # Number of time frames for animation
    t_grid = np.linspace(0, t_max, nt)

    time_field_complex = recover_temporal_profile(complex_fields, omegas, t_grid)

    return t_grid, time_field_complex.real
