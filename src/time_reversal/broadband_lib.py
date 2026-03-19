import concurrent.futures
import dataclasses

import numpy as np
import tqdm

from time_reversal.simulation import run_single_simulation


def _run_single_freq_worker(args):
    """
    Worker function for parallel frequency simulation.
    Args: (cfg, omega, seed)
    """
    cfg, omega, seed = args
    cfg_freq = dataclasses.replace(cfg, w=omega)

    # Set seed for reproducible random medium
    np.random.seed(seed)

    # Run simulation
    res = run_single_simulation(cfg_freq, return_history=True, use_fast_solver=True)

    if res.history is None:
        return None

    # superposition
    nz = res.history.shape[0]
    z_grid = np.arange(nz) * cfg_freq.h
    modulation = np.exp(1j * cfg_freq.k_const * z_grid)

    return res.history * modulation[:, np.newaxis]


def run_broadband_parallel(cfg, n_freq=50, delta_omega=1.5, seed=42):
    """
    Runs broadband simulation using ProcessPoolExecutor.
    """
    omegas = np.linspace(cfg.w - delta_omega, cfg.w + delta_omega, n_freq)

    tasks = [(cfg, w, seed) for w in omegas]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(
            tqdm.tqdm(
                executor.map(_run_single_freq_worker, tasks),
                total=len(tasks),
                desc="Simulating frequencies",
            )
        )

    final_fields = np.array([r for r in results if r is not None])

    return final_fields, omegas


def reconstruct_time_domain(final_fields, omegas, t_max, nt=100):
    """
    Reconstructs time domain signal.
    """
    t_grid = np.linspace(0, t_max, nt)
    n_freq, nz, nx = final_fields.shape

    # Matrix multiplication: Time(t, pixel) = Sum_w Field(w, pixel) * exp(-i w t) * dw
    dw = omegas[1] - omegas[0] if len(omegas) > 1 else 1.0

    # (nt, n_freq)
    transform_matrix = np.exp(-1j * t_grid[:, np.newaxis] * omegas[np.newaxis, :])

    # Flatten spatial: (n_freq, nz*nx)
    flat_fields = final_fields.reshape(n_freq, -1)

    # (nt, nz*nx)
    time_domain_flat = transform_matrix @ flat_fields

    # Reshape: (nt, nz, nx)
    time_domain = time_domain_flat.reshape(nt, nz, nx)

    # Real part is physical field
    return t_grid, (time_domain * dw / (2 * np.pi)).real
