import concurrent.futures
import dataclasses

import numpy as np
import tqdm

from time_reversal.simulation import run_single_simulation


def _run_single_freq_worker(args):
    cfg, omega, seed = args
    cfg_freq = dataclasses.replace(cfg, w=omega)
    np.random.seed(seed)

    res = run_single_simulation(cfg_freq, return_history=True, use_fast_solver=True)
    if res.history is None:
        return None

    nz_total = res.history.shape[0]
    nz_half = nz_total // 2

    phi_fwd = res.history[:nz_half]
    phi_bwd = res.history[nz_half:]

    z_grid_fwd = np.arange(nz_half) * cfg_freq.h
    L = z_grid_fwd[-1]

    z_grid_bwd = np.arange(nz_total - nz_half) * cfg_freq.h + L

    u_fwd = phi_fwd * np.exp(1j * cfg_freq.k_const * z_grid_fwd)[:, np.newaxis]
    u_bwd = (
        phi_bwd * np.exp(1j * cfg_freq.k_const * (z_grid_bwd - 2 * L))[:, np.newaxis]
    )

    return u_fwd, u_bwd


def run_broadband_parallel(cfg, n_freq=50, delta_omega=1.5, seed=42):
    """
    Runs broadband simulation using ProcessPoolExecutor.
    Returns separate forward and backward fields.
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

    valid_results = [r for r in results if r is not None]

    final_fwd = np.array([r[0] for r in valid_results])
    final_bwd = np.array([r[1] for r in valid_results])

    return final_fwd, final_bwd, omegas


def reconstruct_time_domain(final_fields, omegas, t_grid):
    """
    Reconstructs time domain signal for a specific time grid.
    """
    n_freq, nz, nx = final_fields.shape

    # Matrix multiplication: Time(t, pixel) = Sum_w Field(w, pixel) * exp(-i w t) * dw
    dw = omegas[1] - omegas[0] if len(omegas) > 1 else 1.0

    # (nt, n_freq)
    transform_matrix = np.exp(-1j * t_grid[:, np.newaxis] * omegas[np.newaxis, :])

    # Flatten spatial: (n_freq, nz*nx)
    flat_fields = final_fields.reshape(n_freq, -1)

    # (nt, nz*nx)
    time_domain_flat = transform_matrix @ flat_fields

    # Reshape: (len(t_grid), nz, nx)
    time_domain = time_domain_flat.reshape(len(t_grid), nz, nx)

    # Real part is physical field
    return (time_domain * dw / (2 * np.pi)).real
