import concurrent.futures
import dataclasses
from pathlib import Path

import matplotlib
import numpy as np
import tqdm

from time_reversal.config import SimulationConfig
from time_reversal.simulation import run_single_simulation
from time_reversal.viz import (
    animate_sigma_grid,
    animate_wavefield_comparison,
    animate_wavefield_mp4,
    setup_style,
)

matplotlib.use("Agg")


def setup_frequency_grid(
    w_central: float, delta_omega: float, n_freq: int
) -> np.ndarray:
    """
    Generates the frequency grid and spectral weights for a Gaussian pulse.

    Args:
        w_central: Central angular frequency (omega_0).
        delta_omega: interval of freq [w0 - delta_omega, w0 + delta_omega]
        n_freq: Number of frequency points.

    Returns:
        omegas: Array of angular frequencies.
    """
    omegas = np.linspace(w_central - delta_omega, w_central + delta_omega, n_freq)
    return omegas


def _run_single_freq(args: tuple) -> np.ndarray:
    """
    Helper function for multiprocessing. Needs to be defined at the top level
    to be serializable (picklable) by concurrent.futures.
    """
    cfg, omega, seed = args
    cfg_freq = dataclasses.replace(cfg, w=omega)

    # Set seed for reproducible random medium
    np.random.seed(seed)

    # Run simulation
    res = run_single_simulation(cfg_freq, return_history=True, use_fast_solver=True)
    assert res.history is not None, "Expected history to be returned for simulation."

    modulation_factor = np.exp(
        1j * cfg_freq.k_const * np.arange(res.history.shape[0]) * cfg_freq.h
    )

    return res.history * modulation_factor[:, np.newaxis]


def simulate_frequencies(
    cfg: SimulationConfig,
    omegas: np.ndarray,
) -> np.ndarray:
    """
    Runs monochromatic simulations for each frequency in the grid in PARALLEL.

    Args:
        cfg: Base configuration (spatial params, etc.).
        omegas: Array of frequencies to simulate.

    Returns:
        final_fields: Complex field array of shape (n_freq, nz, nx).
    """
    final_fields_per_freq = []

    # Use a fixed seed for all frequencies to ensure the SAME random medium
    seed = 42

    # Pack arguments for the worker function
    tasks = [(cfg, omega, seed) for omega in omegas]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # executor.map maintains the correct output order
        results = executor.map(_run_single_freq, tasks)

        for field in tqdm.tqdm(
            results, total=len(omegas), desc="  Simulating Frequencies"
        ):
            final_fields_per_freq.append(field)

    return np.array(final_fields_per_freq)


def recover_temporal_profile(
    final_fields_per_freq: np.ndarray,
    omegas: np.ndarray,
    t_grid: np.ndarray,
) -> np.ndarray:
    """
    Reconstructs the time-domain signal using highly optimized matrix multiplication.
    """
    dw = omegas[1] - omegas[0] if len(omegas) > 1 else 0.0

    n_freq, nz, nx = final_fields_per_freq.shape

    # Shape: (nt, n_freq)
    modulation_factor = np.exp(-1j * t_grid[:, np.newaxis] * omegas[np.newaxis, :])

    # Flatten spatial dimensions to 1D for fast BLAS matrix multiplication
    # Shape: (n_freq, nz * nx)
    flat_fields = final_fields_per_freq.reshape(n_freq, -1)

    # Fast matrix multiplication: (nt, n_freq) @ (n_freq, nz*nx) -> (nt, nz*nx)
    temporal_flat_complex = modulation_factor @ flat_fields

    # Reshape back to 3D: (nt, nz, nx)
    temporal_signal_complex = temporal_flat_complex.reshape(len(t_grid), nz, nx)

    temporal_signal = (dw / (2 * np.pi)) * temporal_signal_complex
    return temporal_signal


def run_simulation(
    cfg: SimulationConfig,
    n_freq: int,
    delta_omega: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Orchestrates the broadband time reversal simulation.
    """
    print(f"Running Simulation: N={n_freq}")

    omegas = setup_frequency_grid(cfg.w, delta_omega, n_freq)
    final_fields_per_freq = simulate_frequencies(cfg, omegas)

    t_grid = np.linspace(0, cfg.t_max, cfg.nt)

    time_field = recover_temporal_profile(
        final_fields_per_freq,
        omegas,
        t_grid,
    )

    return t_grid, time_field


def main():
    setup_style()

    cfg = SimulationConfig.from_cli()
    n_freq = 100
    delta_omega = 4.5
    sigmas = [0.0, 0.5, 1.0, 2.0, 5.0]
    cfg.t_max = 2 * cfg.L / cfg.c0

    results_dict = {}

    for sigma in sigmas:
        print(f"\n=== Running Simulation for sigma = {sigma} ===")
        cfg = dataclasses.replace(cfg, sigma=sigma)
        _, time_field = run_simulation(cfg, n_freq=n_freq, delta_omega=delta_omega)
        nz = time_field.shape[1]
        intensity = time_field.real  # type: ignore
        results_dict[sigma] = {
            "forward": intensity[:, : int(nz / 2), :],
            "backward": intensity[:, int(nz / 2) + 1 :, :],
        }

        """
        for debug
        animate_wavefield_mp4(
            intensity,
            fps=30,
            t_skip=2,
            x_skip=2,
            save_path=Path(
                f"output/animation/{cfg.mirror_type}/rm_{cfg.r_m:.2f}/r0_{cfg.r0:.2f}/sigma_{sigma:.2f}.mp4"
            ),
        )
        """

        animate_wavefield_comparison(
            results_dict[sigma]["forward"],
            results_dict[sigma]["backward"],
            cfg=cfg,
            fps=30,
            t_skip=5,
            x_skip=2,
            save_path=Path(
                f"output/animation/{cfg.mirror_type}/rm_{cfg.r_m:.2f}/r0_{cfg.r0:.2f}/comparison_sigma_{sigma:.2f}.mp4"
            ),
        )

    animate_sigma_grid(
        results_dict,
        cfg=cfg,
        fps=30,
        t_skip=5,
        x_skip=5,
        save_path=Path(
            f"output/animation/{cfg.mirror_type}/rm_{cfg.r_m:.2f}/r0_{cfg.r0:.2f}/comparison_sigma_grid.mp4"
        ),
    )


if __name__ == "__main__":
    main()
