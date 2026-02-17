import dataclasses

import numpy as np

from time_reversal.config import SimulationConfig
from time_reversal.simulation import run_monte_carlo_simulation
from time_reversal.viz import (
    plot_intensity_grid,
    plot_multiple_comparisons,
    setup_style,
)


def run_comparison():
    setup_style()

    base_cfg = SimulationConfig(
        n_monte_carlo=50, nx=512, L=10.0, x_size=60.0, z_c=1.0, r0=2.0
    )

    r_ms = [2.0, 5.0, 10.0, 20.0]
    sigmas = [0.0, 0.5, 1.0, 2.0]

    intensity_grid_data = []
    results_fields: dict[float, dict[float, np.ndarray]] = {}

    x = np.linspace(base_cfg.x_min, base_cfg.x_max, base_cfg.nx)
    row_labels = [f"$r_m={rm}$" for rm in r_ms]
    col_labels = [rf"$\sigma={s}$" for s in sigmas]
    col_labels[0] = r"Homogeneous ($\sigma=0$)"

    for r_m in r_ms:
        intensity_row = []
        results_fields[r_m] = {}

        print(f"Processing r_m = {r_m}")

        for sigma in sigmas:
            cfg = dataclasses.replace(base_cfg, r_m=r_m, sigma=sigma)
            n_mc = 1 if sigma == 0.0 else base_cfg.n_monte_carlo

            mc_result = run_monte_carlo_simulation(
                cfg, n_simulations=n_mc, compute_intensity_map=True, verbose=False
            )

            # Store Intensity Map
            intensity_row.append(mc_result.mean_intensity_map)

            # Store Mean Field
            results_fields[r_m][sigma] = mc_result.mean_field

        intensity_grid_data.append(intensity_row)

    plot_intensity_grid(
        intensity_grid_data,
        row_labels=row_labels,
        col_labels=col_labels,
        extent=[0, 2 * base_cfg.L, base_cfg.x_min, base_cfg.x_max],
        xlabel="z",
        ylabel="x",
    )

    target_rm = 2.0
    if target_rm in r_ms:
        data_to_plot = []

        # Homogeneous
        homo_field = results_fields[target_rm][0.0]
        data_to_plot.append((homo_field, r"Homogeneous ($\sigma=0$)", "k--"))

        colors = ["b-", "g-", "r-", "m-"]
        idx = 0
        for sigma in sigmas:
            if sigma == 0.0:
                continue

            field = results_fields[target_rm][sigma]
            style = colors[idx % len(colors)]
            data_to_plot.append((field, rf"Random ($\sigma={sigma}$)", style))
            idx += 1

        plot_multiple_comparisons(
            x,
            data_to_plot,
            title=f"Refocusing Quality at $z=2L$ ($r_m={target_rm}$)",
            ylabel=r"$|\mathbb{E}[\phi(x)]|^2$ (Coherent Intensity)",
        )


if __name__ == "__main__":
    run_comparison()
