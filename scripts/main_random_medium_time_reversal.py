import numpy as np

from time_reversal.config import SimulationConfig
from time_reversal.propagation_fun import (
    init_homogeneous,
    mean_field_random_medium_refocused,
)
from time_reversal.simulation import run_monte_carlo_simulation
from time_reversal.viz import plot_comparison, plot_intensity_map, setup_style


def main():
    setup_style()

    cfg = SimulationConfig.from_cli()
    print(f"Running Random Medium Time Reversal with: {cfg}")

    x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)
    phi_init_ref = init_homogeneous(x, cfg.r0)

    # Run Monte Carlo Simulation
    mc_result = run_monte_carlo_simulation(
        cfg,
        n_simulations=cfg.n_monte_carlo,
        compute_intensity_map=True,
        verbose=True,
    )

    if mc_result.single_realization_history is not None:
        intensity_map = np.abs(mc_result.single_realization_history.T) ** 2
        plot_intensity_map(
            intensity_map,
            extent=[0, 2 * cfg.L, cfg.x_min, cfg.x_max],
            title=f"Single Realization Propagation (Forward + Time Reversal)\nz_mirror={cfg.L}, z_focus={2 * cfg.L}",
            xlabel="Propagation distance z",
        )

    if mc_result.mean_intensity_map is not None:
        plot_intensity_map(
            mc_result.mean_intensity_map,
            extent=[0, 2 * cfg.L, cfg.x_min, cfg.x_max],
            title=f"Average Intensity Propagation (N={cfg.n_monte_carlo})\n(Forward + Time Reversal)",
            xlabel="Propagation distance z",
        )

    plot_comparison(
        x,
        mc_result.mean_field,
        phi_init_ref,
        title=f"Mean Refocused Intensity at z=2L vs Initial Source\n(Averaged over {cfg.n_monte_carlo} realizations)",
    )

    mean_theory = mean_field_random_medium_refocused(
        x,
        r0=cfg.r0,
        k_const=cfg.k_const,
        c0=cfg.c0,
        L=cfg.L,
        sigma=cfg.sigma,
        z_c=cfg.z_c,
        r_m=cfg.r_m,
        x_c=cfg.x_c,
    )
    plot_comparison(
        x,
        mc_result.mean_field,
        mean_theory,
        title="Refocused Intensity at z=2L vs Initial Source",
    )


if __name__ == "__main__":
    main()
