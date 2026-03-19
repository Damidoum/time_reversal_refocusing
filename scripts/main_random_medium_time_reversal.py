import pathlib

import numpy as np

from time_reversal.config import SimulationConfig
from time_reversal.propagation_fun import (
    init_homogeneous,
    mean_field_random_medium_refocused,
)
from time_reversal.simulation import run_monte_carlo_simulation
from time_reversal.viz import (
    plot_intensity_map,
    plot_intensity_section,
    plot_multiple_intensity_section,
    setup_style,
)


def monte_carlo_random_medium_time_reversal(
    cfg: SimulationConfig,
    x: np.ndarray,
    phi_init_ref: np.ndarray,
    save_path: pathlib.Path,
):

    # Run Monte Carlo Simulation
    mc_result = run_monte_carlo_simulation(
        cfg,
        n_simulations=cfg.n_monte_carlo,
        compute_intensity_map=True,
        verbose=True,
    )

    if save_path and not save_path.exists():
        save_path.mkdir(parents=True)

    plot_intensity_map(
        mc_result.mean_intensity_map,
        extent=[0, 2 * cfg.L, cfg.x_min, cfg.x_max],
        title=f"Average Intensity Propagation (N={cfg.n_monte_carlo})\n(Forward + Time Reversal)",
        xlabel="Propagation distance z",
        ylabel="Transverse coordinate x",
        save_path=save_path / "intensity_map_time_reversal.pdf",
    )

    plot_intensity_section(
        x,
        mc_result.mean_field,
        phi_init_ref,
        title=f"Mean Refocused Intensity at z=2L vs Initial Source\n(Averaged over {cfg.n_monte_carlo} realizations)",
        xlabel=r"$x$ (Transverse position)",
        ylabel=r"$|\phi(x)|^2$ (Amplitude)",
        label_curve1="Mean Refocused",
        label_curve2="Initial Source",
        save_path=save_path / "intensity_section_time_reversal_initial.pdf",
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

    plot_intensity_section(
        x,
        mc_result.mean_field,
        mean_theory,
        title="Refocused Intensity at z=2L vs Theoretical Prediction",
        xlabel=r"$x$ (Transverse position)",
        ylabel=r"$|\phi(x)|^2$ (Amplitude)",
        label_curve1="Mean Refocused",
        label_curve2="Theoretical Prediction",
        save_path=save_path / "intensity_section_time_reversal_theory.pdf",
    )

    return mc_result.mean_field, mean_theory


def main():
    setup_style()

    cfg = SimulationConfig.from_cli()
    print(f"Running Random Medium Time Reversal with: {cfg}")

    x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)
    phi_init_ref = init_homogeneous(x, cfg.r0)

    save_base_path = pathlib.Path(
        f"output/random_medium_time_reversal/{cfg.mirror_type}/"
    )
    if not save_base_path.exists():
        save_base_path.mkdir(parents=True)

    for rm in [2.0, 5.0, 10.0, 20.0]:
        cfg.r_m = rm
        data_plot = []
        save_path = save_base_path / f"rm_{rm:.2f}"
        if not save_path.exists():
            save_path.mkdir(parents=True)

        for sigma in [0.1, 0.2, 0.3, 0.5, 1.0, 2.0]:
            cfg.sigma = sigma
            # Run Monte Carlo Simulation for Time Reversal in Random Medium
            mean_field, _ = monte_carlo_random_medium_time_reversal(
                cfg,
                x,
                phi_init_ref,
                save_path=save_path / f"sigma_{sigma:.2f}",
            )
            data_plot.append((mean_field, rf"Numerical ($\sigma={sigma:.2f}$)", None))

        plot_multiple_intensity_section(
            x,
            data_list=data_plot,
            title=f"Comparison of Refocused Intensity at $z=2L$ for $r_m={rm}$\n(Averaged over {cfg.n_monte_carlo} realizations)",
            xlabel=r"$x$ (Transverse position)",
            ylabel=r"$|\phi(x)|^2$ (Amplitude)",
            save_path=save_path / "intensity_comparison.pdf",
            show=False,
        )


if __name__ == "__main__":
    main()
