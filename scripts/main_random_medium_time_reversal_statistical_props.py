import pathlib

import numpy as np

from time_reversal.broadband import simulate_frequencies
from time_reversal.config import SimulationConfig
from time_reversal.propagation_fun import compute_theoretical_broadband_refocused
from time_reversal.viz import plot_multiple_intensity_section, setup_style


def run_experiment(
    cfg_base: SimulationConfig,
    omegas: np.ndarray,
    n_realizations: int,
    mixed: bool,
    experiment_name: str,
):
    """
    Runs an experiment (Matched or Mismatched) and saves the plot.
    """
    x = np.linspace(cfg_base.x_min, cfg_base.x_max, cfg_base.nx)

    theory_field = compute_theoretical_broadband_refocused(
        x,
        cfg_base.r0,
        c0=cfg_base.c0,
        L=cfg_base.L,
        sigma=cfg_base.sigma,
        z_c=cfg_base.z_c,
        r_m=cfg_base.r_m,
        x_c=cfg_base.x_c,
        omegas=omegas,
        mixed=mixed,
    )

    data_plot = [(theory_field, "Theory", "k-")]

    print(f"  Running {experiment_name} (mixed={mixed})...")
    for i in range(n_realizations):
        seed = 42 + i
        complex_fields = simulate_frequencies(
            cfg_base, omegas, seed=seed, return_envelope=True, mixed=mixed
        )

        refocused_field = np.sum(complex_fields[:, -1, :], axis=0)

        data_plot.append((refocused_field, f"Realization {i + 1}", "--"))

    suffix = "mismatched" if mixed else "matched"
    save_path = pathlib.Path(
        f"output/broadband/{cfg_base.mirror_type}/rm_{cfg_base.r_m:.2f}/sigma_{cfg_base.sigma:.2f}/statistical_stability_{suffix}.pdf"
    )
    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)

    plot_multiple_intensity_section(
        x,
        data_list=data_plot,  # type: ignore
        title=f"Statistical Stability ({experiment_name})\n(L={cfg_base.L}, r_m={cfg_base.r_m}, sigma={cfg_base.sigma})",
        xlabel=r"$x$ (Transverse position)",
        ylabel=r"$|\Phi|^2$ (Refocused Intensity)",
        save_path=save_path,
        show=True,
    )


def main():
    setup_style()
    cfg = SimulationConfig.from_cli()

    # Parameters
    n_freq = 150
    delta_omega = 0.75
    sigmas = [0.0, 0.5, 1.0, 2.0, 5.0]

    cfg.t_max = 2 * cfg.L / cfg.c0
    omegas = np.linspace(cfg.w - delta_omega, cfg.w + delta_omega, n_freq)

    r_ms = [2.0, 5.0, 10.0, 20.0]

    for rm in r_ms:
        cfg.r_m = rm
        for sigma in sigmas:
            cfg.sigma = sigma
            n_realizations = 5

            print(
                f"\n--- Configuration: L={cfg.L}, r_m={cfg.r_m}, sigma={cfg.sigma} ---"
            )

            run_experiment(
                cfg, omegas, n_realizations, mixed=False, experiment_name="Matched"
            )

            if sigma > 0:
                run_experiment(
                    cfg,
                    omegas,
                    n_realizations,
                    mixed=True,
                    experiment_name="Mismatched",
                )


if __name__ == "__main__":
    main()
