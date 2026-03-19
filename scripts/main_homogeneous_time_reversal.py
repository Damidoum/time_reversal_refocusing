import dataclasses
import pathlib

import numpy as np
import tqdm

from time_reversal.config import SimulationConfig
from time_reversal.field import WaveField
from time_reversal.operators import DiffractionOperator
from time_reversal.propagation_fun import (
    compact_mirror,
    gaussian_mirror,
    homogeneous_time_reversal_analytic_solution,
    init_homogeneous,
)
from time_reversal.propagator import SplitStepPropagator
from time_reversal.solver import AnalyticSolver
from time_reversal.viz import (
    plot_intensity_map,
    plot_intensity_section,
    plot_multiple_intensity_section,
    setup_style,
)


def time_reversal_propagation(cfg: SimulationConfig):
    x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)

    # init Field
    phi0 = init_homogeneous(x=x, r0=cfg.r0)
    field = WaveField(x=x, phi=phi0, k_const=cfg.k_const)
    history = [field.to_real().phi]

    # setup Propagator
    propagator = SplitStepPropagator(steps=[(DiffractionOperator(), AnalyticSolver())])

    # forward Propagation
    for _ in range(int(cfg.L / cfg.h)):
        field = propagator.step(field, dz=cfg.h)
        history.append(field.to_real().phi)

    field = field.to_real()
    phi_at_L = field.phi

    if cfg.mirror_type == "gaussian":
        mirror = gaussian_mirror(x, cfg.r_m)
    elif cfg.mirror_type == "compact":
        mirror = compact_mirror(x, cfg.r_m)
    else:
        raise ValueError(f"Unknown mirror type: {cfg.mirror_type}")

    # Apply phase conjugation and mirror
    phi_mirror = phi_at_L.conj() * mirror
    field = dataclasses.replace(field, phi=phi_mirror)

    # backward propagation
    for _ in range(int(cfg.L / cfg.h)):
        field = propagator.step(field, dz=cfg.h)
        history.append(field.to_real().phi)

    phi_time_reversal = field.to_real().phi

    # theoritical solution
    phi_theory = homogeneous_time_reversal_analytic_solution(
        x, cfg.L, cfg.r0, cfg.k_const, cfg.r_m
    )

    return x, phi0, phi_time_reversal, phi_theory, history


def main():
    setup_style()  # matplotlib style for better visualization

    # Configuration
    cfg = SimulationConfig.from_cli()
    print(f"Running Homogeneous Time Reversal with: {cfg}")

    rm_list = [2.0, 5.0, 10.0, 20.0]  # Mirror widths to compare
    data = []

    for rm in tqdm.tqdm(rm_list):
        cfg.r_m = rm
        x, phi0, phi_time_reversal, phi_theory, history = time_reversal_propagation(cfg)
        data.append([phi_time_reversal, rf"Time Reversal ($r_m = {rm}$)", None])

        save_path = f"output/homogeneous_time_reversal/{cfg.mirror_type}/{cfg.r_m:.2f}/"
        if not pathlib.Path(save_path).exists():
            pathlib.Path(save_path).mkdir(parents=True)

        plot_intensity_section(
            x,
            phi_time_reversal,
            phi0,
            title="Intensity after time reversal ($z=2L$) vs Initial intensity ($z=0$)",
            xlabel=r"$x$ (Transverse position)",
            ylabel=r"$|\phi(x)|^2$ (Amplitude)",
            label_curve1="Time Reversal",
            label_curve2="Initial",
            save_path=pathlib.Path(save_path)
            / "intensity_section_time_reversal_initial.pdf",
        )

        plot_intensity_section(
            x,
            phi_time_reversal,
            phi_theory,
            title="Simulated intensity after time reversal ($z=2L$) vs Theoretical intensity",
            xlabel=r"$x$ (Transverse position)",
            ylabel=r"$|\phi(x)|^2$ (Amplitude)",
            save_path=pathlib.Path(save_path) / "intensity_section_time_reversal.pdf",
        )

        plot_intensity_map(
            intensity_map=np.abs(history).T ** 2,
            extent=[0, 2 * cfg.L, cfg.x_min, cfg.x_max],
            title=r"Intensity Map ($z$ from $0$ to $2L$, with time reversal at $z=L$)",
            xlabel="Propagation distance z",
            ylabel="Transverse coordinate x",
            save_path=pathlib.Path(save_path) / "intensity_map_time_reversal.pdf",
        )

    plot_multiple_intensity_section(
        x,
        data_list=data,
        title="Comparison of Time Reversal Intensity Profiles for Different Mirror Widths",
        xlabel=r"$x$ (Transverse position)",
        ylabel=r"$|\phi(x)|^2$ (Amplitude)",
        save_path=f"output/homogeneous_time_reversal/{cfg.mirror_type}/comparison_time_reversal.pdf",
        show=True,
    )


if __name__ == "__main__":
    main()
