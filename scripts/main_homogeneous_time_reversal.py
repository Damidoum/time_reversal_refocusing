import dataclasses

import numpy as np

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
from time_reversal.viz import plot_comparison, plot_intensity_map, setup_style


def main():
    setup_style()  # matplotlib style for better visualization

    # Configuration
    cfg = SimulationConfig.from_cli()
    print(f"Running Homogeneous Time Reversal with: {cfg}")

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

    plot_comparison(
        x,
        phi_time_reversal,
        phi0,
        title="Time Reversal at $z=2L$ (Homogeneous) vs Initial",
    )

    # Compare with theoretical TR solution
    phi_theory = homogeneous_time_reversal_analytic_solution(
        x, cfg.L, cfg.r0, cfg.k_const, cfg.r_m
    )

    plot_comparison(
        x,
        phi_time_reversal,
        phi_theory,
        title="Time Reversal at $z=2L$ (Theory vs Simulation)",
    )

    plot_intensity_map(
        intensity_map=np.abs(history).T ** 2,
        extent=[0, 2 * cfg.L, cfg.x_min, cfg.x_max],
        title="Intensity Map (Homogeneous medium)",
        xlabel="Propagation distance z",
        ylabel="Transverse coordinate x",
    )


if __name__ == "__main__":
    main()
