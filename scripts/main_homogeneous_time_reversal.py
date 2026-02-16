import dataclasses

import numpy as np

from time_reversal.config import SimulationConfig
from time_reversal.field import WaveField
from time_reversal.operators import DiffractionOperator
from time_reversal.propagation_fun import (
    gaussian_mirror, homogeneous_time_reversal_analytic_solution,
    init_homogeneous)
from time_reversal.propagator import SplitStepPropagator
from time_reversal.solver import AnalyticSolver
from time_reversal.viz import plot_comparison, setup_style


def main():
    setup_style()  # matplotlib style for better visualization

    # Configuration
    cfg = SimulationConfig(c0=1.0, w=1.0, L=10.0, r0=2.0, x_size=60.0, nx=2**10)
    x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)

    # init Field
    phi0 = init_homogeneous(x=x, r0=cfg.r0)
    field = WaveField(x=x, phi=phi0, k_const=cfg.k_const)

    # setup Propagator
    propagator = SplitStepPropagator(steps=[(DiffractionOperator(), AnalyticSolver())])

    # forward Propagation
    field = propagator.step(field, dz=cfg.L)

    field = field.to_real()
    phi_at_L = field.phi

    r_m = 20.0  # mirror radius
    mirror = gaussian_mirror(x, r_m)

    # Apply phase conjugation and mirror
    phi_mirror = phi_at_L.conj() * mirror
    field = dataclasses.replace(field, phi=phi_mirror)

    field = propagator.step(field, dz=cfg.L)

    phi_time_reversal = field.to_real().phi

    plot_comparison(
        x,
        phi_time_reversal,
        phi0,
        title="Time Reversal at $z=2L$ (Homogeneous) vs Initial",
    )

    # Compare with theoretical TR solution
    phi_theory = homogeneous_time_reversal_analytic_solution(
        x, cfg.L, cfg.r0, cfg.k_const, r_m
    )

    plot_comparison(
        x,
        phi_time_reversal,
        phi_theory,
        title="Time Reversal at $z=2L$ (Theory vs Simulation)",
    )


if __name__ == "__main__":
    main()
