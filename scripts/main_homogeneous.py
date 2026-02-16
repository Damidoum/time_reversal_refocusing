import numpy as np

from time_reversal.config import SimulationConfig
from time_reversal.field import WaveField
from time_reversal.operators import DiffractionOperator
from time_reversal.propagation_fun import (homogeneous_analytic_solution,
                                           init_homogeneous)
from time_reversal.propagator import SplitStepPropagator
from time_reversal.solver import AnalyticSolver
from time_reversal.viz import plot_comparison, setup_style


def main():
    setup_style()

    cfg = SimulationConfig(c0=1.0, w=1.0, L=10.0, r0=2.0, x_size=60.0, nx=2**10)

    # theory solution
    x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)
    phi_theory = homogeneous_analytic_solution(x, cfg.L, cfg.r0, cfg.k_const)

    # numerical solution
    phi0 = init_homogeneous(x=x, r0=cfg.r0)
    field = WaveField(x=x, phi=phi0, k_const=cfg.k_const)

    # Using AnalyticSolver for efficiency, but could be RungeKuttaSolver too
    propagator = SplitStepPropagator(steps=[(DiffractionOperator(), AnalyticSolver())])

    # Propagate
    final_field = propagator.step(field, dz=cfg.L)

    # Convert to real space for comparison
    final_field = final_field.to_real()
    phi = final_field.phi

    plot_comparison(
        x, phi, phi_theory, title=f"Beam Profile at $z={cfg.L}$ (Homogeneous)"
    )


if __name__ == "__main__":
    main()
