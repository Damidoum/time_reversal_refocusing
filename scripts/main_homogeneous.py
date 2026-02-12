import numpy as np

from time_reversal.config import SimulationConfig
from time_reversal.homogeneous_forward import (
    homogeneous_analytic_solution,
    homogeneous_forward,
    init_homogeneous_forward,
)
from time_reversal.solver import Propagator
from time_reversal.viz import plot_comparison, setup_style


def main():
    setup_style()

    cfg = SimulationConfig(c0=1.0, w=1.0, L=10.0, r0=2.0, x_size=60.0, nx=2**10)

    # theory solution
    x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)
    phi_theory = homogeneous_analytic_solution(x, cfg.L, cfg.r0, cfg.k_const)

    # numerical solution
    solver = Propagator(homogeneous_forward)
    phi0 = init_homogeneous_forward(x=x, r0=cfg.r0)
    phi = solver.forward(
        phi0=phi0, k_const=cfg.k_const, z_min=0.0, z_max=cfg.L, dx=cfg.dx
    )

    plot_comparison(
        x, phi[:, -1], phi_theory, title=f"Beam Profile at $z={cfg.L}$ (Homogeneous)"
    )


if __name__ == "__main__":
    main()
