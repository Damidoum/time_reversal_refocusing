import numpy as np

from time_reversal.config import SimulationConfig
from time_reversal.homogeneous_forward import (
    homogeneous_analytic_solution,
    homogeneous_forward,
    init_homogeneous_forward,
)
from time_reversal.homogeneous_time_reversal_forward import (
    compactly_supported_mirror,
    gaussian_mirror,
    homogeneous_time_reversal_analytic_solution,
)
from time_reversal.solver import Propagator
from time_reversal.viz import plot_comparison, setup_style


def main():
    setup_style()  # matplotlib style for better visualization

    ## First forward propagation
    cfg = SimulationConfig(c0=1.0, w=1.0, L=10.0, r0=2.0, x_size=60.0, nx=2**10)
    x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)

    # numerical solution
    solver = Propagator(homogeneous_forward)
    phi0 = init_homogeneous_forward(x=x, r0=cfg.r0)
    phi = solver.forward(
        phi0=phi0, k_const=cfg.k_const, z_min=0.0, z_max=cfg.L, dx=cfg.dx
    )

    ## Time reversal
    r_m = 2.0  # mirror radius
    # theory solution
    phi_theory = homogeneous_time_reversal_analytic_solution(
        x, cfg.L, cfg.r0, cfg.k_const, r_m
    )

    # mirror = compactly_supported_mirror(x, r_m)
    mirror = gaussian_mirror(x, r_m)
    phi_mirror = phi[:, -1].conj() * mirror  # apply mirror to the final
    phi_time_reversal = solver.forward(
        phi0=phi_mirror, k_const=cfg.k_const, z_min=cfg.L, z_max=2 * cfg.L, dx=cfg.dx
    )

    plot_comparison(
        x,
        phi_time_reversal[:, -1],
        phi[:, 0],
        title=f"Time Reversal at $z=2L$ (Homogeneous)",
    )

    plot_comparison(
        x,
        phi_time_reversal[:, -1],
        phi_theory,
        title=f"Time Reversal at $z=2L$ (Theory vs Simulation, Homogeneous)",
    )


if __name__ == "__main__":
    main()
