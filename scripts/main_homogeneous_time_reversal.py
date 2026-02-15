import numpy as np
from scipy.fft import fftfreq

from time_reversal.config import SimulationConfig
from time_reversal.propagation_fun import (
    gaussian_mirror, homogeneous_propagation_ode,
    homogeneous_time_reversal_analytic_solution, init_homogeneous)
from time_reversal.solver import PropagatorFourier, RungeKuttaSolver
from time_reversal.viz import plot_comparison, setup_style


def main():
    setup_style()  # matplotlib style for better visualization

    ## First forward propagation
    cfg = SimulationConfig(c0=1.0, w=1.0, L=10.0, r0=2.0, x_size=60.0, nx=2**10)
    x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)

    # numerical solution
    solver = PropagatorFourier(homogeneous_propagation_ode, solver=RungeKuttaSolver())
    phi0 = init_homogeneous(x=x, r0=cfg.r0)
    kappa_vec = (
        fftfreq(len(phi0), d=cfg.dx) * 2 * np.pi
    )  # because fft convention is different from the Fourier Transform convention I used in the derivation of homogeneous forward function

    phi = solver.forward(
        phi0=phi0, z_min=0.0, z_max=cfg.L, k_const=cfg.k_const, kappa_vec=kappa_vec
    )

    ## Time reversal
    r_m = 20.0  # mirror radius
    # theory solution
    phi_theory = homogeneous_time_reversal_analytic_solution(
        x, cfg.L, cfg.r0, cfg.k_const, r_m
    )

    # mirror = compactly_supported_mirror(x, r_m)
    mirror = gaussian_mirror(x, r_m)
    phi_mirror = phi.conj() * mirror  # apply mirror to the final
    phi_time_reversal = solver.forward(
        phi0=phi_mirror,
        z_min=cfg.L,
        z_max=2 * cfg.L,
        k_const=cfg.k_const,
        kappa_vec=kappa_vec,
    )

    plot_comparison(
        x,
        phi_time_reversal,
        phi0,
        title="Time Reversal at $z=2L$ (Homogeneous)",
    )

    plot_comparison(
        x,
        phi_time_reversal,
        phi_theory,
        title="Time Reversal at $z=2L$ (Theory vs Simulation, Homogeneous)",
    )


if __name__ == "__main__":
    main()
