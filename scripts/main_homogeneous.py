import numpy as np
from scipy.fft import fftfreq

from time_reversal.config import SimulationConfig
from time_reversal.propagation_fun import (homogeneous_analytic_solution,
                                           homogeneous_propagation_ode,
                                           init_homogeneous)
from time_reversal.solver import PropagatorFourier, RungeKuttaSolver
from time_reversal.viz import plot_comparison, setup_style


def main():
    setup_style()

    cfg = SimulationConfig(c0=1.0, w=1.0, L=10.0, r0=2.0, x_size=60.0, nx=2**10)

    # theory solution
    x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)
    phi_theory = homogeneous_analytic_solution(x, cfg.L, cfg.r0, cfg.k_const)

    # numerical solution
    solver = PropagatorFourier(homogeneous_propagation_ode, solver=RungeKuttaSolver())
    phi0 = init_homogeneous(x=x, r0=cfg.r0)
    kappa_vec = (
        fftfreq(len(phi0), d=cfg.dx) * 2 * np.pi
    )  # because fft convention is different from the Fourier Transform convention I used in the derivation of homogeneous forward function
    phi = solver.forward(
        phi0=phi0, z_min=0.0, z_max=cfg.L, kappa_vec=kappa_vec, k_const=cfg.k_const
    )

    plot_comparison(
        x, phi, phi_theory, title=f"Beam Profile at $z={cfg.L}$ (Homogeneous)"
    )


if __name__ == "__main__":
    main()
