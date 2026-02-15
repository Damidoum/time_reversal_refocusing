import matplotlib.pyplot as plt
import numpy as np
import tqdm
from scipy.fft import fftfreq

from time_reversal.config import SimulationConfig
from time_reversal.propagation_fun import (covariance_function,
                                           diffraction_propagation_ode,
                                           diffraction_propagation_operator,
                                           init_homogeneous, mean_intensity,
                                           refraction_propagation_ode,
                                           refraction_propagation_operator)
from time_reversal.random_process import StationaryGaussianProcess
from time_reversal.solver import (AnalyticSolver, Propagator,
                                  PropagatorFourier, RungeKuttaSolver)
from time_reversal.viz import setup_style


def solve(cfg, h, z_c, x_c, sigma, use_fast_solver=True):
    """
    Solves the wave propagation in a random medium using the split-step Fourier method.

    Args:
        h (float): Step size for the propagation (sub-step).
        z_c (float): Correlation length in z (size of a random slab).
    """

    if use_fast_solver:
        solver_strategy = AnalyticSolver()
        diffracation_fun = diffraction_propagation_operator
        refraction_fun = refraction_propagation_operator
    else:
        solver_strategy = RungeKuttaSolver()
        diffracation_fun = diffraction_propagation_ode
        refraction_fun = refraction_propagation_ode

    # Initialize propagators with the chosen strategy
    diff_prop = PropagatorFourier(diffracation_fun, solver=solver_strategy)
    refra_prop = Propagator(refraction_fun, solver=solver_strategy)

    n_layers = int(cfg.L / z_c)  # Number of independent random slabs
    n_steps_per_layer = int(z_c / h)  # Number of numerical steps per slab

    x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)

    gaussian_process = StationaryGaussianProcess(
        mean=0.0,
        covariance_function=lambda x: covariance_function(x, x_c, sigma),
    )

    gaussian_samples = gaussian_process.sample(x, n_samples=n_layers)

    # field init
    phi = init_homogeneous(x=x, r0=cfg.r0)
    kappa_vec = fftfreq(len(phi), d=cfg.dx) * 2 * np.pi

    phis = []
    for i in range(n_layers):
        mu_layer = gaussian_samples[i]

        for j in range(n_steps_per_layer):
            z_current = (i * n_steps_per_layer + j) * h
            z_next = z_current + h

            # diffraction step
            phi = diff_prop.forward(
                phi0=phi,
                z_min=z_current,
                z_max=z_next,
                kappa_vec=kappa_vec,
                k_const=cfg.k_const,
            )

            # refraction step
            phi = refra_prop.forward(
                phi0=phi,
                z_min=z_current,
                z_max=z_next,
                mu=mu_layer,
                k_const=cfg.k_const,
            )

            phis.append(phi)

    return phis


def main():
    setup_style()

    # Simulation Configuration
    cfg = SimulationConfig(c0=1.0, w=1.0, L=10.0, r0=2.0, x_size=60.0, nx=2**10)
    x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)

    # Parameters
    USE_FAST_SOLVER = True
    N_MONTE_CARLO = 1000

    h = 0.1
    z_c = 1.0
    x_c = 4.0
    sigma = 1.0

    final_fields = []
    last_run_history = []

    # Monte Carlo Loop
    print(f"Running {N_MONTE_CARLO} simulations with h={h}...")
    for _ in tqdm.tqdm(
        range(N_MONTE_CARLO), desc=f"Monte Carlo (Fast={USE_FAST_SOLVER})"
    ):
        last_run_history = solve(
            cfg, h, z_c, x_c, sigma, use_fast_solver=USE_FAST_SOLVER
        )
        # Store the field at z=L (last step)
        final_fields.append(last_run_history[-1])

    plt.figure(figsize=(10, 6))
    intensity_map = np.abs(np.array(last_run_history).T) ** 2

    plt.imshow(
        intensity_map,
        extent=[0, cfg.L, cfg.x_min, cfg.x_max],
        aspect="auto",
        origin="lower",
        cmap="inferno",
    )
    plt.colorbar(label=r"$|\phi|^2$")
    plt.xlabel("Propagation distance z")
    plt.ylabel("Transverse coordinate x")
    plt.title(
        f"Single Realization Intensity (z_c={z_c}, h={h})\nSolver: {'Analytic' if USE_FAST_SOLVER else 'RK45'}"
    )
    plt.show()

    mean_field_num = np.mean(final_fields, axis=0)

    mean_field_theo = mean_intensity(
        x=x,
        r0=cfg.r0,
        k_const=cfg.k_const,
        L=cfg.L,
        sigma=sigma,
        z_c=z_c,
        c0=cfg.c0,
    )

    plt.figure(figsize=(10, 6))

    # Plot modulus of the Mean Field |E[phi]|
    plt.plot(
        x,
        np.abs(mean_field_num),
        "b-",
        linewidth=2,
        label="Numerical Mean Field",
    )
    plt.plot(
        x,
        np.abs(mean_field_theo),
        "r--",
        linewidth=2,
        label="Theoretical Mean Field",
    )

    plt.xlabel("Transverse coordinate x")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.title(
        f"Comparison of Mean Field Profile at z=L={cfg.L}\n(Averaged over {N_MONTE_CARLO} realizations)"
    )
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()
