import numpy as np
import tqdm

from time_reversal.config import SimulationConfig
from time_reversal.field import WaveField
from time_reversal.operators import DiffractionOperator, RefractionOperator
from time_reversal.propagation_fun import (
    covariance_function,
    init_homogeneous,
    mean_intensity,
)
from time_reversal.propagator import SplitStepPropagator
from time_reversal.random_process import StationaryGaussianProcess
from time_reversal.solver import AnalyticSolver, RungeKuttaSolver
from time_reversal.viz import (
    plot_intensity_map,
    plot_mean_field_comparison,
    setup_style,
)


def solve(cfg: SimulationConfig, use_fast_solver=True):
    """Solve the wave propagation in a random medium."""

    # picking the solver
    if use_fast_solver:
        solver_class = AnalyticSolver
    else:
        solver_class = RungeKuttaSolver

    solver = solver_class()

    # setting up propagator with diffraction and refraction steps
    propagator = SplitStepPropagator(
        steps=[(DiffractionOperator(), solver), (RefractionOperator(), solver)]
    )

    n_layers = int(cfg.L / cfg.z_c)
    n_steps_per_layer = int(cfg.z_c / cfg.h)

    # init field
    x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)
    phi0 = init_homogeneous(x=x, r0=cfg.r0)
    field = WaveField(x=x, phi=phi0, k_const=cfg.k_const)

    # generating random medium layers
    gaussian_process = StationaryGaussianProcess(
        mean=0.0,
        covariance_function=lambda x: covariance_function(x, cfg.x_c, cfg.sigma),  # type: ignore
    )
    # samples for each layer
    gaussian_samples = gaussian_process.sample(x, n_samples=n_layers)
    if n_layers == 1:
        gaussian_samples = gaussian_samples.reshape(1, -1)

    history = []

    # propagation loop
    for i in range(n_layers):
        mu_layer = gaussian_samples[i]

        for _ in range(n_steps_per_layer):
            # propagate by one step h
            field = propagator.step(field, dz=cfg.h, mu=mu_layer)

            # store history
            history.append(field.to_real().phi)

    return history


def main():
    setup_style()

    # sim config
    cfg = SimulationConfig.from_cli()
    print(f"Running Random Medium Simulation with: {cfg}")

    x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)

    # params
    USE_FAST_SOLVER = True
    final_fields = []
    # init history list

    print(f"Running {cfg.n_monte_carlo} simulations with h={cfg.h}...")

    # monte carlo iterator
    iterator = tqdm.tqdm(
        range(cfg.n_monte_carlo), desc=f"Monte Carlo (Fast={USE_FAST_SOLVER})"
    )

    for _ in iterator:
        last_run_history = solve(cfg, use_fast_solver=USE_FAST_SOLVER)
        final_fields.append(last_run_history[-1])

    intensity_map = np.abs(np.array(last_run_history).T) ** 2
    plot_intensity_map(
        intensity_map,
        extent=[0, cfg.L, cfg.x_min, cfg.x_max],
        title=f"Single Realization Intensity (z_c={cfg.z_c}, h={cfg.h})\nSolver: {'Analytic' if USE_FAST_SOLVER else 'RK45'}",
    )

    mean_field_num = np.mean(final_fields, axis=0)

    mean_field_theo = mean_intensity(
        x=x,
        r0=cfg.r0,
        k_const=cfg.k_const,
        L=cfg.L,
        sigma=cfg.sigma,
        z_c=cfg.z_c,
        c0=cfg.c0,
    )

    plot_mean_field_comparison(
        x,
        mean_field_num,
        mean_field_theo,
        title=f"Comparison of Mean Field Profile at z=L={cfg.L}\n(Averaged over {cfg.n_monte_carlo} realizations)",
    )


if __name__ == "__main__":
    main()
