import dataclasses

import numpy as np
import tqdm

from time_reversal.config import SimulationConfig
from time_reversal.field import WaveField
from time_reversal.operators import DiffractionOperator, RefractionOperator
from time_reversal.propagation_fun import (
    covariance_function,
    gaussian_mirror,
    init_homogeneous,
)
from time_reversal.propagator import SplitStepPropagator
from time_reversal.random_process import StationaryGaussianProcess
from time_reversal.solver import AnalyticSolver, RungeKuttaSolver
from time_reversal.viz import plot_comparison, plot_intensity_map, setup_style


def solve_time_reversal(
    cfg: SimulationConfig,
    use_fast_solver=True,
):
    if use_fast_solver:
        solver_class = AnalyticSolver
    else:
        solver_class = RungeKuttaSolver

    solver = solver_class()
    propagator = SplitStepPropagator(
        steps=[(DiffractionOperator(), solver), (RefractionOperator(), solver)]
    )

    n_layers = int(cfg.L / cfg.z_c)
    n_steps_per_layer = int(cfg.z_c / cfg.h)

    x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)
    phi0 = init_homogeneous(x=x, r0=cfg.r0)
    field = WaveField(x=x, phi=phi0, k_const=cfg.k_const)

    # generate medium
    gaussian_process = StationaryGaussianProcess(
        mean=0.0,
        covariance_function=lambda x: covariance_function(x, cfg.x_c, cfg.sigma),  # type: ignore
    )
    gaussian_samples = gaussian_process.sample(x, n_samples=n_layers)
    if n_layers == 1:
        gaussian_samples = gaussian_samples.reshape(1, -1)

    history = []

    history.append(field.to_real().phi)

    # forward propagation
    for i in range(n_layers):
        mu_layer = gaussian_samples[i]
        for _ in range(n_steps_per_layer):
            field = propagator.step(field, dz=cfg.h, mu=mu_layer)
            history.append(field.to_real().phi)

    # time reversal init
    field = field.to_real()
    phi_at_L = field.phi
    mirror = gaussian_mirror(x, cfg.r_m)
    phi_tr = phi_at_L.conj() * mirror
    field = dataclasses.replace(field, phi=phi_tr)

    # time reversal backward propagation
    for i in range(n_layers - 1, -1, -1):
        mu_layer = gaussian_samples[i]
        for _ in range(n_steps_per_layer):
            field = propagator.step(field, dz=cfg.h, mu=mu_layer)
            history.append(field.to_real().phi)

    return field.to_real().phi, phi0, history


def main():
    setup_style()

    cfg = SimulationConfig.from_cli()
    print(f"Running Random Medium Time Reversal with: {cfg}")

    x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)

    USE_FAST_SOLVER = True

    final_fields = []

    print(f"Running {cfg.n_monte_carlo} TR simulations...")
    iterator = tqdm.tqdm(range(cfg.n_monte_carlo))

    phi_init_ref = None
    last_history = None

    all_histories = []

    for _ in iterator:
        result = solve_time_reversal(cfg, use_fast_solver=USE_FAST_SOLVER)

        phi_final, phi_init, hist = result  # type: ignore

        final_fields.append(phi_final)
        all_histories.append(hist)

        if phi_init_ref is None:
            phi_init_ref = phi_init

    # plot the mean intensity of the refocused wave
    mean_intensity_tr = np.mean(np.array(final_fields), axis=0)

    # plot last realization
    last_history = all_histories[-1]
    intensity_map = np.abs(np.array(last_history).T) ** 2
    plot_intensity_map(
        intensity_map,
        extent=[0, 2 * cfg.L, cfg.x_min, cfg.x_max],
        title=f"Single Realization Propagation (Forward + Time Reversal)\nz_mirror={cfg.L}, z_focus={2 * cfg.L}",
        xlabel="Propagation distance z",
    )

    # plot average propagation intensity
    mean_intensity_map = np.mean(np.abs(np.array(all_histories)) ** 2, axis=0).T
    plot_intensity_map(
        mean_intensity_map,
        extent=[0, 2 * cfg.L, cfg.x_min, cfg.x_max],
        title=f"Average Intensity Propagation (N={cfg.n_monte_carlo})\n(Forward + Time Reversal)",
        xlabel="Propagation distance z",
    )

    plot_comparison(
        x,
        mean_intensity_tr,
        phi_init_ref,
        title=f"Mean Refocused Intensity at z=2L vs Initial Source\n(Averaged over {cfg.n_monte_carlo} realizations)",
    )


if __name__ == "__main__":
    main()
