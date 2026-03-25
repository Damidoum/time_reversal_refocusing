import dataclasses
from typing import List, Optional, Tuple

import numpy as np
import tqdm

from time_reversal.config import SimulationConfig
from time_reversal.field import WaveField
from time_reversal.operators import (
    DiffractionOperator,
    LinearOperator,
    RefractionOperator,
)
from time_reversal.propagation_fun import (
    compact_mirror,
    covariance_function,
    gaussian_mirror,
    init_homogeneous,
)
from time_reversal.propagator import SplitStepPropagator
from time_reversal.random_process import StationaryGaussianProcess
from time_reversal.solver import AnalyticSolver, RungeKuttaSolver, Solver


@dataclasses.dataclass
class SimulationResult:
    phi_final: np.ndarray
    phi_init: np.ndarray
    history: Optional[np.ndarray] = None


@dataclasses.dataclass
class MonteCarloResult:
    mean_field: np.ndarray
    mean_intensity: np.ndarray
    mean_intensity_map: Optional[np.ndarray] = (
        None  # mean(|phi(z,x)|^2) for full history
    )
    single_realization_history: Optional[np.ndarray] = (
        None  # One sample history for plotting
    )


def run_single_simulation(
    cfg: SimulationConfig,
    return_history: bool = False,
    use_fast_solver: bool = True,
    seed: Optional[int] = None,
    mixed: bool = False,
) -> SimulationResult:
    """
    Runs a single time reversal simulation (Forward -> Mirror -> Backward).
    Handles both homogeneous (sigma=0) and random media.
    """
    if seed is not None:
        np.random.seed(seed)

    solver_class = AnalyticSolver if use_fast_solver else RungeKuttaSolver
    solver = solver_class()

    is_homogeneous = cfg.sigma == 0.0

    steps: List[Tuple[LinearOperator, Solver]] = []
    steps.append((DiffractionOperator(), solver))
    if not is_homogeneous:
        steps.append((RefractionOperator(), solver))

    propagator = SplitStepPropagator(steps=steps)

    # setup initial field
    x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)
    phi0 = init_homogeneous(x=x, r0=cfg.r0)
    field = WaveField(x=x, phi=phi0, k_const=cfg.k_const)

    # setup random medium layers if needed
    n_layers = int(cfg.L / cfg.z_c)
    n_steps_per_layer = int(cfg.z_c / cfg.h)

    total_steps_one_way = int(cfg.L / cfg.h)

    gaussian_samples = None
    if not is_homogeneous:
        gaussian_process = StationaryGaussianProcess(
            mean=0.0,
            covariance_function=lambda x: covariance_function(x, cfg.x_c, cfg.sigma),  # type: ignore
        )
        gaussian_samples = gaussian_process.sample(x, n_samples=n_layers)
        if n_layers == 1:
            gaussian_samples = gaussian_samples.reshape(1, -1)

    history = []
    if return_history:
        history.append(field.to_real().phi)

    # forward propagation
    if is_homogeneous:
        for _ in range(total_steps_one_way):
            field = propagator.step(field, dz=cfg.h)
            if return_history:
                history.append(field.to_real().phi)
    else:
        # Layered propagation for random medium
        for i in range(n_layers):
            mu_layer = gaussian_samples[i]  # type: ignore
            for _ in range(n_steps_per_layer):
                field = propagator.step(field, dz=cfg.h, mu=mu_layer)
                if return_history:
                    history.append(field.to_real().phi)

    # time reversal
    field = field.to_real()
    phi_at_L = field.phi

    if cfg.mirror_type == "gaussian":
        mirror = gaussian_mirror(x, cfg.r_m)
    elif cfg.mirror_type == "gaussian_adaptive":
        idx_peak = np.argmax(np.abs(phi_at_L))
        x_peak = x[idx_peak]
        mirror = np.exp(-((x - x_peak) ** 2) / cfg.r_m**2)
    elif cfg.mirror_type == "compact":
        mirror = compact_mirror(x, cfg.r_m)
    else:
        raise ValueError(f"Unknown mirror type: {cfg.mirror_type}")

    phi_tr = phi_at_L.conj() * mirror
    field = dataclasses.replace(field, phi=phi_tr)

    # backward propagation
    if is_homogeneous or mixed:
        if mixed:
            propagator = SplitStepPropagator(
                steps=[(DiffractionOperator(), solver)]
            )  # remove refraction for backward pass in mixed case
        for _ in range(total_steps_one_way):
            field = propagator.step(field, dz=cfg.h)
            if return_history:
                history.append(field.to_real().phi)
    else:
        backward_steps = steps[::-1]
        propagator = SplitStepPropagator(steps=backward_steps)
        for i in range(n_layers - 1, -1, -1):
            mu_layer = gaussian_samples[i]  # type: ignore
            for _ in range(n_steps_per_layer):
                field = propagator.step(field, dz=cfg.h, mu=mu_layer)
                if return_history:
                    history.append(field.to_real().phi)

    return SimulationResult(
        phi_final=field.to_real().phi,
        phi_init=phi0,
        history=np.array(history) if return_history else None,
    )


def run_monte_carlo_simulation(
    cfg: SimulationConfig,
    n_simulations: Optional[int] = None,
    compute_intensity_map: bool = True,
    use_fast_solver: bool = True,
    verbose: bool = True,
    mixed: bool = False,
) -> MonteCarloResult:
    """
    Runs Monte Carlo simulations to compute mean field and mean intensity.
    Accumulates statistics on the fly to save memory.
    """
    if n_simulations is None:
        n_simulations = cfg.n_monte_carlo

    if cfg.sigma == 0.0:
        if verbose:
            print("Homogeneous medium (sigma=0), running single simulation.")
        res = run_single_simulation(
            cfg, return_history=compute_intensity_map, use_fast_solver=use_fast_solver
        )

        intensity_map = None
        if compute_intensity_map and res.history is not None:
            intensity_map = np.abs(res.history.T) ** 2

        return MonteCarloResult(
            mean_field=res.phi_final,
            mean_intensity=np.abs(res.phi_final) ** 2,
            mean_intensity_map=intensity_map,
            single_realization_history=res.history,
        )

    sum_field = None
    sum_intensity = None
    sum_intensity_map = None

    last_history = None

    iterator = range(n_simulations)
    if verbose:
        iterator = tqdm.tqdm(iterator, desc=f"Monte Carlo (N={n_simulations})")

    for i in iterator:
        # For the last run, we might want to keep history for visualization
        is_last = i == n_simulations - 1
        need_history = compute_intensity_map or is_last

        res = run_single_simulation(
            cfg,
            return_history=need_history,
            use_fast_solver=use_fast_solver,
            mixed=mixed,
        )

        if sum_field is None:
            sum_field = np.zeros_like(res.phi_final, dtype=np.complex128)
            sum_intensity = np.zeros_like(res.phi_final, dtype=np.float64)

        sum_field += res.phi_final
        sum_intensity += np.abs(res.phi_final) ** 2

        if compute_intensity_map and res.history is not None:
            curr_intensity_map = np.abs(res.history.T) ** 2
            if sum_intensity_map is None:
                sum_intensity_map = np.zeros_like(curr_intensity_map, dtype=np.float64)
            sum_intensity_map += curr_intensity_map

        if is_last:
            last_history = res.history

    assert sum_field is not None
    assert sum_intensity is not None

    mean_field = sum_field / n_simulations  # type: ignore
    mean_intensity = sum_intensity / n_simulations  # type: ignore

    mean_intensity_map = None
    if compute_intensity_map and sum_intensity_map is not None:
        mean_intensity_map = sum_intensity_map / n_simulations

    return MonteCarloResult(
        mean_field=mean_field,
        mean_intensity=mean_intensity,
        mean_intensity_map=mean_intensity_map,
        single_realization_history=last_history,
    )
