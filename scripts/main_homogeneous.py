import numpy as np

from time_reversal.config import SimulationConfig
from time_reversal.field import WaveField
from time_reversal.operators import DiffractionOperator
from time_reversal.propagation_fun import (
    homogeneous_analytic_solution,
    init_homogeneous,
)
from time_reversal.propagator import SplitStepPropagator
from time_reversal.solver import AnalyticSolver
from time_reversal.viz import plot_comparison, plot_intensity_map, setup_style


def main():
    setup_style()

    cfg = SimulationConfig.from_cli()
    print(f"Running Homogeneous Simulation with: {cfg}")

    # theory solution
    x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)
    phi_theory = homogeneous_analytic_solution(x, cfg.L, cfg.r0, cfg.k_const)

    # numerical solution
    phi0 = init_homogeneous(x=x, r0=cfg.r0)
    field = WaveField(x=x, phi=phi0, k_const=cfg.k_const)

    propagator = SplitStepPropagator(steps=[(DiffractionOperator(), AnalyticSolver())])

    # to track the wave profile at each step
    history = []
    for _ in range(int(cfg.L / cfg.h)):
        field = propagator.step(field, dz=cfg.h)
        history.append(field.to_real().phi)

    # convert to real space for comparison
    field = field.to_real()
    phi = field.phi

    plot_comparison(
        x, phi, phi_theory, title=f"Wave Profile at $z={cfg.L}$ (Homogeneous medium)"
    )

    plot_intensity_map(
        intensity_map=np.abs(history).T ** 2,
        extent=[0, cfg.L, cfg.x_min, cfg.x_max],
        title="Intensity Map (Homogeneous medium)",
        xlabel="Propagation distance z",
        ylabel="Transverse coordinate x",
    )


if __name__ == "__main__":
    main()
