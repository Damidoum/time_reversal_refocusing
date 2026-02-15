from abc import ABC, abstractmethod

import numpy as np
from numpy.fft import fft, ifft
from scipy.integrate import solve_ivp


class Solver(ABC):
    """Abstract base class for propagation strategies."""

    @abstractmethod
    def evolve(
        self,
        phi0: np.ndarray,
        z_min: float,
        z_max: float,
        operator_func,
        *args,
        **kwargs,
    ) -> np.ndarray:
        """Evolves the field from z_min to z_max."""
        pass


class AnalyticSolver(Solver):
    """
    Fast solver.
    Uses the exact analytical solution: phi(z+h) = phi(z) * exp(A * h).
    """

    def evolve(
        self,
        phi0: np.ndarray,
        z_min: float,
        z_max: float,
        operator_func,
        *args,
        **kwargs,
    ) -> np.ndarray:
        dz = z_max - z_min

        op = operator_func(z_min, phi0, *args, **kwargs)
        evolution_op = np.exp(op * dz)

        return phi0 * evolution_op


class RungeKuttaSolver(Solver):
    """
    Slow solver.
    Uses scipy.integrate.solve_ivp to solve dphi/dz = A * phi.
    """

    def evolve(
        self,
        phi0: np.ndarray,
        z_min: float,
        z_max: float,
        operator_func,
        *args,
        **kwargs,
    ) -> np.ndarray:

        def wrapper_fun(z, phi_vec):
            return operator_func(z, phi_vec, *args, **kwargs)

        sol = solve_ivp(
            fun=wrapper_fun,
            t_span=(z_min, z_max),
            y0=phi0,
            method="RK45",
            rtol=1e-8,
            atol=1e-11,
        )
        return sol.y[:, -1]


class Propagator:
    """Handles propagation in Real Space"""

    def __init__(self, func, solver: Solver) -> None:
        self.func = func
        self.solver = solver

    def forward(
        self, phi0: np.ndarray, z_min: float, z_max: float, *args, **kwargs
    ) -> np.ndarray:
        return self.solver.evolve(phi0, z_min, z_max, self.func, *args, **kwargs)


class PropagatorFourier:
    """Handles propagation in Fourier Space"""

    def __init__(
        self, func, solver: Solver
    ) -> None:
        self.func = func
        self.solver = solver

    def forward(
        self, phi0: np.ndarray, z_min: float, z_max: float, *args, **kwargs
    ) -> np.ndarray:

        phi_fourier = fft(phi0)

        sol_fourier = self.solver.evolve(
            phi_fourier, z_min, z_max, self.func, *args, **kwargs
        )

        return ifft(sol_fourier, axis=0)
