import dataclasses
from abc import ABC, abstractmethod

import numpy as np
from scipy.integrate import solve_ivp

from time_reversal.field import WaveField
from time_reversal.operators import LinearOperator


class Solver(ABC):
    """Abstract base class for propagation solvers."""

    @abstractmethod
    def evolve(
        self, field: WaveField, z_end: float, operator: LinearOperator, **kwargs
    ) -> WaveField:
        """Evolves the field from field.z to z_end using the given operator."""
        pass


class AnalyticSolver(Solver):
    """
    Fast solver using the operator's analytic solution.
    phi(z+h) = exp(A * h) * phi(z)
    """

    def evolve(
        self, field: WaveField, z_end: float, operator: LinearOperator, **kwargs
    ) -> WaveField:
        dz = z_end - field.z
        if np.isclose(dz, 0):
            return field

        # Apply the operator's analytic evolution
        new_field = operator.apply_op(field, dz, **kwargs)

        # Update the z position
        return dataclasses.replace(new_field, z=z_end)


class RungeKuttaSolver(Solver):
    """
    Slow solver using scipy.integrate.solve_ivp.
    dphi/dz = A * phi
    """

    def evolve(
        self, field: WaveField, z_end: float, operator: LinearOperator, **kwargs
    ) -> WaveField:
        if np.isclose(field.z, z_end):
            return field

        def ode_func(z, phi_flat):
            dphi = operator.compute_derivative(z, phi_flat, field, **kwargs)
            return dphi

        y0 = field.phi.flatten()

        sol = solve_ivp(
            fun=ode_func,
            t_span=(field.z, z_end),
            y0=y0,
            method="RK45",
            rtol=1e-8,
            atol=1e-11,
        )

        new_phi = sol.y[:, -1].reshape(field.phi.shape)

        return dataclasses.replace(field, phi=new_phi, z=z_end)
