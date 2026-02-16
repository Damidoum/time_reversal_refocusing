from abc import ABC, abstractmethod
import dataclasses
import numpy as np

from time_reversal.field import WaveField


class LinearOperator(ABC):
    """Abstract base class for linear operators."""

    @abstractmethod
    def apply_op(self, field: WaveField, dz: float, **kwargs) -> WaveField:
        """Applies the operator analytically: phi(z+dz) = exp(A * dz) * phi(z)."""
        pass

    @abstractmethod
    def compute_derivative(
        self, z: float, phi: np.ndarray, field: WaveField, **kwargs
    ) -> np.ndarray:
        """Computes the derivative for ODE solvers: dphi/dz = A * phi."""
        pass


class DiffractionOperator(LinearOperator):
    """
    Diffraction step in Fourier space.
    Equation: d_phi/dz = [-i * kappa^2 / (2*k)] * phi
    """

    def apply_op(self, field: WaveField, dz: float, **kwargs) -> WaveField:
        op = -1j * field.kappa**2 / (2 * field.k_const)
        evolution_op = np.exp(op * dz)
        return dataclasses.replace(field, phi=field.phi * evolution_op)

    def compute_derivative(
        self, z: float, phi: np.ndarray, field: WaveField, **kwargs
    ) -> np.ndarray:
        op = -1j * field.kappa**2 / (2 * field.k_const)
        return op * phi


class RefractionOperator(LinearOperator):
    """
    Refraction step in Real space.
    Equation: d_phi/dz = [i * k * mu / 2] * phi
    """

    def apply_op(self, field: WaveField, dz: float, **kwargs) -> WaveField:
        mu = kwargs.get("mu")
        if mu is None:
            raise ValueError(
                "RefractionOperator requires 'mu' (refractive index fluctuation) in kwargs."
            )

        op = 0.5j * field.k_const * mu
        evolution_op = np.exp(op * dz)
        return dataclasses.replace(field, phi=field.phi * evolution_op)

    def compute_derivative(
        self, z: float, phi: np.ndarray, field: WaveField, **kwargs
    ) -> np.ndarray:
        mu = kwargs.get("mu")
        if mu is None:
            raise ValueError(
                "RefractionOperator requires 'mu' (refractive index fluctuation) in kwargs."
            )

        op = 0.5j * field.k_const * mu
        return op * phi
