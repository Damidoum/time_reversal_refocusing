import numpy as np


def homogeneous_forward(
    _: np.ndarray, phi_vec: np.ndarray, kappa_vec: np.ndarray, k_const: float
) -> np.ndarray:
    """ODE forward pass for the homogeneous_forward."""
    d_phi = -phi_vec * 1j * kappa_vec**2 / (2 * k_const)
    return d_phi


def init_homogeneous_forward(
    x: np.ndarray,
    r0: float,
) -> np.ndarray:
    """Initialize the field in space domain."""
    return np.exp(-(x**2) / (r0**2))


def homogeneous_analytic_solution(
    x: np.ndarray, L: float, r0: float, k_const: float
) -> np.ndarray:
    """Analytic solution for the homogeneous case."""
    r_t = r0 * np.sqrt(1 + 2j * L / (k_const * r0**2))
    return (r0 / r_t) * np.exp(-(x**2) / (r_t**2))
