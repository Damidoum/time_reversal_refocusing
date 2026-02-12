import numpy as np


def compactly_supported_mirror(x: np.ndarray, r_m: float) -> np.ndarray:
    """Mirror plane"""
    return np.where(np.abs(x) <= 2 * r_m, 1.0, 0.0) * (1.0 - (x / (2 * r_m)) ** 2) ** 2


def gaussian_mirror(x: np.ndarray, r_m: float) -> np.ndarray:
    """Gaussian mirror plane"""
    return np.exp(-(x**2) / (r_m**2))


def homogeneous_time_reversal_analytic_solution(
    x: np.ndarray, L: float, r0: float, k_const: float, r_m: float
) -> np.ndarray:
    """Analytic solution for time reversal in the homogeneous case with gaussian mirror."""
    a = (
        1 + 4 * L**2 / (k_const**2 * r0**2 * r_m**2) + 2j * L / (k_const * r_m**2)
    ) ** 0.5
    r2 = (1 / r_m**2 + 1 / (r0**2 - 2j * L / k_const)) ** -1 + 2j * L / k_const
    return 1 / a * np.exp(-(x**2) / r2)
