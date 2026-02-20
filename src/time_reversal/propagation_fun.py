import numpy as np


def init_homogeneous(
    x: np.ndarray,
    r0: float,
) -> np.ndarray:
    """Initialize the field in space domain for the homogeneous case"""
    return np.exp(-(x**2) / (r0**2))


def homogeneous_analytic_solution(
    x: np.ndarray, length: float, r0: float, k_const: float
) -> np.ndarray:
    """Analytic solution for the homogeneous case."""
    r_t = r0 * np.sqrt(1 + 2j * length / (k_const * r0**2))
    return (r0 / r_t) * np.exp(-(x**2) / (r_t**2))


def compact_mirror(x: np.ndarray, r_m: float) -> np.ndarray:
    """Mirror plane"""
    return np.where(np.abs(x) <= 2 * r_m, 1.0, 0.0) * (1.0 - (x / (2 * r_m)) ** 2) ** 2


def gaussian_mirror(x: np.ndarray, r_m: float) -> np.ndarray:
    """Gaussian mirror plane"""
    return np.exp(-(x**2) / (r_m**2))


def homogeneous_time_reversal_analytic_solution(
    x: np.ndarray, length: float, r0: float, k_const: float, r_m: float
) -> np.ndarray:
    """Analytic solution for time reversal in the homogeneous case with gaussian mirror"""
    a = (
        1
        + 4 * length**2 / (k_const**2 * r0**2 * r_m**2)
        + 2j * length / (k_const * r_m**2)
    ) ** 0.5
    r2 = (
        1 / r_m**2 + 1 / (r0**2 - 2j * length / k_const)
    ) ** -1 + 2j * length / k_const
    return 1 / a * np.exp(-(x**2) / r2)


def covariance_function(x: np.ndarray, x_c: float, sigma: float) -> np.ndarray:
    """Covariance function for the random medium."""
    return sigma**2 * np.exp(-0.5 * x**2 / x_c**2)


def mean_intensity(
    x: np.ndarray,
    r0: float,
    k_const: float,
    L: float,
    sigma: float,
    z_c: float,
    c0: float,
) -> np.ndarray:
    """Mean intensity of the initial field."""
    r_t = r0 * np.sqrt(1 + 2j * L / (k_const * r0**2))
    gamma0 = sigma**2 * z_c
    return (
        (r0 / r_t)
        * np.exp(-((x / r_t) ** 2))
        * np.exp(-((k_const * c0) ** 2) * gamma0 * L / 8)
    )


def mean_field_random_medium_refocused(
    x: np.ndarray,
    r0: float,
    k_const: float,
    c0: float,
    L: float,
    sigma: float,
    z_c: float,
    r_m: float,
    x_c: float,
) -> np.ndarray:
    """Mean field refocus at the source plane."""
    r_tr2 = (1 / r_m**2 + 1 / (r0**2 - 2j * L / k_const)) ** (-1) + 2j * L / k_const
    a_tr = (
        1 + 4 * L**2 / (k_const**2 * r0**2 * r_m**2) + 2j * L / (k_const * r_m**2)
    ) ** 0.5

    gamma2 = 2 * sigma**2 * z_c / (x_c**2)
    inv_r_a2 = gamma2 * (c0 * k_const) ** 2 * L / 48
    return 1 / a_tr * np.exp(-(x**2) / r_tr2) * np.exp(-(x**2) * inv_r_a2)


def mean_field_homogeneous_refocused(
    x: np.ndarray,
    r0: float,
    k_const: float,
    c0: float,
    L: float,
    sigma: float,
    z_c: float,
    r_m: float,
    x_c: float,
) -> np.ndarray:
    """Mean field refocus at the source plane."""
    r_tr2 = (1 / r_m**2 + 1 / (r0**2 - 2j * L / k_const)) ** (-1) + 2j * L / k_const
    a_tr = (
        1 + 4 * L**2 / (k_const**2 * r0**2 * r_m**2) + 2j * L / (k_const * r_m**2)
    ) ** 0.5
    gamma0 = sigma**2 * z_c
    return (
        1
        / a_tr
        * np.exp(-(x**2) / r_tr2)
        * np.exp(-((k_const * c0) ** 2) * gamma0 * L / 8)
    )
