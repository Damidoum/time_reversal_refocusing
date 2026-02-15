import numpy as np


def homogeneous_propagation_ode(
    _: np.ndarray, phi_vec: np.ndarray, kappa_vec: np.ndarray, k_const: float
) -> np.ndarray:
    """ODE forward for the homogeneous case"""
    d_phi = -phi_vec * 1j * kappa_vec**2 / (2 * k_const)
    return d_phi

def homogeneous_operator(
    _: np.ndarray, kappa_vec: np.ndarray, k_const: float
) -> np.ndarray:
    """Operator for the homogeneous case"""
    return -1j * kappa_vec**2 / (2 * k_const)


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


def compactly_supported_mirror(x: np.ndarray, r_m: float) -> np.ndarray:
    """Mirror plane"""
    return np.where(np.abs(x) <= 2 * r_m, 1.0, 0.0) * (1.0 - (x / (2 * r_m)) ** 2) ** 2

def gaussian_mirror(x: np.ndarray, r_m: float) -> np.ndarray:
    """Gaussian mirror plane"""
    return np.exp(-(x**2) / (r_m**2))

def homogeneous_time_reversal_analytic_solution(
    x: np.ndarray, length: float, r0: float, k_const: float, r_m: float
) -> np.ndarray:
    """Analytic solution for time reversal in the homogeneous case with gaussian mirror"""
    a = (1 + 4 * length**2 / (k_const**2 * r0**2 * r_m**2) + 2j * length / (k_const * r_m**2)
    ) ** 0.5
    r2 = (1 / r_m**2 + 1 / (r0**2 - 2j * length / k_const)) ** -1 + 2j * length / k_const
    return 1 / a * np.exp(-(x**2) / r2)


def diffraction_propagation_ode(
    z: float,
    phi_vec: np.ndarray,
    kappa_vec: np.ndarray,
    k_const: float,
) -> np.ndarray:
    """
    Diffraction step in Fourier space.
    Equation: d_phi/dz = [-i * kappa^2 / (2*k)] * phi
    """
    op = -1j * kappa_vec**2 / (2 * k_const)
    return op * phi_vec

def diffraction_propagation_operator(
    z: float,
    phi_vec: np.ndarray,
    kappa_vec: np.ndarray,
    k_const: float,
) -> np.ndarray:
    """
    Diffraction step in Fourier space.
    Equation: d_phi/dz = [-i * kappa^2 / (2*k)] * phi
    """
    op = -1j * kappa_vec**2 / (2 * k_const)
    return op

def refraction_propagation_ode(
    z: float,
    phi_vec: np.ndarray,
    mu: np.ndarray,
    k_const: float,
) -> np.ndarray:
    """
    Refraction step in Real space.
    Equation: d_phi/dz = [i * k * mu / 2] * phi
    """
    op = 0.5j * k_const * mu

    return op * phi_vec

def refraction_propagation_operator(
    z: float,
    phi_vec: np.ndarray,
    mu: np.ndarray,
    k_const: float,
) -> np.ndarray:
    """
    Refraction step in Real space.
    Equation: d_phi/dz = [i * k * mu / 2] * phi
    """
    op = 0.5j * k_const * mu
    return op

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
