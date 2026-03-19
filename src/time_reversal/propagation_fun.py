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
    return sigma**2 * np.exp(-(x**2) / x_c**2)


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
    r_tr2 = ((1 / r_m**2) + 1 / (r0**2 - 2j * L / k_const)) ** (-1) + 2j * L / k_const
    a_tr = (
        1 + 4 * L**2 / (k_const**2 * r0**2 * r_m**2) + 2j * L / (k_const * r_m**2)
    ) ** 0.5

    gamma2 = 2 * sigma**2 * z_c / (x_c**2)
    inv_r_a2 = gamma2 * (c0 * k_const) ** 2 * L / 48
    return (1 / a_tr) * np.exp(-(x**2) / r_tr2) * np.exp(-(x**2) * inv_r_a2)


def mean_field_mixed_medium_refocused(
    x: np.ndarray,
    r0: float,
    k_const: float,
    c0: float,
    L: float,
    sigma: float,
    z_c: float,
    r_m: float,
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


def compute_theoretical_broadband_refocused(
    x: np.ndarray,
    r0: float,
    c0: float,
    L: float,
    sigma: float,
    z_c: float,
    r_m: float,
    x_c: float,
    omegas: np.ndarray,
    mixed: bool = False,
) -> np.ndarray:
    """
    Computes the theoretical refocused profile by summing the theory over all frequencies.
    """
    theory_sum = np.zeros_like(x, dtype=complex)

    for w in omegas:
        k_w = w / c0  # Frequency-dependent wavenumber

        # Compute a_tr and r_tr
        atr = np.sqrt(
            1 + 4 * L**2 / (k_w**2 * r0**2 * r_m**2) + 2j * L / (k_w * r_m**2)
        )

        term1 = 1 / (r_m**2)
        term2 = 1 / (r0**2 - 2j * L / k_w)
        rtr_sq = 1 / (term1 + term2) + 2j * L / k_w

        if mixed:
            # Backward in Homogeneous Medium (using gamma0, no r_a)
            gamma0 = sigma**2 * z_c
            damping = np.exp(-(w**2 / c0**2) * gamma0 * L / 8)
            phi_w = (1 / atr) * np.exp(-(x**2) / rtr_sq) * damping
        else:
            # Backward in Random Medium (using gamma2, r_a)
            gamma2 = 2 * (sigma**2) * z_c / (x_c**2)
            ra_inv2 = gamma2 * (w**2) * L / 48
            phi_w = (1 / atr) * np.exp(-(x**2) / rtr_sq) * np.exp(-(x**2) * ra_inv2)

        theory_sum += phi_w

    return theory_sum


def verify_paraxial_approximation(
    field_history: np.ndarray, h: float, k_const: float, threshold: float = 0.1
) -> bool:
    """
    Verifies if the paraxial approximation is valid for the given field history.
    Checks if the ratio |dzz phi| / |2k dz phi| is small, which corresponds to (kx/2k)^2.

    Args:
        field_history: The field evolution (Nz, Nx).
        h: The step size in z (dz).
        k_const: The wavenumber k = omega / c0.
        threshold: Warning threshold for the ratio.

    Returns:
        True if valid (or warning printed but proceeded), False if critical failure (not used here).
    """
    if field_history.shape[0] < 3:
        return True

    # Compute derivatives using central differences for the inner points
    # We take the middle part of the history to avoid boundary effects
    phi = field_history

    # Calculate derivatives along z (axis 0)
    # dz_phi at index i corresponds to (phi[i+1] - phi[i-1]) / 2h
    # dzz_phi at index i corresponds to (phi[i+1] - 2phi[i] + phi[i-1]) / h^2

    dz_phi = (phi[2:] - phi[:-2]) / (2 * h)
    dzz_phi = (phi[2:] - 2 * phi[1:-1] + phi[:-2]) / (h**2)

    # Avoid division by zero
    den = np.abs(2 * k_const * dz_phi)
    # Use a small epsilon relative to the max gradient
    epsilon = 1e-10 * np.max(den) if np.max(den) > 0 else 1e-10
    den = np.where(den < epsilon, epsilon, den)

    ratio = np.abs(dzz_phi) / den

    # We only care about regions where the field is significant
    intensity = np.abs(phi[1:-1]) ** 2
    max_intensity = np.max(intensity)
    if max_intensity == 0:
        return True

    mask = intensity > 0.01 * max_intensity

    if np.any(mask):
        mean_ratio = np.mean(ratio[mask])
        max_ratio = np.max(ratio[mask])

        # If mean ratio is large, it implies k_x is not negligible compared to k
        if mean_ratio > threshold:
            print(f"Warning: Paraxial approximation validity check (k={k_const:.2f}):")
            print(
                f"  Mean ratio |dzz|/|2k dz| = {mean_ratio:.4f} (Threshold: {threshold})"
            )
            print(f"  Max ratio = {max_ratio:.4f}")
            # We don't return False because we want the simulation to continue, just warn.

    return True
