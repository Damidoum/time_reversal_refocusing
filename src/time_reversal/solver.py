import numpy as np
from numpy.fft import fft, fftfreq, ifft
from scipy.integrate import solve_ivp


class EulerPropagator:
    @staticmethod
    def forward(
        fun,
        phi0: np.ndarray,
        kappa_vec: np.ndarray,
        k_const: float,
        z_min: float,
        z_max: float,
    ) -> np.ndarray:
        """Propagates the field forward in z using the provided function."""
        sol = solve_ivp(
            fun=fun,
            t_span=(z_min, z_max),
            y0=phi0,
            args=(kappa_vec, k_const),
            method="RK45",
            rtol=1e-8,
            atol=1e-11,
        )
        return sol.y


class Propagator:
    def __init__(self, func) -> None:
        self.func = func

    def forward(
        self,
        phi0: np.ndarray,
        k_const: float,
        z_min: float,
        z_max: float,
        dx: float,
    ) -> np.ndarray:
        """Propagates the field forward in z using the provided function."""

        kappa_vec = (
            fftfreq(len(phi0), d=dx) * 2 * np.pi
        )  # because fft convention is different from the Fourier Transform convention I used in the derivation of homogeneous forward function
        phi_fourier = fft(phi0)

        sol_fourier = EulerPropagator.forward(
            self.func, phi_fourier, kappa_vec, k_const, z_min, z_max
        )

        sol = ifft(sol_fourier, axis=0)
        return sol
