from typing import Callable

import numpy as np
from scipy.fft import fft, ifft


class StationaryGaussianProcess:
    def __init__(
        self,
        mean: float,
        covariance_function: Callable[[float | np.ndarray], float | np.ndarray],
    ):
        self.mean = mean
        self.covariance_function = covariance_function

    def cov(self, t1: float, t2: float) -> float | np.ndarray:
        """Calculate the covariance between two time points."""
        return self.covariance_function(t1 - t2)

    def sample(self, x: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """
        Generate N sample paths using FFT-based approximation (Circulant Embedding).

        Args:
            x: The time points (must be equidistant).
            n_samples: Number of independent paths to generate.

        Returns:
            np.ndarray: Shape (n_samples, len(x)) containing the generated paths.
                        If n_samples=1, returns shape (len(x),) for convenience.
        """
        N = len(x)
        if N == 0:
            return np.array([])

        dt = x[1] - x[0] if N > 1 else 1.0
        lags = np.arange(N) * dt

        c = self.covariance_function(lags)
        if np.isscalar(c):
            c = np.array([self.covariance_function(lag) for lag in lags])

        if N > 1:
            c_circ = np.concatenate([c, c[1:-1][::-1]]) # type: ignore
        else:
            c_circ = c

        S = fft(c_circ)
        S_real = S.real # type: ignore
        S_real[S_real < 0] = 0

        spectral_amplitude = np.sqrt(S_real)

        M = len(c_circ) # type: ignore

        noise = np.random.standard_normal(
            (n_samples, M)
        ) + 1j * np.random.standard_normal((n_samples, M))

        weighted_noise = noise * spectral_amplitude
        sample_circ = ifft(weighted_noise, axis=-1) * np.sqrt(M)
        paths = np.real(sample_circ[:, :N]) + self.mean

        if n_samples == 1:
            return paths.flatten()

        return paths
