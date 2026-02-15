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

    def cov(self, t1: float, t2: float) -> float:
        """Calculate the covariance between two time points."""
        return self.covariance_function(t1 - t2)

    def sample(self, n: int, dt: float) -> np.ndarray:
        """Generate a sample path using FFT-based approximation."""
        lags = np.arange(n) * dt

        # Handle both vectorized and scalar covariance functions
        try:
            c = self.covariance_function(lags)
            if not isinstance(c, np.ndarray):
                c = np.array(c)
        except (TypeError, ValueError):
            c = np.array([self.covariance_function(t) for t in lags])

        # Ensure c is 1D array (handling potential 0-d array from scalar return)
        if c.ndim == 0:
            c = np.array([self.covariance_function(t) for t in lags])

        circulant_first_row = np.concatenate([c, c[1:-1][::-1]])

        eig_vals = fft(circulant_first_row).real
        eig_vals = np.maximum(eig_vals, 0)

        m = len(circulant_first_row)
        noise = np.random.normal(0, 1, m) + 1j * np.random.normal(0, 1, m)

        sample_full = ifft(fft(noise) * np.sqrt(eig_vals)).real

        return self.mean + sample_full[:n]
