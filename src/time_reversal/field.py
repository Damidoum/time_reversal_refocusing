from __future__ import annotations

import dataclasses

import numpy as np
from scipy.fft import fft, fftfreq, ifft


@dataclasses.dataclass
class WaveField:
    """Represents the wave field state."""

    x: np.ndarray
    phi: np.ndarray
    k_const: float
    z: float = 0.0
    domain: str = "real"  # "real" or "fourier"

    @property
    def dx(self) -> float:
        """Spatial step size."""
        return self.x[1] - self.x[0]

    @property
    def kappa(self) -> np.ndarray:
        """Returns the spatial frequency grid."""
        return fftfreq(len(self.phi), d=self.dx) * 2 * np.pi

    def copy(self) -> WaveField:
        """Returns a deep copy of the WaveField."""
        return dataclasses.replace(self, x=self.x.copy(), phi=self.phi.copy())

    def to_fourier(self) -> WaveField:
        """Transforms the field to Fourier space."""
        if self.domain == "fourier":
            return self
        return dataclasses.replace(self, phi=fft(self.phi), domain="fourier")

    def to_real(self) -> WaveField:
        """Transforms the field to Real space."""
        if self.domain == "real":
            return self
        return dataclasses.replace(self, phi=ifft(self.phi), domain="real")
