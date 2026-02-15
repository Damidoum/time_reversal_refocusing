from dataclasses import dataclass


@dataclass
class SimulationConfig:
    c0: float = 1.0  # speed
    w: float = 1.0  # frequence
    L: float = 10.0  # propagation distance

    x_size: float = 60.0  # spatial domain size (from -x_max/2 to x_max/2)
    nx: int = 2**10  # number of spatial points

    r0: float = 2.0  # radius of the initial source

    @property
    def dx(self) -> float:
        """Spatial step size."""
        return self.x_size / self.nx

    @property
    def k_const(self) -> float:
        """Wave number."""
        return self.w / self.c0

    @property
    def x_min(self) -> float:
        """Minimum x value."""
        return -self.x_size / 2

    @property
    def x_max(self) -> float:
        """Maximum x value."""
        return self.x_size / 2
