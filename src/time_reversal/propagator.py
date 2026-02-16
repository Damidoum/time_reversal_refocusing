import dataclasses
from typing import List, Tuple

from time_reversal.field import WaveField
from time_reversal.operators import (
    DiffractionOperator,
    LinearOperator,
    RefractionOperator,
)
from time_reversal.solver import Solver


class SplitStepPropagator:
    """
    Orchestrates the propagation using the split-step method.
    Typically alternates between Diffraction (Fourier) and Refraction (Real).
    """

    def __init__(self, steps: List[Tuple[LinearOperator, Solver]]):
        """
        Args:
            steps: A list of (Operator, Solver) pairs to apply in sequence for each step dz.
        """
        self.steps = steps

    def step(self, field: WaveField, dz: float, **kwargs) -> WaveField:
        """
        Propagates the field by a single step dz using the configured operators sequence.

        Args:
            field: The current wave field.
            dz: The step size.
            **kwargs: Additional arguments passed to the operators (e.g., 'mu' for refraction).

        Returns:
            WaveField: The propagated field at z + dz.
        """
        current_field = field
        z_start = field.z
        z_end = z_start + dz

        for operator, solver in self.steps:
            # Ensure correct domain for the operator
            if isinstance(operator, DiffractionOperator):
                if current_field.domain != "fourier":
                    current_field = current_field.to_fourier()
            elif isinstance(operator, RefractionOperator):
                if current_field.domain != "real":
                    current_field = current_field.to_real()

            temp_field = dataclasses.replace(current_field, z=z_start)
            current_field = solver.evolve(temp_field, z_end, operator, **kwargs)

        return dataclasses.replace(current_field, z=z_end)
