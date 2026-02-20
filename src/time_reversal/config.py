import argparse
import sys
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import get_type_hints


@dataclass
class SimulationConfig:
    c0: float = 1.0  # speed
    w: float = 1.0  # frequence
    L: float = 10.0  # propagation distance

    x_size: float = 60.0  # spatial domain size (from -x_max/2 to x_max/2)
    nx: int = 1024  # number of spatial points

    r0: float = 2.0  # radius of the initial source

    h: float = 0.1  # step size for numerical integration
    z_c: float = 1.0
    x_c: float = 4.0
    sigma: float = 1.0

    # mirror
    r_m: float = 2.0
    mirror_type: str = "gaussian"  # options: 'gaussian', 'compact'

    n_monte_carlo: int = 100  # number of realizations

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

    @classmethod
    def from_cli(cls) -> "SimulationConfig":
        """
        Loads configuration from a TOML file and overrides with CLI arguments.
        Usage: python script.py --config config.toml --L 20.0
        """
        parser = argparse.ArgumentParser(
            description="Run simulation with configuration overrides."
        )

        parser.add_argument(
            "--config",
            type=str,
            default="config.toml",
            help="Path to the TOML configuration file.",
        )

        for field in fields(cls):
            help_text = f"Override {field.name} (default: {field.default})"
            parser.add_argument(f"--{field.name}", type=str, help=help_text)

        args = parser.parse_args()

        file_config = {}
        config_path = Path(args.config)
        if config_path.exists():
            print(f"Loading configuration from {config_path}")
            with open(config_path, "rb") as f:
                file_config = tomllib.load(f)
        else:
            print(
                f"Warning: Configuration file {config_path} not found. Using defaults."
            )

        cli_overrides = {}

        # Get correct types, resolving string forward references if any
        type_hints = get_type_hints(cls)

        for key, value in vars(args).items():
            if key == "config" or value is None:
                continue

            # Use type_hints to determine the target type
            target_type = type_hints.get(key)
            if not target_type:
                continue

            try:
                if target_type is int:
                    cli_overrides[key] = int(value)
                elif target_type is float:
                    cli_overrides[key] = float(value)
                else:
                    cli_overrides[key] = value
            except ValueError as e:
                print(
                    f"Error converting argument --{key}='{value}' to {target_type}: {e}"
                )
                sys.exit(1)

        # Start with defaults
        final_config = {f.name: f.default for f in fields(cls)}
        # Override with file config
        final_config.update(file_config)
        # Override with CLI args
        final_config.update(cli_overrides)

        # Filter out unknown keys (though dataclass init would complain anyway)
        valid_keys = {f.name for f in fields(cls)}
        final_config = {k: v for k, v in final_config.items() if k in valid_keys}

        return cls(**final_config)  # type: ignore
