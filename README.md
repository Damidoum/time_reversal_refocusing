# Time-Reversal Refocusing in Heterogeneous Media

This repository contains the numerical simulations and the core Python library developed to study time-reversal refocusing in homogeneous and randomly heterogeneous media.
This project accompanies a theoretical report detailing how wave propagation through a random medium induces a super-resolution effect during the refocusing process, effectively overcoming the classical diffraction limit. This phenomenon has significant applications in high-precision wave focalization, such as the non-invasive destruction of kidney stones.


# Repository Structure

The project is organized into the following components:

- LaTeX Report (`report.pdf`): Contains the theoretical framework, mathematical derivations, and detailed explanations of the paraxial approximation and the Split-Step Fourier Method.
- Jupyter Notebooks (`notebooks/`): Contains the interactive simulations, numerical experiments, and generated plots.
- Core Library (`notebooks/`): The custom Python module implementing the numerical methods (Fourier transforms, time-reversal mirrors, split-step solvers). 
- Animations: A .mp4 file illustrating the dynamic, time-dependent wave refocusing process, generated directly via the notebooks.
 
<video src="./notebooks/time_reversal_demo.mp4" width="100%" controls></video>

# Installation

This project uses uv for fast and reproducible environment and dependency management.

1. Clone the repository:

```bash
git clone git@github.com:Damidoum/time_reversal_refocusing.git
cd time_reversal
```

2. Sync the environment:

This command automatically creates an isolated virtual environment and installs the required dependencies based on the project configuration.
```bash
uv sync
```

# Usage

To run the simulations and reproduce the results presented in the report, launch Jupyter Lab through uv to ensure the correct environment is used.

Launch Jupyter Lab:

```bash
uv run --with jupyter jupyter lab
```

You can also navigate to the `scripts/` directory and execute the Python scripts directly using uv to ensure the correct environment is activated:

```bash
 uv run  scripts/<name_of_script>.py
```
