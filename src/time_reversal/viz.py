import matplotlib.pyplot as plt
import numpy as np


def setup_style():
    """Configures Matplotlib for LaTeX-style rendering."""
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "figure.dpi": 150,
        }
    )


def plot_comparison(x, psi_num, psi_theory, title="Comparison"):
    """Compares numerical simulation with theoretical profile."""
    plt.figure(figsize=(8, 5))

    plt.plot(x, np.abs(psi_theory) ** 2, "k-", lw=1.5, label="Theory")

    subset = slice(None, None, 10)
    plt.plot(x[subset], np.abs(psi_num)[subset] ** 2, "r+", ms=6, label="Simulation")

    plt.xlabel(r"$x$ (Transverse position)")
    plt.ylabel(r"$|\phi(x)|^2$ (Amplitude)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
