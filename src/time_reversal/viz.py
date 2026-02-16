import matplotlib.pyplot as plt
import numpy as np


def setup_style():
    # just setting up some latex style plots
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "figure.dpi": 150,
        }
    )


def plot_comparison(x, psi_num, psi_theory, title="Comparison"):
    # comparing numerical vs theory
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


def plot_intensity_map(
    intensity_map,
    extent,
    title="Intensity Map",
    xlabel="Propagation distance z",
    ylabel="Transverse coordinate x",
):
    # plotting the heatmap of intensity
    plt.figure(figsize=(10, 6))

    plt.imshow(
        intensity_map,
        extent=extent,
        aspect="auto",
        origin="lower",
        cmap="inferno",
    )
    plt.colorbar(label=r"$|\phi|^2$")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_mean_field_comparison(
    x,
    mean_field_num,
    mean_field_theo,
    title="Mean Field Comparison",
    xlabel="Transverse coordinate x",
    ylabel="Magnitude",
):
    # comparing mean fields
    plt.figure(figsize=(10, 6))

    plt.plot(
        x,
        np.abs(mean_field_num),
        "b-",
        linewidth=2,
        label="Numerical Mean Field",
    )
    plt.plot(
        x,
        np.abs(mean_field_theo),
        "r--",
        linewidth=2,
        label="Theoretical Mean Field",
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
