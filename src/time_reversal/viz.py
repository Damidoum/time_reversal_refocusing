from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def setup_style():
    # setting up some latex style plots
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
            "figure.dpi": 150,
        }
    )


def plot_intensity_section(
    x,
    psi_num,
    psi_theory,
    title: str = "Intensity comparison (z = L)",
    xlabel: str = r"$x$ (Transverse position)",
    ylabel: str = r"$|\phi(x)|^2$ (Amplitude)",
    label_curve1: str = "Simulation",
    label_curve2: str = "Theory",
    save_path: str | Path | None = None,
    show: bool = False,
):

    # comparing numerical vs theory
    plt.figure(figsize=(8, 5))

    plt.plot(x, np.abs(psi_theory) ** 2, "k-", lw=1.5, label=label_curve2)
    plt.plot(x, np.abs(psi_num) ** 2, "r--", ms=6, label=label_curve1)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, transparent=True)

    if show:
        plt.show()
    else:
        plt.close()


def plot_intensity_map(
    intensity_map,
    extent,
    title="Intensity Map",
    xlabel="Propagation distance z",
    ylabel="Transverse coordinate x",
    save_path: str | Path | None = None,
    show: bool = False,
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

    if save_path is not None:
        plt.savefig(save_path, dpi=300, transparent=True)
    if show:
        plt.show()
    else:
        plt.close()


def plot_multiple_intensity_section(
    x,
    data_list: list[tuple[np.ndarray, str, str | None]],
    title: str = "Comparison",
    xlabel: str = r"$x$ (Transverse position)",
    ylabel: str = r"$|\phi(x)|^2$ (Amplitude)",
    save_path: str | Path | None = None,
    show: bool = False,
):
    """
    Plots multiple curves for comparison.
    data_list: list of tuples (y_data, label, style_string)
    e.g. [(psi1, "Label 1", "r-"), (psi2, "Label 2", "b--")]
    """

    plt.figure(figsize=(10, 6))

    for y_data, label, style in data_list:
        if style is None:
            style = "-"
        plt.plot(x, np.abs(y_data) ** 2, style, label=label, linewidth=2, alpha=0.8)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, transparent=True)

    if show:
        plt.show()
    else:
        plt.close()


def plot_intensity_grid(
    intensity_grid: list[list[np.ndarray]],
    row_labels: list[str],
    col_labels: list[str],
    extent: list[float],
    xlabel: str = "z",
    ylabel: str = "x",
):
    """
    Plots a grid of intensity maps.
    intensity_grid: List of rows, where each row is a list of intensity maps (2D arrays).
    """
    n_rows = len(intensity_grid)
    n_cols = len(intensity_grid[0])

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), constrained_layout=True
    )

    # Handle single row or single column case where axes might be 1D or scalar
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i, j]
            im = ax.imshow(
                intensity_grid[i][j],
                extent=extent,
                aspect="auto",
                origin="lower",
                cmap="inferno",
            )

            # Add labels only on edges
            if i == n_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)

            # Row titles
            if j == n_cols - 1:
                ax.text(
                    1.05,
                    0.5,
                    row_labels[i],
                    transform=ax.transAxes,
                    rotation=-90,
                    va="center",
                    fontweight="bold",
                )

            # Column titles
            if i == 0:
                ax.set_title(col_labels[j])

    fig.colorbar(im, ax=axes.ravel().tolist(), label=r"$|\phi|^2$")
    plt.show()
