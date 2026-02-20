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


def plot_comparison(
    x,
    psi_num,
    psi_theory,
    title: str = "Comparison",
    xlabel: str = r"$x$ (Transverse position)",
    ylabel: str = r"$|\phi(x)|^2$ (Amplitude)",
):

    # comparing numerical vs theory
    plt.figure(figsize=(8, 5))

    plt.plot(x, np.abs(psi_theory) ** 2, "k-", lw=1.5, label="Theory")
    plt.plot(x, np.abs(psi_num) ** 2, "r--", ms=6, label="Simulation")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
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


def plot_multiple_comparisons(
    x,
    data_list: list[tuple[np.ndarray, str, str]],
    title: str = "Comparison",
    xlabel: str = r"$x$ (Transverse position)",
    ylabel: str = r"$|\phi(x)|^2$ (Amplitude)",
):
    """
    Plots multiple curves for comparison.
    data_list: list of tuples (y_data, label, style_string)
    e.g. [(psi1, "Label 1", "r-"), (psi2, "Label 2", "b--")]
    """
    plt.figure(figsize=(10, 6))

    for y_data, label, style in data_list:
        plt.plot(x, np.abs(y_data) ** 2, style, label=label, linewidth=2)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


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
