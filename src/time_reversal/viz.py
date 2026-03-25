from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from time_reversal.config import SimulationConfig


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
    ax=None,
    title: str = "Intensity comparison (z = L)",
    xlabel: str = r"$x$ (Transverse position)",
    ylabel: str = r"$|\phi(x)|^2$ (Amplitude)",
    label_curve1: str = "Simulation",
    label_curve2: str = "Theory",
    save_path: str | Path | None = None,
    show: bool = False,
):

    # comparing numerical vs theory
    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(x, np.abs(psi_theory) ** 2, "k-", lw=1.5, label=label_curve2)
    ax.plot(x, np.abs(psi_num) ** 2, "r--", ms=6, label=label_curve1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, transparent=True)

    if show:
        plt.show()

    return ax


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
    plt.close()


def plot_multiple_intensity_section(
    x,
    data_list: list[tuple[np.ndarray, str, str | None]],
    ax=None,
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

    if ax is None:
        fig, ax = plt.subplots()

    for y_data, label, style in data_list:
        if style is None:
            style = "-"
        ax.plot(x, np.abs(y_data) ** 2, style, label=label, linewidth=2, alpha=0.8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize="small")

    if save_path is not None:
        fig.savefig(save_path, dpi=300, transparent=True)

    if show:
        plt.show()

    return ax


def plot_intensity_grid(
    intensity_grid: list[list[np.ndarray]],
    row_labels: list[str],
    col_labels: list[str],
    extent: list[float],
    xlabel="Propagation distance z",
    ylabel="Transverse coordinate x",
    title: str = "Intensity Grid",
    save_path: str | Path | None = None,
    show: bool = False,
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
    fig.suptitle(title, fontsize=16, fontweight="bold")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, transparent=True)
    if show:
        plt.show()
    else:
        plt.close()


def animate_wavefield_mp4(
    temporal_signal: np.ndarray,
    t_skip: int = 1,
    x_skip: int = 1,
    fps: int = 30,
    save_path: Path | None = None,
):
    """
    Creates and saves an MP4 animation of the 3D wavefield (time, z, x).
    """
    data = temporal_signal[::t_skip, ::x_skip, ::x_skip]
    nt, _, _ = data.shape
    max_amp = np.max(np.abs(data)) * 0.10

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(
        data[0],
        origin="lower",
        vmin=-max_amp,
        vmax=max_amp,
        cmap="RdBu",
        aspect="auto",
    )

    ax.set_title("Time Reversal Propagation")
    ax.set_xlabel(f"X-axis (subsampled by {x_skip})")
    ax.set_ylabel(f"Z-axis (subsampled by {x_skip})")
    fig.colorbar(im, ax=ax, label="Field Amplitude")

    def update(frame):
        im.set_array(data[frame])
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=nt, blit=True)

    if save_path is not None:
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True)

        ani.save(
            save_path,
            fps=fps,
            writer="ffmpeg",
            extra_args=[
                "-vcodec",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "faststart",
            ],
        )
    plt.close(fig)


def animate_wavefield_comparison(
    forward_signal: np.ndarray,
    backward_signal: np.ndarray,
    cfg: SimulationConfig,
    t_skip: int = 1,
    x_skip: int = 1,
    fps: int = 30,
    save_path: Path | None = None,
):
    f_data = forward_signal[::t_skip, ::x_skip, ::x_skip]
    b_data = backward_signal[::t_skip, ::x_skip, ::x_skip]

    n_frames_prop = f_data.shape[0]
    pause_frames = fps  # 1 second pause

    # Calculate exact index where the wave hits the mirror to synchronize videos
    time_to_mirror = cfg.L / cfg.c0
    frame_mirror = int((time_to_mirror / cfg.t_max) * n_frames_prop)

    total_frames = n_frames_prop + pause_frames + n_frames_prop

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    max_amp = np.max(np.abs(f_data)) * 0.15

    im1 = ax1.imshow(
        np.zeros_like(f_data[0]),
        origin="lower",
        vmin=-max_amp,
        vmax=max_amp,
        cmap="RdBu",
        aspect="auto",
    )
    im2 = ax2.imshow(
        np.zeros_like(f_data[0]),
        origin="lower",
        vmin=-max_amp,
        vmax=max_amp,
        cmap="RdBu",
        aspect="auto",
    )

    title1 = ax1.set_title("")
    title2 = ax2.set_title("")

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])

    def update(frame):
        if frame < n_frames_prop:
            im1.set_array(f_data[frame])
            im2.set_array(np.zeros_like(f_data[0]))
            title1.set_text("FORWARD")
            title2.set_text("Waiting...")

        elif frame < n_frames_prop + pause_frames:
            im1.set_array(f_data[-1])  # Stay at mirror
            im2.set_array(np.flip(b_data[0], 0))
            title1.set_text("RECORDING COMPLETE")
            title2.set_text("TIME REVERSAL PROCESS")

        else:
            idx = frame - (n_frames_prop + pause_frames)

            # Right side: Physical reality flows forward in array
            im2.set_array(np.flip(b_data[idx], 0))

            # Left side: Perfect mathematical rewind synchronized at t = L/c0
            rewind_idx = 2 * frame_mirror - idx
            if 0 <= rewind_idx < n_frames_prop:
                im1.set_array(f_data[rewind_idx])
            else:
                im1.set_array(np.zeros_like(f_data[0]))

            title1.set_text("MOVIE REWIND")
            title2.set_text("TIME REVERSAL")

        return [im1, im2, title1, title2]

    ani = animation.FuncAnimation(fig, update, frames=total_frames, blit=True)

    if save_path is not None:
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True)
        ani.save(
            save_path,
            fps=fps,
            writer="ffmpeg",
            extra_args=[
                "-vcodec",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "faststart",
            ],
        )
    plt.close(fig)


def animate_sigma_grid(
    results_dict: dict,
    cfg: SimulationConfig,
    t_skip: int = 1,
    x_skip: int = 1,
    fps: int = 30,
    save_path: Path | None = None,
):
    """
    Creates a grid animation where each row is a different sigma (noise level).
    Left column: Rewind | Right column: Physical Time Reversal
    """
    sigmas = sorted(results_dict.keys())
    n_rows = len(sigmas)

    # Prepare data for all rows
    all_data = []
    for s in sigmas:
        f = results_dict[s]["forward"][::t_skip, :, ::x_skip]
        b = np.flip(results_dict[s]["backward"][::t_skip, :, ::x_skip], axis=1)
        all_data.append((f, b))

    n_frames_prop = all_data[0][0].shape[0]
    pause_frames = fps

    time_to_mirror = cfg.L / cfg.c0
    frame_mirror = int((time_to_mirror / cfg.t_max) * n_frames_prop)

    total_frames = n_frames_prop + pause_frames + n_frames_prop

    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows))
    plt.subplots_adjust(hspace=0.3, wspace=0.1)

    ims_left = []
    ims_right = []

    global_max = max(np.max(np.abs(d[0])) for d in all_data) * 0.15

    for i, sigma in enumerate(sigmas):
        im_l = axes[i, 0].imshow(
            np.zeros_like(all_data[0][0][0]),
            origin="lower",
            vmin=-global_max,
            vmax=global_max,
            cmap="RdBu",
            aspect="auto",
        )
        im_r = axes[i, 1].imshow(
            np.zeros_like(all_data[0][0][0]),
            origin="lower",
            vmin=-global_max,
            vmax=global_max,
            cmap="RdBu",
            aspect="auto",
        )

        axes[i, 0].set_ylabel(rf"$\sigma$ = {sigma}", fontsize=14, fontweight="bold")
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])

        ims_left.append(im_l)
        ims_right.append(im_r)

    axes[0, 0].set_title("FORWARD / PERFECT REWIND", fontsize=16)
    axes[0, 1].set_title("PHYSICAL TIME REVERSAL", fontsize=16)
    status_text = fig.suptitle("", y=0.95, fontsize=18, color="darkblue")

    def update(frame):
        if frame < n_frames_prop:
            status_text.set_text("FORWARD PROPAGATION (RECORDING)")
            for i in range(n_rows):
                ims_left[i].set_array(all_data[i][0][frame])
                ims_right[i].set_array(np.zeros_like(all_data[0][0][0]))

        elif frame == n_frames_prop:
            status_text.set_text("TIME REVERSAL PROCESS")
            for i in range(n_rows):
                ims_left[i].set_array(all_data[i][0][-1])  # Stay at mirror
                ims_right[i].set_array(all_data[i][1][0])
        elif frame < n_frames_prop + pause_frames:
            pass
        else:
            idx = frame - (n_frames_prop + pause_frames)
            status_text.set_text("REWIND (Left) vs PHYSICS (Right)")

            rewind_idx = 2 * frame_mirror - idx

            for i in range(n_rows):
                if 0 <= rewind_idx < n_frames_prop:
                    ims_left[i].set_array(all_data[i][0][rewind_idx])
                else:
                    ims_left[i].set_array(np.zeros_like(all_data[i][0][0]))

                ims_right[i].set_array(all_data[i][1][idx])

        return ims_left + ims_right + [status_text]

    ani = animation.FuncAnimation(fig, update, frames=total_frames, blit=True)

    if save_path is not None:
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True)

        ani.save(
            save_path,
            fps=fps,
            dpi=80,
            writer="ffmpeg",
            extra_args=[
                "-vcodec",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "faststart",
            ],
        )

    plt.close(fig)


def plot_complex_comparison(
    x,
    psi_num,
    psi_theory,
    title_prefix: str = "Comparison",
    xlabel: str = r"$x$ (Transverse position)",
    label_curve1: str = "Simulation",
    label_curve2: str = "Theory",
    save_path: str | Path | None = None,
    show: bool = False,
):
    """
    Plots the comparison of Real and Imaginary parts of the wave field side by side.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Real Part
    ax1.plot(x, np.real(psi_theory), "k-", lw=1.5, label=label_curve2)
    ax1.plot(x, np.real(psi_num), "r--", ms=6, label=label_curve1)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(r"Re($\phi$)")
    ax1.set_title(f"{title_prefix} - Real Part")
    ax1.legend(fontsize="small")
    ax1.grid(True, alpha=0.3)

    # Imaginary Part
    ax2.plot(x, np.imag(psi_theory), "k-", lw=1.5, label=label_curve2)
    ax2.plot(x, np.imag(psi_num), "r--", ms=6, label=label_curve1)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(r"Im($\phi$)")
    ax2.set_title(f"{title_prefix} - Imaginary Part")
    ax2.legend(fontsize="small")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, transparent=True)

    if show:
        plt.show()

    return fig, (ax1, ax2)
