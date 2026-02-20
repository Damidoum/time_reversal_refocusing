import dataclasses

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from time_reversal.config import SimulationConfig
from time_reversal.simulation import run_single_simulation
from time_reversal.viz import setup_style

matplotlib.use("Agg")


def setup_frequency_grid(
    w_central: float, delta_omega: float, n_freq: int
) -> np.ndarray:
    """
    Generates the frequency grid and spectral weights for a Gaussian pulse.

    Args:
        w_central: Central angular frequency (omega_0).
        delta_omega: interval of freq [w0 - delta_omega, w0 + delta_omega]
        n_freq: Number of frequency points.

    Returns:
        omegas: Array of angular frequencies.
    """
    omegas = np.linspace(w_central - delta_omega, w_central + delta_omega, n_freq)
    return omegas


def simulate_frequencies(
    base_cfg: SimulationConfig,
    omegas: np.ndarray,
) -> np.ndarray:
    """
    Runs monochromatic simulations for each frequency in the grid.

    Args:
        base_cfg: Base configuration (spatial params, etc.).
        omegas: Array of frequencies to simulate.

    Returns:
        final_fields: Complex field array of shape (n_freq, nz, nx).
    """
    final_fields_per_freq = []

    for omega in tqdm.tqdm(omegas, desc="  Simulating Frequencies"):
        # Update config with new frequency
        cfg_freq = dataclasses.replace(base_cfg, w=omega)

        # Run simulation
        res = run_single_simulation(cfg_freq, return_history=True, use_fast_solver=True)
        assert (
            res.history is not None
        ), "Expected history to be returned for simulation."

        modulation_factor = np.exp(
            1j * cfg_freq.k_const * np.arange(res.history.shape[0]) * cfg_freq.h
        )

        final_fields_per_freq.append(res.history * modulation_factor[:, np.newaxis])

    return np.array(final_fields_per_freq)


def recover_temporal_profile(
    final_fields_per_freq: np.ndarray,
    omegas: np.ndarray,
    t_grid: np.ndarray,
) -> np.ndarray:
    dw = omegas[1] - omegas[0] if len(omegas) > 1 else 0.0
    modulation_factor = np.exp(-1j * t_grid[:, np.newaxis] * omegas[np.newaxis, :])
    temporal_signal_complex = np.einsum(
        "tf, fzx -> tzx", modulation_factor, final_fields_per_freq
    )
    temporal_signal = (dw / (2 * np.pi)) * temporal_signal_complex
    return temporal_signal


def run_simulation(
    base_cfg: SimulationConfig,
    n_freq: int,
    delta_omega: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Orchestrates the broadband time reversal simulation.
    """
    print(f"Running Broadband Simulation: N={n_freq}")

    omegas = setup_frequency_grid(base_cfg.w, delta_omega, n_freq)
    final_fields_per_freq = simulate_frequencies(base_cfg, omegas)

    t_grid = np.linspace(0, base_cfg.t_max, base_cfg.nt)

    time_field = recover_temporal_profile(
        final_fields_per_freq,
        omegas,
        t_grid,
    )

    return t_grid, time_field


def animate_comparison_split(
    forward_signal: np.ndarray,
    backward_signal: np.ndarray,
    save_filename: str = "tr_educational.mp4",
    t_skip: int = 1,
    x_skip: int = 1,
    fps: int = 30,
):

    f_data = forward_signal[::t_skip, :, ::x_skip]
    b_data = backward_signal[::t_skip, :, ::x_skip]
    rewind_data = np.flip(f_data, axis=0)

    n_frames_prop = f_data.shape[0]
    pause_frames = fps  # 2 seconds pause

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
            title1.set_text("STEP 1: FORWARD\n(Signal recorded by mirror)")
            title2.set_text("Waiting...")

        elif frame < n_frames_prop + pause_frames:
            im1.set_array(f_data[-1])  # Stay at mirror
            im2.set_array(np.flip(b_data[0], 0))
            title1.set_text("--- RECORDING COMPLETE ---")
            title2.set_text("--- TIME REVERSAL PROCESS ---")

        else:
            idx = frame - (n_frames_prop + pause_frames)
            im1.set_array(rewind_data[idx])
            im2.set_array(np.flip(b_data[idx], 0))
            title1.set_text("MOVIE REWIND\n(Pure Math / Inverted Film)")
            title2.set_text("TIME REVERSAL\n(Physical Wave converging)")

        return [im1, im2, title1, title2]

    ani = animation.FuncAnimation(fig, update, frames=total_frames, blit=True)

    ani.save(
        save_filename,
        fps=fps,
        writer="ffmpeg",
        extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"],
    )
    plt.close(fig)


def animate_sigma_grid(
    results_dict: dict,  # {sigma_value: {"forward": signal, "backward": signal}}
    save_filename: str = "output/comparison_sigma_grid.mp4",
    t_skip: int = 1,
    x_skip: int = 1,
    fps: int = 30,
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
        rewind = np.flip(f, axis=0)
        all_data.append((f, rewind, b))

    n_frames_prop = all_data[0][0].shape[0]
    pause_frames = fps
    total_frames = n_frames_prop + pause_frames + n_frames_prop

    # Dynamic scaling: adjust figsize based on number of sigmas
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows))
    plt.subplots_adjust(hspace=0.3, wspace=0.1)

    ims_left = []
    ims_right = []

    # Global max amplitude across all sigmas for a consistent scale
    global_max = max(np.max(np.abs(d[0])) for d in all_data) * 0.15

    for i, sigma in enumerate(sigmas):
        # Setup each row
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

    # Main title titles
    axes[0, 0].set_title("FORWARD / REWIND", fontsize=16)
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
                ims_left[i].set_array(all_data[i][1][0])  # Stay at mirror
                ims_right[i].set_array(all_data[i][2][0])
        elif frame < n_frames_prop + pause_frames:
            pass
        else:
            idx = frame - (n_frames_prop + pause_frames)
            status_text.set_text("REWIND (Left) vs PHYSICS (Right)")
            for i in range(n_rows):
                ims_left[i].set_array(all_data[i][1][idx])
                ims_right[i].set_array(all_data[i][2][idx])

        return ims_left + ims_right + [status_text]

    ani = animation.FuncAnimation(fig, update, frames=total_frames, blit=True)

    ani.save(
        save_filename,
        fps=fps,
        dpi=80,
        writer="ffmpeg",
        extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"],
    )
    plt.close(fig)


def animate_wavefield_mp4(
    temporal_signal: np.ndarray,
    save_filename: str = "time_reversal.mp4",
    t_skip: int = 1,
    space_skip: int = 1,
    fps: int = 30,
):
    """
    Creates and saves an MP4 animation of the 3D wavefield (time, z, x).

    Args:
        temporal_signal: 3D array of shape (nt, nz, nx).
        save_filename: Path to save the .mp4 file.
        t_skip: Subsampling factor for time (1 = all frames, 10 = every 10th frame).
        space_skip: Subsampling factor for spatial axes to reduce file size.
        fps: Frames per second for the output video.
    """
    data = temporal_signal[::t_skip, ::space_skip, ::space_skip]
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
    ax.set_xlabel(f"X-axis (subsampled by {space_skip})")
    ax.set_ylabel(f"Z-axis (subsampled by {space_skip})")
    fig.colorbar(im, ax=ax, label="Field Amplitude")

    def update(frame):
        im.set_array(data[frame])

        if frame % max(1, nt // 10) == 0:
            print(f"Rendering: {frame}/{nt} frames ({(frame/nt)*100:.0f}%)")
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=nt, blit=True)

    ani.save(
        save_filename,
        fps=fps,
        writer="ffmpeg",
        extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"],
    )

    plt.close(fig)


def main():
    setup_style()

    nx = 2**10
    L = 10.0
    x_size = 60.0
    c0 = 1.0
    w_central = 1.0
    delta_omega = 1.5

    n_freq = 150
    sigmas = [0.0, 0.5, 1.0, 2.0, 5.0]

    base_cfg = SimulationConfig(
        nx=nx,
        L=L,
        x_size=x_size,
        w=w_central,
        c0=c0,
        r0=1,
        r_m=10,
        z_c=1.0,
        x_c=1.0,
        sigma=0.0,
        h=0.05,
        t_max=20.0,
    )

    results_dict = {}

    for sigma in sigmas:
        print(f"\n=== Running Simulation for sigma = {sigma} ===")
        base_cfg = dataclasses.replace(base_cfg, sigma=sigma)
        _, time_field = run_simulation(base_cfg, n_freq=n_freq, delta_omega=delta_omega)
        nz = time_field.shape[1]
        intensity = time_field.real  # type: ignore
        results_dict[sigma] = {
            "forward": intensity[:, : int(nz / 2), :],
            "backward": intensity[:, int(nz / 2) + 1 :, :],
        }
        # animate_comparison_split(
        #     intensity[:, : int(nz / 2), :],
        #     intensity[:, int(nz / 2) + 1 :, :],
        #     f"output/time_reversal_split_sigma_{sigma}.mp4",
        #     fps=30,
        #     t_skip=5,
        #     x_skip=5,
        # )

    animate_sigma_grid(
        results_dict,
        f"data/2comparison_sigma_grid_{base_cfg.r_m}_{base_cfg.r0}.mp4",
        fps=30,
        t_skip=5,
        x_skip=5,
    )


if __name__ == "__main__":
    main()
