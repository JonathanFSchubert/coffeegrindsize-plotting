"""
Notes:
    - All plot / smoothing parameters are defined in the PARAMETERS section below
"""

import sys
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.lines as mlines

try:
    from KDEpy import FFTKDE
except Exception:
    FFTKDE = None


# ----------------------------
# PARAMETERS
# ----------------------------
OUTPUT_PNG = "plot.png"

# KDE parameters (these operate in log10(mm) space)

# "ISJ": use KDEpy's Improved Sheather-Jones (solve-the-equation style) selector
# None: Silverman rule-of-thumb
# otherwise use a number between 0 and 1 (try for example 0.035)
# i recommend using "ISJ" for acuracy with a big sample size. for smaller sample sizes it might be more helpful to use None (easier to see general trends)
DEFAULT_BANDWIDTH = 0.035

GRID_POINTS = 800          # number of x points to evaluate KDE

# Plot appearance
FIGSIZE_KDE = (11, 6)
COLOR_CYCLE = None  # None -> use matplotlib default cycle

# Memory safety: internal chunking for KDE computation (set None to auto-choose)
KDE_SAMPLE_CHUNK = None


# ----------------------------
# SUBSAMPLED KDE (optional)
# ----------------------------
SHOW_SUBSAMPLED_KDE = True         # True = overlay subsampled KDEs
SUBSAMPLE_FRACTION = 0.8           # fraction of particles to use per subsample
SUBSAMPLED_COLOR_ALPHA = 0.10      # transparency for subsampled KDE lines
SUBSAMPLED_REPEAT = 20             # how many times to randomly sample and plot




# ----------------------------
# Utility functions
# ----------------------------
def load_csv_sizes_volumes(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load SHORT_AXIS, VOLUME, PIXEL_SCALE and return sizes_mm, volumes arrays."""
    df = pd.read_csv(path)
    required = ("SHORT_AXIS", "VOLUME", "PIXEL_SCALE")
    for c in required:
        if c not in df.columns:
            raise ValueError(f"CSV {path!r} missing required column {c!r}")
    df = df[["SHORT_AXIS", "VOLUME", "PIXEL_SCALE"]].dropna()
    sizes_mm = df["SHORT_AXIS"].to_numpy(dtype=float) / df["PIXEL_SCALE"].to_numpy(dtype=float)
    volumes = df["VOLUME"].to_numpy(dtype=float)
    mask = (sizes_mm > 0) & (volumes >= 0)
    sizes_mm = sizes_mm[mask]
    volumes = volumes[mask]
    if sizes_mm.size == 0:
        raise ValueError(f"No positive SHORT_AXIS values found in {path!r}")
    return sizes_mm, volumes


def silverman_bandwidth(samples_log: np.ndarray) -> float:
    """Silverman rule-of-thumb for bandwidth (applied to log10-space samples)."""
    n = len(samples_log)
    if n <= 1:
        return 0.1
    std = np.std(samples_log, ddof=1)
    iqr = np.subtract(*np.percentile(samples_log, [75, 25]))
    sigma = min(std, iqr / 1.349) if iqr > 0 else std
    if sigma <= 0:
        sigma = max(std, 1e-3)
    bw = 1.06 * sigma * (n ** (-1 / 5))
    return float(bw)


def compute_isj_bandwidth(values_log: np.ndarray) -> float:
    """
    Compute an ISJ / Sheather-Jones style bandwidth (in log10 units) using KDEpy FFTKDE.
    Returns a float bandwidth.

    - For very small samples -> fallback to Silverman for stability.
    """
    if FFTKDE is None:
        raise ImportError(
            "KDEpy is required to compute ISJ bandwidth. Install it with: pip install KDEpy"
        )
    values = np.asarray(values_log, dtype=float)
    n = values.size
    if n <= 8:
        # ISJ tends to need a reasonable sample size; fallback to Silverman
        return silverman_bandwidth(values)

    # Fit KDEpy's FFTKDE with bw='ISJ' (it will compute an internal numeric bw)
    kde = FFTKDE(bw="ISJ", kernel="gaussian")
    kde.fit(values)
    # calling evaluate forces computation and sets kde.bw in KDEpy implementations
    try:
        # use a small evaluation (default grid) just to force bandwidth computation
        kde.evaluate(256)
    except Exception:
        # fallback to evaluate() without explicit points if above fails
        kde.evaluate()
    bw = getattr(kde, "bw", None)
    if bw is None:
        raise RuntimeError("KDEpy did not return a numeric bandwidth (kde.bw is None).")
    return float(bw)


def numpy_weighted_gaussian_kde_logspace(
    values_mm: np.ndarray,
    weights: np.ndarray,
    x_grid_mm: np.ndarray,
    bandwidth: float | None = None,
    chunk_size: int | None = None,
) -> np.ndarray:
    """
    Weighted Gaussian KDE in log10(size) space

    Returns density per dex evaluated at x_grid_mm (same interpretation as KDE in log10-space).
    """
    values_log = np.log10(values_mm).astype(float)
    grid_log = np.log10(x_grid_mm).astype(float)

    w = np.asarray(weights, dtype=float)
    if w.sum() <= 0:
        raise ValueError("weights must sum to > 0")
    w = w / w.sum()

    if isinstance(bandwidth, str):
        method = bandwidth.lower()
        if method in ("isj"):
            bandwidth = compute_isj_bandwidth(values_log)
        else:
           # try sensible conversion or error
            try:
                bandwidth = float(bandwidth)
            except Exception:
               raise ValueError(f"Unknown bandwidth method string: {bandwidth!r}")

    if bandwidth is None:
        bandwidth = silverman_bandwidth(values_log)
        if bandwidth <= 0:
            bandwidth = 0.1
    h = float(bandwidth)
    norm_factor = 1.0 / (np.sqrt(2.0 * np.pi) * h)
    n = values_log.size
    m = grid_log.size

    if chunk_size is None:
        if n * m < 5e7:
            chunk_size = n
        else:
            chunk_size = max(1000, int(5e7 // m))

    dens = np.zeros(m, dtype=float)
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        vals_block = values_log[start:end]
        w_block = w[start:end]
        diff = (grid_log[:, None] - vals_block[None, :]) / h  # shape (m, b)
        contrib = np.exp(-0.5 * (diff ** 2))
        dens += (contrib * w_block[None, :]).sum(axis=1)
    dens *= norm_factor

    return dens



def interp_density_at_x(x_grid_mm: np.ndarray, density_per_dex: np.ndarray, x_mm: float) -> float:
    """Interpolate density (per dex) at a given x_mm (simple linear interp in log10 domain)."""
    log_grid = np.log10(x_grid_mm)
    log_x = np.log10(x_mm)
    return float(np.interp(log_x, log_grid, density_per_dex))


def fraction_volume_in_relative_tile(sizes_mm: np.ndarray, volumes: np.ndarray, center: float, rel_width: float) -> float:
    """Fraction of total volume inside [center*(1 - rel/2), center*(1 + rel/2)]."""
    half = rel_width / 2.0
    low = center * (1.0 - half)
    high = center * (1.0 + half)
    mask = (sizes_mm >= low) & (sizes_mm < high)
    total = float(volumes.sum())
    if total <= 0:
        return 0.0
    return float(volumes[mask].sum() / total)


def calculate_d10_raw_data(sizes_mm: np.ndarray, volumes: np.ndarray) -> float:
    """
    Calculate D10 directly from raw particle data (volume-weighted).
    Returns the particle diameter (mm) below which 10% of the total volume lies.
    """
    if sizes_mm.size == 0 or volumes.sum() <= 0:
        return np.nan

    # sort by size
    order = np.argsort(sizes_mm)
    sizes_sorted = sizes_mm[order]
    vol_sorted = volumes[order]

    # cumulative volume fraction
    cum_vol = np.cumsum(vol_sorted)
    frac = cum_vol / cum_vol[-1]

    # find where it crosses 0.10
    return float(np.interp(0.10, frac, sizes_sorted))


def calculate_d10_KDE(
    sizes_mm: np.ndarray,
    volumes: np.ndarray,
    x_grid_mm: np.ndarray,
    bandwidth: float | str | None = None,
) -> float:
    """
    Calculate D10 from the KDE-smoothed volume distribution.
    Uses numerical integration of the KDE curve (volume-weighted).
    """
    if sizes_mm.size == 0 or volumes.sum() <= 0:
        return np.nan

    # compute KDE for volume weights
    dens_vol = numpy_weighted_gaussian_kde_logspace(
        sizes_mm, volumes, x_grid_mm, bandwidth=bandwidth
    )

    # integrate density in log-space
    log_x = np.log10(x_grid_mm)
    dx = np.diff(log_x)
    # trapezoidal integration
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (dens_vol[1:] + dens_vol[:-1]) * dx)])
    cum /= cum[-1]  # normalize to [0,1]

    # find D10 in mm
    return float(np.interp(0.10, cum, x_grid_mm))



# ----------------------------
# Main plotting logic
# ----------------------------
def plot_multiple_files(file_paths: List[str]) -> None:
    
    print("Bigger differce in KDE vs raw indicate that the KDE might be not as acurate")
    
    # set color cycle
    if COLOR_CYCLE is not None:
        plt.rcParams["axes.prop_cycle"] = plt.cycler(color=COLOR_CYCLE)

    # load all data
    datasets = []
    for p in file_paths:
        sizes_mm, volumes = load_csv_sizes_volumes(p)
        datasets.append((p, sizes_mm, volumes))

    # combined x grid
    all_sizes = np.hstack([sizes for (_, sizes, _) in datasets])
    x_min = max(all_sizes.min() * 0.9, 1e-6)
    x_max = all_sizes.max() * 1.1
    x_grid = np.logspace(np.log10(x_min), np.log10(x_max), GRID_POINTS)

    # create KDE plot
    fig, ax = plt.subplots(figsize=FIGSIZE_KDE)
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", None)

    d10_summary = []  # store (basename, D10_mm, tile_frac) for tile scatter

    for i, (path, sizes_mm, volumes) in enumerate(datasets):
        basename = os.path.basename(path)
        color = colors[i % len(colors)] if colors else None

        # weights
        n = len(sizes_mm)
        num_weights = np.ones(n, dtype=float)
        vol_weights = volumes.astype(float).copy()
        if vol_weights.sum() <= 0:
            raise ValueError(f"Total volume <= 0 in file {path!r}")

        # compute KDEs (density per dex)
        dens_num = numpy_weighted_gaussian_kde_logspace(
            sizes_mm, num_weights, x_grid, bandwidth=DEFAULT_BANDWIDTH, chunk_size=KDE_SAMPLE_CHUNK
        )
        dens_vol = numpy_weighted_gaussian_kde_logspace(
            sizes_mm, vol_weights, x_grid, bandwidth=DEFAULT_BANDWIDTH, chunk_size=KDE_SAMPLE_CHUNK
        )

        # SUBSAMPLE KDE's
        if SHOW_SUBSAMPLED_KDE and sizes_mm.size > 1:
            for r in range(SUBSAMPLED_REPEAT):
                n_sub = max(2, int(len(sizes_mm) * SUBSAMPLE_FRACTION))
                subsample_indices = np.random.choice(len(sizes_mm), n_sub, replace=False)
                sizes_sub = sizes_mm[subsample_indices]
                volumes_sub = volumes[subsample_indices]

                # compute KDEs for subsample
                dens_num_sub = numpy_weighted_gaussian_kde_logspace(
                    sizes_sub, np.ones(n_sub), x_grid, bandwidth=DEFAULT_BANDWIDTH, chunk_size=KDE_SAMPLE_CHUNK
                )
                dens_vol_sub = numpy_weighted_gaussian_kde_logspace(
                    sizes_sub, volumes_sub, x_grid, bandwidth=DEFAULT_BANDWIDTH, chunk_size=KDE_SAMPLE_CHUNK
                )

                # plot pale lines with same color as main line
                ax.plot(x_grid, dens_num_sub, linestyle="--", linewidth=1.5, color=color, alpha=SUBSAMPLED_COLOR_ALPHA)
                ax.plot(x_grid, dens_vol_sub, linestyle="-", linewidth=1.5, color=color, alpha=SUBSAMPLED_COLOR_ALPHA)

        d10_raw = calculate_d10_raw_data(sizes_mm, volumes)
        d10_kde = calculate_d10_KDE(sizes_mm, volumes, x_grid, bandwidth=DEFAULT_BANDWIDTH)
        print(f"{basename}: D10 raw = {d10_raw:.3g} mm, D10 KDE = {d10_kde:.3g} mm")
        percent_diff = 100.0 * (d10_kde - d10_raw) / d10_raw if d10_raw != 0 else np.nan
        print(f"{basename}: % difference (KDE vs raw) = {percent_diff:.2f}%")

        # plotting of the D10
        offset_step = 0.1
        offset = (i - (len(datasets) - 1) / 2) * offset_step
        y_pos = 1.42 + offset

        # plot raw D10 marker
        ax.scatter(
            [d10_raw], [y_pos],
            color=color,
            marker="o", edgecolor="black", zorder=5
        )

        # plot KDE D10 marker
        ax.scatter(
            [d10_kde], [y_pos],
            color=color,
            marker="x", s=80, zorder=5
        )

        # plot: solid = number, dashed = volume
        ax.plot(x_grid, dens_num, linestyle="--", linewidth=1.5, label=f"{basename} — number", color=color)
        ax.plot(x_grid, dens_vol, linestyle="-", linewidth=1.5, label=f"{basename} — volume", color=color)


    # format KDE axis

    raw_marker = mlines.Line2D([], [], color="black", marker="o", linestyle="None", label="D10 raw")
    kde_marker = mlines.Line2D([], [], color="black", marker="x", linestyle="None", label="D10 KDE")
    
    handles, labels = ax.get_legend_handles_labels()
    handles.extend([raw_marker, kde_marker])
    labels.extend([raw_marker.get_label(), kde_marker.get_label()])
    ax.legend(handles=handles, labels=labels, fontsize=8)

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.ticklabel_format(axis="x", style="plain")
    ax.set_xlabel("Short diameter (mm)")
    ax.set_ylabel("Fraction per decade (fraction / dex)")
    ax.set_title("Particle size distributions")
    ax.grid(which="both", linestyle=":", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300)
    print(f"Saved plot: {OUTPUT_PNG}")


# ----------------------------
# CLI entrypoint
# ----------------------------
def main(argv: List[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) < 1:
        print("Usage: python plot_psd_multi.py file1.csv file2.csv ...")
        return
    file_paths = argv
    # sanity check file existence
    for p in file_paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"File not found: {p}")
    plot_multiple_files(file_paths)


if __name__ == "__main__":
    main()
