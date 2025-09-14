"""
Notes:

Currently only a ChatGPT Slop. Will update the code in the future!

    - All plot / smoothing parameters are defined in the PARAMETERS section below
      (edit those constants directly if you want to change behavior).
    - The script intentionally avoids optional command-line parameters: you pass only the files
      as arguments.
"""

from __future__ import annotations

import sys
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# PARAMETERS (edit here)
# ----------------------------
OUTPUT_KDE_PNG = "particle_distributions.png"
OUTPUT_TILE_PNG = "D10_tile_fraction.png"

# KDE parameters (these operate in log10(mm) space)
DEFAULT_BANDWIDTH = 0.035   # None -> Silverman rule-of-thumb will be used; otherwise use a number between 0 and 1
GRID_POINTS = 800          # number of x points to evaluate KDE

# Tile around D10: relative width (e.g. 0.10 = ±5% around D10 => total width = 10% of D10).
TILE_REL_WIDTH = 0.10

# Plot appearance
FIGSIZE_KDE = (11, 6)
FIGSIZE_TILE = (6.5, 5)
COLOR_CYCLE = None  # None -> use matplotlib default cycle

# Memory safety: internal chunking for KDE computation (set None to auto-choose)
KDE_SAMPLE_CHUNK = None

# Units display (True: annotate D10 in µm)
ANNOTATE_D10_IN_MICRONS = True

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


def numpy_weighted_gaussian_kde_logspace(
    values_mm: np.ndarray,
    weights: np.ndarray,
    x_grid_mm: np.ndarray,
    bandwidth: float | None = None,
    chunk_size: int | None = None,
) -> np.ndarray:
    """
    Weighted Gaussian KDE in log10(size) space using numpy alone.

    Returns density per dex evaluated at x_grid_mm (same interpretation as KDE in log10-space).
    """
    values_log = np.log10(values_mm).astype(float)
    grid_log = np.log10(x_grid_mm).astype(float)

    w = np.asarray(weights, dtype=float)
    if w.sum() <= 0:
        raise ValueError("weights must sum to > 0")
    w = w / w.sum()

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


def compute_percentile_from_kde(x_grid_mm: np.ndarray, density_per_dex: np.ndarray, q: float) -> float:
    """
    Compute the diameter (mm) corresponding to quantile q (0 < q < 1) from KDE density
    defined on x_grid_mm (density is per dex). Integration done in log-space.
    """
    logx = np.log10(x_grid_mm)
    # trapezoidal CDF over log-space
    cdf = np.cumsum(0.5 * (density_per_dex[:-1] + density_per_dex[1:]) * np.diff(logx))
    cdf = np.concatenate(([0.0], cdf))
    if cdf[-1] <= 0:
        return float("nan")
    cdf = cdf / cdf[-1]
    q = float(q)
    if q <= 0:
        return x_grid_mm[0]
    if q >= 1:
        return x_grid_mm[-1]
    idx = np.searchsorted(cdf, q)
    if idx == 0:
        return x_grid_mm[0]
    if idx >= len(cdf):
        return x_grid_mm[-1]
    # linear interpolate in log-space between idx-1 and idx
    t = (q - cdf[idx - 1]) / (cdf[idx] - cdf[idx - 1])
    logv = np.log10(x_grid_mm[idx - 1]) * (1 - t) + np.log10(x_grid_mm[idx]) * t
    return 10 ** logv


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


# ----------------------------
# Main plotting logic
# ----------------------------
def plot_multiple_files(file_paths: List[str]) -> None:
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

        # plot: solid = number, dashed = volume
        ax.plot(x_grid, dens_num, linestyle="-", linewidth=1.5, label=f"{basename} — number", color=color)
        ax.plot(x_grid, dens_vol, linestyle="--", linewidth=1.5, label=f"{basename} — volume", color=color)

        # compute D10 (volume-based)
        D10_vol = compute_percentile_from_kde(x_grid, dens_vol, 0.10)
        d10_marker_y = interp_density_at_x(x_grid, dens_vol, D10_vol)

        # draw D10 marker & vertical line
        ax.axvline(D10_vol, color=color, linestyle=":", linewidth=0.9, alpha=0.9)
        ax.scatter([D10_vol], [d10_marker_y], marker="s", s=64, facecolors=color, edgecolors="k", zorder=6)

        # annotate D10 (in µm if requested)
        if ANNOTATE_D10_IN_MICRONS:
            ann_text = f"D10 = {D10_vol*1000:.0f} µm"
        else:
            ann_text = f"D10 = {D10_vol:.4f} mm"
        ax.annotate(
            ann_text,
            xy=(D10_vol, d10_marker_y),
            xytext=(D10_vol * 1.12, d10_marker_y * 0.95),
            fontsize=9,
            color=color,
            arrowprops=dict(arrowstyle="->", lw=0.7, color=color),
        )

        # compute tile fraction around D10 (relative)
        tile_frac = fraction_volume_in_relative_tile(sizes_mm, volumes, D10_vol, TILE_REL_WIDTH)
        d10_summary.append((basename, D10_vol, tile_frac))

    # format KDE axis
    ax.set_xscale("log")
    ax.set_xlabel("Short diameter (mm) — log scale")
    ax.set_ylabel("Fraction per decade (fraction / dex)")
    ax.set_title("Particle size distributions — KDE in log10(size)\nSolid=line: number, dashed=line: volume")
    ax.legend(fontsize=8)
    ax.grid(which="both", linestyle=":", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUTPUT_KDE_PNG, dpi=300)
    print(f"Saved KDE plot: {OUTPUT_KDE_PNG}")

    # print D10 summary
    print("\nD10 (volume) summary:")
    for name, d10_mm, frac in d10_summary:
        print(f" - {name}: D10 = {d10_mm*1000:.1f} µm, tile_frac (±{TILE_REL_WIDTH*50:.1f}%): {frac*100:.3f} %")


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
