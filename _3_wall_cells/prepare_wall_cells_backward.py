#!/usr/bin/env python3

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


# ============================================================
# PATHS
# ============================================================
GP_ROOT = Path("/homes/aiskhak/projects/GP").resolve()

CASE_DIR = GP_ROOT / "1_mixing_length" / "backward"
OUT_DIR = GP_ROOT / "3_wall_cells" / "backward"

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# UTILITIES
# ============================================================
def latest_file(directory: Path, pattern: str) -> Path:
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {directory}")
    return files[-1]


def read_elv_csv(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)

    if data.dtype.names is None:
        raise RuntimeError(f"Could not read named columns from {path}")

    required = ["id", "x", "y", "z", "yw_aux_var", "elvol_aux_var"]
    for name in required:
        if name not in data.dtype.names:
            raise RuntimeError(
                f"Required column '{name}' not found in {path}. "
                f"Available columns: {data.dtype.names}"
            )

    return data


def find_wall_cell_threshold(yw: np.ndarray) -> float:
    """
    Detect threshold separating the first near-wall cell band from the next layer.

    For the current uniform BFS mesh, this should identify the first wall-adjacent
    cell layer, e.g. yw ≈ 0.125 if dy = 0.25.
    """
    yw_valid = np.sort(yw[np.isfinite(yw) & (yw > 0.0)])

    if yw_valid.size < 3:
        raise RuntimeError("Not enough positive finite yw values.")

    # Unique values are better here because many cells share the same yw.
    yw_unique = np.unique(np.round(yw_valid, decimals=14))

    if yw_unique.size < 2:
        raise RuntimeError("Could not detect multiple wall-distance layers.")

    ratios = yw_unique[1:] / yw_unique[:-1]
    idx = int(np.argmax(ratios))

    lower = yw_unique[idx]
    upper = yw_unique[idx + 1]

    threshold = float(np.sqrt(lower * upper))
    return threshold


def main():
    elv_file = latest_file(CASE_DIR, "bfs_2d_fv_csv_elv_*.csv")

    print("Reading ELV:", elv_file)

    data = read_elv_csv(elv_file)

    x = np.asarray(data["x"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=np.float64)
    z = np.asarray(data["z"], dtype=np.float64)
    ids = np.asarray(data["id"], dtype=np.int64)

    yw = np.asarray(data["yw_aux_var"], dtype=np.float64)
    vol = np.asarray(data["elvol_aux_var"], dtype=np.float64)

    if np.any(~np.isfinite(yw)):
        raise RuntimeError("Non-finite yw values found.")

    if np.any(yw < 0.0):
        raise RuntimeError("Negative yw values found.")

    if np.any(vol <= 0.0):
        raise RuntimeError("Non-positive element volume/area values found.")

    threshold = find_wall_cell_threshold(yw)

    wall_mask = (yw > 0.0) & (yw < threshold)

    if not np.any(wall_mask):
        raise RuntimeError("No wall cells detected.")

    wall_yw = yw[wall_mask]
    wall_vol = vol[wall_mask]

    # For 2D FV mesh, VolumeAux is area. A simple grid-size measure is sqrt(area).
    h_cell = np.sqrt(vol)
    h_wall = h_cell[wall_mask]

    # For this BFS geometry, the largest possible wall distance in the fluid is O(0.5)
    # for each channel half-height, but for consistency with the functional closure
    # use max(yw) from this MOOSE mesh.
    L_outer = float(np.max(yw))

    eta_h_wall = h_wall / L_outer

    wall_cells_path = OUT_DIR / "wall_cells.csv"

    with wall_cells_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x", "y", "z", "yw", "elvol", "h_cell", "eta_h"])

        for i in np.where(wall_mask)[0]:
            writer.writerow([
                int(ids[i]),
                f"{x[i]:.16e}",
                f"{y[i]:.16e}",
                f"{z[i]:.16e}",
                f"{yw[i]:.16e}",
                f"{vol[i]:.16e}",
                f"{h_cell[i]:.16e}",
                f"{h_cell[i] / L_outer:.16e}",
            ])

    sorted_yw = np.sort(np.unique(np.round(yw[yw > 0.0], decimals=14)))
    below = sorted_yw[sorted_yw < threshold]
    above = sorted_yw[sorted_yw >= threshold]

    largest_below = float(below[-1]) if below.size else np.nan
    smallest_above = float(above[0]) if above.size else np.nan
    gap_ratio = smallest_above / largest_below if largest_below > 0.0 else np.nan

    summary = {
        "case": "backward",
        "elv_file": str(elv_file),
        "wall_cells_file": str(wall_cells_path),
        "n_total_cells": int(yw.size),
        "n_wall_cells": int(wall_yw.size),
        "threshold": float(threshold),
        "yw_min": float(np.min(yw)),
        "yw_max": float(np.max(yw)),
        "L_outer": L_outer,
        "yw_wall_min": float(np.min(wall_yw)),
        "yw_wall_mean": float(np.mean(wall_yw)),
        "yw_wall_median": float(np.median(wall_yw)),
        "yw_wall_max": float(np.max(wall_yw)),
        "h_wall_min": float(np.min(h_wall)),
        "h_wall_mean": float(np.mean(h_wall)),
        "h_wall_median": float(np.median(h_wall)),
        "h_wall_max": float(np.max(h_wall)),
        "eta_h_wall_min": float(np.min(eta_h_wall)),
        "eta_h_wall_mean": float(np.mean(eta_h_wall)),
        "eta_h_wall_median": float(np.median(eta_h_wall)),
        "eta_h_wall_max": float(np.max(eta_h_wall)),
        "largest_yw_below_threshold": largest_below,
        "smallest_yw_above_threshold": smallest_above,
        "gap_ratio_next_over_wall": float(gap_ratio),
    }

    summary_path = OUT_DIR / "wall_cells_summary.csv"

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print()
    print("===== Wall-cell summary =====")
    print("n total cells:       ", summary["n_total_cells"])
    print("n wall cells:        ", summary["n_wall_cells"])
    print("threshold:           ", f"{summary['threshold']:.8e}")
    print("yw wall median:      ", f"{summary['yw_wall_median']:.8e}")
    print("h wall median:       ", f"{summary['h_wall_median']:.8e}")
    print("L_outer=max(yw):     ", f"{summary['L_outer']:.8e}")
    print("eta_h wall median:   ", f"{summary['eta_h_wall_median']:.8e}")
    print("gap ratio:           ", f"{summary['gap_ratio_next_over_wall']:.8e}")
    print()
    print("Wrote:", wall_cells_path)
    print("Wrote:", summary_path)


if __name__ == "__main__":
    main()