#!/usr/bin/env python3
# Example:
# python prepare_wall_cells.py --gp-root .. --re 3000 --cgs 1 2 3 4

"""
Prepare near-wall cell data for GP grid-resolution closure.

This script:
  1. Goes through 1_mixing_length/<RE>/<CG>/ for selected CG grids.
  2. Finds the latest tamu_2d_fv_csv_elv_*.csv file.
  3. Reads element id, coordinates, and corrected yw_aux_var.
  4. Detects the first near-wall cell band using the largest gap in sorted yw.
  5. Saves:
       - wall_cells_CG*.csv:
           id, x, y, z, yw
       - wall_cells_median_summary.csv:
           RE, CG, threshold, n_wall_cells, yw_median, etc.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def latest_file(directory: Path, pattern: str) -> Path:
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {directory}")
    return files[-1]


def read_elv_csv(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)

    if data.dtype.names is None:
        raise RuntimeError(f"Could not read named columns from {path}")

    required = ["id", "x", "y", "z", "yw_aux_var"]
    for name in required:
        if name not in data.dtype.names:
            raise RuntimeError(
                f"Required column '{name}' not found in {path}. "
                f"Available columns: {data.dtype.names}"
            )

    return data


def find_wall_cell_threshold(yw: np.ndarray) -> float:
    """
    Detect the threshold separating the first near-wall cell band
    from the next layer.

    Method:
      - keep positive finite wall distances
      - sort them
      - find the largest ratio jump
      - set threshold as the geometric mean across that jump
    """
    yw_valid = np.sort(yw[np.isfinite(yw) & (yw > 0.0)])

    if yw_valid.size < 3:
        raise RuntimeError("Not enough positive finite yw values to detect wall-cell threshold.")

    ratios = yw_valid[1:] / yw_valid[:-1]
    idx = int(np.argmax(ratios))

    lower = yw_valid[idx]
    upper = yw_valid[idx + 1]

    threshold = float(np.sqrt(lower * upper))
    return threshold


def write_wall_cells(output_path: Path, data: np.ndarray, wall_mask: np.ndarray):
    """
    Save only wall-cell information needed later:
      id, x, y, z, yw
    """
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x", "y", "z", "yw"])

        for i in np.where(wall_mask)[0]:
            writer.writerow([
                int(data["id"][i]),
                f"{data['x'][i]:.16e}",
                f"{data['y'][i]:.16e}",
                f"{data['z'][i]:.16e}",
                f"{data['yw_aux_var'][i]:.16e}",
            ])


def process_cg(gp_root: Path, re: str, cg: str, output_dir: Path) -> dict:
    case_dir = gp_root / "1_mixing_length" / re / cg
    elv_file = latest_file(case_dir, "tamu_2d_fv_csv_elv_*.csv")

    data = read_elv_csv(elv_file)
    yw = np.asarray(data["yw_aux_var"], dtype=np.float64)

    if np.any(~np.isfinite(yw)):
        raise RuntimeError(f"Non-finite yw values found in {elv_file}")

    if np.any(yw < 0.0):
        raise RuntimeError(f"Negative yw values found in {elv_file}")

    threshold = find_wall_cell_threshold(yw)

    wall_mask = (yw > 0.0) & (yw < threshold)
    wall_yw = yw[wall_mask]

    if wall_yw.size == 0:
        raise RuntimeError(f"No wall cells detected for RE={re}, CG={cg}")

    wall_cells_path = output_dir / f"wall_cells_CG{cg}.csv"
    write_wall_cells(wall_cells_path, data, wall_mask)

    sorted_yw = np.sort(yw[yw > 0.0])
    below = sorted_yw[sorted_yw < threshold]
    above = sorted_yw[sorted_yw >= threshold]

    largest_below = float(below[-1]) if below.size else np.nan
    smallest_above = float(above[0]) if above.size else np.nan
    gap_ratio = smallest_above / largest_below if largest_below > 0.0 else np.nan

    summary = {
        "RE": re,
        "CG": cg,
        "elv_file": str(elv_file),
        "wall_cells_file": str(wall_cells_path),
        "n_total_cells": int(yw.size),
        "n_wall_cells": int(wall_yw.size),
        "threshold": float(threshold),
        "yw_wall_min": float(np.min(wall_yw)),
        "yw_wall_mean": float(np.mean(wall_yw)),
        "yw_wall_median": float(np.median(wall_yw)),
        "yw_wall_max": float(np.max(wall_yw)),
        "largest_yw_below_threshold": largest_below,
        "smallest_yw_above_threshold": smallest_above,
        "gap_ratio_next_over_wall": float(gap_ratio),
    }

    return summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gp-root",
        default=".",
        help="Path to GP root directory. Default: current directory.",
    )
    parser.add_argument(
        "--re",
        default="3000",
        help="Reynolds number directory used to extract grid data. Default: 3000.",
    )
    parser.add_argument(
        "--cgs",
        nargs="+",
        default=["1", "2", "3", "4"],
        help="Coarse grids to process. Default: 1 2 3 4.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: <gp-root>/3_wall_cells.",
    )

    args = parser.parse_args()

    gp_root = Path(args.gp_root).resolve()

    if args.output_dir is None:
        output_dir = gp_root / "3_wall_cells"
    else:
        output_dir = Path(args.output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"GP root:    {gp_root}")
    print(f"RE source:  {args.re}")
    print(f"CGs:        {args.cgs}")
    print(f"Output dir: {output_dir}")
    print()

    summaries = []

    for cg in args.cgs:
        print(f"===== Processing CG{cg} =====")

        summary = process_cg(
            gp_root=gp_root,
            re=args.re,
            cg=cg,
            output_dir=output_dir,
        )

        summaries.append(summary)

        print(f"  ELV file:          {summary['elv_file']}")
        print(f"  threshold:         {summary['threshold']:.8e}")
        print(f"  wall cells:        {summary['n_wall_cells']}")
        print(f"  yw wall median:    {summary['yw_wall_median']:.8e}")
        print(f"  gap ratio:         {summary['gap_ratio_next_over_wall']:.3e}")
        print(f"  wrote:             {summary['wall_cells_file']}")
        print()

    summary_path = output_dir / "wall_cells_median_summary.csv"

    fieldnames = [
        "RE",
        "CG",
        "elv_file",
        "wall_cells_file",
        "n_total_cells",
        "n_wall_cells",
        "threshold",
        "yw_wall_min",
        "yw_wall_mean",
        "yw_wall_median",
        "yw_wall_max",
        "largest_yw_below_threshold",
        "smallest_yw_above_threshold",
        "gap_ratio_next_over_wall",
    ]

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary)

    print("Wrote summary:")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()