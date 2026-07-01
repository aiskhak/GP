#!/usr/bin/env python3

import os
import fnmatch
from pathlib import Path

import numpy as np


# ============================================================
# PATHS
# ============================================================
GP_ROOT = Path("/homes/aiskhak/projects/GP").resolve()

TEST_ROOT = GP_ROOT / "test" / "backward"
MAPPED_FILE = GP_ROOT / "2_mapped" / "backward" / "mapped.csv"

# Candidate VPP patterns.
# Add more if your output names differ.
VPP_PATTERNS = [
    "bfs_2d_fv_gp_csv_vpp_*.csv",
    "bfs_2d_fv_csv_vpp_*.csv",
    "*vpp*.csv",
]


# ============================================================
# UTILITIES
# ============================================================
def read_named_csv(path: Path):
    with path.open("r") as f:
        header = f.readline().strip().split(",")

    header = [h.strip() for h in header]
    col = {name: i for i, name in enumerate(header)}

    data = np.loadtxt(path, delimiter=",", skiprows=1)

    return header, col, data


def find_latest_vpp_files(root: Path):
    """
    Find latest VPP in each immediate candidate folder.

    Example:
      test/backward/wall_only/...
      test/backward/wall_grid/...
    """
    candidates = []

    # First: if TEST_ROOT itself contains VPP files.
    direct = []
    for pat in VPP_PATTERNS:
        direct.extend(root.glob(pat))
    if direct:
        direct = sorted(set(direct), key=lambda p: p.stat().st_mtime)
        candidates.append(("backward", direct[-1]))

    # Then search subfolders.
    for sub in sorted([p for p in root.iterdir() if p.is_dir()]):
        files = []
        for dirpath, _, filenames in os.walk(sub):
            dirpath = Path(dirpath)
            for pat in VPP_PATTERNS:
                for name in fnmatch.filter(filenames, pat):
                    files.append(dirpath / name)

        if files:
            files = sorted(set(files), key=lambda p: p.stat().st_mtime)
            candidates.append((sub.name, files[-1]))

    return candidates


def nearest_map(x_src, y_src, x_ref, y_ref):
    """
    Robust nearest-neighbor alignment.

    For this backward case, ordering should often already match,
    but this protects us if candidate output order differs.
    """
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(np.column_stack((x_ref, y_ref)))
        dist, idx = tree.query(np.column_stack((x_src, y_src)), k=1)
        return idx, dist
    except Exception:
        idx = np.zeros(x_src.size, dtype=np.int64)
        dist = np.zeros(x_src.size, dtype=np.float64)

        for i in range(x_src.size):
            d2 = (x_src[i] - x_ref) ** 2 + (y_src[i] - y_ref) ** 2
            j = np.argmin(d2)
            idx[i] = j
            dist[i] = np.sqrt(d2[j])

        return idx, dist


def compute_x_region_errors(x, err2_base, err2_gp, dx_chunk=6.0):
    """
    Compute:
      1. cumulative errors for x < x_cut
      2. chunk errors over physically meaningful BFS intervals

    Uses:
      err2 = (u - u_dns)^2 + (v - v_dns)^2
    """

    # Custom BFS regions:
    #   -6..0       upstream inlet channel
    #    0..6       near-step separated region
    #    6..8.8     pre/near reattachment
    #    8.8..12    immediate post-reattachment recovery
    #    12..20     downstream recovery
    #    20..32     far downstream
    #    32..36     catches last MOOSE cell centers if needed
    edges = np.array([-6.0, 0.0, 6.0, 8.8, 12.0, 20.0, 32.0, 36.0])

    cumulative = []
    chunks = []

    # Cumulative: x < cutoff
    for x_cut in edges[1:]:
        mask = x < x_cut

        if np.any(mask):
            base_mse = float(np.mean(err2_base[mask]))
            gp_mse = float(np.mean(err2_gp[mask]))
            norm = gp_mse / base_mse if base_mse > 0.0 else np.nan

            cumulative.append({
                "x_cut": float(x_cut),
                "n_points": int(np.sum(mask)),
                "baseline_mse": base_mse,
                "gp_mse": gp_mse,
                "norm": norm,
            })

    # Chunks: x in [a,b)
    for a, b in zip(edges[:-1], edges[1:]):
        mask = (x >= a) & (x < b)

        if np.any(mask):
            base_mse = float(np.mean(err2_base[mask]))
            gp_mse = float(np.mean(err2_gp[mask]))
            norm = gp_mse / base_mse if base_mse > 0.0 else np.nan

            chunks.append({
                "x0": float(a),
                "x1": float(b),
                "n_points": int(np.sum(mask)),
                "baseline_mse": base_mse,
                "gp_mse": gp_mse,
                "norm": norm,
            })

    return cumulative, chunks

def compute_error_for_vpp(label: str, vpp_file: Path, mapped):
    # Read candidate VPP
    header, col, data = read_named_csv(vpp_file)

    required = ["x", "y", "u", "v"]
    for name in required:
        if name not in col:
            raise RuntimeError(
                f"Required column '{name}' not found in {vpp_file}. "
                f"Available columns: {header}"
            )

    x = data[:, col["x"]]
    y = data[:, col["y"]]
    u_gp = data[:, col["u"]]
    v_gp = data[:, col["v"]]

    # Optional GP mixing length column
    if "mixing_length_gp_aux_var" in col:
        lm_gp = data[:, col["mixing_length_gp_aux_var"]]
    elif "mixing_length_aux_var" in col:
        lm_gp = data[:, col["mixing_length_aux_var"]]
    else:
        lm_gp = np.full_like(u_gp, np.nan)

    # Read mapped DNS/reference
    x_ref = mapped["x_cg"]
    y_ref = mapped["y_cg"]
    u_dns = mapped["u_dns"]
    v_dns = mapped["v_dns"]
    u_base = mapped["u_baseline"]
    v_base = mapped["v_baseline"]

    idx, dist = nearest_map(x, y, x_ref, y_ref)

    u_dns_i = u_dns[idx]
    v_dns_i = v_dns[idx]
    u_base_i = u_base[idx]
    v_base_i = v_base[idx]

    du_gp = u_gp - u_dns_i
    dv_gp = v_gp - v_dns_i

    du_base = u_base_i - u_dns_i
    dv_base = v_base_i - v_dns_i

    err2_gp = du_gp**2 + dv_gp**2
    err2_base = du_base**2 + dv_base**2
    
    cumulative_x_errors, chunk_x_errors = compute_x_region_errors(
        x=x,
        err2_base=err2_base,
        err2_gp=err2_gp,
        dx_chunk=6.0,
    )

    mse_u = np.mean(du_gp**2)
    mse_v = np.mean(dv_gp**2)
    mse_uv = np.mean(err2_gp)

    # Save pointwise fields for ParaView
    out_detail = TEST_ROOT / f"{label}_dns_baseline_gp_fields.csv"

    detail = np.column_stack(
        (
            x,
            y,
            np.zeros_like(x),
            u_dns_i,
            v_dns_i,
            u_base_i,
            v_base_i,
            u_gp,
            v_gp,
            lm_gp,
            err2_base,
            err2_gp,
            err2_gp - err2_base,
            dist,
        )
    )

    np.savetxt(
        out_detail,
        detail,
        delimiter=",",
        header=(
            "x,y,z,"
            "u_dns,v_dns,"
            "u_baseline,v_baseline,"
            "u_gp,v_gp,"
            "mixing_length_gp,"
            "err2_baseline,err2_gp,err2_gp_minus_baseline,"
            "map_dist"
        ),
        comments="",
        fmt="%.12e",
    )
    
    # Save cumulative x-cutoff errors
    out_cumulative = TEST_ROOT / f"{label}_x_cumulative_errors.csv"

    with out_cumulative.open("w") as f:
        f.write("x_cut,n_points,baseline_mse,gp_mse,norm\n")
        for row in cumulative_x_errors:
            f.write(
                f"{row['x_cut']:.12e},"
                f"{row['n_points']},"
                f"{row['baseline_mse']:.16e},"
                f"{row['gp_mse']:.16e},"
                f"{row['norm']:.16e}\n"
            )

    # Save chunked x-interval errors
    out_chunks = TEST_ROOT / f"{label}_x_chunk_errors_dx6.csv"

    with out_chunks.open("w") as f:
        f.write("x0,x1,n_points,baseline_mse,gp_mse,norm\n")
        for row in chunk_x_errors:
            f.write(
                f"{row['x0']:.12e},"
                f"{row['x1']:.12e},"
                f"{row['n_points']},"
                f"{row['baseline_mse']:.16e},"
                f"{row['gp_mse']:.16e},"
                f"{row['norm']:.16e}\n"
            )

    return {
        "label": label,
        "vpp_file": str(vpp_file),
        "detail_file": str(out_detail),
        "cumulative_file": str(out_cumulative),
        "chunk_file": str(out_chunks),
        "n_points": int(x.size),
        "mse_u": mse_u,
        "mse_v": mse_v,
        "mse_uv": mse_uv,
        "map_dist_min": float(dist.min()),
        "map_dist_mean": float(dist.mean()),
        "map_dist_max": float(dist.max()),
        "map_dist_p95": float(np.percentile(dist, 95)),
        "u_min": float(u_gp.min()),
        "u_max": float(u_gp.max()),
        "v_min": float(v_gp.min()),
        "v_max": float(v_gp.max()),
    }


def main():
    if not MAPPED_FILE.exists():
        raise FileNotFoundError(f"Mapped file not found: {MAPPED_FILE}")

    print("TEST_ROOT:  ", TEST_ROOT)
    print("MAPPED_FILE:", MAPPED_FILE)
    print()

    # Load mapped.csv
    mapped_header, mapped_col, mapped_data = read_named_csv(MAPPED_FILE)

    needed = [
        "x_cg",
        "y_cg",
        "u_dns",
        "v_dns",
        "u_baseline",
        "v_baseline",
    ]

    for name in needed:
        if name not in mapped_col:
            raise RuntimeError(
                f"Required column '{name}' not found in mapped.csv. "
                f"Available columns: {mapped_header}"
            )

    mapped = {
        name: mapped_data[:, mapped_col[name]]
        for name in needed
    }

    # Baseline error from mapped.csv itself
    du_base = mapped["u_baseline"] - mapped["u_dns"]
    dv_base = mapped["v_baseline"] - mapped["v_dns"]

    baseline_mse_u = np.mean(du_base ** 2)
    baseline_mse_v = np.mean(dv_base ** 2)
    baseline_mse_uv = np.mean(du_base ** 2 + dv_base ** 2)

    print("===== Baseline from mapped.csv =====")
    print("MSE_u        =", f"{baseline_mse_u:.12e}")
    print("MSE_v        =", f"{baseline_mse_v:.12e}")
    print("MSE_uv       =", f"{baseline_mse_uv:.12e}")
    print()

    candidates = find_latest_vpp_files(TEST_ROOT)

    if not candidates:
        raise FileNotFoundError(f"No candidate VPP files found under {TEST_ROOT}")

    results = []

    for label, vpp_file in candidates:
        print(f"===== Candidate: {label} =====")
        print("VPP:", vpp_file)

        r = compute_error_for_vpp(label, vpp_file, mapped)
        r["norm"] = r["mse_uv"] / baseline_mse_uv
        results.append(r)

        print("n_points     =", r["n_points"])
        print("u min/max    =", f"{r['u_min']:.6e}", f"{r['u_max']:.6e}")
        print("v min/max    =", f"{r['v_min']:.6e}", f"{r['v_max']:.6e}")
        print("map dist max =", f"{r['map_dist_max']:.6e}")
        print("map dist mean=", f"{r['map_dist_mean']:.6e}")
        print("MSE_u        =", f"{r['mse_u']:.12e}")
        print("MSE_v        =", f"{r['mse_v']:.12e}")
        print("MSE_uv       =", f"{r['mse_uv']:.12e}")
        print("norm         =", f"{r['norm']:.12e}")
        print("detail file  =", r["detail_file"])
        print("cumulative x =", r["cumulative_file"])
        print("chunked x    =", r["chunk_file"])
        
        # Print chunked errors directly for quick inspection
        print()
        print("Chunked x-error summary, dx = 6:")
        print("  x0       x1       n        base_mse        gp_mse          norm")

        chunk_data = np.genfromtxt(
            r["chunk_file"],
            delimiter=",",
            names=True,
            dtype=None,
            encoding=None,
        )

        if chunk_data.shape == ():
            chunk_data = np.array([chunk_data])

        for row in chunk_data:
            print(
                f"{row['x0']:8.2f} {row['x1']:8.2f} "
                f"{int(row['n_points']):7d} "
                f"{row['baseline_mse']:14.6e} "
                f"{row['gp_mse']:14.6e} "
                f"{row['norm']:12.6e}"
            )

        if r["norm"] < 1.0:
            print("IMPROVEMENT  = yes")
        else:
            print("IMPROVEMENT  = no")

        print()

    # Save summary
    out_file = TEST_ROOT / "backward_candidate_error_summary.csv"

    with out_file.open("w") as f:
        f.write(
            "label,vpp_file,n_points,"
            "mse_u,mse_v,mse_uv,norm,"
            "map_dist_min,map_dist_mean,map_dist_max,map_dist_p95,"
            "u_min,u_max,v_min,v_max\n"
        )

        for r in results:
            f.write(
                f"{r['label']},{r['vpp_file']},{r['n_points']},"
                f"{r['mse_u']:.16e},{r['mse_v']:.16e},{r['mse_uv']:.16e},{r['norm']:.16e},"
                f"{r['map_dist_min']:.16e},{r['map_dist_mean']:.16e},"
                f"{r['map_dist_max']:.16e},{r['map_dist_p95']:.16e},"
                f"{r['u_min']:.16e},{r['u_max']:.16e},"
                f"{r['v_min']:.16e},{r['v_max']:.16e}\n"
            )

    print("Saved summary:", out_file)


if __name__ == "__main__":
    main()