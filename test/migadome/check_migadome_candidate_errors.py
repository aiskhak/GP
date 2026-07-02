#!/usr/bin/env python3

# python3 check_migadome_candidate_errors.py | tee migadome_error_check.log

import os
import fnmatch
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================
GP_ROOT = Path("/homes/aiskhak/projects/GP").resolve()

TEST_ROOT = GP_ROOT / "test" / "migadome"

# Your mapped file has columns:
#   u_hf, v_hf, w_hf      high-fidelity projected/mapped to coarse grid
#   u_cg, v_cg, w_cg      baseline coarse-grid mixing-length result
MAPPED_FILE = GP_ROOT / "2_mapped" / "migadome" / "mapped.csv"

if not MAPPED_FILE.exists():
    alt = GP_ROOT / "2_mapped" / "migadome" / "mapped_velocity_compare.csv"
    if alt.exists():
        MAPPED_FILE = alt

VPP_PATTERNS = [
    "migadome_3d_gpml_csv_vpp_*.csv",
    "migadome_3d_ml_csv_vpp_*.csv",
    "*vpp*.csv",
]


# ============================================================
# REGION DEFINITION
# ============================================================
# Adjust this after checking printed y min/max and region counts.
# For now:
#   inlet_tubes : y < INLET_TUBE_Y_MAX
#   dome        : y >= INLET_TUBE_Y_MAX
INLET_TUBE_Y_MAX = 1.0


def assign_regions(y):
    regions = np.empty(y.size, dtype=object)

    inlet_mask = y < INLET_TUBE_Y_MAX
    regions[inlet_mask] = "inlet_tubes"
    regions[~inlet_mask] = "dome"

    return regions


# ============================================================
# UTILITIES
# ============================================================
def find_latest_vpp_files(root: Path):
    candidates = []

    direct = []
    for pat in VPP_PATTERNS:
        direct.extend(root.glob(pat))

    if direct:
        direct = sorted(set(direct), key=lambda p: p.stat().st_mtime)
        candidates.append(("migadome", direct[-1]))

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


def nearest_map_3d(x_src, y_src, z_src, x_ref, y_ref, z_ref):
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(np.column_stack((x_ref, y_ref, z_ref)))
        dist, idx = tree.query(np.column_stack((x_src, y_src, z_src)), k=1)
        return idx, dist

    except Exception:
        idx = np.zeros(x_src.size, dtype=np.int64)
        dist = np.zeros(x_src.size, dtype=np.float64)

        for i in range(x_src.size):
            d2 = (
                (x_src[i] - x_ref) ** 2
                + (y_src[i] - y_ref) ** 2
                + (z_src[i] - z_ref) ** 2
            )
            j = int(np.argmin(d2))
            idx[i] = j
            dist[i] = np.sqrt(d2[j])

        return idx, dist


def require_columns(df, cols, label):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"{label} is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def optional_col(df, name):
    if name in df.columns:
        return df[name].to_numpy(dtype=float)
    return np.full(len(df), np.nan)


def compute_region_errors(regions, err2_base, err2_gp):
    rows = []

    for region_name in ["inlet_tubes", "dome", "all"]:
        if region_name == "all":
            mask = np.ones(regions.size, dtype=bool)
        else:
            mask = regions == region_name

        if not np.any(mask):
            rows.append(
                {
                    "region": region_name,
                    "n_points": 0,
                    "baseline_mse": np.nan,
                    "gp_mse": np.nan,
                    "norm": np.nan,
                }
            )
            continue

        base_mse = float(np.mean(err2_base[mask]))
        gp_mse = float(np.mean(err2_gp[mask]))
        norm = gp_mse / base_mse if base_mse > 0.0 else np.nan

        rows.append(
            {
                "region": region_name,
                "n_points": int(np.sum(mask)),
                "baseline_mse": base_mse,
                "gp_mse": gp_mse,
                "norm": norm,
            }
        )

    return rows


def align_candidate_to_mapped(vpp_df, mapped_df):
    """
    Prefer exact element-id merge. If ID is unavailable, fall back to 3D nearest mapping.
    """

    if "id" in vpp_df.columns and "id" in mapped_df.columns:
        tmp = vpp_df.rename(
            columns={
                "x": "x_gp",
                "y": "y_gp",
                "z": "z_gp",
                "u": "u_gp",
                "v": "v_gp",
                "w": "w_gp",
            }
        )

        merged = pd.merge(
            tmp,
            mapped_df,
            on="id",
            how="inner",
            validate="one_to_one",
        )

        if len(merged) != len(vpp_df):
            raise RuntimeError(
                f"ID merge mismatch: merged={len(merged)}, VPP={len(vpp_df)}, mapped={len(mapped_df)}"
            )

        dx = merged["x_gp"].to_numpy() - merged["x_cg"].to_numpy()
        dy = merged["y_gp"].to_numpy() - merged["y_cg"].to_numpy()
        dz = merged["z_gp"].to_numpy() - merged["z_cg"].to_numpy()
        dist = np.sqrt(dx**2 + dy**2 + dz**2)

        return merged, dist, "id_merge"

    # Fallback: nearest neighbor by coordinates
    require_columns(vpp_df, ["x", "y", "z", "u", "v", "w"], "VPP file")
    require_columns(mapped_df, ["x_cg", "y_cg", "z_cg"], "mapped file")

    x = vpp_df["x"].to_numpy(dtype=float)
    y = vpp_df["y"].to_numpy(dtype=float)
    z = vpp_df["z"].to_numpy(dtype=float)

    x_ref = mapped_df["x_cg"].to_numpy(dtype=float)
    y_ref = mapped_df["y_cg"].to_numpy(dtype=float)
    z_ref = mapped_df["z_cg"].to_numpy(dtype=float)

    idx, dist = nearest_map_3d(x, y, z, x_ref, y_ref, z_ref)

    gp = vpp_df.reset_index(drop=True).rename(
        columns={
            "x": "x_gp",
            "y": "y_gp",
            "z": "z_gp",
            "u": "u_gp",
            "v": "v_gp",
            "w": "w_gp",
        }
    )

    ref = mapped_df.iloc[idx].reset_index(drop=True)
    merged = pd.concat([gp, ref], axis=1)

    return merged, dist, "nearest_neighbor"


def compute_error_for_vpp(label: str, vpp_file: Path, mapped_df: pd.DataFrame):
    print(f"===== Candidate: {label} =====")
    print("VPP:", vpp_file)

    vpp_df = pd.read_csv(vpp_file)

    require_columns(vpp_df, ["x", "y", "z", "u", "v", "w"], "VPP file")

    merged, dist, alignment_mode = align_candidate_to_mapped(vpp_df, mapped_df)

    # Coordinates from GP/VPP sample
    x = merged["x_gp"].to_numpy(dtype=float)
    y = merged["y_gp"].to_numpy(dtype=float)
    z = merged["z_gp"].to_numpy(dtype=float)

    # HF mapped/projected to coarse grid
    u_hf = merged["u_hf"].to_numpy(dtype=float)
    v_hf = merged["v_hf"].to_numpy(dtype=float)
    w_hf = merged["w_hf"].to_numpy(dtype=float)

    # Baseline coarse-grid ML solution
    u_base = merged["u_cg"].to_numpy(dtype=float)
    v_base = merged["v_cg"].to_numpy(dtype=float)
    w_base = merged["w_cg"].to_numpy(dtype=float)

    # GP candidate solution
    u_gp = merged["u_gp"].to_numpy(dtype=float)
    v_gp = merged["v_gp"].to_numpy(dtype=float)
    w_gp = merged["w_gp"].to_numpy(dtype=float)

    # Optional GP fields
    lm_gp = (
        optional_col(merged, "mixing_length_gp_aux_var")
        if "mixing_length_gp_aux_var" in merged.columns
        else optional_col(merged, "mixing_length_aux_var")
    )

    lm_std = optional_col(merged, "mixing_length_std_aux_var")
    gp_factor = optional_col(merged, "gp_factor_aux_var")
    eddy_visc = optional_col(merged, "eddy_viscosity_aux_var")
    yw = optional_col(merged, "yw_aux_var")

    # Error definitions
    du_base = u_base - u_hf
    dv_base = v_base - v_hf
    dw_base = w_base - w_hf

    du_gp = u_gp - u_hf
    dv_gp = v_gp - v_hf
    dw_gp = w_gp - w_hf

    err2_base = du_base**2 + dv_base**2 + dw_base**2
    err2_gp = du_gp**2 + dv_gp**2 + dw_gp**2

    regions = assign_regions(y)
    region_rows = compute_region_errors(regions, err2_base, err2_gp)

    mse_u = float(np.mean(du_gp**2))
    mse_v = float(np.mean(dv_gp**2))
    mse_w = float(np.mean(dw_gp**2))
    mse_uvw = float(np.mean(err2_gp))

    baseline_mse_u = float(np.mean(du_base**2))
    baseline_mse_v = float(np.mean(dv_base**2))
    baseline_mse_w = float(np.mean(dw_base**2))
    baseline_mse_uvw = float(np.mean(err2_base))

    # Save pointwise fields
    out_detail = TEST_ROOT / f"{label}_hf_baseline_gp_fields.csv"

    region_id = np.array([0 if r == "inlet_tubes" else 1 for r in regions], dtype=float)

    detail = np.column_stack(
        (
            x,
            y,
            z,
            u_hf,
            v_hf,
            w_hf,
            u_base,
            v_base,
            w_base,
            u_gp,
            v_gp,
            w_gp,
            lm_gp,
            lm_std,
            gp_factor,
            eddy_visc,
            yw,
            err2_base,
            err2_gp,
            err2_gp - err2_base,
            dist,
            region_id,
        )
    )

    np.savetxt(
        out_detail,
        detail,
        delimiter=",",
        header=(
            "x,y,z,"
            "u_hf,v_hf,w_hf,"
            "u_baseline,v_baseline,w_baseline,"
            "u_gp,v_gp,w_gp,"
            "mixing_length_gp,mixing_length_std,gp_factor,eddy_viscosity,yw,"
            "err2_baseline,err2_gp,err2_gp_minus_baseline,"
            "map_dist,region_id"
        ),
        comments="",
        fmt="%.12e",
    )

    # Save region errors
    out_regions = TEST_ROOT / f"{label}_region_errors.csv"

    with out_regions.open("w") as f:
        f.write("region,n_points,baseline_mse,gp_mse,norm\n")
        for row in region_rows:
            f.write(
                f"{row['region']},"
                f"{row['n_points']},"
                f"{row['baseline_mse']:.16e},"
                f"{row['gp_mse']:.16e},"
                f"{row['norm']:.16e}\n"
            )

    result = {
        "label": label,
        "vpp_file": str(vpp_file),
        "detail_file": str(out_detail),
        "region_file": str(out_regions),
        "alignment_mode": alignment_mode,
        "n_points": int(len(merged)),
        "baseline_mse_u": baseline_mse_u,
        "baseline_mse_v": baseline_mse_v,
        "baseline_mse_w": baseline_mse_w,
        "baseline_mse_uvw": baseline_mse_uvw,
        "mse_u": mse_u,
        "mse_v": mse_v,
        "mse_w": mse_w,
        "mse_uvw": mse_uvw,
        "norm": mse_uvw / baseline_mse_uvw if baseline_mse_uvw > 0.0 else np.nan,
        "map_dist_min": float(np.min(dist)),
        "map_dist_mean": float(np.mean(dist)),
        "map_dist_max": float(np.max(dist)),
        "map_dist_p95": float(np.percentile(dist, 95)),
        "x_min": float(np.min(x)),
        "x_max": float(np.max(x)),
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
        "z_min": float(np.min(z)),
        "z_max": float(np.max(z)),
        "lm_min": float(np.nanmin(lm_gp)),
        "lm_max": float(np.nanmax(lm_gp)),
        "factor_min": float(np.nanmin(gp_factor)),
        "factor_max": float(np.nanmax(gp_factor)),
        "region_rows": region_rows,
    }

    return result


def main():
    if not MAPPED_FILE.exists():
        raise FileNotFoundError(f"Mapped file not found: {MAPPED_FILE}")

    print("TEST_ROOT:        ", TEST_ROOT)
    print("MAPPED_FILE:      ", MAPPED_FILE)
    print("INLET_TUBE_Y_MAX: ", INLET_TUBE_Y_MAX)
    print()

    mapped_df = pd.read_csv(MAPPED_FILE)

    required_mapped = [
        "x_cg",
        "y_cg",
        "z_cg",
        "u_cg",
        "v_cg",
        "w_cg",
        "u_hf",
        "v_hf",
        "w_hf",
    ]

    require_columns(mapped_df, required_mapped, "mapped file")

    candidates = find_latest_vpp_files(TEST_ROOT)

    if not candidates:
        raise FileNotFoundError(f"No candidate VPP files found under {TEST_ROOT}")

    results = []

    for label, vpp_file in candidates:
        r = compute_error_for_vpp(label, vpp_file, mapped_df)
        results.append(r)

        print("alignment mode =", r["alignment_mode"])
        print("n_points       =", r["n_points"])
        print("x min/max      =", f"{r['x_min']:.6e}", f"{r['x_max']:.6e}")
        print("y min/max      =", f"{r['y_min']:.6e}", f"{r['y_max']:.6e}")
        print("z min/max      =", f"{r['z_min']:.6e}", f"{r['z_max']:.6e}")
        print("lm min/max     =", f"{r['lm_min']:.6e}", f"{r['lm_max']:.6e}")
        print("factor min/max =", f"{r['factor_min']:.6e}", f"{r['factor_max']:.6e}")
        print("map dist max   =", f"{r['map_dist_max']:.6e}")
        print("map dist mean  =", f"{r['map_dist_mean']:.6e}")
        print()
        print("Baseline MSE against HF:")
        print("MSE_u_base     =", f"{r['baseline_mse_u']:.12e}")
        print("MSE_v_base     =", f"{r['baseline_mse_v']:.12e}")
        print("MSE_w_base     =", f"{r['baseline_mse_w']:.12e}")
        print("MSE_uvw_base   =", f"{r['baseline_mse_uvw']:.12e}")
        print()
        print("GP MSE against HF:")
        print("MSE_u_gp       =", f"{r['mse_u']:.12e}")
        print("MSE_v_gp       =", f"{r['mse_v']:.12e}")
        print("MSE_w_gp       =", f"{r['mse_w']:.12e}")
        print("MSE_uvw_gp     =", f"{r['mse_uvw']:.12e}")
        print("norm           =", f"{r['norm']:.12e}")
        print("detail file    =", r["detail_file"])
        print("region file    =", r["region_file"])
        print()

        print("Region error summary:")
        print("  region          n        base_mse        gp_mse          norm")

        for row in r["region_rows"]:
            print(
                f"  {row['region']:12s} "
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

    out_file = TEST_ROOT / "migadome_candidate_error_summary.csv"

    with out_file.open("w") as f:
        f.write(
            "label,vpp_file,alignment_mode,n_points,"
            "baseline_mse_u,baseline_mse_v,baseline_mse_w,baseline_mse_uvw,"
            "mse_u,mse_v,mse_w,mse_uvw,norm,"
            "map_dist_min,map_dist_mean,map_dist_max,map_dist_p95,"
            "x_min,x_max,y_min,y_max,z_min,z_max,"
            "lm_min,lm_max,factor_min,factor_max\n"
        )

        for r in results:
            f.write(
                f"{r['label']},{r['vpp_file']},{r['alignment_mode']},{r['n_points']},"
                f"{r['baseline_mse_u']:.16e},{r['baseline_mse_v']:.16e},"
                f"{r['baseline_mse_w']:.16e},{r['baseline_mse_uvw']:.16e},"
                f"{r['mse_u']:.16e},{r['mse_v']:.16e},{r['mse_w']:.16e},"
                f"{r['mse_uvw']:.16e},{r['norm']:.16e},"
                f"{r['map_dist_min']:.16e},{r['map_dist_mean']:.16e},"
                f"{r['map_dist_max']:.16e},{r['map_dist_p95']:.16e},"
                f"{r['x_min']:.16e},{r['x_max']:.16e},"
                f"{r['y_min']:.16e},{r['y_max']:.16e},"
                f"{r['z_min']:.16e},{r['z_max']:.16e},"
                f"{r['lm_min']:.16e},{r['lm_max']:.16e},"
                f"{r['factor_min']:.16e},{r['factor_max']:.16e}\n"
            )

    print("Saved summary:", out_file)


if __name__ == "__main__":
    main()
