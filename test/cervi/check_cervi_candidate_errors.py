#!/usr/bin/env python3

# python3 check_cervi_candidate_errors.py | tee cervi_error_check.log

import os
import fnmatch
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================
GP_ROOT = Path("/homes/aiskhak/projects/GP").resolve()

TEST_ROOT = GP_ROOT / "test" / "cervi"

# Standard-ML mapped baseline:
#   u_dns, v_dns              OpenFOAM SST reference mapped to MOOSE grid
#   u_baseline, v_baseline    standard ML result, delta=0.10
MAPPED_FILE = GP_ROOT / "2_mapped" / "cervi" / "03_ml_dx025_delta0p1" / "mapped.csv"

VPP_PATTERNS = [
    "cervi_ml_dx025_gp_vpp_*.csv",
    "*vpp*.csv",
]

ELV_PATTERNS = [
    "cervi_ml_dx025_gp_elv_*.csv",
    "*elv*.csv",
]


# ============================================================
# REGION DEFINITIONS
# ============================================================
def region_masks(x, y):
    return {
        "all": np.ones_like(x, dtype=bool),

        "main_tank": (
            (x >= 0.0) & (x <= 1.0) &
            (y >= 0.0) & (y <= 1.0)
        ),

        "inlet_duct": (
            (x >= 0.30) & (x <= 0.50) &
            (y >= -0.40) & (y <= 0.0)
        ),

        "outlet_duct": (
            (x >= 1.0) & (x <= 1.40) &
            (y >= 0.50) & (y <= 0.70)
        ),

        "lower_tank": (
            (x >= 0.0) & (x <= 1.0) &
            (y >= 0.0) & (y < 0.50)
        ),

        "upper_tank": (
            (x >= 0.0) & (x <= 1.0) &
            (y >= 0.50) & (y <= 1.0)
        ),
    }


def primary_region_id(x, y):
    # 0 inlet, 1 outlet, 2 lower tank, 3 upper tank, 4 other
    rid = np.full(x.size, 4.0)

    inlet = (
        (x >= 0.30) & (x <= 0.50) &
        (y >= -0.40) & (y <= 0.0)
    )
    outlet = (
        (x >= 1.0) & (x <= 1.40) &
        (y >= 0.50) & (y <= 0.70)
    )
    lower = (
        (x >= 0.0) & (x <= 1.0) &
        (y >= 0.0) & (y < 0.50)
    )
    upper = (
        (x >= 0.0) & (x <= 1.0) &
        (y >= 0.50) & (y <= 1.0)
    )

    rid[inlet] = 0.0
    rid[outlet] = 1.0
    rid[lower] = 2.0
    rid[upper] = 3.0

    return rid


# ============================================================
# UTILITIES
# ============================================================
def find_latest_matching_file(root: Path, patterns):
    files = []

    for pat in patterns:
        files.extend(root.glob(pat))

    if not files:
        return None

    files = sorted(set(files), key=lambda p: p.stat().st_mtime)
    return files[-1]


def find_latest_vpp_files(root: Path):
    candidates = []

    direct = []
    for pat in VPP_PATTERNS:
        direct.extend(root.glob(pat))

    if direct:
        direct = sorted(set(direct), key=lambda p: p.stat().st_mtime)
        candidates.append(("cervi", direct[-1]))

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


def nearest_map_2d(x_src, y_src, x_ref, y_ref):
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(np.column_stack((x_ref, y_ref)))
        dist, idx = tree.query(np.column_stack((x_src, y_src)), k=1)
        return idx.astype(np.int64), dist.astype(np.float64)

    except Exception:
        idx = np.zeros(x_src.size, dtype=np.int64)
        dist = np.zeros(x_src.size, dtype=np.float64)

        chunk = 250
        for i0 in range(0, x_src.size, chunk):
            i1 = min(i0 + chunk, x_src.size)

            dx = x_src[i0:i1, None] - x_ref[None, :]
            dy = y_src[i0:i1, None] - y_ref[None, :]
            d2 = dx * dx + dy * dy

            j = np.argmin(d2, axis=1)
            idx[i0:i1] = j
            dist[i0:i1] = np.sqrt(d2[np.arange(i1 - i0), j])

        return idx, dist


def require_columns(df, cols, label):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"{label} is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def first_existing_col(df, names, label):
    for n in names:
        if n in df.columns:
            return n

    raise RuntimeError(
        f"Could not find {label}. Tried {names}.\n"
        f"Available columns: {list(df.columns)}"
    )


def optional_col(df, name):
    if name in df.columns:
        return df[name].to_numpy(dtype=float)
    return np.full(len(df), np.nan)


def safe_label(label):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in label)


def add_elv_fields_if_available(vpp_df, vpp_file):
    """
    VPP often has final velocity and final GP fields.
    ELV may contain yw_aux_var and initial diagnostic fields.
    If a useful field is missing from VPP, map ELV to VPP by x,y.
    """

    parent = vpp_file.parent
    elv_file = find_latest_matching_file(parent, ELV_PATTERNS)

    if elv_file is None:
        return vpp_df, None, None

    try:
        elv_df = pd.read_csv(elv_file)
    except Exception:
        return vpp_df, None, None

    if not {"x", "y"}.issubset(elv_df.columns):
        return vpp_df, elv_file, None

    x_vpp = vpp_df["x"].to_numpy(dtype=float)
    y_vpp = vpp_df["y"].to_numpy(dtype=float)

    x_elv = elv_df["x"].to_numpy(dtype=float)
    y_elv = elv_df["y"].to_numpy(dtype=float)

    idx, dist = nearest_map_2d(x_vpp, y_vpp, x_elv, y_elv)

    out = vpp_df.copy()

    for c in [
        "yw_aux_var",
        "mixing_length_aux_var",
        "mixing_length_std_aux_var",
        "gp_factor_aux_var",
        "elvol_aux_var",
    ]:
        if c not in out.columns and c in elv_df.columns:
            out[c] = elv_df.iloc[idx][c].to_numpy(dtype=float)

    return out, elv_file, dist


def standardize_mapped_columns(mapped_df):
    """
    Accept either old or new mapped-file column names.
    """

    require_columns(mapped_df, ["x_cg", "y_cg"], "mapped file")

    u_hf_col = first_existing_col(
        mapped_df,
        ["u_hf", "u_ref", "u_of", "u_dns"],
        "mapped reference u",
    )
    v_hf_col = first_existing_col(
        mapped_df,
        ["v_hf", "v_ref", "v_of", "v_dns"],
        "mapped reference v",
    )

    u_base_col = first_existing_col(
        mapped_df,
        ["u_cg", "u_baseline"],
        "baseline u",
    )
    v_base_col = first_existing_col(
        mapped_df,
        ["v_cg", "v_baseline"],
        "baseline v",
    )

    out = mapped_df.copy()
    out["u_hf"] = mapped_df[u_hf_col].to_numpy(dtype=float)
    out["v_hf"] = mapped_df[v_hf_col].to_numpy(dtype=float)
    out["u_base"] = mapped_df[u_base_col].to_numpy(dtype=float)
    out["v_base"] = mapped_df[v_base_col].to_numpy(dtype=float)

    return out


def align_candidate_to_mapped(vpp_df, mapped_df):
    """
    Prefer exact element-id merge if available.
    Otherwise use 2D nearest neighbor by x,y.
    """

    if "id" in vpp_df.columns and "id" in mapped_df.columns:
        tmp = vpp_df.rename(
            columns={
                "x": "x_gp",
                "y": "y_gp",
                "u": "u_gp",
                "v": "v_gp",
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
                f"ID merge mismatch: merged={len(merged)}, "
                f"VPP={len(vpp_df)}, mapped={len(mapped_df)}"
            )

        dx = merged["x_gp"].to_numpy() - merged["x_cg"].to_numpy()
        dy = merged["y_gp"].to_numpy() - merged["y_cg"].to_numpy()
        dist = np.sqrt(dx**2 + dy**2)

        return merged, dist, "id_merge"

    require_columns(vpp_df, ["x", "y", "u", "v"], "VPP file")
    require_columns(mapped_df, ["x_cg", "y_cg"], "mapped file")

    x = vpp_df["x"].to_numpy(dtype=float)
    y = vpp_df["y"].to_numpy(dtype=float)

    x_ref = mapped_df["x_cg"].to_numpy(dtype=float)
    y_ref = mapped_df["y_cg"].to_numpy(dtype=float)

    idx, dist = nearest_map_2d(x, y, x_ref, y_ref)

    gp = vpp_df.reset_index(drop=True).rename(
        columns={
            "x": "x_gp",
            "y": "y_gp",
            "u": "u_gp",
            "v": "v_gp",
        }
    )

    ref = mapped_df.iloc[idx].reset_index(drop=True)
    merged = pd.concat([gp, ref], axis=1)

    return merged, dist, "nearest_neighbor"


def compute_region_errors(x, y, err2_base, err2_gp, ref2):
    rows = []

    masks = region_masks(x, y)

    for region_name in [
        "all",
        "main_tank",
        "inlet_duct",
        "outlet_duct",
        "lower_tank",
        "upper_tank",
    ]:
        mask = masks[region_name]

        if not np.any(mask):
            rows.append(
                {
                    "region": region_name,
                    "n_points": 0,
                    "baseline_mse": np.nan,
                    "gp_mse": np.nan,
                    "norm": np.nan,
                    "ref_mean_U2": np.nan,
                    "baseline_rel_mse": np.nan,
                    "gp_rel_mse": np.nan,
                }
            )
            continue

        base_mse = float(np.mean(err2_base[mask]))
        gp_mse = float(np.mean(err2_gp[mask]))
        ref_mean = float(np.mean(ref2[mask]))

        rows.append(
            {
                "region": region_name,
                "n_points": int(np.sum(mask)),
                "baseline_mse": base_mse,
                "gp_mse": gp_mse,
                "norm": gp_mse / base_mse if base_mse > 0.0 else np.nan,
                "ref_mean_U2": ref_mean,
                "baseline_rel_mse": base_mse / ref_mean if ref_mean > 0.0 else np.nan,
                "gp_rel_mse": gp_mse / ref_mean if ref_mean > 0.0 else np.nan,
            }
        )

    return rows


def compute_error_for_vpp(label: str, vpp_file: Path, mapped_df: pd.DataFrame):
    print(f"===== Candidate: {label} =====")
    print("VPP:", vpp_file)

    raw_vpp_df = pd.read_csv(vpp_file)
    require_columns(raw_vpp_df, ["x", "y", "u", "v"], "VPP file")

    vpp_df, elv_file, elv_dist = add_elv_fields_if_available(raw_vpp_df, vpp_file)

    if elv_file is not None:
        print("ELV:", elv_file)
        if elv_dist is not None:
            print(
                "VPP->ELV dist min/mean/max =",
                f"{np.min(elv_dist):.6e}",
                f"{np.mean(elv_dist):.6e}",
                f"{np.max(elv_dist):.6e}",
            )

    merged, dist, alignment_mode = align_candidate_to_mapped(vpp_df, mapped_df)

    x = merged["x_gp"].to_numpy(dtype=float)
    y = merged["y_gp"].to_numpy(dtype=float)

    u_hf = merged["u_hf"].to_numpy(dtype=float)
    v_hf = merged["v_hf"].to_numpy(dtype=float)

    u_base = merged["u_base"].to_numpy(dtype=float)
    v_base = merged["v_base"].to_numpy(dtype=float)

    u_gp = merged["u_gp"].to_numpy(dtype=float)
    v_gp = merged["v_gp"].to_numpy(dtype=float)

    lm_gp = optional_col(merged, "mixing_length_aux_var")
    lm_std = optional_col(merged, "mixing_length_std_aux_var")
    gp_factor = optional_col(merged, "gp_factor_aux_var")
    eddy_visc = optional_col(merged, "eddy_viscosity_aux_var")
    yw = optional_col(merged, "yw_aux_var")

    du_base = u_base - u_hf
    dv_base = v_base - v_hf

    du_gp = u_gp - u_hf
    dv_gp = v_gp - v_hf

    err2_base = du_base**2 + dv_base**2
    err2_gp = du_gp**2 + dv_gp**2
    ref2 = u_hf**2 + v_hf**2

    region_rows = compute_region_errors(x, y, err2_base, err2_gp, ref2)

    mse_u = float(np.mean(du_gp**2))
    mse_v = float(np.mean(dv_gp**2))
    mse_uv = float(np.mean(err2_gp))

    baseline_mse_u = float(np.mean(du_base**2))
    baseline_mse_v = float(np.mean(dv_base**2))
    baseline_mse_uv = float(np.mean(err2_base))

    ref_mean_U2 = float(np.mean(ref2))

    label_safe = safe_label(label)

    out_detail = TEST_ROOT / f"{label_safe}_hf_baseline_gp_fields.csv"

    detail = np.column_stack(
        (
            x,
            y,
            u_hf,
            v_hf,
            u_base,
            v_base,
            u_gp,
            v_gp,
            lm_gp,
            lm_std,
            gp_factor,
            eddy_visc,
            yw,
            err2_base,
            err2_gp,
            err2_gp - err2_base,
            dist,
            primary_region_id(x, y),
        )
    )

    np.savetxt(
        out_detail,
        detail,
        delimiter=",",
        header=(
            "x,y,"
            "u_hf,v_hf,"
            "u_baseline,v_baseline,"
            "u_gp,v_gp,"
            "mixing_length_gp,mixing_length_std,gp_factor,eddy_viscosity,yw,"
            "err2_baseline,err2_gp,err2_gp_minus_baseline,"
            "map_dist,region_id"
        ),
        comments="",
        fmt="%.12e",
    )

    out_regions = TEST_ROOT / f"{label_safe}_region_errors.csv"

    with out_regions.open("w") as f:
        f.write(
            "region,n_points,baseline_mse,gp_mse,norm,"
            "ref_mean_U2,baseline_rel_mse,gp_rel_mse\n"
        )
        for row in region_rows:
            f.write(
                f"{row['region']},"
                f"{row['n_points']},"
                f"{row['baseline_mse']:.16e},"
                f"{row['gp_mse']:.16e},"
                f"{row['norm']:.16e},"
                f"{row['ref_mean_U2']:.16e},"
                f"{row['baseline_rel_mse']:.16e},"
                f"{row['gp_rel_mse']:.16e}\n"
            )

    result = {
        "label": label,
        "vpp_file": str(vpp_file),
        "elv_file": str(elv_file) if elv_file is not None else "",
        "detail_file": str(out_detail),
        "region_file": str(out_regions),
        "alignment_mode": alignment_mode,
        "n_points": int(len(merged)),
        "baseline_mse_u": baseline_mse_u,
        "baseline_mse_v": baseline_mse_v,
        "baseline_mse_uv": baseline_mse_uv,
        "mse_u": mse_u,
        "mse_v": mse_v,
        "mse_uv": mse_uv,
        "norm": mse_uv / baseline_mse_uv if baseline_mse_uv > 0.0 else np.nan,
        "ref_mean_U2": ref_mean_U2,
        "baseline_rel_mse": baseline_mse_uv / ref_mean_U2 if ref_mean_U2 > 0.0 else np.nan,
        "gp_rel_mse": mse_uv / ref_mean_U2 if ref_mean_U2 > 0.0 else np.nan,
        "map_dist_min": float(np.min(dist)),
        "map_dist_mean": float(np.mean(dist)),
        "map_dist_max": float(np.max(dist)),
        "map_dist_p95": float(np.percentile(dist, 95)),
        "x_min": float(np.min(x)),
        "x_max": float(np.max(x)),
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
        "lm_min": float(np.nanmin(lm_gp)),
        "lm_max": float(np.nanmax(lm_gp)),
        "lm_std_min": float(np.nanmin(lm_std)),
        "lm_std_max": float(np.nanmax(lm_std)),
        "factor_min": float(np.nanmin(gp_factor)),
        "factor_max": float(np.nanmax(gp_factor)),
        "nut_min": float(np.nanmin(eddy_visc)),
        "nut_max": float(np.nanmax(eddy_visc)),
        "yw_min": float(np.nanmin(yw)),
        "yw_max": float(np.nanmax(yw)),
        "region_rows": region_rows,
    }

    return result


def main():
    if not MAPPED_FILE.exists():
        raise FileNotFoundError(f"Mapped file not found: {MAPPED_FILE}")

    print("TEST_ROOT:   ", TEST_ROOT)
    print("MAPPED_FILE: ", MAPPED_FILE)
    print()

    mapped_raw = pd.read_csv(MAPPED_FILE)
    mapped_df = standardize_mapped_columns(mapped_raw)

    print("Mapped columns:")
    print(list(mapped_raw.columns))
    print("Mapped points:", len(mapped_df))
    print()

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
        print("yw min/max     =", f"{r['yw_min']:.6e}", f"{r['yw_max']:.6e}")
        print("lm_std min/max =", f"{r['lm_std_min']:.6e}", f"{r['lm_std_max']:.6e}")
        print("lm_gp min/max  =", f"{r['lm_min']:.6e}", f"{r['lm_max']:.6e}")
        print("factor min/max =", f"{r['factor_min']:.6e}", f"{r['factor_max']:.6e}")
        print("nut min/max    =", f"{r['nut_min']:.6e}", f"{r['nut_max']:.6e}")
        print("map dist max   =", f"{r['map_dist_max']:.6e}")
        print("map dist mean  =", f"{r['map_dist_mean']:.6e}")
        print()
        print("Baseline MSE against OpenFOAM:")
        print("MSE_u_base       =", f"{r['baseline_mse_u']:.12e}")
        print("MSE_v_base       =", f"{r['baseline_mse_v']:.12e}")
        print("MSE_uv_base      =", f"{r['baseline_mse_uv']:.12e}")
        print("baseline rel MSE =", f"{r['baseline_rel_mse']:.12e}")
        print()
        print("GP MSE against OpenFOAM:")
        print("MSE_u_gp         =", f"{r['mse_u']:.12e}")
        print("MSE_v_gp         =", f"{r['mse_v']:.12e}")
        print("MSE_uv_gp        =", f"{r['mse_uv']:.12e}")
        print("GP rel MSE       =", f"{r['gp_rel_mse']:.12e}")
        print("norm GP/base     =", f"{r['norm']:.12e}")
        print("detail file      =", r["detail_file"])
        print("region file      =", r["region_file"])
        print()

        print("Region error summary:")
        print("  region          n        base_mse        gp_mse          norm      gp_rel_mse")

        for row in r["region_rows"]:
            print(
                f"  {row['region']:12s} "
                f"{int(row['n_points']):7d} "
                f"{row['baseline_mse']:14.6e} "
                f"{row['gp_mse']:14.6e} "
                f"{row['norm']:12.6e} "
                f"{row['gp_rel_mse']:12.6e}"
            )

        if r["norm"] < 1.0:
            print("IMPROVEMENT  = yes")
        else:
            print("IMPROVEMENT  = no")

        print()

    out_file = TEST_ROOT / "cervi_candidate_error_summary.csv"

    with out_file.open("w") as f:
        f.write(
            "label,vpp_file,elv_file,alignment_mode,n_points,"
            "baseline_mse_u,baseline_mse_v,baseline_mse_uv,"
            "mse_u,mse_v,mse_uv,norm,"
            "ref_mean_U2,baseline_rel_mse,gp_rel_mse,"
            "map_dist_min,map_dist_mean,map_dist_max,map_dist_p95,"
            "x_min,x_max,y_min,y_max,"
            "yw_min,yw_max,lm_std_min,lm_std_max,lm_min,lm_max,"
            "factor_min,factor_max,nut_min,nut_max\n"
        )

        for r in results:
            f.write(
                f"{r['label']},{r['vpp_file']},{r['elv_file']},"
                f"{r['alignment_mode']},{r['n_points']},"
                f"{r['baseline_mse_u']:.16e},{r['baseline_mse_v']:.16e},"
                f"{r['baseline_mse_uv']:.16e},"
                f"{r['mse_u']:.16e},{r['mse_v']:.16e},"
                f"{r['mse_uv']:.16e},{r['norm']:.16e},"
                f"{r['ref_mean_U2']:.16e},{r['baseline_rel_mse']:.16e},"
                f"{r['gp_rel_mse']:.16e},"
                f"{r['map_dist_min']:.16e},{r['map_dist_mean']:.16e},"
                f"{r['map_dist_max']:.16e},{r['map_dist_p95']:.16e},"
                f"{r['x_min']:.16e},{r['x_max']:.16e},"
                f"{r['y_min']:.16e},{r['y_max']:.16e},"
                f"{r['yw_min']:.16e},{r['yw_max']:.16e},"
                f"{r['lm_std_min']:.16e},{r['lm_std_max']:.16e},"
                f"{r['lm_min']:.16e},{r['lm_max']:.16e},"
                f"{r['factor_min']:.16e},{r['factor_max']:.16e},"
                f"{r['nut_min']:.16e},{r['nut_max']:.16e}\n"
            )

    print("Saved summary:", out_file)


if __name__ == "__main__":
    main()
