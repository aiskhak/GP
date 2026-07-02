import os
import numpy as np
import pandas as pd

# ============================================================
# USER SETTINGS
# ============================================================
PROJECT_ROOT = os.path.abspath("..")

HF_FILE = os.path.join(
    PROJECT_ROOT,
    "0_data_les",
    "migadome",
    "nek_avg_on_moose.csv",
)

CG_FILE = os.path.join(
    PROJECT_ROOT,
    "1_mixing_length",
    "migadome",
    "migadome_3d_ml_csv_vpp_0080.csv",
)

OUT_DIR = os.path.join(PROJECT_ROOT, "2_mapped", "migadome")
OUT_FILE = os.path.join(OUT_DIR, "mapped.csv")
SUMMARY_FILE = os.path.join(OUT_DIR, "velocity_error_summary.txt")

# Coordinate tolerance for checking that both files use same cell centers
COORD_TOL_WARN = 1.0e-10


# ============================================================
# UTILITIES
# ============================================================
def read_hf_file(filename):
    print("Reading high-fidelity projected file:")
    print("  ", filename)

    df = pd.read_csv(filename)

    required = ["id", "x", "y", "z", "U", "V", "W"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"HF file is missing columns: {missing}")

    # Keep only what we need; rename velocity columns clearly
    df = df[["id", "x", "y", "z", "volume", "yw", "U", "V", "W"]].copy()

    df = df.rename(
        columns={
            "x": "x_hf",
            "y": "y_hf",
            "z": "z_hf",
            "volume": "volume_hf",
            "yw": "yw_hf",
            "U": "u_hf",
            "V": "v_hf",
            "W": "w_hf",
        }
    )

    df["id"] = df["id"].astype(int)
    return df


def read_cg_file(filename):
    print("Reading MOOSE coarse-grid VPP file:")
    print("  ", filename)

    df = pd.read_csv(filename)

    required = ["id", "x", "y", "z", "u", "v", "w"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"CG file is missing columns: {missing}\n"
            f"Available columns are: {list(df.columns)}"
        )

    # Keep useful columns if present
    keep = ["id", "x", "y", "z", "u", "v", "w"]

    optional = [
        "pressure",
        "mixing_length_aux_var",
        "eddy_viscosity_aux_var",
        "yw_aux_var",
    ]
    for c in optional:
        if c in df.columns:
            keep.append(c)

    df = df[keep].copy()

    rename = {
        "x": "x_cg",
        "y": "y_cg",
        "z": "z_cg",
        "u": "u_cg",
        "v": "v_cg",
        "w": "w_cg",
        "pressure": "p_cg",
        "mixing_length_aux_var": "lmix_cg",
        "eddy_viscosity_aux_var": "nut_cg",
        "yw_aux_var": "yw_cg",
    }

    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    df["id"] = df["id"].astype(int)
    return df


def compute_error_stats(df):
    lines = []

    lines.append("============================================================")
    lines.append("MiGaDome coarse-grid velocity comparison")
    lines.append("============================================================")
    lines.append(f"Number of matched cells: {len(df)}")
    lines.append("")

    # Coordinate mismatch
    dx = df["x_cg"] - df["x_hf"]
    dy = df["y_cg"] - df["y_hf"]
    dz = df["z_cg"] - df["z_hf"]
    dist = np.sqrt(dx**2 + dy**2 + dz**2)

    lines.append("Coordinate mismatch:")
    lines.append(f"  max |dx|   = {np.max(np.abs(dx)):.12e}")
    lines.append(f"  max |dy|   = {np.max(np.abs(dy)):.12e}")
    lines.append(f"  max |dz|   = {np.max(np.abs(dz)):.12e}")
    lines.append(f"  max dist   = {np.max(dist):.12e}")
    lines.append(f"  mean dist  = {np.mean(dist):.12e}")
    lines.append("")

    if np.max(dist) > COORD_TOL_WARN:
        lines.append(
            f"WARNING: coordinate mismatch larger than {COORD_TOL_WARN:.1e}. "
            "Check whether files came from the same MOOSE mesh / same scaling."
        )
        lines.append("")
    else:
        lines.append("Coordinate check: OK")
        lines.append("")

    # Velocity errors
    for comp, cg, hf in [
        ("u", "u_cg", "u_hf"),
        ("v", "v_cg", "v_hf"),
        ("w", "w_cg", "w_hf"),
    ]:
        err = df[cg] - df[hf]
        mae = np.mean(np.abs(err))
        rmse = np.sqrt(np.mean(err**2))
        max_abs = np.max(np.abs(err))

        hf_rms = np.sqrt(np.mean(df[hf] ** 2))
        rel_rmse = rmse / hf_rms if hf_rms > 0 else np.nan

        lines.append(f"{comp}-velocity error, CG - HF:")
        lines.append(f"  HF min/max      = {df[hf].min(): .12e}, {df[hf].max(): .12e}")
        lines.append(f"  CG min/max      = {df[cg].min(): .12e}, {df[cg].max(): .12e}")
        lines.append(f"  mean error      = {np.mean(err): .12e}")
        lines.append(f"  MAE             = {mae: .12e}")
        lines.append(f"  RMSE            = {rmse: .12e}")
        lines.append(f"  max |error|     = {max_abs: .12e}")
        lines.append(f"  relative RMSE   = {rel_rmse: .12e}")
        lines.append("")

    # Vector velocity error
    eu = df["u_cg"] - df["u_hf"]
    ev = df["v_cg"] - df["v_hf"]
    ew = df["w_cg"] - df["w_hf"]

    e_mag = np.sqrt(eu**2 + ev**2 + ew**2)
    hf_mag = np.sqrt(df["u_hf"]**2 + df["v_hf"]**2 + df["w_hf"]**2)
    cg_mag = np.sqrt(df["u_cg"]**2 + df["v_cg"]**2 + df["w_cg"]**2)

    lines.append("Vector velocity error:")
    lines.append(f"  mean |e|        = {np.mean(e_mag): .12e}")
    lines.append(f"  RMS  |e|        = {np.sqrt(np.mean(e_mag**2)): .12e}")
    lines.append(f"  max  |e|        = {np.max(e_mag): .12e}")
    lines.append(f"  mean |U_HF|     = {np.mean(hf_mag): .12e}")
    lines.append(f"  mean |U_CG|     = {np.mean(cg_mag): .12e}")
    lines.append("")

    return "\n".join(lines)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    hf = read_hf_file(HF_FILE)
    cg = read_cg_file(CG_FILE)

    print("")
    print("HF rows:", len(hf))
    print("CG rows:", len(cg))

    # Merge by MOOSE element id. This is safer than relying on row order,
    # because VPP files may be sorted by x rather than by id.
    df = pd.merge(cg, hf, on="id", how="inner", validate="one_to_one")

    if len(df) != len(cg) or len(df) != len(hf):
        raise ValueError(
            f"Merge mismatch: merged={len(df)}, CG={len(cg)}, HF={len(hf)}. "
            "Check IDs in both files."
        )

    # Sort for reproducible output
    df = df.sort_values("id").reset_index(drop=True)

    # Coordinate differences
    df["dx"] = df["x_cg"] - df["x_hf"]
    df["dy"] = df["y_cg"] - df["y_hf"]
    df["dz"] = df["z_cg"] - df["z_hf"]
    df["coord_dist"] = np.sqrt(df["dx"]**2 + df["dy"]**2 + df["dz"]**2)

    # Velocity errors
    df["du"] = df["u_cg"] - df["u_hf"]
    df["dv"] = df["v_cg"] - df["v_hf"]
    df["dw"] = df["w_cg"] - df["w_hf"]
    df["err_mag"] = np.sqrt(df["du"]**2 + df["dv"]**2 + df["dw"]**2)

    df["speed_hf"] = np.sqrt(df["u_hf"]**2 + df["v_hf"]**2 + df["w_hf"]**2)
    df["speed_cg"] = np.sqrt(df["u_cg"]**2 + df["v_cg"]**2 + df["w_cg"]**2)
    df["speed_error"] = df["speed_cg"] - df["speed_hf"]

    # Save mapped/comparison file
    df.to_csv(OUT_FILE, index=False, float_format="%.12e")
    print("")
    print("Saved comparison file:")
    print("  ", OUT_FILE)

    # Save summary
    summary = compute_error_stats(df)

    with open(SUMMARY_FILE, "w") as f:
        f.write(summary)

    print("")
    print(summary)
    print("Saved summary:")
    print("  ", SUMMARY_FILE)


if __name__ == "__main__":
    main()