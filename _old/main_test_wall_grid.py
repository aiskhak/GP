#!/usr/bin/env python3

# Run from project root:
#   python -u main_test_wall_grid.py 2>&1 | tee wall_grid_all_cases.log

import os
import fnmatch
import shutil
import subprocess
import time
import csv
import numpy as np

# ============================================================
# Settings
# ============================================================
PROJECT_ROOT = os.path.abspath(".")

RE_LIST = ["3000", "3413", "5963", "7912", "9000", "10622", "12819", "14000"]
CG_LIST = ["1", "2", "3", "4"]

NP_BY_CG = {
    "1": 4,
    "2": 4,
    "3": 8,
    "4": 12,
}

CAND_NAME = "cand_wall_grid"

STEADY_RUN_SCRIPT = "fv_app_run.sh"
UNSTEADY_RUN_SCRIPT = "fv_app_run_unsteady.sh"

LM_FILE = "lm_pred.csv"
LOG_FILE = "log.csv"

BASELINE_VPP_PATTERN = "tamu_2d_fv_csv_vpp_*.csv"
VPP_PATTERN = "tamu_2d_fv_gp_csv_vpp_*.csv"

SOLVER_TIMEOUT = 2700.0
YW_WL = 0.0

# Best grid-aware candidate:
# raw = -1.1*eta_h - 4.8*eta_y - 0.12
C1 = -1.1
C2 = -4.8
C3 = -0.12

KAPPA0 = 0.41
C_EFF_MIN_FACTOR = 0.05
C_EFF_MAX_FACTOR = 2.0

RESULTS_DIR = os.path.join(PROJECT_ROOT, "csv_wall_grid_candidate_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SUMMARY_FILE = os.path.join(
    RESULTS_DIR,
    "csv_wall_grid_candidate_all_cases_summary.csv",
)

WALL_CELL_SUMMARY_FILE = os.path.join(
    PROJECT_ROOT,
    "3_wall_cells",
    "wall_cells_median_summary.csv",
)

# ============================================================
# Path utilities
# ============================================================
def get_paths(RE, CG):
    case_root = os.path.join(PROJECT_ROOT, "runs", RE, CG)
    template_dir = os.path.join(case_root, "template")
    cand_dir = os.path.join(case_root, CAND_NAME)
    mapped_file = os.path.join(PROJECT_ROOT, "2_mapped", RE, CG, "mapped.csv")
    baseline_dir = os.path.join(PROJECT_ROOT, "1_mixing_length", RE, CG)
    return case_root, template_dir, cand_dir, mapped_file, baseline_dir


# ============================================================
# Data utilities
# ============================================================
def read_mapped(mapped_file):
    print(f"Reading mapped file: {mapped_file}", flush=True)

    if not os.path.isfile(mapped_file):
        raise FileNotFoundError(f"Mapped file not found: {mapped_file}")

    data = np.loadtxt(mapped_file, delimiter=",", skiprows=1, dtype=np.float64)

    r_cg = data[:, 0]
    z_cg = data[:, 1]
    ur_les = data[:, 2]
    uz_les = data[:, 3]
    yw = data[:, 4]
    nut = data[:, 5]

    return r_cg, z_cg, ur_les, uz_les, yw, nut

def read_h_wall_by_cg(summary_file):
    if not os.path.isfile(summary_file):
        raise FileNotFoundError(f"Wall-cell summary file not found: {summary_file}")

    data = np.genfromtxt(summary_file, delimiter=",", names=True, dtype=None, encoding=None)

    if "CG" not in data.dtype.names or "yw_wall_median" not in data.dtype.names:
        raise RuntimeError(
            f"Required columns CG and yw_wall_median not found in {summary_file}. "
            f"Available columns: {data.dtype.names}"
        )

    h_wall_by_cg = {}

    for row in np.atleast_1d(data):
        cg = str(row["CG"])
        h_wall = float(row["yw_wall_median"])

        if h_wall <= 0.0 or not np.isfinite(h_wall):
            raise ValueError(f"Invalid h_wall for CG={cg}: {h_wall}")

        h_wall_by_cg[cg] = h_wall

    print("Read wall-cell median sizes:", flush=True)
    for cg in sorted(h_wall_by_cg.keys(), key=lambda x: int(x)):
        print(f"  CG{cg}: h_wall = {h_wall_by_cg[cg]:.12e}", flush=True)

    return h_wall_by_cg

def read_velocity_result(case_path, pattern):
    files = fnmatch.filter(os.listdir(case_path), pattern)
    files = [os.path.join(case_path, f) for f in files]
    files.sort(key=lambda x: os.path.getmtime(x))

    if not files:
        raise FileNotFoundError(
            f"No VPP file found in {case_path} with pattern {pattern}"
        )

    path = files[-1]
    print(f"Using VPP file: {path}", flush=True)

    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    print("VPP columns:", data.dtype.names, flush=True)

    # In your VPP, MOOSE coordinates are:
    # x -> z_cg, y -> r_cg
    uz_cg = data["u"]
    ur_cg = data["v"]
    z_cg = data["x"]
    r_cg = data["y"]

    return ur_cg, uz_cg, r_cg, z_cg, path


def velocity_mse(ur_les, uz_les, ur_cg, uz_cg, z_cg, yw, yw_wl):
    dome = z_cg < 0
    keep = yw[dome] > yw_wl

    if keep.sum() == 0:
        return np.nan

    u_mse = (ur_les[dome] - ur_cg[dome]) ** 2
    v_mse = (uz_les[dome] - uz_cg[dome]) ** 2

    return float(u_mse[keep].mean() + v_mse[keep].mean())


def check_coordinates(label, r_ref, z_ref, r_res, z_res, tol=1e-10):
    if r_ref.size != r_res.size:
        raise ValueError(
            f"{label}: size mismatch: result has {r_res.size}, mapped has {r_ref.size}"
        )

    dr = float(np.max(np.abs(r_res - r_ref)))
    dz = float(np.max(np.abs(z_res - z_ref)))

    print(f"{label}: max |r_res-r_mapped| = {dr:.12e}", flush=True)
    print(f"{label}: max |z_res-z_mapped| = {dz:.12e}", flush=True)

    if dr > tol or dz > tol:
        raise ValueError(f"{label}: VPP row order does not match mapped.csv")

    return dr, dz


# ============================================================
# Closure utilities
# ============================================================
def build_lm_from_raw(raw, yw):
    raw = np.asarray(raw, dtype=np.float64)

    if np.any(~np.isfinite(raw)):
        raise ValueError("Non-finite raw closure values")

    t = np.tanh(raw)

    log_factor = np.where(
        t < 0.0,
        np.log(C_EFF_MIN_FACTOR) * (-t),
        np.log(C_EFF_MAX_FACTOR) * t,
    )

    C_eff = KAPPA0 * np.exp(log_factor)
    lm = C_eff * yw

    return lm, C_eff


def save_lm_csv(z_cg, r_cg, lm, out_file):
    arr = np.column_stack((z_cg, r_cg, lm))
    np.savetxt(
        out_file,
        arr,
        delimiter=",",
        header="z,r,lm",
        comments="",
        fmt="%.12e",
    )


# ============================================================
# Folder and solver utilities
# ============================================================
def prepare_candidate_folder(template_dir, cand_dir):
    if not os.path.isdir(template_dir):
        raise FileNotFoundError(f"Template folder not found: {template_dir}")

    if os.path.exists(cand_dir):
        print(f"Removing existing candidate folder: {cand_dir}", flush=True)
        shutil.rmtree(cand_dir)

    print(f"Copying template to: {cand_dir}", flush=True)
    shutil.copytree(template_dir, cand_dir)


def clean_case_folder(case_path):
    for fname in [LOG_FILE, LM_FILE]:
        fpath = os.path.join(case_path, fname)
        if os.path.exists(fpath):
            os.remove(fpath)

    for fname in fnmatch.filter(os.listdir(case_path), VPP_PATTERN):
        os.remove(os.path.join(case_path, fname))


def search_last(file_name, string, n_lines=200):
    with open(file_name, "r", errors="ignore") as f:
        lines = f.readlines()[-n_lines:]
    return any(string in line for line in lines)


def run_case(case_path, run_script):
    str_err = "Solve Did NOT Converge"
    str_conv = "Solve Converged"
    str_ss = "Steady-State Solution Achieved"
    str_fin = "Finished Executing!!!"

    cmd = f'cd "{case_path}" && bash "{run_script}"'

    print(f"Running case in: {case_path}", flush=True)
    print(f"Using script: {run_script}", flush=True)

    start = time.time()

    try:
        ret = subprocess.call(
            cmd,
            shell=True,
            executable="/bin/bash",
            timeout=SOLVER_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"Solver timed out after {elapsed:.2f} sec", flush=True)
        return False, elapsed

    elapsed = time.time() - start

    print(f"Return code = {ret}", flush=True)
    print(f"Elapsed = {elapsed:.2f} sec", flush=True)

    log_path = os.path.join(case_path, LOG_FILE)

    if not os.path.exists(log_path):
        print("Log file not found.", flush=True)
        return False, elapsed

    has_err = search_last(log_path, str_err)
    has_conv = search_last(log_path, str_conv)
    has_ss = search_last(log_path, str_ss)
    has_fin = search_last(log_path, str_fin)

    print(f"has_err  = {has_err}", flush=True)
    print(f"has_conv = {has_conv}", flush=True)
    print(f"has_ss   = {has_ss}", flush=True)
    print(f"has_fin  = {has_fin}", flush=True)

    ok = (has_conv or has_ss) and has_fin and not has_err
    return ok, elapsed


# ============================================================
# Baseline and candidate evaluation
# ============================================================
def evaluate_baseline_error(baseline_dir, r_cg, z_cg, ur_les, uz_les, yw):
    ur_base, uz_base, r_base, z_base, baseline_vpp = read_velocity_result(
        baseline_dir,
        BASELINE_VPP_PATTERN,
    )

    check_coordinates("baseline", r_cg, z_cg, r_base, z_base)

    baseline_err = velocity_mse(
        ur_les,
        uz_les,
        ur_base,
        uz_base,
        z_base,
        yw,
        YW_WL,
    )

    return baseline_err, baseline_vpp


def run_one_case(RE, CG, h_wall_by_cg):
    print("\n" + "=" * 80, flush=True)
    print(f"Running RE={RE}, CG={CG}", flush=True)
    print("=" * 80, flush=True)

    case_root, template_dir, cand_dir, mapped_file, baseline_dir = get_paths(RE, CG)

    r_cg, z_cg, ur_les, uz_les, yw, nut = read_mapped(mapped_file)

    L_outer = float(np.max(yw))
    if L_outer <= 0.0 or not np.isfinite(L_outer):
        raise ValueError(f"Invalid L_outer = {L_outer}")

    if CG not in h_wall_by_cg:
        raise KeyError(f"No h_wall value found for CG={CG}")

    h_wall = float(h_wall_by_cg[CG])
    eta_h_scalar = h_wall / L_outer

    eta_y = yw / L_outer
    raw = C1 * eta_h_scalar + C2 * eta_y + C3
    lm, C_eff = build_lm_from_raw(raw, yw)

    print(f"L_outer = {L_outer:.12e}", flush=True)
    print(f"h_wall = {h_wall:.12e}", flush=True)
    print(f"eta_h = {eta_h_scalar:.12e}", flush=True)
    print(f"eta_y min/max = {eta_y.min():.12e}, {eta_y.max():.12e}", flush=True)
    print(f"raw min/max = {raw.min():.12e}, {raw.max():.12e}", flush=True)
    print(f"C_eff min/max = {C_eff.min():.12e}, {C_eff.max():.12e}", flush=True)
    print(f"lm min/max = {lm.min():.12e}, {lm.max():.12e}", flush=True)

    baseline_err, baseline_vpp = evaluate_baseline_error(
        baseline_dir,
        r_cg,
        z_cg,
        ur_les,
        uz_les,
        yw,
    )

    print(f"Baseline MSE = {baseline_err:.12e}", flush=True)

    prepare_candidate_folder(template_dir, cand_dir)
    clean_case_folder(cand_dir)

    lm_path = os.path.join(cand_dir, LM_FILE)
    save_lm_csv(z_cg, r_cg, lm, lm_path)

    print(f"Wrote: {lm_path}", flush=True)

    run_mode = "steady"
    ok, steady_elapsed = run_case(cand_dir, STEADY_RUN_SCRIPT)
    unsteady_elapsed = 0.0

    if not ok:
        print("Steady failed; trying unsteady fallback.", flush=True)

        clean_case_folder(cand_dir)
        save_lm_csv(z_cg, r_cg, lm, lm_path)

        run_mode = "unsteady_fallback"
        ok, unsteady_elapsed = run_case(cand_dir, UNSTEADY_RUN_SCRIPT)

    if not ok:
        return {
            "RE": RE,
            "CG": CG,
            "np": NP_BY_CG[CG],
            "status": "failed",
            "run_mode": run_mode,
            "baseline_mse": baseline_err,
            "candidate_mse": np.nan,
            "normalized_mse": np.nan,
            "L_outer": L_outer,
            "h_wall": h_wall,
            "eta_h": eta_h_scalar,
            "eta_y_min": float(np.min(eta_y)),
            "eta_y_max": float(np.max(eta_y)),
            "raw_min": float(np.min(raw)),
            "raw_max": float(np.max(raw)),
            "C_eff_min": float(np.min(C_eff)),
            "C_eff_max": float(np.max(C_eff)),
            "lm_min": float(np.min(lm)),
            "lm_max": float(np.max(lm)),
            "steady_elapsed": steady_elapsed,
            "unsteady_elapsed": unsteady_elapsed,
            "candidate_folder": cand_dir,
            "candidate_vpp": "",
            "baseline_vpp": baseline_vpp,
        }

    ur_res, uz_res, r_res, z_res, candidate_vpp = read_velocity_result(
        cand_dir,
        VPP_PATTERN,
    )

    check_coordinates("candidate", r_cg, z_cg, r_res, z_res)

    candidate_err = velocity_mse(
        ur_les,
        uz_les,
        ur_res,
        uz_res,
        z_res,
        yw,
        YW_WL,
    )

    normalized = candidate_err / baseline_err

    print(f"Candidate MSE = {candidate_err:.12e}", flush=True)
    print(f"Normalized MSE = {normalized:.12e}", flush=True)

    return {
        "RE": RE,
        "CG": CG,
        "np": NP_BY_CG[CG],
        "status": "ok",
        "run_mode": run_mode,
        "baseline_mse": baseline_err,
        "candidate_mse": candidate_err,
        "normalized_mse": normalized,
        "L_outer": L_outer,
        "h_wall": h_wall,
        "eta_h": eta_h_scalar,
        "eta_y_min": float(np.min(eta_y)),
        "eta_y_max": float(np.max(eta_y)),
        "raw_min": float(np.min(raw)),
        "raw_max": float(np.max(raw)),
        "C_eff_min": float(np.min(C_eff)),
        "C_eff_max": float(np.max(C_eff)),
        "lm_min": float(np.min(lm)),
        "lm_max": float(np.max(lm)),
        "steady_elapsed": steady_elapsed,
        "unsteady_elapsed": unsteady_elapsed,
        "candidate_folder": cand_dir,
        "candidate_vpp": candidate_vpp,
        "baseline_vpp": baseline_vpp,
    }


def write_summary(rows):
    fieldnames = [
        "RE",
        "CG",
        "np",
        "status",
        "run_mode",
        "baseline_mse",
        "candidate_mse",
        "normalized_mse",
        "L_outer",
        "h_wall",
        "eta_h",
        "eta_y_min",
        "eta_y_max",
        "raw_min",
        "raw_max",
        "C_eff_min",
        "C_eff_max",
        "lm_min",
        "lm_max",
        "steady_elapsed",
        "unsteady_elapsed",
        "candidate_folder",
        "candidate_vpp",
        "baseline_vpp",
    ]

    with open(SUMMARY_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nWrote summary: {SUMMARY_FILE}", flush=True)


# ============================================================
# Main
# ============================================================
def main():
    print("===== CSV wall-grid candidate: all cases =====", flush=True)
    print(f"PROJECT_ROOT = {PROJECT_ROOT}", flush=True)
    print(f"Candidate: raw = {C1}*eta_h + {C2}*eta_y + {C3}", flush=True)
    print(f"Summary file: {SUMMARY_FILE}", flush=True)

    h_wall_by_cg = read_h_wall_by_cg(WALL_CELL_SUMMARY_FILE)

    rows = []

    for RE in RE_LIST:
        for CG in CG_LIST:
            try:
                row = run_one_case(RE, CG, h_wall_by_cg)
            except Exception as exc:
                print(f"FAILED before/inside run for RE={RE}, CG={CG}: {exc}", flush=True)
                row = {
                    "RE": RE,
                    "CG": CG,
                    "np": NP_BY_CG[CG],
                    "status": "exception",
                    "run_mode": "exception",
                    "baseline_mse": np.nan,
                    "candidate_mse": np.nan,
                    "normalized_mse": np.nan,
                    "L_outer": np.nan,
                    "h_wall": np.nan,
                    "eta_h": np.nan,
                    "eta_y_min": np.nan,
                    "eta_y_max": np.nan,
                    "raw_min": np.nan,
                    "raw_max": np.nan,
                    "C_eff_min": np.nan,
                    "C_eff_max": np.nan,
                    "lm_min": np.nan,
                    "lm_max": np.nan,
                    "steady_elapsed": np.nan,
                    "unsteady_elapsed": np.nan,
                    "candidate_folder": "",
                    "candidate_vpp": "",
                    "baseline_vpp": "",
                }

            rows.append(row)
            write_summary(rows)

            print("\nCurrent completed results:", flush=True)
            for r in rows:
                print(
                    f"RE={r['RE']:>5}, CG={r['CG']}, "
                    f"status={r['status']}, mode={r['run_mode']}, "
                    f"norm={r['normalized_mse']}",
                    flush=True,
                )

    ok_rows = [r for r in rows if r["status"] == "ok"]

    print("\n===== Final summary =====", flush=True)
    print(f"Completed OK: {len(ok_rows)} / {len(rows)}", flush=True)

    if ok_rows:
        norms = np.array([float(r["normalized_mse"]) for r in ok_rows], dtype=float)
        print(f"Mean normalized MSE = {np.mean(norms):.12e}", flush=True)
        print(f"Min normalized MSE  = {np.min(norms):.12e}", flush=True)
        print(f"Max normalized MSE  = {np.max(norms):.12e}", flush=True)

    print("===== Done =====", flush=True)


if __name__ == "__main__":
    main()