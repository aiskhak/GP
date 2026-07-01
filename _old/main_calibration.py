#!/usr/bin/env python3
"""
Deterministic calibration of linear closure models.

Run from the project root, e.g.

    python -u main_calibration.py --chunk 1 2>&1 | tee linear_chunk1.log

Models:
    Grid-aware: raw = C1*eta_h + C2*eta_y + C3
    Wall-only:  raw = C2*eta_y + C3  (equivalent to C1 = 0)

This is adapted from the tournament workflow: it reuses the same mapped data,
baseline normalization, lm_pred.csv generation, steady/unsteady solver calls,
and velocity-MSE fitness definition.
"""

import os
import fnmatch
import time
import json
import shutil
import subprocess
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "--chunk",
    required=True,
    type=int,
    choices=[1, 2, 3, 4],
    help="Candidate chunk to run: 1, 2, 3, or 4."
)
args = parser.parse_args()
CHUNK_ID = args.chunk


# ============================================================
# USER SETTINGS
# ============================================================
PROJECT_ROOT = os.path.abspath(".")

RE_LIST = ["3000", "3413", "5963", "7912", "9000", "10622", "12819", "14000"]
TEST_RE_LIST = ["5963", "12819"]

# Reduced set for initial coarse screening. Change if needed.
SCREEN_RE_LIST = ["3413", "9000", "14000"]
SCREEN_CG_LIST = ["1", "2", "3", "4"]

CASES = []
for RE in RE_LIST:
    CASES += [
        {"RE": RE, "CG": "1", "np": 4},
        {"RE": RE, "CG": "2", "np": 4},
        {"RE": RE, "CG": "3", "np": 8},
        {"RE": RE, "CG": "4", "np": 12},
    ]

BASE_RUN_TAG = "linear_refine_best_screen_C1etaH_C2etaY_C3_CeffMin005"
RUN_TAG = f"{BASE_RUN_TAG}_chunk{CHUNK_ID}"
RESULTS_DIR = os.path.join(PROJECT_ROOT, f"linear_results_{RUN_TAG}")
os.makedirs(RESULTS_DIR, exist_ok=True)

CORE_LIMIT = 127
SOLVER_TIMEOUT = 2700.0
YW_WL = 0.0

KAPPA0 = 0.41
C_EFF_MIN_FACTOR = 0.05
C_EFF_MAX_FACTOR = 2.0
C_EFF_MIN = C_EFF_MIN_FACTOR * KAPPA0
C_EFF_MAX = C_EFF_MAX_FACTOR * KAPPA0

BIG_PENALTY = 1.0e9

# Maximum number of candidates evaluated at once for one case.
# For CG4, 8 candidates use 8*12 = 96 MPI ranks.
N_PARALLEL_CANDIDATES = 8

STEADY_RUN_SCRIPT = "fv_app_run.sh"
UNSTEADY_RUN_SCRIPT = "fv_app_run_unsteady.sh"

BASELINE_VPP_PATTERN = "tamu_2d_fv_csv_vpp_*.csv"
VPP_PATTERN = "tamu_2d_fv_gp_csv_vpp_*.csv"
LOG_FILE = "log.csv"
LM_FILE = "lm_pred.csv"

WALL_CELL_SUMMARY_FILE = os.path.join(
    PROJECT_ROOT,
    "3_wall_cells",
    "wall_cells_median_summary.csv",
)

# Local perturbation study around previously discovered good linear closures.
LOCAL_CENTERS = [
    # Wall-only previous winners:
    # raw = C2*eta_y + C3
    {"model": "wall", "C1": 0.0, "C2": -5.3, "C3": -0.1886769525},
    {"model": "wall", "C1": 0.0, "C2": -5.3, "C3": -0.2833383671},
    {"model": "wall", "C1": 0.0, "C2": -3.0, "C3": -0.3790154144},
    {"model": "wall", "C1": 0.0, "C2": -2.0, "C3": -0.3859274008},

    # Grid-aware interpretable variants:
    # raw = C1*eta_h + C2*eta_y + C3
    {"model": "grid", "C1": 1.0, "C2": -5.3, "C3": 0.0},
    {"model": "grid", "C1": 1.0, "C2": -5.3, "C3": -0.3334771200},
    {"model": "grid", "C1": 1.0, "C2": -3.0, "C3": -0.38},
]

C1_DELTA = [-1.0, 0.0, 1.0]
C2_DELTA = [-0.5, 0.0, 0.5]
C3_DELTA = [-0.05, 0.0, 0.05]

TOP_K_SCREEN_EACH_MODEL = 4
TOP_K_FINAL = 8

# Small refinement around the best screen region:
# screen best was approximately raw = -4.8*eta_y - 0.1386769525.
REFINE_C1_VALUES = [-1.1, -1.0, -0.9]
REFINE_C2_VALUES = [-5.2, -5.0, -4.8]
REFINE_C3_VALUES = [-0.20, -0.16, -0.14, -0.12, -0.08]


# ============================================================
# DATA AND FILE UTILITIES
# ============================================================
def get_case_paths(RE, CG):
    case_root = os.path.join(PROJECT_ROOT, "runs", RE, CG)
    template_dir = os.path.join(case_root, "template")
    baseline_dir = os.path.join(PROJECT_ROOT, "1_mixing_length", RE, CG)
    mapped_file = os.path.join(PROJECT_ROOT, "2_mapped", RE, CG, "mapped.csv")
    return case_root, template_dir, baseline_dir, mapped_file


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


def read_mapped(mapped_file):
    print("Reading mapped file:", mapped_file, flush=True)
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


def read_velocity_result(case_path, vpp_pattern):
    files = fnmatch.filter(os.listdir(case_path), vpp_pattern)
    files = [os.path.join(case_path, f) for f in files]
    files.sort(key=lambda x: os.path.getmtime(x))

    if len(files) == 0:
        raise FileNotFoundError(
            f"No VPP file found in {case_path} with pattern {vpp_pattern}"
        )

    res_path = files[-1]
    print("Using VPP file:", res_path, flush=True)

    data = np.genfromtxt(res_path, delimiter=",", names=True, dtype=np.float64)
    print("VPP columns:", data.dtype.names, flush=True)

    uz_cg = data["u"]
    ur_cg = data["v"]
    z_cg = data["x"]
    r_cg = data["y"]

    return ur_cg, uz_cg, r_cg, z_cg


def velocity_mse(ur_les, uz_les, ur_cg, uz_cg, z_cg, yw, yw_wl):
    dome = z_cg < 0
    keep = yw[dome] > yw_wl

    if keep.sum() == 0:
        return BIG_PENALTY

    u_mse = (ur_les[dome] - ur_cg[dome]) ** 2
    v_mse = (uz_les[dome] - uz_cg[dome]) ** 2
    return u_mse[keep].mean() + v_mse[keep].mean()


def evaluate_baseline_mixing_length(case):
    print("\n===== Evaluating baseline mixing-length case =====", flush=True)
    print("Baseline dir:", case["baseline_dir"], flush=True)

    ur_base, uz_base, r_base, z_base = read_velocity_result(
        case["baseline_dir"],
        BASELINE_VPP_PATTERN,
    )

    if r_base.size != case["r_cg"].size:
        raise ValueError(
            f"Baseline size mismatch for RE={case['RE']}, CG={case['CG']}: "
            f"result has {r_base.size}, mapped has {case['r_cg'].size}"
        )

    dr = np.max(np.abs(r_base - case["r_cg"]))
    dz = np.max(np.abs(z_base - case["z_cg"]))

    print("baseline coordinate check: max |r_base-r_cg| =", dr, flush=True)
    print("baseline coordinate check: max |z_base-z_cg| =", dz, flush=True)

    if dr > 1e-10 or dz > 1e-10:
        raise ValueError(
            f"Baseline VPP row order mismatch for RE={case['RE']}, CG={case['CG']}"
        )

    baseline_err = velocity_mse(
        case["ur_les"],
        case["uz_les"],
        ur_base,
        uz_base,
        z_base,
        case["yw"],
        YW_WL,
    )

    print(
        f"Baseline MSE for RE={case['RE']}, CG={case['CG']} = "
        f"{baseline_err:.8e}",
        flush=True,
    )

    return baseline_err


# ============================================================
# SOLVER UTILITIES
# ============================================================
def prepare_candidate_case(case_root, local_id):
    template_dir = os.path.join(case_root, "template")
    cand_dir = os.path.join(case_root, f"cand_{local_id:03d}")

    if os.path.exists(cand_dir):
        shutil.rmtree(cand_dir)

    shutil.copytree(template_dir, cand_dir)
    return cand_dir


def ensure_candidate_folders(case_root, n_candidates):
    for local_id in range(n_candidates):
        prepare_candidate_case(case_root, local_id)


def clean_case_folder(case_path):
    for fname in (LOG_FILE, LM_FILE):
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
    print("Running case in:", case_path, "with", run_script, flush=True)

    solver_start = time.time()

    try:
        ret = subprocess.call(
            cmd,
            shell=True,
            executable="/bin/bash",
            timeout=SOLVER_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        solver_elapsed = time.time() - solver_start
        print("Solver timed out", flush=True)
        print("Elapsed:", round(solver_elapsed, 2), "sec", flush=True)
        return False, solver_elapsed

    solver_elapsed = time.time() - solver_start

    print("Return code =", ret, flush=True)
    print("Elapsed:", round(solver_elapsed, 2), "sec", flush=True)

    log_path = os.path.join(case_path, LOG_FILE)
    if not os.path.exists(log_path):
        print("Log file not found", flush=True)
        return False, solver_elapsed

    has_err = search_last(log_path, str_err)
    has_conv = search_last(log_path, str_conv)
    has_ss = search_last(log_path, str_ss)
    has_fin = search_last(log_path, str_fin)

    print("has_err =", has_err, flush=True)
    print("has_conv =", has_conv, flush=True)
    print("has_ss =", has_ss, flush=True)
    print("has_fin =", has_fin, flush=True)

    is_done = (has_conv or has_ss) and has_fin
    return is_done and not has_err, solver_elapsed


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
# CLOSURE AND EVALUATION
# ============================================================
def build_lm_from_raw(raw, yw):
    raw = np.asarray(raw, dtype=np.float64).copy()

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


def candidate_expression(candidate):
    if candidate["model"] == "grid":
        return (
            f"raw = {candidate['C1']:.12g}*eta_h "
            f"+ {candidate['C2']:.12g}*eta_y "
            f"+ {candidate['C3']:.12g}"
        )
    if candidate["model"] == "wall":
        return (
            f"raw = {candidate['C2']:.12g}*eta_y "
            f"+ {candidate['C3']:.12g}"
        )
    raise ValueError(f"Unknown model type: {candidate['model']}")


def compute_raw(candidate, case):
    if candidate["model"] == "grid":
        return (
            candidate["C1"] * case["eta_h"]
            + candidate["C2"] * case["eta_y"]
            + candidate["C3"]
        )
    if candidate["model"] == "wall":
        return candidate["C2"] * case["eta_y"] + candidate["C3"]
    raise ValueError(f"Unknown model type: {candidate['model']}")


def evaluate_candidate_case_worker(args):
    local_id, candidate, case = args

    case_path = os.path.join(case["case_root"], f"cand_{local_id:03d}")

    print(
        f"\n===== Linear candidate local_id={local_id:03d}, "
        f"RE={case['RE']}, CG={case['CG']} =====",
        flush=True,
    )
    print(candidate_expression(candidate), flush=True)

    steady_elapsed = 0.0
    unsteady_elapsed = 0.0

    try:
        raw = compute_raw(candidate, case)

        if raw.shape != case["r_cg"].shape:
            raise ValueError(f"raw shape mismatch: {raw.shape} vs {case['r_cg'].shape}")

        lm, C_eff = build_lm_from_raw(raw, case["yw"])

        print("raw min/max =", float(np.min(raw)), float(np.max(raw)), flush=True)
        print("C_eff min/max =", float(np.min(C_eff)), float(np.max(C_eff)), flush=True)
        print("lm min/max =", float(np.min(lm)), float(np.max(lm)), flush=True)

        clean_case_folder(case_path)
        save_lm_csv(case["z_cg"], case["r_cg"], lm, os.path.join(case_path, LM_FILE))

        run_mode = "steady"
        ok, steady_elapsed = run_case(case_path, STEADY_RUN_SCRIPT)

        if not ok:
            print("Steady failed; trying unsteady fallback", flush=True)
            clean_case_folder(case_path)
            save_lm_csv(case["z_cg"], case["r_cg"], lm, os.path.join(case_path, LM_FILE))
            run_mode = "unsteady_fallback"
            ok, unsteady_elapsed = run_case(case_path, UNSTEADY_RUN_SCRIPT)

        total_solver_elapsed = steady_elapsed + unsteady_elapsed

        if not ok:
            raise RuntimeError("Both steady and unsteady fallback failed")

        ur_cg, uz_cg, r_res, z_res = read_velocity_result(case_path, VPP_PATTERN)

        if r_res.size != case["r_cg"].size:
            raise ValueError(f"Size mismatch: result has {r_res.size}, mapped has {case['r_cg'].size}")

        dr = np.max(np.abs(r_res - case["r_cg"]))
        dz = np.max(np.abs(z_res - case["z_cg"]))

        print("coordinate check: max |r_res-r_cg| =", dr, flush=True)
        print("coordinate check: max |z_res-z_cg| =", dz, flush=True)

        if dr > 1e-10 or dz > 1e-10:
            raise ValueError("VPP row order does not match mapped.csv row order")

        raw_fit = velocity_mse(
            case["ur_les"],
            case["uz_les"],
            ur_cg,
            uz_cg,
            z_res,
            case["yw"],
            YW_WL,
        )

        norm_fit = raw_fit / case["baseline_err"]

        return {
            "candidate_id": candidate["candidate_id"],
            "local_id": local_id,
            "model": candidate["model"],
            "C1": candidate.get("C1", 0.0),
            "C2": candidate["C2"],
            "C3": candidate["C3"],
            "RE": case["RE"],
            "CG": case["CG"],
            "raw_fitness": float(raw_fit),
            "baseline_fitness": float(case["baseline_err"]),
            "norm_fitness": float(norm_fit),
            "run_mode": run_mode,
            "steady_elapsed": float(steady_elapsed),
            "unsteady_elapsed": float(unsteady_elapsed),
            "total_solver_elapsed": float(total_solver_elapsed),
            "raw_min": float(np.min(raw)),
            "raw_max": float(np.max(raw)),
            "C_eff_min": float(np.min(C_eff)),
            "C_eff_max": float(np.max(C_eff)),
            "Ceff_frac_within_5pct_lower": float(np.mean(C_eff < 1.05*C_EFF_MIN)),
            "Ceff_frac_within_10pct_lower": float(np.mean(C_eff < 1.10*C_EFF_MIN)),
            "expr": candidate_expression(candidate),
        }

    except Exception as exc:
        print("Exception during evaluation:", exc, flush=True)

        return {
            "candidate_id": candidate["candidate_id"],
            "local_id": local_id,
            "model": candidate["model"],
            "C1": candidate.get("C1", 0.0),
            "C2": candidate["C2"],
            "C3": candidate["C3"],
            "RE": case["RE"],
            "CG": case["CG"],
            "raw_fitness": BIG_PENALTY,
            "baseline_fitness": float(case["baseline_err"]),
            "norm_fitness": BIG_PENALTY,
            "run_mode": "failed",
            "steady_elapsed": float(steady_elapsed) if np.isfinite(steady_elapsed) else -1.0,
            "unsteady_elapsed": float(unsteady_elapsed) if np.isfinite(unsteady_elapsed) else -1.0,
            "total_solver_elapsed": float(steady_elapsed + unsteady_elapsed) if np.isfinite(steady_elapsed + unsteady_elapsed) else -1.0,
            "raw_min": np.nan,
            "raw_max": np.nan,
            "C_eff_min": np.nan,
            "C_eff_max": np.nan,
            "Ceff_frac_within_5pct_lower": np.nan,
            "Ceff_frac_within_10pct_lower": np.nan,
            "expr": candidate_expression(candidate),
        }


def make_case_batches(case_data, population_size, core_limit):
    batches = []
    current = []
    current_cores = 0

    for case in case_data:
        needed = population_size * case["np"]

        if current and current_cores + needed > core_limit:
            batches.append(current)
            current = []
            current_cores = 0

        if needed > core_limit:
            raise RuntimeError(
                f"Single case RE={case['RE']}, CG={case['CG']} requires "
                f"{needed} cores for population_size={population_size}, "
                f"which exceeds CORE_LIMIT={core_limit}"
            )

        current.append(case)
        current_cores += needed

    if current:
        batches.append(current)

    return batches


def evaluate_candidates(candidates, case_data, stage_name):
    print(
        f"\n===== Evaluating {len(candidates)} candidates for stage {stage_name} "
        f"on {len(case_data)} cases =====",
        flush=True,
    )

    details_path = os.path.join(RESULTS_DIR, f"{RUN_TAG}_{stage_name}_details.csv")
    summary_path = os.path.join(RESULTS_DIR, f"{RUN_TAG}_{stage_name}_summary.csv")

    all_results = []

    for cand_start in range(0, len(candidates), N_PARALLEL_CANDIDATES):
        cand_batch = candidates[cand_start:cand_start + N_PARALLEL_CANDIDATES]

        print(
            f"\n===== Candidate batch {cand_start}..{cand_start + len(cand_batch)-1} =====",
            flush=True,
        )

        for case in case_data:
            ensure_candidate_folders(case["case_root"], len(cand_batch))

        case_batches = make_case_batches(
            case_data,
            population_size=len(cand_batch),
            core_limit=CORE_LIMIT,
        )

        for ib, case_batch in enumerate(case_batches):
            print(
                f"\n===== Stage {stage_name}, candidate batch starting {cand_start}, "
                f"case batch {ib+1}/{len(case_batches)} =====",
                flush=True,
            )

            tasks = []
            for local_id, candidate in enumerate(cand_batch):
                for case in case_batch:
                    tasks.append((local_id, candidate, case))

            used_cores = sum(task[2]["np"] for task in tasks)
            if used_cores > CORE_LIMIT:
                raise RuntimeError(f"Internal error: used_cores={used_cores} > CORE_LIMIT={CORE_LIMIT}")

            print(f"Running {len(tasks)} tasks with {used_cores} MPI ranks", flush=True)

            with ProcessPoolExecutor(max_workers=len(tasks)) as ex:
                futures = [ex.submit(evaluate_candidate_case_worker, task) for task in tasks]

                for fut in as_completed(futures):
                    all_results.append(fut.result())
                    
            # Live checkpoint after each case batch.
            checkpoint_details_path = os.path.join(
                RESULTS_DIR,
                f"{RUN_TAG}_{stage_name}_details_checkpoint.csv",
            )
            checkpoint_summary_path = os.path.join(
                RESULTS_DIR,
                f"{RUN_TAG}_{stage_name}_summary_checkpoint.csv",
            )

            write_details_csv(checkpoint_details_path, all_results)
            checkpoint_summary_rows = summarize_results(all_results, candidates)
            write_summary_csv(checkpoint_summary_path, checkpoint_summary_rows)

            print("\n===== LIVE CHECKPOINT =====", flush=True)
            print(
                f"Chunk {CHUNK_ID}, stage {stage_name}, "
                f"candidate batch starting {cand_start}, "
                f"case batch {ib+1}/{len(case_batches)}",
                flush=True,
            )
            print(f"Completed result rows: {len(all_results)}", flush=True)
            print("Current best partial candidates:", flush=True)

            for row in checkpoint_summary_rows[:5]:
                print(
                    f"  n_cases={row['n_cases']:3d}, "
                    f"mean={row['mean_norm_fitness']:.8f}, "
                    f"{row['expr']}",
                    flush=True,
                )

            print(f"Checkpoint details: {checkpoint_details_path}", flush=True)
            print(f"Checkpoint summary: {checkpoint_summary_path}", flush=True)
            print("===========================\n", flush=True)

    write_details_csv(details_path, all_results)
    summary_rows = summarize_results(all_results, candidates)
    write_summary_csv(summary_path, summary_rows)

    print(f"\nWrote details: {details_path}", flush=True)
    print(f"Wrote summary: {summary_path}", flush=True)

    return summary_rows, all_results


def summarize_results(results, candidates):
    by_id = {}
    for r in results:
        by_id.setdefault(r["candidate_id"], []).append(r)

    summary_rows = []

    for candidate in candidates:
        cid = candidate["candidate_id"]
        vals = by_id.get(cid, [])

        mean_norm = float(np.mean([v["norm_fitness"] for v in vals])) if vals else BIG_PENALTY

        by_cg = {}
        for cg in ["1", "2", "3", "4"]:
            cg_vals = [v["norm_fitness"] for v in vals if v["CG"] == cg]
            by_cg[cg] = float(np.mean(cg_vals)) if cg_vals else np.nan

        lower5 = [v["Ceff_frac_within_5pct_lower"] for v in vals if np.isfinite(v["Ceff_frac_within_5pct_lower"])]
        lower10 = [v["Ceff_frac_within_10pct_lower"] for v in vals if np.isfinite(v["Ceff_frac_within_10pct_lower"])]

        summary_rows.append({
            "candidate_id": cid,
            "model": candidate["model"],
            "C1": candidate.get("C1", 0.0),
            "C2": candidate["C2"],
            "C3": candidate["C3"],
            "mean_norm_fitness": mean_norm,
            "CG1_mean": by_cg["1"],
            "CG2_mean": by_cg["2"],
            "CG3_mean": by_cg["3"],
            "CG4_mean": by_cg["4"],
            "mean_frac_5pct_lower": float(np.mean(lower5)) if lower5 else np.nan,
            "mean_frac_10pct_lower": float(np.mean(lower10)) if lower10 else np.nan,
            "n_cases": len(vals),
            "expr": candidate_expression(candidate),
        })

    summary_rows.sort(key=lambda x: x["mean_norm_fitness"])
    return summary_rows


def write_details_csv(path, rows):
    header = [
        "candidate_id", "local_id", "model", "C1", "C2", "C3", "RE", "CG",
        "raw_fitness", "baseline_fitness", "norm_fitness", "run_mode",
        "steady_elapsed", "unsteady_elapsed", "total_solver_elapsed",
        "raw_min", "raw_max", "C_eff_min", "C_eff_max",
        "Ceff_frac_within_5pct_lower", "Ceff_frac_within_10pct_lower", "expr",
    ]

    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(
                f'{r["candidate_id"]},{r["local_id"]},{r["model"]},'
                f'{r["C1"]:.12e},{r["C2"]:.12e},{r["C3"]:.12e},'
                f'{r["RE"]},{r["CG"]},'
                f'{r["raw_fitness"]:.12e},{r["baseline_fitness"]:.12e},'
                f'{r["norm_fitness"]:.12e},{r["run_mode"]},'
                f'{r["steady_elapsed"]:.6e},{r["unsteady_elapsed"]:.6e},'
                f'{r["total_solver_elapsed"]:.6e},'
                f'{r["raw_min"]:.12e},{r["raw_max"]:.12e},'
                f'{r["C_eff_min"]:.12e},{r["C_eff_max"]:.12e},'
                f'{r["Ceff_frac_within_5pct_lower"]:.12e},'
                f'{r["Ceff_frac_within_10pct_lower"]:.12e},'
                f'"{r["expr"]}"\n'
            )


def write_summary_csv(path, rows):
    header = [
        "rank", "candidate_id", "model", "C1", "C2", "C3",
        "mean_norm_fitness", "CG1_mean", "CG2_mean", "CG3_mean", "CG4_mean",
        "mean_frac_5pct_lower", "mean_frac_10pct_lower", "n_cases", "expr",
    ]

    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for rank, r in enumerate(rows):
            f.write(
                f'{rank},{r["candidate_id"]},{r["model"]},'
                f'{r["C1"]:.12e},{r["C2"]:.12e},{r["C3"]:.12e},'
                f'{r["mean_norm_fitness"]:.12e},'
                f'{r["CG1_mean"]:.12e},{r["CG2_mean"]:.12e},'
                f'{r["CG3_mean"]:.12e},{r["CG4_mean"]:.12e},'
                f'{r["mean_frac_5pct_lower"]:.12e},'
                f'{r["mean_frac_10pct_lower"]:.12e},'
                f'{r["n_cases"]},"{r["expr"]}"\n'
            )


# ============================================================
# CANDIDATE GENERATION
# ============================================================
def make_grid_aware_candidates(C1_values, C2_values, C3_values, start_id=0):
    candidates = []
    cid = start_id
    for C1 in C1_values:
        for C2 in C2_values:
            for C3 in C3_values:
                candidates.append({
                    "candidate_id": cid,
                    "model": "grid",
                    "C1": float(C1),
                    "C2": float(C2),
                    "C3": float(C3),
                })
                cid += 1
    return candidates


def make_wall_only_candidates(C2_values, C3_values, start_id=0):
    candidates = []
    cid = start_id
    for C2 in C2_values:
        for C3 in C3_values:
            candidates.append({
                "candidate_id": cid,
                "model": "wall",
                "C1": 0.0,
                "C2": float(C2),
                "C3": float(C3),
            })
            cid += 1
    return candidates

def make_best_screen_refine_candidates(start_id=0):
    candidates = []
    cid = start_id

    # Grid-aware refinement:
    # raw = C1*eta_h + C2*eta_y + C3
    for C1 in REFINE_C1_VALUES:
        for C2 in REFINE_C2_VALUES:
            for C3 in REFINE_C3_VALUES:
                candidates.append({
                    "candidate_id": cid,
                    "model": "grid",
                    "C1": float(C1),
                    "C2": float(C2),
                    "C3": float(C3),
                })
                cid += 1

    # Wall-only refinement:
    # raw = C2*eta_y + C3, equivalent to C1 = 0
    for C2 in REFINE_C2_VALUES:
        for C3 in REFINE_C3_VALUES:
            candidates.append({
                "candidate_id": cid,
                "model": "wall",
                "C1": 0.0,
                "C2": float(C2),
                "C3": float(C3),
            })
            cid += 1

    return candidates

def select_candidate_chunk(all_candidates, chunk_id):
    """
    Split candidates into 4 simple chunks.

    Current candidate order from make_best_screen_refine_candidates:
      - first 125 are grid-aware candidates
      - last 25 are wall-only candidates

    Chunks:
      1: first half of grid-aware candidates
      2: second half of grid-aware candidates
      3: first half of wall-only candidates
      4: second half of wall-only candidates
    """
    grid_candidates = [c for c in all_candidates if c["model"] == "grid"]
    wall_candidates = [c for c in all_candidates if c["model"] == "wall"]

    grid_mid = (len(grid_candidates) + 1) // 2
    wall_mid = (len(wall_candidates) + 1) // 2

    if chunk_id == 1:
        selected = grid_candidates[:grid_mid]
    elif chunk_id == 2:
        selected = grid_candidates[grid_mid:]
    elif chunk_id == 3:
        selected = wall_candidates[:wall_mid]
    elif chunk_id == 4:
        selected = wall_candidates[wall_mid:]
    else:
        raise ValueError(f"Unknown chunk_id: {chunk_id}")

    # Renumber locally so candidate IDs are simple inside each chunk.
    for new_id, cand in enumerate(selected):
        cand["global_candidate_id"] = cand["candidate_id"]
        cand["candidate_id"] = new_id

    return selected

def make_local_candidates(centers, start_id=0):
    candidates = []
    seen = set()
    cid = start_id

    for center in centers:
        model = center["model"]

        if model == "grid":
            for d1 in C1_DELTA:
                for d2 in C2_DELTA:
                    for d3 in C3_DELTA:
                        C1 = center["C1"] + d1
                        C2 = center["C2"] + d2
                        C3 = center["C3"] + d3

                        key = (model, round(C1, 10), round(C2, 10), round(C3, 10))
                        if key in seen:
                            continue
                        seen.add(key)

                        candidates.append({
                            "candidate_id": cid,
                            "model": "grid",
                            "C1": float(C1),
                            "C2": float(C2),
                            "C3": float(C3),
                        })
                        cid += 1

        elif model == "wall":
            for d2 in C2_DELTA:
                for d3 in C3_DELTA:
                    C1 = 0.0
                    C2 = center["C2"] + d2
                    C3 = center["C3"] + d3

                    key = (model, round(C1, 10), round(C2, 10), round(C3, 10))
                    if key in seen:
                        continue
                    seen.add(key)

                    candidates.append({
                        "candidate_id": cid,
                        "model": "wall",
                        "C1": 0.0,
                        "C2": float(C2),
                        "C3": float(C3),
                    })
                    cid += 1

        else:
            raise ValueError(f"Unknown model type: {model}")

    return candidates

def linspace_around(center, half_width, n):
    return np.linspace(center - half_width, center + half_width, n)


def make_refine_candidates_from_best(best_rows, start_id=0):
    candidates = []
    cid = start_id
    seen = set()

    for row in best_rows:
        model = row["model"]

        if model == "grid":
            C1_values = linspace_around(row["C1"], REFINE_C1_HALF_WIDTH, REFINE_N_C1)
            C2_values = linspace_around(row["C2"], REFINE_C2_HALF_WIDTH, REFINE_N_C2)
            C3_values = linspace_around(row["C3"], REFINE_C3_HALF_WIDTH, REFINE_N_C3)

            for C1 in C1_values:
                for C2 in C2_values:
                    for C3 in C3_values:
                        key = ("grid", round(float(C1), 10), round(float(C2), 10), round(float(C3), 10))
                        if key in seen:
                            continue
                        seen.add(key)
                        candidates.append({
                            "candidate_id": cid,
                            "model": "grid",
                            "C1": float(C1),
                            "C2": float(C2),
                            "C3": float(C3),
                        })
                        cid += 1

        elif model == "wall":
            C2_values = linspace_around(row["C2"], REFINE_C2_HALF_WIDTH, REFINE_N_C2)
            C3_values = linspace_around(row["C3"], REFINE_C3_HALF_WIDTH, REFINE_N_C3)

            for C2 in C2_values:
                for C3 in C3_values:
                    key = ("wall", 0.0, round(float(C2), 10), round(float(C3), 10))
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append({
                        "candidate_id": cid,
                        "model": "wall",
                        "C1": 0.0,
                        "C2": float(C2),
                        "C3": float(C3),
                    })
                    cid += 1

    return candidates


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n===== Linear closure calibration =====", flush=True)
    print("Project root:", PROJECT_ROOT, flush=True)
    print("Results dir:", RESULTS_DIR, flush=True)

    h_wall_by_cg = read_h_wall_by_cg(WALL_CELL_SUMMARY_FILE)

    case_data = []
    train_case_data = []
    test_case_data = []
    screen_case_data = []

    for case in CASES:
        RE = case["RE"]
        CG = case["CG"]
        np_ranks = case["np"]

        case_root, template_dir, baseline_dir, mapped_file = get_case_paths(RE, CG)

        print(f"\n===== Preparing RE={RE}, CG={CG}, np={np_ranks} =====", flush=True)

        r_cg, z_cg, ur_les, uz_les, yw, nut = read_mapped(mapped_file)

        if CG not in h_wall_by_cg:
            raise KeyError(f"No h_wall value found for CG={CG}")

        h_wall = h_wall_by_cg[CG]
        L_outer = np.max(yw)

        if L_outer <= 0.0 or not np.isfinite(L_outer):
            raise ValueError(f"Invalid L_outer for RE={RE}, CG={CG}")

        eta_h_scalar = h_wall / L_outer
        eta_h = np.full_like(yw, eta_h_scalar, dtype=np.float64)
        eta_y = yw / L_outer

        print(f"h_wall = {h_wall:.12e}", flush=True)
        print(f"L_outer = {L_outer:.12e}", flush=True)
        print(f"eta_h = {eta_h_scalar:.12e}", flush=True)
        print(f"eta_y min/max = {eta_y.min():.12e}, {eta_y.max():.12e}", flush=True)

        case_entry = {
            "RE": RE,
            "CG": CG,
            "np": np_ranks,
            "case_root": case_root,
            "template_dir": template_dir,
            "baseline_dir": baseline_dir,
            "mapped_file": mapped_file,
            "r_cg": r_cg,
            "z_cg": z_cg,
            "ur_les": ur_les,
            "uz_les": uz_les,
            "yw": yw,
            "nut": nut,
            "h_wall": float(h_wall),
            "L_outer": float(L_outer),
            "eta_h": eta_h,
            "eta_y": eta_y,
            "eta_h_scalar": float(eta_h_scalar),
            "eta_y_min": float(np.min(eta_y)),
            "eta_y_max": float(np.max(eta_y)),
        }

        case_entry["baseline_err"] = evaluate_baseline_mixing_length(case_entry)

        case_data.append(case_entry)

        if RE in TEST_RE_LIST:
            test_case_data.append(case_entry)
        else:
            train_case_data.append(case_entry)

        if (RE in SCREEN_RE_LIST) and (CG in SCREEN_CG_LIST) and (RE not in TEST_RE_LIST):
            screen_case_data.append(case_entry)

    metadata = {
        "CHUNK_ID": CHUNK_ID,
        "RUN_TAG": RUN_TAG,
        "models": {
            "grid": "raw = C1*eta_h + C2*eta_y + C3",
            "wall": "raw = C2*eta_y + C3, equivalent to C1=0",
        },
        "RE_LIST": RE_LIST,
        "TEST_RE_LIST": TEST_RE_LIST,
        "SCREEN_RE_LIST": SCREEN_RE_LIST,
        "SCREEN_CG_LIST": SCREEN_CG_LIST,
        "KAPPA0": KAPPA0,
        "C_EFF_MIN_FACTOR": C_EFF_MIN_FACTOR,
        "C_EFF_MAX_FACTOR": C_EFF_MAX_FACTOR,
        "C_EFF_MIN": C_EFF_MIN,
        "C_EFF_MAX": C_EFF_MAX,
        "CORE_LIMIT": CORE_LIMIT,
        "N_PARALLEL_CANDIDATES": N_PARALLEL_CANDIDATES,
        "refinement_grid": {
            "C1_values": REFINE_C1_VALUES,
            "C2_values": REFINE_C2_VALUES,
            "C3_values": REFINE_C3_VALUES,
        },
        "cases": [
            {
                "RE": c["RE"],
                "CG": c["CG"],
                "np": c["np"],
                "baseline_err": c["baseline_err"],
                "h_wall": c["h_wall"],
                "L_outer": c["L_outer"],
                "eta_h_scalar": c["eta_h_scalar"],
                "eta_y_min": c["eta_y_min"],
                "eta_y_max": c["eta_y_max"],
            }
            for c in case_data
        ],
    }

    with open(os.path.join(RESULTS_DIR, f"{RUN_TAG}_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # ------------------------------------------------------------
    # Stage 1: refinement around best screen region on full training set.
    # ------------------------------------------------------------
    print("\n===== Stage 1: refinement around best screen region =====", flush=True)

    refine_candidates = make_grid_aware_candidates(
        REFINE_C1_VALUES,
        REFINE_C2_VALUES,
        REFINE_C3_VALUES,
        start_id=0,
    )

    print(f"Running candidate chunk: {CHUNK_ID}", flush=True)
    print(f"Number of refinement candidates in this chunk: {len(refine_candidates)}", flush=True)

    refine_summary, _ = evaluate_candidates(
        refine_candidates,
        train_case_data,
        stage_name="refine_train",
    )

    print("\n===== Top refined training candidates =====", flush=True)
    for row in refine_summary[:TOP_K_FINAL]:
        print(
            f"{row['mean_norm_fitness']:.8f} | {row['expr']}",
            flush=True,
        )

    # ------------------------------------------------------------
    # Stage 3: held-out test evaluation of top refined candidates.
    # ------------------------------------------------------------
    print("\n===== Stage 3: final held-out test evaluation =====", flush=True)

    top_rows = refine_summary[:TOP_K_FINAL]
    final_candidates = []

    for i, row in enumerate(top_rows):
        final_candidates.append({
            "candidate_id": i,
            "model": row["model"],
            "C1": float(row["C1"]),
            "C2": float(row["C2"]),
            "C3": float(row["C3"]),
        })

    final_summary, _ = evaluate_candidates(
        final_candidates,
        test_case_data,
        stage_name="final_test",
    )

    train_score_by_params = {}
    for row in refine_summary:
        key = (
            row["model"],
            round(row["C1"], 12),
            round(row["C2"], 12),
            round(row["C3"], 12),
        )
        train_score_by_params[key] = row

    combined_path = os.path.join(RESULTS_DIR, f"{RUN_TAG}_final_combined_summary.csv")
    with open(combined_path, "w") as f:
        f.write(
            "rank_test,model,C1,C2,C3,train_mean_norm,test_mean_norm,"
            "test_CG1,test_CG2,test_CG3,test_CG4,"
            "test_frac_5pct_lower,test_frac_10pct_lower,expr\n"
        )

        for rank, test_row in enumerate(final_summary):
            key = (
                test_row["model"],
                round(test_row["C1"], 12),
                round(test_row["C2"], 12),
                round(test_row["C3"], 12),
            )
            train_row = train_score_by_params.get(key)
            train_mean = train_row["mean_norm_fitness"] if train_row else np.nan

            f.write(
                f'{rank},{test_row["model"]},'
                f'{test_row["C1"]:.12e},{test_row["C2"]:.12e},{test_row["C3"]:.12e},'
                f'{train_mean:.12e},{test_row["mean_norm_fitness"]:.12e},'
                f'{test_row["CG1_mean"]:.12e},{test_row["CG2_mean"]:.12e},'
                f'{test_row["CG3_mean"]:.12e},{test_row["CG4_mean"]:.12e},'
                f'{test_row["mean_frac_5pct_lower"]:.12e},'
                f'{test_row["mean_frac_10pct_lower"]:.12e},'
                f'"{test_row["expr"]}"\n'
            )

    print("\n===== Final test ranking =====", flush=True)
    for row in final_summary:
        key = (
            row["model"],
            round(row["C1"], 12),
            round(row["C2"], 12),
            round(row["C3"], 12),
        )
        train_row = train_score_by_params.get(key)
        train_mean = train_row["mean_norm_fitness"] if train_row else np.nan
        print(
            f"train={train_mean:.8f}, test={row['mean_norm_fitness']:.8f} | {row['expr']}",
            flush=True,
        )

    print("\nWrote final combined summary:", combined_path, flush=True)
    print("\n===== Done =====", flush=True)


if __name__ == "__main__":
    main()
