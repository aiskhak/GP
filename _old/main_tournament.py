# # python -u main_tournament.py 2>&1 | tee gp_run_CG1234_asymLogCeff_etaH_etaY_allDome_train6Re_test2Re_champion_tournament_10seeds_seed200_avg5.log

# salloc -p ksu-mne-train.q --nodelist=warlock35 --nodes=1 --ntasks=128 --mem=700G --time=72:00:00

# CHECK FILES
'''
for RE in 3000 3413 5963 7912 9000 10622 12819 14000; do
  for CG in 1 2 3 4; do
    echo "Checking RE=$RE CG=$CG"
    ls runs/$RE/$CG/template >/dev/null || echo "missing template"
    ls 2_mapped/$RE/$CG/mapped.csv >/dev/null || echo "missing mapped"
    ls 1_mixing_length/$RE/$CG/tamu_2d_fv_csv_vpp_*.csv >/dev/null || echo "missing baseline"
  done
done
'''

import copy
import os
import fnmatch
import time
import random
import operator
import subprocess
from functools import partial

import numpy as np
from deap import base, creator, gp, tools

import shutil
import json
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============================================================
# USER SETTINGS
# ============================================================
RUN_TEMPLATE_FIRST = False

RE_LIST = ["3000", "3413", "5963", "7912", "9000", "10622", "12819", "14000"]
TEST_RE_LIST = ["5963", "12819"]

CASES = []
for RE in RE_LIST:
    CASES += [
        {"RE": RE, "CG": "1", "np": 4},
        {"RE": RE, "CG": "2", "np": 4},
        {"RE": RE, "CG": "3", "np": 8},
        {"RE": RE, "CG": "4", "np": 12},
    ]

RANDOM_SEED = 200
RUN_TAG = "CG1234_asymLogCeff_etaH_etaY_allDome_train6Re_test2Re_champion_tournament_10seeds_seed200_avg5"

RESULTS_FILE = f"gp_results_{RUN_TAG}.csv"
DETAILS_FILE = f"gp_results_{RUN_TAG}_details.csv"
METADATA_FILE = f"gp_results_{RUN_TAG}_metadata.json"
FINAL_BEST_FILE = f"gp_results_{RUN_TAG}_final_best.json"
CORE_LIMIT = 127
SOLVER_TIMEOUT = 2700.0  # seconds

# One stochastic mini-batch per generation.
# Select random cases so that:
# N_CANDIDATES * sum(np_case) <= CORE_LIMIT

# same logic as your old script
YW_WL = 0.0

# paths: adjust if needed
# old NN workflow used this root for solver runs
PROJECT_ROOT = os.path.abspath(".")

def get_case_paths(RE, CG):
    case_root = os.path.join(PROJECT_ROOT, "runs", RE, CG)
    template_dir = os.path.join(case_root, "template")
    baseline_dir = os.path.join(PROJECT_ROOT, "1_mixing_length", RE, CG)
    mapped_file = os.path.join(PROJECT_ROOT, "2_mapped", RE, CG, "mapped.csv")
    return case_root, template_dir, baseline_dir, mapped_file

# old workflow activated the moose env, cd'ed into the case folder,
# then sourced fv_app_run.sh
STEADY_RUN_SCRIPT = "fv_app_run.sh"
UNSTEADY_RUN_SCRIPT = "fv_app_run_unsteady.sh"
STEADY_INPUT_FILE = "tamu_2d_fv_gp.i"
UNSTEADY_INPUT_FILE = "tamu_2d_fv_gp_unsteady.i"
RESTART_LINE = "restart_file_base = tamu_2d_fv_gp_out_cp/LATEST"
COMMENTED_RESTART_LINE = "#restart_file_base = tamu_2d_fv_gp_out_cp/LATEST"
BASELINE_VPP_PATTERN = "tamu_2d_fv_csv_vpp_*.csv"
VPP_PATTERN = "tamu_2d_fv_gp_csv_vpp_*.csv"
ELV_PATTERN = "tamu_2d_fv_csv_elv_*.csv"
WALL_CELL_SUMMARY_FILE = os.path.join(
    PROJECT_ROOT,
    "3_wall_cells",
    "wall_cells_median_summary.csv",
)

# files used in the old workflow
LOG_FILE = "log.csv"
LM_FILE = "lm_pred.csv"

# complexity penalty
LAMBDA_SIZE = 1.0e-5
KAPPA0 = 0.41

C_EFF_MIN_FACTOR = 0.05
C_EFF_MAX_FACTOR = 2.0

C_EFF_MIN = C_EFF_MIN_FACTOR * KAPPA0
C_EFF_MAX = C_EFF_MAX_FACTOR * KAPPA0

LAMBDA_RAW = 1.0e-6

# big penalty for invalid / diverged candidates
BIG_PENALTY = 1.0e9

FITNESS_CACHE_FILE = f"gp_results_{RUN_TAG}_fitness_cache.json"
USE_FITNESS_CACHE = True

N_CANDIDATES = 10
N_GENERATIONS = 20
BATCHES_PER_SELECTION = 5
TOURNAMENT_SIZE = 3
MUTATION_PROB = 0.6
CROSSOVER_PROB = 0.4
ELITE_COUNT = 4

CHAMPION_EXPR_STRINGS = [
    # seed 6: best overall
    "sub(-0.37901541435676955, protected_div(eta_y, 0.33314395947154063))",

    # seed 6 simplified
    "sub(-0.38, mul(3.0, eta_y))",

    # seed 7: new strong candidate
    "sub(eta_y, protected_div(eta_y, 0.1297762159749819))",

    # seed 7 simplified: approximately -6.70 eta_y
    "mul(-6.7, eta_y)",

    # seed 3
    "add(eta_h, sub(protected_div(eta_y, -0.18867695249518834), 0.33347711999614726))",

    # seed 4
    "sub(-0.41338404990671096, sub(add(eta_y, eta_y), sub(protected_tanh(mul(0.3809913094372004, eta_h)), eta_y)))",

    # seed 8: approximately -0.386 - 2 eta_y
    "neg(sub(0.38592740083515376, neg(add(eta_y, eta_y))))",

    # seed 5
    "neg(protected_div(sub(eta_y, eta_h), protected_tanh(protected_tanh(0.13585021784260087))))",

    # simple variant around seed 3/7 family
    "sub(-0.333, mul(5.3, eta_y))",

    # diversity candidate from seed 1
    "add(mul(0.12305820209505292, neg(protected_div(eta_y, eta_h))), eta_y)",
]

# ============================================================
# SMALL UTILITIES
# ============================================================
def write_metadata_file(case_data, train_case_data, test_case_data):
    metadata = {
        "random_seed": RANDOM_SEED,
        "re_list": RE_LIST,
        "data_split": {
            "test_re_list": TEST_RE_LIST,
            "training_cases": [
                f"RE{case['RE']}_CG{case['CG']}" for case in train_case_data
            ],
            "test_cases": [
                f"RE{case['RE']}_CG{case['CG']}" for case in test_case_data
            ],
        },
        "cases": [
            {
                "RE": case["RE"],
                "CG": case["CG"],
                "np": case["np"],
                "mapped_file": case["mapped_file"],
                "baseline_dir": case["baseline_dir"],
                "baseline_err": case["baseline_err"],
                "L_outer": case["L_outer"],
                "h_wall": case["h_wall"],
                "eta_h_scalar": case["eta_h_scalar"],
                "eta_h_min": float(np.min(case["eta_h"])),
                "eta_h_max": float(np.max(case["eta_h"])),
                "eta_y_min": float(np.min(case["eta_y"])),
                "eta_y_max": float(np.max(case["eta_y"])),
                "yw_min": float(np.min(case["yw"])),
                "yw_max": float(np.max(case["yw"])),
                "n_points": int(case["yw"].size),
            }
            for case in case_data
        ],
        "gp_settings": {
            "n_candidates": N_CANDIDATES,
            "n_generations": N_GENERATIONS,
            "tournament_size": TOURNAMENT_SIZE,
            "mutation_prob": MUTATION_PROB,
            "crossover_prob": CROSSOVER_PROB,
            "elite_count": ELITE_COUNT,
            "lambda_size": LAMBDA_SIZE,
            "lambda_raw": LAMBDA_RAW,
            "kappa0": KAPPA0,
            "C_eff_min_factor": C_EFF_MIN_FACTOR,
            "C_eff_max_factor": C_EFF_MAX_FACTOR,
            "C_eff_min": C_EFF_MIN,
            "C_eff_max": C_EFF_MAX,
            "max_tree_size": 24,
            "primitive_set": [
                "protected_div",
                "add",
                "sub",
                "mul",
                "neg",
                "protected_tanh",
            ],
            "ephemeral_constant_range": [-0.5, 0.5],
        },
        "closure_mapping": {
            "input_variables": {
                "eta_h": "h_wall(CG)/L_outer",
                "eta_y": "yw/L_outer",
            },
            "h_wall": "median corrected wall distance of first near-wall cell band for each CG",
            "L_outer": "max(yw) for each case",
            "raw": "GP expression evaluated using eta_h and eta_y",
            "coefficient_mapping": (
                "C_eff = KAPPA0*exp(log_factor), where "
                "t=tanh(raw), log_factor=log(C_EFF_MIN_FACTOR)*(-t) for t<0 "
                "and log(C_EFF_MAX_FACTOR)*t for t>=0"
            ),
            "C_eff_bounds": [C_EFF_MIN, C_EFF_MAX],
            "C_eff_neutral": "raw = 0 gives C_eff = KAPPA0",
            "kappa0": KAPPA0,
            "C_eff_min_factor": C_EFF_MIN_FACTOR,
            "C_eff_max_factor": C_EFF_MAX_FACTOR,
            "lm": "C_eff*yw",
        },
        "fitness_definition": {
            "quantity": "velocity MSE",
            "normalization": "candidate fitness / baseline mixing-length MSE",
            "region": "z < 0 and yw > 0; effectively all dome cells because corrected yw is positive",
            "yw_wall_cutoff": YW_WL,
            "big_penalty": BIG_PENALTY,
        },
        "execution": {
            "core_limit": CORE_LIMIT,
            "solver_timeout": SOLVER_TIMEOUT,
            "steady_run_script": STEADY_RUN_SCRIPT,
            "unsteady_run_script": UNSTEADY_RUN_SCRIPT,
            "python_version": platform.python_version(),
            "working_directory": PROJECT_ROOT,
        },
    }

    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

def latest_file(folder, pattern):
    files = fnmatch.filter(os.listdir(folder), pattern)
    files = [os.path.join(folder, f) for f in files]
    files.sort(key=lambda x: os.path.getmtime(x))

    if len(files) == 0:
        raise FileNotFoundError(f"No file found in {folder} with pattern {pattern}")

    return files[-1]

def read_h_wall_by_cg(summary_file):
    """
    Read characteristic first-wall-cell size for each CG from
    3_wall_cells/wall_cells_median_summary.csv.

    Uses yw_wall_median as h_wall(CG).
    """
    if not os.path.isfile(summary_file):
        raise FileNotFoundError(f"Wall-cell summary file not found: {summary_file}")

    data = np.genfromtxt(summary_file, delimiter=",", names=True, dtype=None, encoding=None)

    required = ["CG", "yw_wall_median"]
    for name in required:
        if name not in data.dtype.names:
            raise RuntimeError(
                f"Required column '{name}' not found in {summary_file}. "
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

def set_restart_flag(case_path, enable_restart):
    for fname in (STEADY_INPUT_FILE, UNSTEADY_INPUT_FILE):
        fpath = os.path.join(case_path, fname)

        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Input file not found: {fpath}")

        with open(fpath, "r") as f:
            lines = f.readlines()

        new_lines = []
        found = False

        for line in lines:
            stripped = line.strip()

            if stripped == RESTART_LINE or stripped == COMMENTED_RESTART_LINE:
                found = True
                indent = line[:len(line) - len(line.lstrip())]

                if enable_restart:
                    new_lines.append(indent + RESTART_LINE + "\n")
                else:
                    new_lines.append(indent + COMMENTED_RESTART_LINE + "\n")
            else:
                new_lines.append(line)

        if not found:
            raise RuntimeError(f"Restart line not found in {fpath}")

        with open(fpath, "w") as f:
            f.writelines(new_lines)

def reset_candidate_folders(case_root, n_candidates):
    for name in os.listdir(case_root):
        path = os.path.join(case_root, name)
        if os.path.isdir(path) and name.startswith("cand_"):
            shutil.rmtree(path)

    for cand_id in range(n_candidates):
        prepare_candidate_case(case_root, cand_id)

def prepare_candidate_case(case_root, cand_id):
    template_dir = os.path.join(case_root, "template")
    cand_dir = os.path.join(case_root, f"cand_{cand_id:03d}")

    if os.path.exists(cand_dir):
        shutil.rmtree(cand_dir)

    shutil.copytree(template_dir, cand_dir)

    return cand_dir

def search_last(file_name, string, n_lines=200):
    with open(file_name, "r", errors="ignore") as f:
        lines = f.readlines()[-n_lines:]
        return any(string in line for line in lines)


def protected_tanh(x):
    return np.tanh(x)

def protected_div(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)

    left, right = np.broadcast_arrays(left, right)

    out = np.ones_like(right, dtype=np.float64)

    mask = np.abs(right) > 1e-6
    out[mask] = left[mask] / right[mask]

    return out

# ============================================================
# DATA PREP: matches your old workflow
# ============================================================
def read_mapped(mapped_file):
    print("Reading mapped file:", mapped_file, flush=True)

    if not os.path.isfile(mapped_file):
        raise FileNotFoundError(f"Mapped file not found: {mapped_file}")

    data = np.loadtxt(mapped_file, delimiter=",", skiprows=1, dtype=np.float64)

    # columns:
    # r_cg,z_cg,ur_les,uz_les,yw,nut
    r_cg = data[:, 0]
    z_cg = data[:, 1]
    ur_les = data[:, 2]
    uz_les = data[:, 3]
    yw = data[:, 4]
    nut = data[:, 5]

    return r_cg, z_cg, ur_les, uz_les, yw, nut



# ============================================================
# GP SETUP
# ============================================================
pset = gp.PrimitiveSet("MAIN", 2)
pset.renameArguments(ARG0="eta_h")
pset.renameArguments(ARG1="eta_y")

pset.addPrimitive(protected_div, 2)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(protected_tanh, 1)

# no lambda warning
pset.addEphemeralConstant("rand", partial(random.uniform, -0.5, 0.5))

if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("compile", gp.compile, pset=pset)

toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=len, max_value=24))
toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=24))
toolbox.register("clone", copy.deepcopy)

def build_lm_from_yw_raw(raw, yw):
    raw = np.asarray(raw, dtype=np.float64).copy()

    if np.any(~np.isfinite(raw)):
        raise ValueError("Non-finite raw closure values")

    raw_min = float(np.min(raw))
    raw_max = float(np.max(raw))

    t = np.tanh(raw)

    log_factor = np.where(
        t < 0.0,
        np.log(C_EFF_MIN_FACTOR) * (-t),  # t=-1 -> log(0.05)
        np.log(C_EFF_MAX_FACTOR) * t,     # t=+1 -> log(2.0)
    )

    C_eff = KAPPA0 * np.exp(log_factor)

    lm = C_eff * yw

    print("raw min/max =", raw_min, raw_max, flush=True)
    print("C_eff min/max =", np.min(C_eff), np.max(C_eff), flush=True)
    print("lm min/max =", np.min(lm), np.max(lm), flush=True)

    return lm, C_eff

def select_random_case_batch(case_data):
    shuffled = list(case_data)
    random.shuffle(shuffled)

    selected = []
    used_cores = 0

    for case in shuffled:
        added_cores = N_CANDIDATES * case["np"]

        if selected and used_cores + added_cores > CORE_LIMIT:
            continue

        if not selected and added_cores > CORE_LIMIT:
            raise RuntimeError(
                f"Single case RE={case['RE']}, CG={case['CG']} requires "
                f"{added_cores} cores, which exceeds CORE_LIMIT={CORE_LIMIT}"
            )

        selected.append(case)
        used_cores += added_cores

    if not selected:
        raise RuntimeError("No cases selected for this generation.")

    return selected, used_cores

def make_cache_key(individual, case):
    return f"RE{case['RE']}_CG{case['CG']}__{str(individual)}"


def load_fitness_cache():
    if not USE_FITNESS_CACHE:
        return {}

    if not os.path.isfile(FITNESS_CACHE_FILE):
        return {}

    with open(FITNESS_CACHE_FILE, "r") as f:
        return json.load(f)


def save_fitness_cache(cache):
    if not USE_FITNESS_CACHE:
        return

    tmp_file = FITNESS_CACHE_FILE + ".tmp"

    with open(tmp_file, "w") as f:
        json.dump(cache, f, indent=2)

    os.replace(tmp_file, FITNESS_CACHE_FILE)

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


def clean_case_folder(case_path):
    for fname in (LOG_FILE, LM_FILE):
        fpath = os.path.join(case_path, fname)
        if os.path.exists(fpath):
            os.remove(fpath)

    for fname in fnmatch.filter(os.listdir(case_path), VPP_PATTERN):
        os.remove(os.path.join(case_path, fname))


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

    print("VPP mtime =", os.path.getmtime(res_path), flush=True)
    print("First 3 u values:", uz_cg[:3], flush=True)
    print("First 3 x/z values:", z_cg[:3], flush=True)
    print("First 3 y/r values:", r_cg[:3], flush=True)

    return ur_cg, uz_cg, r_cg, z_cg

def velocity_mse(ur_les, uz_les, ur_cg, uz_cg, z_cg, yw, yw_wl):
    """
    Same dome-only and wall-cutoff logic as your old Test_NN.
    """
    dome = z_cg < 0
    keep = yw[dome] > yw_wl

    if keep.sum() == 0:
        return BIG_PENALTY

    u_mse = (ur_les[dome] - ur_cg[dome]) ** 2
    v_mse = (uz_les[dome] - uz_cg[dome]) ** 2
    return u_mse[keep].mean() + v_mse[keep].mean()

def evaluate_individual(individual, r_cg, z_cg, yw, eta_h, eta_y, ur_les, uz_les, case_path):
    """
    One GP candidate -> lm_pred.csv -> solver -> velocity-based fitness
    """
    func = toolbox.compile(expr=individual)

    try:
        raw = func(eta_h, eta_y)

        raw = np.asarray(raw, dtype=np.float64)

        # Constant GP expressions may return either a Python scalar
        # or a 0-D NumPy array. Broadcast them to all cells.
        if raw.shape == ():
            raw = np.full_like(r_cg, float(raw), dtype=np.float64)

        if raw.shape != r_cg.shape:
            print(
                f"Invalid shape from candidate: raw.shape={raw.shape}, "
                f"expected={r_cg.shape}",
                flush=True,
            )
            return BIG_PENALTY, "failed", np.nan, np.nan, np.nan

        if np.any(~np.isfinite(raw)):
            print("Non-finite values from candidate", flush=True)
            return BIG_PENALTY, "failed", np.nan, np.nan, np.nan

        lm, C_eff = build_lm_from_yw_raw(raw, yw)
        print("C_eff min/max =", np.min(C_eff), np.max(C_eff), flush=True)
        print("lm min/max =", np.min(lm), np.max(lm), flush=True)

        clean_case_folder(case_path)

        lm_path = os.path.join(case_path, LM_FILE)
        save_lm_csv(z_cg, r_cg, lm, lm_path)

        run_mode = "steady"

        steady_elapsed = 0.0
        unsteady_elapsed = 0.0

        ok, steady_elapsed = run_case(case_path, STEADY_RUN_SCRIPT)

        if not ok:
            print("Steady failed; trying unsteady fallback", flush=True)

            clean_case_folder(case_path)

            lm_path = os.path.join(case_path, LM_FILE)
            save_lm_csv(z_cg, r_cg, lm, lm_path)

            run_mode = "unsteady_fallback"
            ok, unsteady_elapsed = run_case(case_path, UNSTEADY_RUN_SCRIPT)

        total_solver_elapsed = steady_elapsed + unsteady_elapsed

        if not ok:
            print("Both steady and unsteady fallback failed for candidate", flush=True)
            return BIG_PENALTY, "failed", steady_elapsed, unsteady_elapsed, total_solver_elapsed

        ur_cg, uz_cg, r_res, z_res = read_velocity_result(case_path, VPP_PATTERN)

        if r_res.size != r_cg.size:
            raise ValueError(f"Size mismatch: result has {r_res.size}, mapped has {r_cg.size}")

        dr = np.max(np.abs(r_res - r_cg))
        dz = np.max(np.abs(z_res - z_cg))

        print("coordinate check: max |r_res-r_cg| =", dr, flush=True)
        print("coordinate check: max |z_res-z_cg| =", dz, flush=True)

        if dr > 1e-10 or dz > 1e-10:
            raise ValueError("VPP row order does not match mapped.csv row order")

        # use result z for dome mask, same as old code
        vel_err = velocity_mse(ur_les, uz_les, ur_cg, uz_cg, z_res, yw, YW_WL)

        complexity_penalty = LAMBDA_SIZE * len(individual)
        raw_penalty = LAMBDA_RAW * float(np.mean(raw**2))
        fitness = vel_err + complexity_penalty + raw_penalty

        print("raw min/max =", np.min(raw), np.max(raw), flush=True)
        print("lm min/max =", np.min(lm), np.max(lm), flush=True)

        print("Expression:", individual, flush=True)
        print("Velocity MSE:", vel_err, flush=True)
        print("Complexity penalty:", complexity_penalty, flush=True)
        print("Raw penalty:", raw_penalty, flush=True)
        print("Fitness:", fitness, flush=True)

        return fitness, run_mode, steady_elapsed, unsteady_elapsed, total_solver_elapsed

    except Exception as e:
        print("Exception during evaluation:", e, flush=True)
        return BIG_PENALTY, "failed", np.nan, np.nan, np.nan

def evaluate_candidate_case_worker(args):
    cand_id, individual, case = args

    case_path = os.path.join(case["case_root"], f"cand_{cand_id:03d}")

    print(
        f"\n===== Candidate {cand_id:03d}, "
        f"RE={case['RE']}, CG={case['CG']} =====",
        flush=True,
    )
    print("Expression:", individual, flush=True)

    fit, run_mode, steady_elapsed, unsteady_elapsed, total_solver_elapsed = evaluate_individual(
        individual,
        case["r_cg"],
        case["z_cg"],
        case["yw"],
        case["eta_h"],
        case["eta_y"],
        case["ur_les"],
        case["uz_les"],
        case_path,
    )

    if case["baseline_err"] <= 0.0 or not np.isfinite(case["baseline_err"]):
        norm_fit = BIG_PENALTY
    else:
        norm_fit = fit / case["baseline_err"]

    tree_size = len(individual)
    tree_height = individual.height

    return (
        cand_id,
        case["RE"],
        case["CG"],
        fit,
        norm_fit,
        run_mode,
        steady_elapsed,
        unsteady_elapsed,
        total_solver_elapsed,
        tree_size,
        tree_height,
        str(individual),
    )

def initialize_template_from_population(population, case):
    print("\n===== Initializing template from average GP candidate =====", flush=True)
    r_cg = case["r_cg"]
    z_cg = case["z_cg"]
    yw = case["yw"]
    eta_h = case["eta_h"]
    eta_y = case["eta_y"]
    template_dir = case["template_dir"]

    lm_list = []

    for ind in population:
        func = toolbox.compile(expr=ind)
        raw = func(eta_h, eta_y)

        if np.isscalar(raw):
            raw = np.full_like(r_cg, raw, dtype=np.float64)

        raw = np.asarray(raw, dtype=np.float64)

        if raw.shape != r_cg.shape or np.any(~np.isfinite(raw)):
            continue

        lm, _ = build_lm_from_yw_raw(raw, yw)
        lm_list.append(lm)

    if len(lm_list) == 0:
        raise RuntimeError("No valid GP candidates for template initialization.")

    lm_avg = np.mean(np.vstack(lm_list), axis=0)

    clean_case_folder(template_dir)
    set_restart_flag(template_dir, enable_restart=True)
    save_lm_csv(z_cg, r_cg, lm_avg, os.path.join(template_dir, LM_FILE))

    #ok = run_case(template_dir, STEADY_RUN_SCRIPT)

    #if not ok:
    #    print("Template steady initialization failed; trying unsteady.", flush=True)
    #    clean_case_folder(template_dir)
    #    set_restart_flag(template_dir, enable_restart=False)
    #    save_lm_csv(z_cg, r_cg, lm_avg, os.path.join(template_dir, LM_FILE))
    #    ok = run_case(template_dir, UNSTEADY_RUN_SCRIPT)

    print("Skipping steady initialization; running unsteady directly.", flush=True)
    ok, _ = run_case(template_dir, UNSTEADY_RUN_SCRIPT)

    if not ok:
        raise RuntimeError("Template initialization failed.")

    set_restart_flag(template_dir, enable_restart=True)

    print("Template initialization finished successfully.", flush=True)

def evaluate_population(population, selected_cases):
    fitness_cache = load_fitness_cache()

    tasks = []
    cached_results = []
    pending_by_key = {}

    for case in selected_cases:
        for cand_id, individual in enumerate(population):
            cache_key = make_cache_key(individual, case)

            if cache_key in fitness_cache:
                item = fitness_cache[cache_key]

                cached_results.append(
                    (
                        cand_id,
                        case["RE"],
                        case["CG"],
                        item["raw_fit"],
                        item["norm_fit"],
                        item["run_mode"],
                        item.get("steady_elapsed", item.get("solver_elapsed", 0.0)),
                        item.get("unsteady_elapsed", 0.0),
                        item.get("total_solver_elapsed", item.get("solver_elapsed", 0.0)),
                        item["tree_size"],
                        item["tree_height"],
                        str(individual),
                    )
                )

            elif cache_key in pending_by_key:
                pending_by_key[cache_key]["duplicate_cand_ids"].append(cand_id)

            else:
                tasks.append((cand_id, individual, case))

                pending_by_key[cache_key] = {
                    "primary_cand_id": cand_id,
                    "duplicate_cand_ids": [],
                    "case": case,
                    "individual": individual,
                }

    batch_cores = sum(task[2]["np"] for task in tasks)

    if batch_cores > CORE_LIMIT:
        raise RuntimeError(
            f"Selected batch uses {batch_cores} cores, "
            f"which exceeds CORE_LIMIT={CORE_LIMIT}"
        )

    random.shuffle(tasks)

    candidate_norm_errors = {cand_id: [] for cand_id in range(len(population))}
    results = []

    for r in cached_results:
        cand_id, RE, CG, raw_fit, norm_fit, run_mode, steady_elapsed, unsteady_elapsed, total_solver_elapsed, tree_size, tree_height, expr = r
        candidate_norm_errors[cand_id].append(norm_fit)
        results.append(r)

    if cached_results:
        print(f"Used {len(cached_results)} cached evaluations", flush=True)

    n_duplicates = sum(
        len(item["duplicate_cand_ids"])
        for item in pending_by_key.values()
    )

    if n_duplicates:
        print(f"Skipped {n_duplicates} duplicate in-batch evaluations", flush=True)

    print(
        f"\n===== Running one stochastic batch with "
        f"{len(tasks)} unique tasks, {batch_cores} MPI ranks =====",
        flush=True,
    )

    if tasks:
        with ProcessPoolExecutor(max_workers=len(tasks)) as ex:
            futures = [
                ex.submit(evaluate_candidate_case_worker, task)
                for task in tasks
            ]

            for fut in as_completed(futures):
                (
                    cand_id,
                    RE,
                    CG,
                    raw_fit,
                    norm_fit,
                    run_mode,
                    steady_elapsed,
                    unsteady_elapsed,
                    total_solver_elapsed,
                    tree_size,
                    tree_height,
                    expr,
                ) = fut.result()

                candidate_norm_errors[cand_id].append(norm_fit)
                results.append(
                    (
                        cand_id,
                        RE,
                        CG,
                        raw_fit,
                        norm_fit,
                        run_mode,
                        steady_elapsed,
                        unsteady_elapsed,
                        total_solver_elapsed,
                        tree_size,
                        tree_height,
                        expr,
                    )
                )

                case_match = next(
                    case for case in selected_cases
                    if case["RE"] == RE and case["CG"] == CG
                )

                individual = population[cand_id]
                cache_key = make_cache_key(individual, case_match)

                duplicate_cand_ids = pending_by_key.get(cache_key, {}).get(
                    "duplicate_cand_ids",
                    [],
                )

                for dup_cand_id in duplicate_cand_ids:
                    candidate_norm_errors[dup_cand_id].append(norm_fit)
                    results.append(
                        (
                            dup_cand_id,
                            RE,
                            CG,
                            raw_fit,
                            norm_fit,
                            run_mode,
                            steady_elapsed,
                            unsteady_elapsed,
                            total_solver_elapsed,
                            tree_size,
                            tree_height,
                            expr,
                        )
                    )

                    if run_mode != "failed" and np.isfinite(raw_fit) and raw_fit < BIG_PENALTY:
                        primary_case_path = os.path.join(
                            case_match["case_root"],
                            f"cand_{cand_id:03d}",
                        )
                        duplicate_case_path = os.path.join(
                            case_match["case_root"],
                            f"cand_{dup_cand_id:03d}",
                        )

                        primary_cp = os.path.join(primary_case_path, "tamu_2d_fv_gp_out_cp")
                        duplicate_cp = os.path.join(duplicate_case_path, "tamu_2d_fv_gp_out_cp")

                        if os.path.isdir(primary_cp):
                            if os.path.exists(duplicate_cp):
                                shutil.rmtree(duplicate_cp)
                            shutil.copytree(primary_cp, duplicate_cp)

                        primary_lm = os.path.join(primary_case_path, LM_FILE)
                        duplicate_lm = os.path.join(duplicate_case_path, LM_FILE)

                        if os.path.isfile(primary_lm):
                            shutil.copy2(primary_lm, duplicate_lm)

                if run_mode != "failed" and np.isfinite(raw_fit) and raw_fit < BIG_PENALTY:
                    fitness_cache[cache_key] = {
                        "RE": RE,
                        "CG": CG,
                        "expr": str(individual),
                        "raw_fit": float(raw_fit),
                        "norm_fit": float(norm_fit),
                        "run_mode": run_mode,
                        "steady_elapsed": float(steady_elapsed) if np.isfinite(steady_elapsed) else -1.0,
                        "unsteady_elapsed": float(unsteady_elapsed) if np.isfinite(unsteady_elapsed) else -1.0,
                        "total_solver_elapsed": float(total_solver_elapsed) if np.isfinite(total_solver_elapsed) else -1.0,
                        "tree_size": int(tree_size),
                        "tree_height": int(tree_height),
                    }

                    save_fitness_cache(fitness_cache)
    else:
        print("All evaluations in this batch were loaded from cache", flush=True)

    for cand_id in range(len(population)):
        errs = candidate_norm_errors[cand_id]
        population[cand_id].fitness.values = (
            float(np.mean(errs)) if errs else BIG_PENALTY,
        )

    return results

def evaluate_baseline_mixing_length(case):
    print("\n===== Evaluating baseline mixing-length case =====", flush=True)
    print("Baseline dir:", case["baseline_dir"], flush=True)

    r_cg = case["r_cg"]
    z_cg = case["z_cg"]
    yw = case["yw"]
    ur_les = case["ur_les"]
    uz_les = case["uz_les"]

    ur_base, uz_base, r_base, z_base = read_velocity_result(
        case["baseline_dir"],
        BASELINE_VPP_PATTERN,
    )

    if r_base.size != r_cg.size:
        raise ValueError(
            f"Baseline size mismatch for RE={case['RE']}, CG={case['CG']}: "
            f"result has {r_base.size}, mapped has {r_cg.size}"
        )

    dr = np.max(np.abs(r_base - r_cg))
    dz = np.max(np.abs(z_base - z_cg))

    print("baseline coordinate check: max |r_base-r_cg| =", dr, flush=True)
    print("baseline coordinate check: max |z_base-z_cg| =", dz, flush=True)

    if dr > 1e-10 or dz > 1e-10:
        raise ValueError(
            f"Baseline VPP row order mismatch for RE={case['RE']}, CG={case['CG']}"
        )

    baseline_err = velocity_mse(
        ur_les,
        uz_les,
        ur_base,
        uz_base,
        z_base,
        yw,
        YW_WL,
    )

    print(
        f"Baseline MSE for RE={case['RE']}, CG={case['CG']} = "
        f"{baseline_err:.8e}",
        flush=True,
    )

    return baseline_err


def make_case_batches_for_population(case_data, population_size, core_limit):
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

def select_top_unique_candidate_ids(population, final_train_scores, top_k):
    """
    Select top candidates by full-training score, but do not return
    duplicate symbolic expressions.
    """
    ranked_ids = sorted(
        final_train_scores.keys(),
        key=lambda cid: np.mean(final_train_scores[cid]),
    )

    selected_ids = []
    seen_expr = set()

    for cid in ranked_ids:
        expr = str(population[cid])

        if expr in seen_expr:
            continue

        selected_ids.append(cid)
        seen_expr.add(expr)

        if len(selected_ids) >= top_k:
            break

    if len(selected_ids) == 0:
        raise RuntimeError("No unique final candidates were selected.")

    return selected_ids

# ============================================================
# MAIN: evaluate ONE DEAP candidate with real solver fitness
# ============================================================
def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("\n===== Loading all available cases =====", flush=True)

    h_wall_by_cg = read_h_wall_by_cg(WALL_CELL_SUMMARY_FILE)

    case_data = []
    train_case_data = []
    test_case_data = []

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
            "eta_h_min": float(np.min(eta_h)),
            "eta_h_max": float(np.max(eta_h)),
            "eta_y_min": float(np.min(eta_y)),
            "eta_y_max": float(np.max(eta_y)),
        }

        case_entry["baseline_err"] = evaluate_baseline_mixing_length(case_entry)
        case_data.append(case_entry)

        if RE in TEST_RE_LIST:
            test_case_data.append(case_entry)
        else:
            train_case_data.append(case_entry)

    if len(train_case_data) == 0:
        raise RuntimeError("No training cases selected.")

    if len(test_case_data) == 0:
        raise RuntimeError("No test cases selected.")

    write_metadata_file(case_data, train_case_data, test_case_data)

    with open(RESULTS_FILE, "w") as f:
        f.write("seed,gen,rank,norm_fitness,cases,expr\n")

    with open(DETAILS_FILE, "w") as f:
        f.write(
            "seed,gen,cand_id,RE,CG,raw_fitness,baseline_fitness,"
            "norm_fitness,run_mode,steady_elapsed,unsteady_elapsed,"
            "total_solver_elapsed,tree_size,tree_height,expr\n"
        )

    if len(CHAMPION_EXPR_STRINGS) != N_CANDIDATES:
        raise RuntimeError(
            f"Expected {N_CANDIDATES} champion expressions, "
            f"but got {len(CHAMPION_EXPR_STRINGS)}"
        )

    population = [
        creator.Individual(gp.PrimitiveTree.from_string(expr, pset))
        for expr in CHAMPION_EXPR_STRINGS
    ]

    print("\n===== Initial champion population =====", flush=True)
    for i, ind in enumerate(population):
        print(f"Candidate {i:03d}: {ind}", flush=True)

    if RUN_TEMPLATE_FIRST:
        for case in case_data:
            print(
                f"\n===== Template initialization for "
                f"RE={case['RE']}, CG={case['CG']} =====",
                flush=True,
            )
            initialize_template_from_population(population, case)

    for case in case_data:
        reset_candidate_folders(case["case_root"], N_CANDIDATES)

    for gen in range(N_GENERATIONS):
        print(f"\n===== Generation {gen} =====", flush=True)

        all_batch_results = []
        fitness_accum = {cand_id: [] for cand_id in range(len(population))}
        selected_case_names_all = []

        for ibatch in range(BATCHES_PER_SELECTION):
            print(
                f"\n===== Generation {gen}, stochastic batch "
                f"{ibatch + 1}/{BATCHES_PER_SELECTION} =====",
                flush=True,
            )

            selected_cases, batch_cores = select_random_case_batch(train_case_data)

            selected_case_names = [
                f"RE{case['RE']}_CG{case['CG']}"
                for case in selected_cases
            ]

            selected_case_names_all.extend(selected_case_names)

            print("Selected cases:", selected_case_names, flush=True)
            print("Selected batch cores:", batch_cores, flush=True)

            batch_results = evaluate_population(population, selected_cases)
            all_batch_results.extend(batch_results)

            for cand_id, RE, CG, raw_fit, norm_fit, run_mode, steady_elapsed, unsteady_elapsed, total_solver_elapsed, tree_size, tree_height, expr in batch_results:
                fitness_accum[cand_id].append(norm_fit)

                with open(DETAILS_FILE, "a") as f:
                    baseline_fit = next(
                        case["baseline_err"]
                        for case in selected_cases
                        if case["RE"] == RE and case["CG"] == CG
                    )

                    f.write(
                        f'{RANDOM_SEED},{gen},{cand_id},{RE},{CG},'
                        f'{raw_fit:.12e},{baseline_fit:.12e},{norm_fit:.12e},'
                        f'{run_mode},{steady_elapsed:.6e},{unsteady_elapsed:.6e},'
                        f'{total_solver_elapsed:.6e},{tree_size},{tree_height},"{expr}"\n'
                    )

        for cand_id in range(len(population)):
            vals = fitness_accum[cand_id]
            population[cand_id].fitness.values = (
                float(np.mean(vals)) if vals else BIG_PENALTY,
            )

        selected_case_names = selected_case_names_all
        results = all_batch_results

        ranked = sorted(
            [(ind.fitness.values[0], str(ind), ind) for ind in population],
            key=lambda x: x[0],
        )

        print("\n--- Generation summary ---", flush=True)
        for fit, expr, _ in ranked:
            print(f"fitness = {fit:.8f} | expr = {expr}", flush=True)

        with open(RESULTS_FILE, "a") as f:
            cases_str = ";".join(selected_case_names)
            for rank, (fit, expr, _) in enumerate(ranked):
                f.write(
                    f'{RANDOM_SEED},{gen},{rank},{fit:.12e},'
                    f'"{cases_str}","{expr}"\n'
                )

        if gen == N_GENERATIONS - 1:
            break

        elites = [toolbox.clone(ind) for _, _, ind in ranked[:ELITE_COUNT]]

        selected = toolbox.select(population, len(population) - ELITE_COUNT)
        offspring = [toolbox.clone(ind) for ind in selected]

        for i in range(0, len(offspring) - 1, 2):
            if random.random() < CROSSOVER_PROB:
                offspring[i], offspring[i + 1] = toolbox.mate(
                    offspring[i],
                    offspring[i + 1],
                )
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values

        for i in range(len(offspring)):
            if random.random() < MUTATION_PROB:
                offspring[i], = toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        population = elites + offspring[:len(population) - ELITE_COUNT]

    print("\n===== Final best =====", flush=True)

    # ------------------------------------------------------------
    # Evaluate final population on all training cases.
    # This avoids selecting the final best candidate from only the
    # last stochastic mini-batch.
    # ------------------------------------------------------------
    print("\n===== Evaluating final population on all training cases =====", flush=True)

    final_training_results = []

    train_case_batches = make_case_batches_for_population(
        train_case_data,
        population_size=len(population),
        core_limit=CORE_LIMIT,
    )

    for ib, case_batch in enumerate(train_case_batches):
        print(
            f"\n===== Final training population batch {ib+1}/{len(train_case_batches)} =====",
            flush=True,
        )

        batch_results = evaluate_population(population, case_batch)
        final_training_results.extend(batch_results)

    final_train_scores = {}
    for cand_id, RE, CG, raw_fit, norm_fit, run_mode, steady_elapsed, unsteady_elapsed, total_solver_elapsed, tree_size, tree_height, expr in final_training_results:
        final_train_scores.setdefault(cand_id, []).append(norm_fit)

    TOP_K_FINAL = 5

    top_cand_ids = select_top_unique_candidate_ids(
        population,
        final_train_scores,
        TOP_K_FINAL,
    )

    best_cand_id = top_cand_ids[0]
    best = population[best_cand_id]
    best_train_mean_normalized_fitness = float(np.mean(final_train_scores[best_cand_id]))

    print("Top final unique candidate IDs:", top_cand_ids, flush=True)
    for cid in top_cand_ids:
        print(
            f"  cand_id={cid}: train mean normalized fitness = "
            f"{np.mean(final_train_scores[cid]):.8f} | expr = {population[cid]}",
            flush=True,
        )

    # Save full-training evaluation for later inspection.
    FINAL_TRAIN_EVAL_FILE = f"gp_results_{RUN_TAG}_final_train_eval.csv"

    with open(FINAL_TRAIN_EVAL_FILE, "w") as f:
        f.write(
            "seed,RE,CG,cand_id,raw_fitness,baseline_fitness,"
            "norm_fitness,run_mode,steady_elapsed,unsteady_elapsed,"
            "total_solver_elapsed,tree_size,tree_height,expr\n"
        )

        for cand_id, RE, CG, raw_fit, norm_fit, run_mode, steady_elapsed, unsteady_elapsed, total_solver_elapsed, tree_size, tree_height, expr in final_training_results:
            baseline_fit = next(
                case["baseline_err"]
                for case in case_data
                if case["RE"] == RE and case["CG"] == CG
            )

            f.write(
                f'{RANDOM_SEED},{RE},{CG},{cand_id},'
                f'{raw_fit:.12e},{baseline_fit:.12e},{norm_fit:.12e},'
                f'{run_mode},{steady_elapsed:.6e},{unsteady_elapsed:.6e},'
                f'{total_solver_elapsed:.6e},{tree_size},{tree_height},"{expr}"\n'
            )

    # ------------------------------------------------------------
    # Evaluate only the top final training candidates on held-out
    # test cases. Test performance is reported separately and is
    # not used to select the candidates.
    # ------------------------------------------------------------
    print("\n===== Evaluating top final candidates on held-out test cases =====", flush=True)

    top_population = [population[cid] for cid in top_cand_ids]

    final_population_results = []

    final_case_batches = make_case_batches_for_population(
        test_case_data,
        population_size=len(top_population),
        core_limit=CORE_LIMIT,
    )

    for ib, case_batch in enumerate(final_case_batches):
        print(
            f"\n===== Final test batch {ib+1}/{len(final_case_batches)} =====",
            flush=True,
        )

        batch_results = evaluate_population(top_population, case_batch)

        # evaluate_population uses local candidate IDs 0,1,...
        # Convert them back to the original population candidate IDs.
        for local_cand_id, RE, CG, raw_fit, norm_fit, run_mode, steady_elapsed, unsteady_elapsed, total_solver_elapsed, tree_size, tree_height, expr in batch_results:
            original_cand_id = top_cand_ids[local_cand_id]
            final_population_results.append(
                (
                    original_cand_id,
                    RE,
                    CG,
                    raw_fit,
                    norm_fit,
                    run_mode,
                    steady_elapsed,
                    unsteady_elapsed,
                    total_solver_elapsed,
                    tree_size,
                    tree_height,
                    expr,
                )
            )

    final_scores = {}
    for cand_id, RE, CG, raw_fit, norm_fit, run_mode, steady_elapsed, unsteady_elapsed, total_solver_elapsed, tree_size, tree_height, expr in final_population_results:
        final_scores.setdefault(cand_id, []).append(norm_fit)

    best_test_mean_normalized_fitness = float(np.mean(final_scores[best_cand_id]))

    final_results = final_population_results

    FINAL_EVAL_FILE = f"gp_results_{RUN_TAG}_final_eval.csv"

    with open(FINAL_EVAL_FILE, "w") as f:
        f.write("seed,RE,CG,cand_id,raw_fitness,baseline_fitness,norm_fitness,run_mode,steady_elapsed,unsteady_elapsed,total_solver_elapsed,tree_size,tree_height,expr\n")

        for cand_id, RE, CG, raw_fit, norm_fit, run_mode, steady_elapsed, unsteady_elapsed, total_solver_elapsed, tree_size, tree_height, expr in final_results:
            baseline_fit = next(
                case["baseline_err"]
                for case in case_data
                if case["RE"] == RE and case["CG"] == CG
            )

            f.write(
                f'{RANDOM_SEED},{RE},{CG},{cand_id},'
                f'{raw_fit:.12e},{baseline_fit:.12e},{norm_fit:.12e},'
                f'{run_mode},{steady_elapsed:.6e},{unsteady_elapsed:.6e},'
                f'{total_solver_elapsed:.6e},{tree_size},{tree_height},"{expr}"\n'
            )

    final_best = {
        "seed": RANDOM_SEED,
        "best_train_mean_normalized_fitness": best_train_mean_normalized_fitness,
        "best_test_mean_normalized_fitness": best_test_mean_normalized_fitness,
        "best_cand_id": int(best_cand_id),
        "top_cand_ids": [int(cid) for cid in top_cand_ids],
        "top_expressions": {
            str(cid): str(population[cid]) for cid in top_cand_ids
        },
        "top_train_mean_normalized_fitness": {
            str(cid): float(np.mean(final_train_scores[cid])) for cid in top_cand_ids
        },
        "top_test_mean_normalized_fitness": {
            str(cid): float(np.mean(final_scores[cid])) for cid in top_cand_ids
        },
        "tree_size": int(len(best)),
        "tree_height": int(best.height),
        "best_expression": str(best),
        "input_variables": ["eta_h", "eta_y"],
        "closure_definition": {
            "eta_h": "h_wall(CG)/L_outer",
            "eta_y": "yw/L_outer",
            "h_wall": "median corrected wall distance of first near-wall cell band for each CG",
            "raw": "GP expression evaluated using eta_h and eta_y",
            "C_eff": (
                "KAPPA0*exp(log_factor), where "
                "t=tanh(raw), log_factor=log(C_EFF_MIN_FACTOR)*(-t) for t<0 "
                "and log(C_EFF_MAX_FACTOR)*t for t>=0"
            ),
            "C_eff_bounds": [C_EFF_MIN, C_EFF_MAX],
            "C_eff_neutral": "raw = 0 gives C_eff = KAPPA0",
            "lm": "C_eff*yw",
            "KAPPA0": KAPPA0,
            "C_EFF_MIN_FACTOR": C_EFF_MIN_FACTOR,
            "C_EFF_MAX_FACTOR": C_EFF_MAX_FACTOR,
            "C_EFF_MIN": C_EFF_MIN,
            "C_EFF_MAX": C_EFF_MAX,
        },
        "test_re_list": TEST_RE_LIST,
        "training_cases": [
            f"RE{case['RE']}_CG{case['CG']}" for case in train_case_data
        ],
        "test_cases": [
            f"RE{case['RE']}_CG{case['CG']}" for case in test_case_data
        ],
    }

    with open(FINAL_BEST_FILE, "w") as f:
        json.dump(final_best, f, indent=2)

    print(f"Best training mean normalized fitness = {best_train_mean_normalized_fitness:.8f}", flush=True)
    print(f"Held-out test mean normalized fitness = {best_test_mean_normalized_fitness:.8f}", flush=True)
    print(f"Best expr = {best}", flush=True)
    print(f"Final best saved to {FINAL_BEST_FILE}", flush=True)

    print("\n===== Baselines =====", flush=True)
    for case in case_data:
        print(
            f"RE={case['RE']}, CG={case['CG']}: "
            f"baseline MSE = {case['baseline_err']:.8e}",
            flush=True,
        )

if __name__ == "__main__":
    main()
