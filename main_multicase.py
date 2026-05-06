# salloc -p ksu-mne-train.q --nodelist=warlock35 --nodes=1 --ntasks=128 --mem=700G --time=72:00:00
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
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============================================================
# USER SETTINGS
# ============================================================
RUN_TEMPLATE_FIRST = False

CASES = [
    {"RE": "12819", "CG": "1", "np": 4},
    {"RE": "12819", "CG": "2", "np": 4},
    {"RE": "12819", "CG": "3", "np": 8},
    {"RE": "12819", "CG": "4", "np": 12},
]

RANDOM_SEED = 1
RESULTS_FILE = "gp_results_RE12819_RANDOMCG_seed1.csv"
CORE_LIMIT = 127

# One stochastic mini-batch per generation.
# Select random cases so that:
# N_CANDIDATES * sum(np_case) <= CORE_LIMIT
CASES_PER_GENERATION = None

# same logic as your old script
YW_WL = 0.2
TIME_MAX = 7000.0

#if CG == "4":
#    Z_POS = 3.0
#else:
#    Z_POS = 1.0e6
Z_POS = 1.0e6

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

# files used in the old workflow
LOG_FILE = "log.csv"
LM_FILE = "lm_pred.csv"
EID_FILE = "elem_id.csv"

# complexity penalty
LAMBDA_SIZE = 0.0 #1.0e-5

# big penalty for invalid / diverged candidates
BIG_PENALTY = 1.0e9

N_CANDIDATES = 10
N_WORKERS = None
N_GENERATIONS = 20
TOURNAMENT_SIZE = 3
MUTATION_PROB = 0.5
CROSSOVER_PROB = 0.5
ELITE_COUNT = 2

# ============================================================
# SMALL UTILITIES
# ============================================================
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
    if np.isscalar(right):
        if abs(right) < 1e-6:
            return 1.0
        return left / right

    out = np.ones_like(left, dtype=np.float64)

    # IMPORTANT: avoid division when left ≈ right
    mask = (np.abs(right) > 1e-6) & (np.abs(left - right) > 1e-6)

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
pset = gp.PrimitiveSet("MAIN", 1)
pset.renameArguments(ARG0="yw")

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

toolbox.decorate("mate", gp.staticLimit(key=len, max_value=18))
toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=18))
toolbox.register("clone", copy.deepcopy)

def build_lm_from_yw_raw(raw, yw, z_cg, z_pos):
    raw = np.asarray(raw, dtype=np.float64).copy()
    raw = np.clip(raw, -1.0, 1.0)

    kappa0 = 0.41
    alpha = 0.5
    lm_cap = 0.1

    kappa_eff = kappa0 * (1.0 + alpha * raw)
    kappa_eff = np.maximum(kappa_eff, 0.0)
    kappa_eff = np.minimum(kappa_eff, 1.2)

    lm_raw = kappa_eff * yw
    lm = lm_raw / (1.0 + lm_raw / lm_cap)

    lm[z_cg > z_pos] = 0.0
    lm = np.maximum(lm, 0.0)

    print("lm_raw min/max =", np.min(lm_raw), np.max(lm_raw), flush=True)
    return lm, kappa_eff

def make_core_limited_batches(tasks, core_limit):
    batches = []
    current = []
    current_cores = 0

    for task in tasks:
        _, _, case = task
        np_ranks = case["np"]

        if current and current_cores + np_ranks > core_limit:
            batches.append(current)
            current = []
            current_cores = 0

        current.append(task)
        current_cores += np_ranks

    if current:
        batches.append(current)

    return batches

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

def evaluate_population(population, selected_cases):
    tasks = []

    for case in selected_cases:
        for cand_id, individual in enumerate(population):
            tasks.append((cand_id, individual, case))

    batch_cores = sum(task[2]["np"] for task in tasks)

    if batch_cores > CORE_LIMIT:
        raise RuntimeError(
            f"Selected batch uses {batch_cores} cores, "
            f"which exceeds CORE_LIMIT={CORE_LIMIT}"
        )

    random.shuffle(tasks)

    candidate_norm_errors = {cand_id: [] for cand_id in range(len(population))}
    results = []

    print(
        f"\n===== Running one stochastic batch with "
        f"{len(tasks)} tasks, {batch_cores} MPI ranks =====",
        flush=True,
    )

    with ProcessPoolExecutor(max_workers=len(tasks)) as ex:
        futures = [
            ex.submit(evaluate_candidate_case_worker, task)
            for task in tasks
        ]

        for fut in as_completed(futures):
            cand_id, RE, CG, raw_fit, norm_fit, expr = fut.result()
            candidate_norm_errors[cand_id].append(norm_fit)
            results.append((cand_id, RE, CG, raw_fit, norm_fit, expr))

    for cand_id in range(len(population)):
        errs = candidate_norm_errors[cand_id]
        population[cand_id].fitness.values = (
            float(np.mean(errs)) if errs else BIG_PENALTY,
        )

    return results

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

    start = time.time()
    ret = subprocess.call(cmd, shell=True, executable="/bin/bash")
    elapsed = time.time() - start

    print("Return code =", ret, flush=True)
    print("Elapsed:", round(elapsed, 2), "sec", flush=True)

    log_path = os.path.join(case_path, LOG_FILE)
    if not os.path.exists(log_path):
        print("Log file not found", flush=True)
        return False

    has_err = search_last(log_path, str_err)
    has_conv = search_last(log_path, str_conv)
    has_ss = search_last(log_path, str_ss)
    has_fin = search_last(log_path, str_fin)

    print("has_err =", has_err, flush=True)
    print("has_conv =", has_conv, flush=True)
    print("has_ss =", has_ss, flush=True)
    print("has_fin =", has_fin, flush=True)

    is_done = (has_conv or has_ss) and has_fin
    return is_done and not has_err

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

def evaluate_individual(individual, r_cg, z_cg, yw, ur_les, uz_les, case_path):
    """
    One GP candidate -> lm_pred.csv -> solver -> velocity-based fitness
    """
    func = toolbox.compile(expr=individual)

    try:
        yw_scale = np.max(np.abs(yw))
        yw_n = yw / yw_scale

        raw = func(yw_n)

        if np.isscalar(raw):
            raw = np.full_like(r_cg, raw, dtype=np.float64)

        raw = np.asarray(raw, dtype=np.float64)

        if raw.shape != r_cg.shape:
            print("Invalid shape from candidate", flush=True)
            return BIG_PENALTY,

        if np.any(~np.isfinite(raw)):
            print("Non-finite values from candidate", flush=True)
            return BIG_PENALTY,

        lm, kappa_eff = build_lm_from_yw_raw(raw, yw, z_cg, Z_POS)
        print("kappa_eff min/max =", np.min(kappa_eff), np.max(kappa_eff), flush=True)
        print("lm min/max =", np.min(lm), np.max(lm), flush=True)

        clean_case_folder(case_path)

        lm_path = os.path.join(case_path, LM_FILE)
        save_lm_csv(z_cg, r_cg, lm, lm_path)

        ok = run_case(case_path, STEADY_RUN_SCRIPT)

        if not ok:
            print("Steady failed; trying unsteady fallback", flush=True)

            clean_case_folder(case_path)

            lm_path = os.path.join(case_path, LM_FILE)
            save_lm_csv(z_cg, r_cg, lm, lm_path)

            ok = run_case(case_path, UNSTEADY_RUN_SCRIPT)

        if not ok:
            print("Both steady and unsteady fallback failed for candidate", flush=True)
            return BIG_PENALTY,

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
        fitness = vel_err + complexity_penalty

        print("raw min/max =", np.min(raw), np.max(raw), flush=True)
        print("lm min/max =", np.min(lm), np.max(lm), flush=True)

        print("Expression:", individual, flush=True)
        print("Velocity MSE:", vel_err, flush=True)
        print("Complexity penalty:", complexity_penalty, flush=True)
        print("Fitness:", fitness, flush=True)

        return fitness,

    except Exception as e:
        print("Exception during evaluation:", e, flush=True)
        return BIG_PENALTY,

def evaluate_candidate_case_worker(args):
    cand_id, individual, case = args

    case_path = os.path.join(case["case_root"], f"cand_{cand_id:03d}")

    print(
        f"\n===== Candidate {cand_id:03d}, "
        f"RE={case['RE']}, CG={case['CG']} =====",
        flush=True,
    )
    print("Expression:", individual, flush=True)

    fit = evaluate_individual(
        individual,
        case["r_cg"],
        case["z_cg"],
        case["yw"],
        case["ur_les"],
        case["uz_les"],
        case_path,
    )[0]

    if case["baseline_err"] <= 0.0 or not np.isfinite(case["baseline_err"]):
        norm_fit = BIG_PENALTY
    else:
        norm_fit = fit / case["baseline_err"]

    return cand_id, case["RE"], case["CG"], fit, norm_fit, str(individual)

def initialize_template_from_population(population, case):
    print("\n===== Initializing template from average GP candidate =====", flush=True)
    r_cg = case["r_cg"]
    z_cg = case["z_cg"]
    yw = case["yw"]
    template_dir = case["template_dir"]

    lm_list = []

    yw_scale = np.max(np.abs(yw))
    yw_n = yw / yw_scale

    for ind in population:
        func = toolbox.compile(expr=ind)
        raw = func(yw_n)

        if np.isscalar(raw):
            raw = np.full_like(r_cg, raw, dtype=np.float64)

        raw = np.asarray(raw, dtype=np.float64)

        if raw.shape != r_cg.shape or np.any(~np.isfinite(raw)):
            continue

        lm, _ = build_lm_from_yw_raw(raw, yw, z_cg, Z_POS)
        lm_list.append(lm)

    if len(lm_list) == 0:
        raise RuntimeError("No valid GP candidates for template initialization.")

    lm_avg = np.mean(np.vstack(lm_list), axis=0)

    clean_case_folder(template_dir)
    set_restart_flag(template_dir, enable_restart=False)
    save_lm_csv(z_cg, r_cg, lm_avg, os.path.join(template_dir, LM_FILE))

    ok = run_case(template_dir, STEADY_RUN_SCRIPT)

    if not ok:
        print("Template steady initialization failed; trying unsteady.", flush=True)
        clean_case_folder(template_dir)
        set_restart_flag(template_dir, enable_restart=False)
        save_lm_csv(z_cg, r_cg, lm_avg, os.path.join(template_dir, LM_FILE))
        ok = run_case(template_dir, UNSTEADY_RUN_SCRIPT)

    if not ok:
        raise RuntimeError("Template initialization failed.")

    set_restart_flag(template_dir, enable_restart=True)

    print("Template initialization finished successfully.", flush=True)

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

# ============================================================
# MAIN: evaluate ONE DEAP candidate with real solver fitness
# ============================================================
def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("\n===== Loading all available cases =====", flush=True)

    case_data = []

    for case in CASES:
        RE = case["RE"]
        CG = case["CG"]
        np_ranks = case["np"]

        case_root, template_dir, baseline_dir, mapped_file = get_case_paths(RE, CG)

        print(f"\n===== Preparing RE={RE}, CG={CG}, np={np_ranks} =====", flush=True)

        r_cg, z_cg, ur_les, uz_les, yw, nut = read_mapped(mapped_file)

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
        }

        case_entry["baseline_err"] = evaluate_baseline_mixing_length(case_entry)
        case_data.append(case_entry)

    with open(RESULTS_FILE, "w") as f:
        f.write("seed,gen,rank,fitness,cases,expr\n")

    population = [toolbox.individual() for _ in range(N_CANDIDATES)]

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

        selected_cases, batch_cores = select_random_case_batch(case_data)

        selected_case_names = [
            f"RE{case['RE']}_CG{case['CG']}"
            for case in selected_cases
        ]

        print("Selected cases:", selected_case_names, flush=True)
        print("Selected batch cores:", batch_cores, flush=True)

        results = evaluate_population(population, selected_cases)

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

    best = min(population, key=lambda ind: ind.fitness.values[0])

    print(f"Best sampled normalized fitness = {best.fitness.values[0]:.8f}", flush=True)
    print(f"Best expr = {best}", flush=True)

    print("\n===== Baselines =====", flush=True)
    for case in case_data:
        print(
            f"RE={case['RE']}, CG={case['CG']}: "
            f"baseline MSE = {case['baseline_err']:.8e}",
            flush=True,
        )

if __name__ == "__main__":
    main()
