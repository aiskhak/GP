import os
import fnmatch
import numpy as np

# ============================================================
# USER SETTINGS
# ============================================================
RE_LIST = ["3000", "3413", "5963", "7912", "9000", "14000"] #["10622", "12819", ]
CG_LIST = ["1", "2", "3"] #, "4"

PROJECT_ROOT = os.path.abspath("..")

DATA_LES_ROOT = os.path.join(PROJECT_ROOT, "0_data_les")
MIXING_LENGTH_ROOT = os.path.join(PROJECT_ROOT, "1_mixing_length")
MAPPED_ROOT = os.path.join(PROJECT_ROOT, "2_mapped")


# ============================================================
# UTILITIES
# ============================================================
def do_nearest_neighbor_map(r_cg, z_cg, r_les, z_les):
    i_map = np.zeros(r_cg.shape, dtype=np.int64)
    dist = np.zeros(r_cg.shape, dtype=np.float64)

    for j in range(r_cg.size):
        d = np.sqrt((r_cg[j] - r_les) ** 2 + (z_cg[j] - z_les) ** 2)
        i_best = np.argmin(d)
        i_map[j] = i_best
        dist[j] = d[i_best]

    return i_map, dist


def find_latest_file(folder, pattern, label):
    files = fnmatch.filter(os.listdir(folder), pattern)
    files = [os.path.join(folder, f) for f in files]

    if len(files) == 0:
        raise FileNotFoundError(f"No {label} files found in {folder} with pattern {pattern}")

    files.sort(key=lambda x: os.path.getmtime(x))
    return files[-1]


def find_latest_elv_file(reynolds, coarse_grid):
    folder = os.path.join(MIXING_LENGTH_ROOT, reynolds, coarse_grid)
    return find_latest_file(folder, "tamu_2d_fv_csv_elv_*.csv", "ELV")


def find_latest_vpp_file(reynolds, coarse_grid):
    folder = os.path.join(MIXING_LENGTH_ROOT, reynolds, coarse_grid)
    return find_latest_file(folder, "tamu_2d_fv_csv_vpp_*.csv", "VPP")


def read_cg_elv(reynolds, coarse_grid):
    elv_file = find_latest_elv_file(reynolds, coarse_grid)
    print("Reading ELV:", elv_file)

    # Actual ELV columns:
    # elvol_aux_var,id,x,y,yw_aux_var,z
    data = np.loadtxt(
        elv_file,
        delimiter=",",
        skiprows=1,
        usecols=(2, 3, 4),
        dtype=np.float64,
    )

    z_cg = data[:, 0]   # x
    r_cg = data[:, 1]   # y
    yw = data[:, 2]     # yw_aux_var

    return r_cg, z_cg, yw


def read_baseline_nut(reynolds, coarse_grid):
    vpp_file = find_latest_vpp_file(reynolds, coarse_grid)
    print("Reading VPP:", vpp_file)

    # Expected VPP columns include eddy_viscosity_aux_var as the first column:
    # eddy_viscosity_aux_var,id,u,v,x,y,z
    nut = np.loadtxt(
        vpp_file,
        delimiter=",",
        skiprows=1,
        usecols=(0,),
        dtype=np.float64,
    )

    return nut


def read_les(reynolds):
    les_file = os.path.join(DATA_LES_ROOT, reynolds, "grid4_avg.csv")
    print("Reading LES:", les_file)

    # LES columns: id,r,t,z,u_r,u_z,p
    data_les = np.loadtxt(
        les_file,
        delimiter=",",
        skiprows=1,
        usecols=(1, 3, 4, 5),
        dtype=np.float64,
    )

    r_les = data_les[:, 0]
    z_les = -data_les[:, 1]
    ur_les = data_les[:, 2]
    uz_les = -data_les[:, 3]

    return r_les, z_les, ur_les, uz_les


def build_mapped(reynolds, coarse_grid):
    print(f"\n=== Mapping Re={reynolds}, CG={coarse_grid} ===")

    r_cg, z_cg, yw = read_cg_elv(reynolds, coarse_grid)
    nut = read_baseline_nut(reynolds, coarse_grid)
    r_les, z_les, ur_les, uz_les = read_les(reynolds)

    if nut.size != r_cg.size:
        raise ValueError(
            f"Size mismatch for Re={reynolds}, CG={coarse_grid}: "
            f"nut.size={nut.size}, r_cg.size={r_cg.size}"
        )

    i_map, dist = do_nearest_neighbor_map(r_cg, z_cg, r_les, z_les)

    if np.unique(i_map).size != i_map.size:
        print("WARNING: repeated LES indices in mapping")

    print("max distance =", np.max(dist))
    print("mean distance =", np.mean(dist))

    ur_map = ur_les[i_map]
    uz_map = uz_les[i_map]

    out_dir = os.path.join(MAPPED_ROOT, reynolds, coarse_grid)
    os.makedirs(out_dir, exist_ok=True)

    out_file = os.path.join(out_dir, "mapped.csv")

    arr = np.column_stack(
        (
            r_cg,
            z_cg,
            ur_map,
            uz_map,
            yw,
            nut,
        )
    )

    np.savetxt(
        out_file,
        arr,
        delimiter=",",
        header="r_cg,z_cg,ur_les,uz_les,yw,nut",
        comments="",
        fmt="%.12e",
    )

    print("Saved:", out_file)


def main():
    for reynolds in RE_LIST:
        for coarse_grid in CG_LIST:
            build_mapped(reynolds, coarse_grid)


if __name__ == "__main__":
    main()