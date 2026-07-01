import os
import fnmatch
import numpy as np

try:
    from scipy.spatial import cKDTree
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# ============================================================
# PATHS
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DNS_FILE = os.path.join(
    PROJECT_ROOT,
    "0_data_les",
    "backward",
    "BFS_Ret395_ER2_XY.dat",
)

CASE_DIR = os.path.join(
    PROJECT_ROOT,
    "1_mixing_length",
    "backward",
)

OUT_DIR = os.path.join(
    PROJECT_ROOT,
    "2_mapped",
    "backward",
)

OUT_FILE = os.path.join(OUT_DIR, "mapped.csv")


# ============================================================
# UTILITIES
# ============================================================
def find_latest_file(folder, pattern, label):
    files = fnmatch.filter(os.listdir(folder), pattern)
    files = [os.path.join(folder, f) for f in files]

    if len(files) == 0:
        raise FileNotFoundError(
            f"No {label} files found in {folder} with pattern {pattern}"
        )

    files.sort(key=lambda x: os.path.getmtime(x))
    return files[-1]


def get_header_indices(csv_file):
    with open(csv_file, "r") as f:
        header = f.readline().strip().split(",")

    header = [h.strip() for h in header]
    col = {name: i for i, name in enumerate(header)}

    return header, col


def read_moose_vpp():
    vpp_file = find_latest_file(
        CASE_DIR,
        "bfs_2d_fv_csv_vpp_*.csv",
        "MOOSE VPP",
    )

    print("Reading MOOSE VPP:", vpp_file)

    header, col = get_header_indices(vpp_file)
    print("VPP columns:")
    print(header)

    needed = [
        "x",
        "y",
        "u",
        "v",
        "yw_aux_var",
        "eddy_viscosity_aux_var",
    ]

    for name in needed:
        if name not in col:
            raise ValueError(
                f"Required column '{name}' not found in VPP file.\n"
                f"Available columns: {header}"
            )

    usecols = tuple(col[name] for name in needed)

    data = np.loadtxt(
        vpp_file,
        delimiter=",",
        skiprows=1,
        usecols=usecols,
        dtype=np.float64,
    )

    x_cg = data[:, 0]
    y_cg = data[:, 1]
    u_cg = data[:, 2]
    v_cg = data[:, 3]
    yw = data[:, 4]
    nut = data[:, 5]

    print("MOOSE points:", x_cg.size)
    print("MOOSE x min/max:", x_cg.min(), x_cg.max())
    print("MOOSE y min/max:", y_cg.min(), y_cg.max())
    print("MOOSE u min/max:", u_cg.min(), u_cg.max())
    print("MOOSE v min/max:", v_cg.min(), v_cg.max())
    print("MOOSE yw min/max:", yw.min(), yw.max())
    print("MOOSE nut min/max:", nut.min(), nut.max())

    return x_cg, y_cg, u_cg, v_cg, yw, nut


def read_dns():
    print("Reading DNS:", DNS_FILE)

    # Tecplot POINT-packed:
    # columns:
    # x, y, <u>, <v>, <u'u'>, ...
    data = np.loadtxt(
        DNS_FILE,
        skiprows=3,
        usecols=(0, 1, 2, 3),
        dtype=np.float64,
    )

    x_dns = data[:, 0]
    y_dns = data[:, 1]
    u_dns = data[:, 2]
    v_dns = data[:, 3]

    # The DNS map includes the rectangular plotting region.
    # The upstream lower block x<0,y<0 is solid/zero-filled.
    # Remove it so nearest-neighbor mapping cannot select solid points.
    fluid = ~((x_dns < 0.0) & (y_dns < 0.0))

    x_dns = x_dns[fluid]
    y_dns = y_dns[fluid]
    u_dns = u_dns[fluid]
    v_dns = v_dns[fluid]

    print("DNS fluid points:", x_dns.size)
    print("DNS x min/max:", x_dns.min(), x_dns.max())
    print("DNS y min/max:", y_dns.min(), y_dns.max())
    print("DNS u min/max:", u_dns.min(), u_dns.max())
    print("DNS v min/max:", v_dns.min(), v_dns.max())

    return x_dns, y_dns, u_dns, v_dns


def nearest_neighbor_map(x_cg, y_cg, x_dns, y_dns):
    cg_points = np.column_stack((x_cg, y_cg))
    dns_points = np.column_stack((x_dns, y_dns))

    if HAVE_SCIPY:
        print("Using scipy.spatial.cKDTree for nearest-neighbor mapping.")
        tree = cKDTree(dns_points)
        dist, i_map = tree.query(cg_points, k=1)
        return i_map.astype(np.int64), dist.astype(np.float64)

    print("WARNING: scipy not available. Using slow chunked NumPy mapping.")

    i_map = np.zeros(x_cg.shape, dtype=np.int64)
    dist = np.zeros(x_cg.shape, dtype=np.float64)

    chunk = 250

    for i0 in range(0, x_cg.size, chunk):
        i1 = min(i0 + chunk, x_cg.size)

        dx = x_cg[i0:i1, None] - x_dns[None, :]
        dy = y_cg[i0:i1, None] - y_dns[None, :]
        d2 = dx * dx + dy * dy

        idx = np.argmin(d2, axis=1)
        i_map[i0:i1] = idx
        dist[i0:i1] = np.sqrt(d2[np.arange(i1 - i0), idx])

    return i_map, dist


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    x_cg, y_cg, u_cg, v_cg, yw, nut = read_moose_vpp()
    x_dns, y_dns, u_dns, v_dns = read_dns()

    print("\nMapping DNS to MOOSE coarse-grid points...")
    i_map, map_dist = nearest_neighbor_map(x_cg, y_cg, x_dns, y_dns)

    if np.unique(i_map).size != i_map.size:
        print("WARNING: repeated DNS indices in mapping.")
        print("This can be okay if the MOOSE grid is locally finer than the DNS map.")

    print("map distance min/max:", map_dist.min(), map_dist.max())
    print("map distance mean:", map_dist.mean())
    print("map distance 95th percentile:", np.percentile(map_dist, 95))

    u_dns_map = u_dns[i_map]
    v_dns_map = v_dns[i_map]

    # Baseline MSE on the mapped field
    mse_u = np.mean((u_cg - u_dns_map) ** 2)
    mse_v = np.mean((v_cg - v_dns_map) ** 2)
    mse_uv = np.mean((u_cg - u_dns_map) ** 2 + (v_cg - v_dns_map) ** 2)

    dns_norm = np.mean(u_dns_map ** 2 + v_dns_map ** 2)
    rel_mse = mse_uv / dns_norm

    print("\n===== Baseline error on mapped DNS field =====")
    print("MSE_u        =", f"{mse_u:.12e}")
    print("MSE_v        =", f"{mse_v:.12e}")
    print("MSE_uv       =", f"{mse_uv:.12e}")
    print("DNS mean |U|2=", f"{dns_norm:.12e}")
    print("relative MSE =", f"{rel_mse:.12e}")

    arr = np.column_stack(
        (
            x_cg,
            y_cg,
            u_dns_map,
            v_dns_map,
            yw,
            nut,
            u_cg,
            v_cg,
            map_dist,
        )
    )

    np.savetxt(
        OUT_FILE,
        arr,
        delimiter=",",
        header=(
            "x_cg,y_cg,u_dns,v_dns,yw,nut,"
            "u_baseline,v_baseline,map_dist"
        ),
        comments="",
        fmt="%.12e",
    )

    print("\nSaved:", OUT_FILE)


if __name__ == "__main__":
    main()