import os
import fnmatch
import xml.etree.ElementTree as ET
import numpy as np

#cd /homes/aiskhak/projects/GP/2_mapped
#CERVI_CASE=03_ml_dx025_delta0p1 python map_cervi_openfoam_to_moose.py


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

# Default MOOSE case. Override with:
#   CERVI_CASE=01_ml_coarse python map_cervi_openfoam_to_moose.py
#   CERVI_CASE=02_ml_dx025  python map_cervi_openfoam_to_moose.py
CASE_NAME = os.environ.get("CERVI_CASE", "02_ml_dx025")

CASE_DIR = os.path.join(
    PROJECT_ROOT,
    "1_mixing_length",
    "cervi",
    CASE_NAME,
)

OF_CASE_DIR = os.path.join(
    PROJECT_ROOT,
    "0_data_les",
    "cervi",
    "test1_sst_simpleFoam_qs",
)

OF_VTU_DEFAULT = os.path.join(
    OF_CASE_DIR,
    "VTK",
    "test1_sst_simpleFoam_qs_20000",
    "internal.vtu",
)

OUT_DIR = os.path.join(
    PROJECT_ROOT,
    "2_mapped",
    "cervi",
    CASE_NAME,
)

OUT_FILE = os.path.join(OUT_DIR, "mapped.csv")
METRICS_FILE = os.path.join(OUT_DIR, "metrics.csv")


# ============================================================
# UTILITIES
# ============================================================
def find_latest_file_recursive(folder, pattern, label):
    matches = []

    for root, _, files in os.walk(folder):
        for f in fnmatch.filter(files, pattern):
            matches.append(os.path.join(root, f))

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No {label} files found in {folder} with pattern {pattern}"
        )

    matches.sort(key=lambda x: os.path.getmtime(x))
    return matches[-1]


def find_openfoam_vtu():
    if os.path.isfile(OF_VTU_DEFAULT):
        return OF_VTU_DEFAULT

    return find_latest_file_recursive(
        OF_CASE_DIR,
        "internal.vtu",
        "OpenFOAM internal.vtu",
    )


def get_header_indices(csv_file):
    with open(csv_file, "r") as f:
        header = f.readline().strip().split(",")

    header = [h.strip() for h in header]
    col = {name: i for i, name in enumerate(header)}

    return header, col


def get_col(col, header, name):
    if name not in col:
        raise ValueError(
            f"Required column '{name}' not found.\n"
            f"Available columns: {header}"
        )
    return col[name]


# ============================================================
# READ MOOSE
# ============================================================
def read_csv_selected(csv_file, needed):
    header, col = get_header_indices(csv_file)
    print("Columns in", csv_file)
    print(header)

    usecols = tuple(get_col(col, header, name) for name in needed)

    data = np.loadtxt(
        csv_file,
        delimiter=",",
        skiprows=1,
        usecols=usecols,
        dtype=np.float64,
    )

    if data.ndim == 1:
        data = data.reshape(1, -1)

    return data, header


def map_elv_to_vpp(x_vpp, y_vpp, x_elv, y_elv, yw_elv):
    vpp_points = np.column_stack((x_vpp, y_vpp))
    elv_points = np.column_stack((x_elv, y_elv))

    if HAVE_SCIPY:
        tree = cKDTree(elv_points)
        dist, ind = tree.query(vpp_points, k=1)
    else:
        ind = np.zeros(x_vpp.shape, dtype=np.int64)
        dist = np.zeros(x_vpp.shape, dtype=np.float64)
        chunk = 250
        for i0 in range(0, x_vpp.size, chunk):
            i1 = min(i0 + chunk, x_vpp.size)
            dx = x_vpp[i0:i1, None] - x_elv[None, :]
            dy = y_vpp[i0:i1, None] - y_elv[None, :]
            d2 = dx * dx + dy * dy
            idx = np.argmin(d2, axis=1)
            ind[i0:i1] = idx
            dist[i0:i1] = np.sqrt(d2[np.arange(i1 - i0), idx])

    print("VPP->ELV map distance min/max:", dist.min(), dist.max())
    print("VPP->ELV map distance mean:", dist.mean())

    if dist.max() > 1e-10:
        print("WARNING: VPP and ELV element centers are not identical to roundoff.")

    return yw_elv[ind]


def read_moose_vpp():
    vpp_file = find_latest_file_recursive(
        CASE_DIR,
        "*vpp*.csv",
        "MOOSE VPP",
    )

    print("Reading MOOSE VPP:", vpp_file)

    # Prefer yw from VPP if present. Otherwise read from ELV and map by coordinates.
    header, col = get_header_indices(vpp_file)

    if "yw_aux_var" in col:
        needed = [
            "x",
            "y",
            "u",
            "v",
            "yw_aux_var",
            "eddy_viscosity_aux_var",
        ]

        data, _ = read_csv_selected(vpp_file, needed)

        x_cg = data[:, 0]
        y_cg = data[:, 1]
        u_cg = data[:, 2]
        v_cg = data[:, 3]
        yw = data[:, 4]
        nut = data[:, 5]

    else:
        needed_vpp = [
            "x",
            "y",
            "u",
            "v",
            "eddy_viscosity_aux_var",
        ]

        data, _ = read_csv_selected(vpp_file, needed_vpp)

        x_cg = data[:, 0]
        y_cg = data[:, 1]
        u_cg = data[:, 2]
        v_cg = data[:, 3]
        nut = data[:, 4]

        elv_file = find_latest_file_recursive(
            CASE_DIR,
            "*elv*.csv",
            "MOOSE ELV",
        )

        print("Reading MOOSE ELV:", elv_file)

        needed_elv = [
            "x",
            "y",
            "yw_aux_var",
        ]

        elv, _ = read_csv_selected(elv_file, needed_elv)

        x_elv = elv[:, 0]
        y_elv = elv[:, 1]
        yw_elv = elv[:, 2]

        yw = map_elv_to_vpp(x_cg, y_cg, x_elv, y_elv, yw_elv)

    print("MOOSE points:", x_cg.size)
    print("MOOSE x min/max:", x_cg.min(), x_cg.max())
    print("MOOSE y min/max:", y_cg.min(), y_cg.max())
    print("MOOSE u min/max:", u_cg.min(), u_cg.max())
    print("MOOSE v min/max:", v_cg.min(), v_cg.max())
    print("MOOSE yw min/max:", yw.min(), yw.max())
    print("MOOSE nut min/max:", nut.min(), nut.max())

    return x_cg, y_cg, u_cg, v_cg, yw, nut


# ============================================================
# READ OPENFOAM VTU
# ============================================================
def read_data_array(parent, name=None):
    if parent is None:
        return None

    for da in parent.findall("DataArray"):
        if name is None or da.attrib.get("Name") == name:
            ncomp = int(da.attrib.get("NumberOfComponents", "1"))
            text = da.text or ""
            vals = np.fromstring(text, sep=" ")
            if ncomp > 1:
                vals = vals.reshape((-1, ncomp))
            return vals

    return None


def read_openfoam_vtu():
    vtu_file = find_openfoam_vtu()

    print("Reading OpenFOAM VTU:", vtu_file)

    root = ET.parse(vtu_file).getroot()
    piece = root.find(".//Piece")

    if piece is None:
        raise RuntimeError(f"Could not find VTU Piece in {vtu_file}")

    points = read_data_array(piece.find("Points"))
    if points is None:
        raise RuntimeError("Could not read VTU points.")

    cells_node = piece.find("Cells")
    connectivity = read_data_array(cells_node, "connectivity")
    offsets = read_data_array(cells_node, "offsets")

    if connectivity is None or offsets is None:
        raise RuntimeError("Could not read VTU cell connectivity/offsets.")

    connectivity = connectivity.astype(np.int64)
    offsets = offsets.astype(np.int64)

    centers = []
    start = 0
    for off in offsets:
        conn = connectivity[start:off]
        centers.append(points[conn].mean(axis=0))
        start = off

    centers = np.asarray(centers)

    cell_data = piece.find("CellData")
    U = read_data_array(cell_data, "U")

    if U is None:
        raise RuntimeError("Could not find OpenFOAM U in VTU CellData.")

    x_of = centers[:, 0]
    y_of = centers[:, 1]
    u_of = U[:, 0]
    v_of = U[:, 1]

    print("OpenFOAM points:", x_of.size)
    print("OpenFOAM x min/max:", x_of.min(), x_of.max())
    print("OpenFOAM y min/max:", y_of.min(), y_of.max())
    print("OpenFOAM u min/max:", u_of.min(), u_of.max())
    print("OpenFOAM v min/max:", v_of.min(), v_of.max())

    return x_of, y_of, u_of, v_of


# ============================================================
# MAPPING
# ============================================================
def nearest_neighbor_map(x_cg, y_cg, x_ref, y_ref):
    cg_points = np.column_stack((x_cg, y_cg))
    ref_points = np.column_stack((x_ref, y_ref))

    if HAVE_SCIPY:
        print("Using scipy.spatial.cKDTree for nearest-neighbor mapping.")
        tree = cKDTree(ref_points)
        dist, i_map = tree.query(cg_points, k=1)
        return i_map.astype(np.int64), dist.astype(np.float64)

    print("WARNING: scipy not available. Using slow chunked NumPy mapping.")

    i_map = np.zeros(x_cg.shape, dtype=np.int64)
    dist = np.zeros(x_cg.shape, dtype=np.float64)

    chunk = 250

    for i0 in range(0, x_cg.size, chunk):
        i1 = min(i0 + chunk, x_cg.size)

        dx = x_cg[i0:i1, None] - x_ref[None, :]
        dy = y_cg[i0:i1, None] - y_ref[None, :]
        d2 = dx * dx + dy * dy

        idx = np.argmin(d2, axis=1)
        i_map[i0:i1] = idx
        dist[i0:i1] = np.sqrt(d2[np.arange(i1 - i0), idx])

    return i_map, dist


# ============================================================
# METRICS
# ============================================================
def metric_row(name, mask, u_cg, v_cg, u_ref, v_ref):
    if np.count_nonzero(mask) == 0:
        return None

    du = u_cg[mask] - u_ref[mask]
    dv = v_cg[mask] - v_ref[mask]

    mse_u = np.mean(du**2)
    mse_v = np.mean(dv**2)
    mse_uv = np.mean(du**2 + dv**2)

    ref_norm = np.mean(u_ref[mask]**2 + v_ref[mask]**2)
    rel_mse = mse_uv / max(ref_norm, 1e-300)
    rel_l2 = np.sqrt(rel_mse)

    return [
        name,
        int(np.count_nonzero(mask)),
        mse_u,
        mse_v,
        mse_uv,
        ref_norm,
        rel_mse,
        rel_l2,
        np.mean(np.sqrt(u_ref[mask]**2 + v_ref[mask]**2)),
        np.mean(np.sqrt(u_cg[mask]**2 + v_cg[mask]**2)),
    ]


def write_metrics(x, y, u_cg, v_cg, u_ref, v_ref):
    masks = {
        "all": np.ones_like(x, dtype=bool),
        "main_tank": (x >= 0.0) & (x <= 1.0) & (y >= 0.0) & (y <= 1.0),
        "inlet_duct": (x >= 0.30) & (x <= 0.50) & (y >= -0.40) & (y <= 0.0),
        "outlet_duct": (x >= 1.0) & (x <= 1.40) & (y >= 0.50) & (y <= 0.70),
        "lower_tank": (x >= 0.0) & (x <= 1.0) & (y >= 0.0) & (y < 0.50),
        "upper_tank": (x >= 0.0) & (x <= 1.0) & (y >= 0.50) & (y <= 1.0),
    }

    header = (
        "region,n,mse_u,mse_v,mse_uv,ref_mean_U2,"
        "relative_mse,relative_l2,mean_Uref_mag,mean_Umoose_mag"
    )

    rows = []
    for name, mask in masks.items():
        row = metric_row(name, mask, u_cg, v_cg, u_ref, v_ref)
        if row is not None:
            rows.append(row)

    with open(METRICS_FILE, "w") as f:
        f.write(header + "\n")
        for row in rows:
            f.write(
                f"{row[0]},{row[1]},"
                + ",".join(f"{v:.12e}" for v in row[2:])
                + "\n"
            )

    print("\n===== Velocity-error metrics =====")
    print(header)
    for row in rows:
        print(
            f"{row[0]},{row[1]},"
            + ",".join(f"{v:.6e}" for v in row[2:])
        )


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("CASE_NAME:", CASE_NAME)
    print("CASE_DIR:", CASE_DIR)
    print("OF_CASE_DIR:", OF_CASE_DIR)
    print("OUT_DIR:", OUT_DIR)

    x_cg, y_cg, u_cg, v_cg, yw, nut = read_moose_vpp()
    x_of, y_of, u_of, v_of = read_openfoam_vtu()

    print("\nMapping OpenFOAM reference to MOOSE coarse-grid points...")
    i_map, map_dist = nearest_neighbor_map(x_cg, y_cg, x_of, y_of)

    if np.unique(i_map).size != i_map.size:
        print("WARNING: repeated OpenFOAM indices in mapping.")
        print("This is usually okay because OpenFOAM is much finer than MOOSE.")

    print("map distance min/max:", map_dist.min(), map_dist.max())
    print("map distance mean:", map_dist.mean())
    print("map distance 95th percentile:", np.percentile(map_dist, 95))

    u_of_map = u_of[i_map]
    v_of_map = v_of[i_map]

    # Baseline MSE on mapped OpenFOAM field
    mse_u = np.mean((u_cg - u_of_map) ** 2)
    mse_v = np.mean((v_cg - v_of_map) ** 2)
    mse_uv = np.mean((u_cg - u_of_map) ** 2 + (v_cg - v_of_map) ** 2)

    of_norm = np.mean(u_of_map**2 + v_of_map**2)
    rel_mse = mse_uv / max(of_norm, 1e-300)
    rel_l2 = np.sqrt(rel_mse)

    print("\n===== Baseline error on mapped OpenFOAM field =====")
    print("MSE_u              =", f"{mse_u:.12e}")
    print("MSE_v              =", f"{mse_v:.12e}")
    print("MSE_uv             =", f"{mse_uv:.12e}")
    print("OpenFOAM mean |U|2 =", f"{of_norm:.12e}")
    print("relative MSE       =", f"{rel_mse:.12e}")
    print("relative L2        =", f"{rel_l2:.12e}")

    arr = np.column_stack(
        (
            x_cg,
            y_cg,
            u_of_map,
            v_of_map,
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

    write_metrics(x_cg, y_cg, u_cg, v_cg, u_of_map, v_of_map)

    print("\nSaved:", OUT_FILE)
    print("Saved:", METRICS_FILE)


if __name__ == "__main__":
    main()
