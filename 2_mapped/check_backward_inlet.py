import os
import fnmatch
import numpy as np
import pandas as pd

PROJECT_ROOT = "/homes/aiskhak/projects/GP"

CASE_DIR = os.path.join(PROJECT_ROOT, "1_mixing_length", "backward")
INLET_FILE = os.path.join(CASE_DIR, "dns_inlet_u_xminus5p98.csv")
MAPPED_FILE = os.path.join(PROJECT_ROOT, "2_mapped", "backward", "mapped.csv")


def find_latest_file(folder, pattern):
    files = fnmatch.filter(os.listdir(folder), pattern)
    files = [os.path.join(folder, f) for f in files]
    files.sort(key=lambda x: os.path.getmtime(x))
    return files[-1]


vpp_file = find_latest_file(CASE_DIR, "bfs_2d_fv_csv_vpp_*.csv")

print("VPP:", vpp_file)
print("Inlet file:", INLET_FILE)
print("Mapped:", MAPPED_FILE)

vpp = pd.read_csv(vpp_file)
mapped = pd.read_csv(MAPPED_FILE)

inlet = np.loadtxt(INLET_FILE, delimiter=",")
y_prof = inlet[:, 0]
u_prof = inlet[:, 1]

# First MOOSE x-plane
x0 = np.min(vpp["x"].values)
tol = 1e-10

near = vpp[np.isclose(vpp["x"], x0, atol=tol)].copy()
near = near.sort_values("y")

# Interpolate prescribed inlet profile to MOOSE y locations
u_in_interp = np.interp(near["y"].values, y_prof, u_prof)

near["u_in_prescribed"] = u_in_interp
near["du_moose_minus_inlet"] = near["u"] - near["u_in_prescribed"]

print("\nFirst MOOSE plane x =", x0)
print("n points =", len(near))

print("\nNear-inlet comparison: MOOSE u vs prescribed inlet u")
print(near[["y", "u", "u_in_prescribed", "du_moose_minus_inlet", "v"]].to_string(index=False))

print("\nSummary:")
print("max |u - u_in| =", np.max(np.abs(near["du_moose_minus_inlet"].values)))
print("mean |u - u_in| =", np.mean(np.abs(near["du_moose_minus_inlet"].values)))
print("max |v| =", np.max(np.abs(near["v"].values)))

# Compare mapped DNS at same first plane
m0 = np.min(mapped["x_cg"].values)
mnear = mapped[np.isclose(mapped["x_cg"], m0, atol=tol)].copy()
mnear = mnear.sort_values("y_cg")

print("\nMapped first plane x =", m0)
print("\nMOOSE baseline vs mapped DNS at first plane:")
print(mnear[["y_cg", "u_dns", "u_baseline", "v_dns", "v_baseline", "map_dist"]].to_string(index=False))

print("\nFirst-plane mapped error:")
du = mnear["u_baseline"].values - mnear["u_dns"].values
dv = mnear["v_baseline"].values - mnear["v_dns"].values
print("MSE_u first plane =", np.mean(du**2))
print("MSE_v first plane =", np.mean(dv**2))
print("MSE_uv first plane =", np.mean(du**2 + dv**2))