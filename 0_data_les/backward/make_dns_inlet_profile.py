import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# USER SETTINGS
# ============================================================
fname = Path("BFS_Ret395_ER2_XY.dat")

# Must match the refined Gmsh mesh
x_inlet = -6.0
x_step = 0.0
Nx_up = 150

# First MOOSE cell-center plane in the upstream block
dx_up = (x_step - x_inlet) / Nx_up
x_target = x_inlet + 0.5 * dx_up

# Output names tied to this mesh
out_u = "dns_inlet_u_xminus5p98.csv"
out_v = "dns_inlet_v_xminus5p98.csv"

# ============================================================
# READ DNS MAP
# ============================================================
data = np.loadtxt(fname, skiprows=3)

x = data[:, 0]
y = data[:, 1]
u = data[:, 2]
v = data[:, 3]

# Pick nearest DNS x-plane
x_unique = np.unique(x)
x_prof = x_unique[np.argmin(np.abs(x_unique - x_target))]

print("x_inlet =", x_inlet)
print("x_step  =", x_step)
print("Nx_up   =", Nx_up)
print("dx_up   =", dx_up)
print("Requested first MOOSE cell-center x_target =", x_target)
print("Using nearest DNS x =", x_prof)
print("Difference =", abs(x_prof - x_target))

mask = np.isclose(x, x_prof, atol=1e-12)

prof = pd.DataFrame({
    "y": y[mask],
    "u": u[mask],
    "v": v[mask],
})

# Keep only inlet channel y = 0..1
prof = prof[(prof["y"] >= -1e-8) & (prof["y"] <= 1.0 + 1e-8)]
prof = prof.sort_values("y").reset_index(drop=True)

# Add exact no-slip wall endpoints
wall_rows = pd.DataFrame({
    "y": [0.0, 1.0],
    "u": [0.0, 0.0],
    "v": [0.0, 0.0],
})

prof = pd.concat([prof, wall_rows], ignore_index=True)
prof = prof.drop_duplicates(subset="y", keep="first")
prof = prof.sort_values("y").reset_index(drop=True)

print()
print(prof.head(10))
print(prof.tail(10))
print("n profile points =", len(prof))
print("u min/max =", prof["u"].min(), prof["u"].max())
print("v min/max =", prof["v"].min(), prof["v"].max())

prof[["y", "u"]].to_csv(out_u, index=False, header=False)
prof[["y", "v"]].to_csv(out_v, index=False, header=False)

print()
print("Wrote", out_u)
print("Wrote", out_v)