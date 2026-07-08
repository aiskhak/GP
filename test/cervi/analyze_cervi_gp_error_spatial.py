#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INFILE = Path("cervi_hf_baseline_gp_fields.csv")
OUTDIR = Path("figs_error_diagnostics")
OUTDIR.mkdir(exist_ok=True)

if not INFILE.exists():
    raise FileNotFoundError(f"Missing {INFILE}. Run check_cervi_candidate_errors.py first.")

df = pd.read_csv(INFILE)

required = [
    "x", "y",
    "u_hf", "v_hf",
    "u_baseline", "v_baseline",
    "u_gp", "v_gp",
    "mixing_length_gp", "mixing_length_std",
    "gp_factor", "eddy_viscosity", "yw",
    "err2_baseline", "err2_gp", "err2_gp_minus_baseline",
    "region_id",
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing columns: {missing}\nAvailable: {list(df.columns)}")

x = df["x"].to_numpy()
y = df["y"].to_numpy()

err_b = df["err2_baseline"].to_numpy()
err_g = df["err2_gp"].to_numpy()
derr = df["err2_gp_minus_baseline"].to_numpy()

u_hf = df["u_hf"].to_numpy()
v_hf = df["v_hf"].to_numpy()
u_b = df["u_baseline"].to_numpy()
v_b = df["v_baseline"].to_numpy()
u_g = df["u_gp"].to_numpy()
v_g = df["v_gp"].to_numpy()

umag_hf = np.sqrt(u_hf**2 + v_hf**2)
umag_b = np.sqrt(u_b**2 + v_b**2)
umag_g = np.sqrt(u_g**2 + v_g**2)

gp_factor = df["gp_factor"].to_numpy()
lm_std = df["mixing_length_std"].to_numpy()
lm_gp = df["mixing_length_gp"].to_numpy()
nut = df["eddy_viscosity"].to_numpy()
yw = df["yw"].to_numpy()

regions = {
    "all": np.ones_like(x, dtype=bool),
    "main_tank": (x >= 0.0) & (x <= 1.0) & (y >= 0.0) & (y <= 1.0),
    "inlet_duct": (x >= 0.30) & (x <= 0.50) & (y >= -0.40) & (y <= 0.0),
    "outlet_duct": (x >= 1.0) & (x <= 1.40) & (y >= 0.50) & (y <= 0.70),
    "lower_tank": (x >= 0.0) & (x <= 1.0) & (y >= 0.0) & (y < 0.50),
    "upper_tank": (x >= 0.0) & (x <= 1.0) & (y >= 0.50) & (y <= 1.0),
}

print("\n===== Signed GP impact =====")
print("err2_gp_minus_baseline < 0 means GP improves locally")
print("err2_gp_minus_baseline > 0 means GP worsens locally\n")

rows = []
for name, m in regions.items():
    if not np.any(m):
        continue

    n = int(np.sum(m))
    base_mse = np.mean(err_b[m])
    gp_mse = np.mean(err_g[m])
    dmean = np.mean(derr[m])

    improved = derr[m] < 0.0
    worsened = derr[m] > 0.0

    rows.append({
        "region": name,
        "n": n,
        "base_mse": base_mse,
        "gp_mse": gp_mse,
        "ratio": gp_mse / base_mse if base_mse > 0 else np.nan,
        "mean_delta": dmean,
        "improved_frac": np.mean(improved),
        "worsened_frac": np.mean(worsened),
        "sum_improvement": np.sum(np.minimum(derr[m], 0.0)),
        "sum_worsening": np.sum(np.maximum(derr[m], 0.0)),
    })

summary = pd.DataFrame(rows)
print(summary.to_string(index=False, float_format=lambda z: f"{z:.6e}"))
summary.to_csv(OUTDIR / "regional_signed_gp_impact.csv", index=False)

print("\n===== Largest local worsenings =====")
idx_worst = np.argsort(derr)[-15:][::-1]
for i in idx_worst:
    print(
        f"x={x[i]:.5f}, y={y[i]:.5f}, "
        f"derr={derr[i]:.6e}, "
        f"err_base={err_b[i]:.6e}, err_gp={err_g[i]:.6e}, "
        f"Uhf={umag_hf[i]:.6e}, Ub={umag_b[i]:.6e}, Ugp={umag_g[i]:.6e}, "
        f"factor={gp_factor[i]:.4f}, yw={yw[i]:.4f}"
    )

print("\n===== Largest local improvements =====")
idx_best = np.argsort(derr)[:15]
for i in idx_best:
    print(
        f"x={x[i]:.5f}, y={y[i]:.5f}, "
        f"derr={derr[i]:.6e}, "
        f"err_base={err_b[i]:.6e}, err_gp={err_g[i]:.6e}, "
        f"Uhf={umag_hf[i]:.6e}, Ub={umag_b[i]:.6e}, Ugp={umag_g[i]:.6e}, "
        f"factor={gp_factor[i]:.4f}, yw={yw[i]:.4f}"
    )


def scatter_plot(name, values, title, label, symmetric=False):
    plt.figure(figsize=(8, 6))

    kwargs = {}
    if symmetric:
        vmax = np.nanpercentile(np.abs(values), 99)
        kwargs["vmin"] = -vmax
        kwargs["vmax"] = vmax
        kwargs["cmap"] = "coolwarm"

    sc = plt.scatter(x, y, c=values, s=9, linewidths=0, **kwargs)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(title)
    cb = plt.colorbar(sc)
    cb.set_label(label)
    plt.tight_layout()
    out = OUTDIR / f"{name}.png"
    plt.savefig(out, dpi=250)
    plt.close()
    print("Wrote", out)


scatter_plot(
    "err2_gp_minus_baseline",
    derr,
    "GP error minus baseline error",
    "err2_gp - err2_baseline",
    symmetric=True,
)

scatter_plot(
    "err2_baseline",
    err_b,
    "Standard ML pointwise velocity error",
    "err2_baseline",
)

scatter_plot(
    "err2_gp",
    err_g,
    "GP-corrected ML pointwise velocity error",
    "err2_gp",
)

scatter_plot(
    "gp_factor",
    gp_factor,
    "GP mixing-length multiplier",
    "gp_factor",
)

scatter_plot(
    "lm_std",
    lm_std,
    "Standard mixing length",
    "l_m,std [m]",
)

scatter_plot(
    "lm_gp",
    lm_gp,
    "GP-corrected mixing length",
    "l_m,gp [m]",
)

scatter_plot(
    "umag_hf",
    umag_hf,
    "OpenFOAM reference velocity magnitude",
    "|U|_HF [m/s]",
)

scatter_plot(
    "umag_baseline_minus_hf",
    umag_b - umag_hf,
    "Standard ML |U| minus OpenFOAM |U|",
    "|U|_baseline - |U|_HF [m/s]",
    symmetric=True,
)

scatter_plot(
    "umag_gp_minus_hf",
    umag_g - umag_hf,
    "GP ML |U| minus OpenFOAM |U|",
    "|U|_GP - |U|_HF [m/s]",
    symmetric=True,
)

# A compact vector-difference diagnostic
stride = max(1, len(x) // 800)
sel = np.arange(0, len(x), stride)

plt.figure(figsize=(8, 6))
plt.scatter(x, y, c=derr, s=9, linewidths=0, cmap="coolwarm",
            vmin=-np.nanpercentile(np.abs(derr), 99),
            vmax=np.nanpercentile(np.abs(derr), 99))
plt.quiver(
    x[sel], y[sel],
    (u_g - u_b)[sel], (v_g - v_b)[sel],
    angles="xy", scale_units="xy", scale=0.5, width=0.002,
)
plt.gca().set_aspect("equal", adjustable="box")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("GP-induced velocity change over signed error impact")
cb = plt.colorbar()
cb.set_label("err2_gp - err2_baseline")
plt.tight_layout()
out = OUTDIR / "gp_velocity_change_over_error_delta.png"
plt.savefig(out, dpi=250)
plt.close()
print("Wrote", out)

print("\nSaved summary:", OUTDIR / "regional_signed_gp_impact.csv")
print("Figures are in:", OUTDIR)
