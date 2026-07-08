from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

case = Path(".")
vtu_files = sorted((case / "VTK").glob("**/internal.vtu"))
if not vtu_files:
    raise SystemExit("No internal.vtu found. Run foamToVTK first.")

vtu_path = vtu_files[-1]
print(f"Reading {vtu_path}")

root = ET.parse(vtu_path).getroot()

piece = root.find(".//Piece")
if piece is None:
    raise SystemExit("Could not find VTU Piece.")

def read_data_array(parent, name=None):
    for da in parent.findall("DataArray"):
        if name is None or da.attrib.get("Name") == name:
            ncomp = int(da.attrib.get("NumberOfComponents", "1"))
            text = da.text or ""
            vals = np.fromstring(text, sep=" ")
            if ncomp > 1:
                vals = vals.reshape((-1, ncomp))
            return vals
    return None

# Points
points_node = piece.find("Points")
points = read_data_array(points_node)
if points is None:
    raise SystemExit("Could not read points.")

# Cells/connectivity
cells_node = piece.find("Cells")
connectivity = read_data_array(cells_node, "connectivity").astype(int)
offsets = read_data_array(cells_node, "offsets").astype(int)

cells = []
start = 0
for off in offsets:
    cells.append(connectivity[start:off])
    start = off

centers = np.array([points[c].mean(axis=0) for c in cells])
x = centers[:, 0]
y = centers[:, 1]

# Cell data
fields = {}
cell_data = piece.find("CellData")
if cell_data is not None:
    for da in cell_data.findall("DataArray"):
        name = da.attrib.get("Name")
        if not name:
            continue
        ncomp = int(da.attrib.get("NumberOfComponents", "1"))
        vals = np.fromstring(da.text or "", sep=" ")
        if ncomp > 1:
            vals = vals.reshape((-1, ncomp))
        fields[name] = vals

print("Available cell fields:")
for k, v in fields.items():
    print(f"  {k}: shape {np.shape(v)}")

if "U" not in fields:
    raise SystemExit("U field not found in CellData.")

U = fields["U"]
Ux = U[:, 0]
Uy = U[:, 1]
Umag = np.sqrt(Ux**2 + Uy**2)

plot_data = {
    "Ux": (Ux, "Ux [m/s]"),
    "Uy": (Uy, "Uy [m/s]"),
    "Umag": (Umag, "|U| [m/s]"),
}

for name in ["nut", "k", "omega", "p", "yPlus"]:
    if name in fields:
        arr = fields[name]
        if arr.ndim > 1:
            arr = arr[:, 0]
        plot_data[name] = (arr, name)

def plot_scalar(name, values, label):
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(x, y, c=values, s=3, linewidths=0)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(label)
    cb = plt.colorbar(sc)
    cb.set_label(label)
    plt.tight_layout()
    out = Path("figs") / f"{name}_20000.png"
    plt.savefig(out, dpi=250)
    plt.close()
    print(f"Wrote {out}")

Path("figs").mkdir(exist_ok=True)

for name, (values, label) in plot_data.items():
    plot_scalar(name, values, label)

# Velocity vector overlay
stride = max(1, len(x) // 2500)
idx = np.arange(0, len(x), stride)

plt.figure(figsize=(8, 6))
plt.scatter(x, y, c=Umag, s=3, linewidths=0)
plt.quiver(x[idx], y[idx], Ux[idx], Uy[idx], angles="xy", scale_units="xy", scale=1.0, width=0.0015)
plt.gca().set_aspect("equal", adjustable="box")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Velocity vectors over |U|")
cb = plt.colorbar()
cb.set_label("|U| [m/s]")
plt.tight_layout()
out = Path("figs") / "velocity_vectors_20000.png"
plt.savefig(out, dpi=250)
plt.close()
print(f"Wrote {out}")
