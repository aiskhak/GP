#!/usr/bin/env python3

import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET

infile = Path("BFS_Ret395_ER2_XY.dat")
outfile = Path("BFS_Ret395_ER2_XY_point.vts")

I = 1512
J = 304
NHEADER = 3

names = [
    "x", "y", "u_mean", "v_mean",
    "uu", "vv", "ww", "uv",
    "Ep", "k", "k_prime",
    "Psuu", "Psvv", "Psww", "Psuv",
    "Pruu", "Prvv", "Pruv",
    "Epuu", "Epvv", "Epww", "Epuv",
    "TCDuu", "TCDvv", "TCDww", "TCDuv",
    "TPDuu", "TPDvv", "TPDuv",
]

data = np.loadtxt(infile, skiprows=NHEADER)

print("data shape =", data.shape)
print("expected =", I * J, "points")

x = data[:, 0].reshape(J, I)
y = data[:, 1].reshape(J, I)
u = data[:, 2].reshape(J, I)
v = data[:, 3].reshape(J, I)

print("x range =", x.min(), x.max())
print("y range =", y.min(), y.max())
print("u range =", u.min(), u.max())
print("v range =", v.min(), v.max())

vtk = ET.Element(
    "VTKFile",
    type="StructuredGrid",
    version="0.1",
    byte_order="LittleEndian",
)

grid = ET.SubElement(vtk, "StructuredGrid", WholeExtent=f"0 {I-1} 0 {J-1} 0 0")
piece = ET.SubElement(grid, "Piece", Extent=f"0 {I-1} 0 {J-1} 0 0")

point_data = ET.SubElement(piece, "PointData")

for col, name in enumerate(names[2:], start=2):
    field = data[:, col].reshape(J, I)

    da = ET.SubElement(
        point_data,
        "DataArray",
        type="Float64",
        Name=name,
        NumberOfComponents="1",
        format="ascii",
    )

    da.text = "\n" + "\n".join(f"{val:.12e}" for val in field.ravel()) + "\n"

vel_da = ET.SubElement(
    point_data,
    "DataArray",
    type="Float64",
    Name="velocity",
    NumberOfComponents="3",
    format="ascii",
)

vel_da.text = "\n" + "\n".join(
    f"{uu:.12e} {vv:.12e} 0.0" for uu, vv in zip(u.ravel(), v.ravel())
) + "\n"

ET.SubElement(piece, "CellData")

points_node = ET.SubElement(piece, "Points")
points_da = ET.SubElement(
    points_node,
    "DataArray",
    type="Float64",
    NumberOfComponents="3",
    format="ascii",
)

points_da.text = "\n" + "\n".join(
    f"{xx:.12e} {yy:.12e} 0.0" for xx, yy in zip(x.ravel(), y.ravel())
) + "\n"

tree = ET.ElementTree(vtk)
ET.indent(tree, space="  ")
tree.write(outfile, encoding="utf-8", xml_declaration=True)

print("Wrote:", outfile)