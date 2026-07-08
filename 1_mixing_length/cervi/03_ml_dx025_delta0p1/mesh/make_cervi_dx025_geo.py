from pathlib import Path
from collections import defaultdict
import math

dx = 0.025
lc = dx

# Cervi test1 geometry
xs = [0.0, 0.30, 0.50, 1.00, 1.40]
ys = [-0.40, 0.0, 0.50, 0.70, 1.00]

# Blocks:
# inlet duct: (x1,x2) x (y0,y1)
# tank: x0-x3, y1-y4 split by inlet/outlet coordinates
# outlet duct: (x3,x4) x (y2,y3)
blocks = []
blocks.append((1, 0))  # inlet duct

for i in [0, 1, 2]:
    for j in [1, 2, 3]:
        blocks.append((i, j))  # main tank

blocks.append((3, 2))  # outlet duct

# Create only points used by block corners
used_points = set()
for i, j in blocks:
    used_points.update([(i, j), (i+1, j), (i+1, j+1), (i, j+1)])

geo = []
point_ids = {}
pid = 1

for ij in sorted(used_points):
    i, j = ij
    point_ids[ij] = pid
    geo.append(f"Point({pid}) = {{{xs[i]}, {ys[j]}, 0.0, {lc}}};")
    pid += 1

line_ids = {}
line_orient = {}  # key -> actual orientation used when Line was created
line_counts = defaultdict(int)
lid = 1

def edge_key(a, b):
    return tuple(sorted([a, b]))

def get_line(a, b):
    global lid

    key = edge_key(a, b)

    if key not in line_ids:
        line_ids[key] = lid
        line_orient[key] = (a, b)

        pa = point_ids[a]
        pb = point_ids[b]
        geo.append(f"Line({lid}) = {{{pa}, {pb}}};")

        x0, y0 = xs[a[0]], ys[a[1]]
        x1, y1 = xs[b[0]], ys[b[1]]
        length = math.hypot(x1 - x0, y1 - y0)
        npts = int(round(length / dx)) + 1
        geo.append(f"Transfinite Line {{{lid}}} = {npts};")

        lid += 1

    created_a, created_b = line_orient[key]
    if (a, b) == (created_a, created_b):
        return line_ids[key]
    elif (a, b) == (created_b, created_a):
        return -line_ids[key]
    else:
        raise RuntimeError("Bad line orientation logic")

surfaces = []
sid = 1
loop_id = 1

for i, j in blocks:
    p00 = (i, j)
    p10 = (i+1, j)
    p11 = (i+1, j+1)
    p01 = (i, j+1)

    # Counter-clockwise loop
    l1 = get_line(p00, p10)
    l2 = get_line(p10, p11)
    l3 = get_line(p11, p01)
    l4 = get_line(p01, p00)

    geo.append(f"Curve Loop({loop_id}) = {{{l1}, {l2}, {l3}, {l4}}};")
    geo.append(f"Plane Surface({sid}) = {{{loop_id}}};")
    geo.append(f"Transfinite Surface {{{sid}}};")
    geo.append(f"Recombine Surface {{{sid}}};")

    surfaces.append(sid)

    for e in [edge_key(p00, p10), edge_key(p10, p11), edge_key(p11, p01), edge_key(p01, p00)]:
        line_counts[e] += 1

    loop_id += 1
    sid += 1

# External boundary classification
inlet_lines = []
outlet_lines = []
wall_lines = []

for key, count in line_counts.items():
    if count != 1:
        continue

    a, b = key
    line_id = line_ids[key]

    xa, ya = xs[a[0]], ys[a[1]]
    xb, yb = xs[b[0]], ys[b[1]]

    # bottom inlet opening: y = -0.4, x in [0.3,0.5]
    if abs(ya + 0.40) < 1e-12 and abs(yb + 0.40) < 1e-12:
        inlet_lines.append(line_id)

    # right outlet opening: x = 1.4, y in [0.5,0.7]
    elif abs(xa - 1.40) < 1e-12 and abs(xb - 1.40) < 1e-12:
        outlet_lines.append(line_id)

    else:
        wall_lines.append(line_id)

geo.append(f'Physical Surface("fluid") = {{{", ".join(map(str, surfaces))}}};')
geo.append(f'Physical Line("inlet") = {{{", ".join(map(str, inlet_lines))}}};')
geo.append(f'Physical Line("outlet") = {{{", ".join(map(str, outlet_lines))}}};')
geo.append(f'Physical Line("walls") = {{{", ".join(map(str, wall_lines))}}};')

geo.append("Mesh.RecombineAll = 1;")
geo.append("Mesh.MshFileVersion = 2.2;")

Path("cervi_test1_dx025.geo").write_text("\n".join(geo) + "\n")

print("Wrote cervi_test1_dx025.geo")
print(f"surfaces = {len(surfaces)}")
print(f"inlet lines = {inlet_lines}")
print(f"outlet lines = {outlet_lines}")
print(f"wall lines = {wall_lines}")
