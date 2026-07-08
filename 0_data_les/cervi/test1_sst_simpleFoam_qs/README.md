# Cervi test1 OpenFOAM SST reference

Case:
- Geometry: Cervi test1/run_0001 tank-like 2D benchmark geometry.
- Solver: incompressible OpenFOAM simpleFoam.
- Turbulence model: k-omega SST.
- Mesh: structured graded blockMesh, 46,400 hexahedral cells.
- Final accepted field: 20000.

Purpose:
- Practical quasi-steady velocity reference for comparison with MOOSE mixing-length / GP-corrected closures.
- This case is intended for velocity-field comparison, consistent with TAMU/Migadome-style comparisons.

Convergence note:
- Velocity change from 18000 to 20000:
  - Ux relative_L2_change   = 7.471681e-03
  - Uy relative_L2_change   = 6.182399e-03
  - Uvec relative_L2_change = 6.808849e-03
  - Umag relative_L2_change = 6.539445e-03
- The velocity field is accepted as a practical quasi-steady reference.
- Turbulence variables were still changing more noticeably, so this should not be described as a fully converged turbulence-field reference.

Final yPlus:
- min     = 0.2053644738
- max     = 206.7484058
- average = 45.46563788

Re-run/continue:
```bash
./run_simpleFoam_serial.sh

The case is archived primarily as a frozen velocity-reference field at iteration 20000.
