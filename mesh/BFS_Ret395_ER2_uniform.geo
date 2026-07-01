// ============================================================
// Structured 2D backward-facing step mesh
// Matching CTTC DNS BFS_Ret395_ER2 coordinate convention
//
// DNS data:
//   x = -6 to 32
//   y = -1 to 1
//   expansion ratio ER = 2
//   step corner at x = 0, y = 0
//
// Fluid domain:
//   upstream:          x = -6..0,  y = 0..1
//   downstream lower:  x =  0..32, y = -1..0
//   downstream upper:  x =  0..32, y = 0..1
//
// Boundaries:
//   inlet
//   outlet
//   top_wall
//   bottom_wall
//   step_wall
//
// Mesh rationale:
//   Target first-cell eta_y = yw_first / yw_max ≈ 0.02,
//   comparable to TAMU CG1 near-wall normalized distance.
//   With Ny = 25 per unit height:
//      dy = 1/25 = 0.04
//      yw_first = dy/2 = 0.02
//      yw_max ≈ 0.98
//      eta_y_first ≈ 0.0204
// ============================================================

SetFactory("OpenCASCADE");

// ----------------------
// Geometry
// ----------------------
x0 = -6.0;
x1 =  0.0;
x2 = 32.0;

y0 = -1.0;
y1 =  0.0;
y2 =  1.0;

// ----------------------
// Refined approximately uniform mesh controls
// dx ≈ dy ≈ 0.04
// ----------------------
Nx_up   = 150;   // length 6  / 150 = 0.040000
Nx_down = 800;   // length 32 / 800 = 0.040000
Ny_low  = 25;    // height 1 / 25  = 0.040000
Ny_up   = 25;    // height 1 / 25  = 0.040000

// ----------------------
// Points
// ----------------------
Point(1) = {x0, y1, 0, 1.0};  // inlet lower, y=0
Point(2) = {x1, y1, 0, 1.0};  // step corner, y=0
Point(3) = {x0, y2, 0, 1.0};  // inlet upper
Point(4) = {x1, y2, 0, 1.0};  // top at step plane

Point(5) = {x1, y0, 0, 1.0};  // step lower corner
Point(6) = {x2, y0, 0, 1.0};  // outlet bottom
Point(7) = {x2, y1, 0, 1.0};  // outlet centerline y=0
Point(8) = {x2, y2, 0, 1.0};  // outlet top

// ----------------------
// Lines
// ----------------------
// Upstream upper block: x=-6..0, y=0..1
Line(1) = {1, 2};   // upstream bottom wall / lower wall before step
Line(2) = {2, 4};   // internal vertical interface
Line(3) = {4, 3};   // upstream top wall
Line(4) = {3, 1};   // inlet

// Downstream lower block: x=0..32, y=-1..0
Line(5) = {5, 6};   // downstream bottom wall
Line(6) = {6, 7};   // outlet lower
Line(7) = {7, 2};   // internal horizontal interface y=0
Line(8) = {2, 5};   // vertical step wall

// Downstream upper block outer lines
Line(10) = {7, 8};  // outlet upper
Line(11) = {8, 4};  // downstream top wall

// ----------------------
// Surfaces
// ----------------------
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

Curve Loop(2) = {8, 5, 6, 7};
Plane Surface(2) = {2};

Curve Loop(3) = {-7, 10, 11, -2};
Plane Surface(3) = {3};

// ----------------------
// Uniform transfinite mesh
// ----------------------
Transfinite Curve {1, 3} = Nx_up + 1 Using Progression 1.0;
Transfinite Curve {5, 7, 11} = Nx_down + 1 Using Progression 1.0;

Transfinite Curve {8, 6} = Ny_low + 1 Using Progression 1.0;
Transfinite Curve {2, 4, 10} = Ny_up + 1 Using Progression 1.0;

Transfinite Surface {1};
Transfinite Surface {2};
Transfinite Surface {3};

Recombine Surface {1, 2, 3};

// ----------------------
// Physical groups
// ----------------------
Physical Surface("fluid") = {1, 2, 3};

Physical Curve("inlet") = {4};
Physical Curve("outlet") = {6, 10};

Physical Curve("top_wall") = {3, 11};
Physical Curve("bottom_wall") = {1, 5};
Physical Curve("step_wall") = {8};

Mesh.Algorithm = 8;
Mesh.RecombineAll = 1;
Mesh.ElementOrder = 1;