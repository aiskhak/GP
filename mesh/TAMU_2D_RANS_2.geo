R_inlet         =    19.05/2;
gap             =    R_inlet*0.17;
R_dome          =    408.18/2;
H_dome          =    196.85;
R_dome_p        =    Sqrt(R_dome*R_dome+((R_dome*R_dome-H_dome*H_dome)/(2*H_dome))*((R_dome*R_dome-H_dome*H_dome)/(2*H_dome)));
R_outlet        =    R_dome-2*R_inlet;
R_dome_b        =    (R_outlet-R_inlet)/2; TETA = Asin((R_dome_p-H_dome)/(R_dome_p-2*R_inlet));

//Blocking the axisymmetric model:
pref=newp; Printf("pref =%g", pref);
Point(1) = {0, -30*R_inlet, 0, 1.0};
Point(3) = {R_inlet, -30*R_inlet, 0, 1.0};
Point(4) = {R_outlet, -4*R_inlet, 0, 1.0};
Point(5) = {R_dome, -4*R_inlet, 0, 1.0};
Point(6) = {0, 0, 0, 1.0};
Point(8) = {R_inlet, 0, 0, 1.0};
Point(10) = {2*R_inlet, 0, 0, 1.0};
Point(11) = {R_dome_b, 0, 0, 1.0};
Point(12) = {(R_dome_p-2*R_inlet)*Sin(Pi/2-TETA), 0, 0, 1.0};
Point(13) = {R_dome, 0, 0, 1.0};
Point(14) = {0, R_dome_b, 0, 1.0};
Point(16) = {R_inlet, R_dome_b, 0, 1.0};
Point(18) = {2*R_inlet, R_dome_b, 0, 1.0};
Point(19) = {R_dome_b, R_dome_b-(R_dome_p-H_dome), 0, 1.0};
Point(20) = {(R_dome_p-2*R_inlet)*Cos(Pi/4), (R_dome_p-2*R_inlet)*Sin(Pi/4)-(R_dome_p-H_dome), 0, 1.0};
Point(21) = {2*R_inlet, Sqrt((R_dome_p-2*R_inlet)*(R_dome_p-2*R_inlet)-(2*R_inlet)*(2*R_inlet))-(R_dome_p-H_dome), 0, 1.0};
Point(23) = {R_inlet, Sqrt((R_dome_p-2*R_inlet)*(R_dome_p-2*R_inlet)-(R_inlet)*(R_inlet))-(R_dome_p-H_dome), 0, 1.0};
Point(25) = {0, R_dome_p-2*R_inlet-(R_dome_p-H_dome), 0, 1.0};
Point(26) = {R_dome_p*Cos(Pi/4), R_dome_p*Sin(Pi/4)-(R_dome_p-H_dome), 0, 1.0};
Point(27) = {2*R_inlet, Sqrt(R_dome_p*R_dome_p-(2*R_inlet)*(2*R_inlet))-(R_dome_p-H_dome), 0, 1.0};
Point(29) = {R_inlet, Sqrt(R_dome_p*R_dome_p-(R_inlet)*(R_inlet))-(R_dome_p-H_dome), 0, 1.0};
Point(31) = {0, H_dome, 0, 1.0};
Point(32) = {0, -(R_dome_p-H_dome), 0, 1.0};
Rotate {{0, 0, 1}, {0, 0, 0}, Pi/2} {
  Point{1}; Point{3}; Point{4}; Point{5}; Point{6}; Point{8}; Point{10}; Point{11}; Point{12}; Point{13}; Point{14}; Point{16}; Point{18}; Point{19}; Point{20}; Point{21}; Point{23}; Point{25}; Point{26}; Point{27}; Point{29}; Point{31}; Point{32};
}
Line(1) = {31, 25};
Line(2) = {25, 14};
Line(3) = {14, 6};
Line(4) = {6, 1};
Line(5) = {1, 3};
Line(7) = {3, 8};
Line(9) = {8, 10};
Line(10) = {10, 11};
Line(11) = {11, 12};
Line(12) = {12, 4};
Line(13) = {4, 5};
Line(14) = {5, 13};
Circle(15) = {13, 32, 26};
Circle(16) = {26, 32, 27};
Circle(17) = {27, 32, 29};
Circle(20) = {29, 32, 31};
Line(22) = {23, 25};
Line(24) = {16, 14};
Line(26) = {8, 6};
Line(28) = {29, 23};
Line(30) = {23, 16};
Line(32) = {16, 8};
Line(39) = {27, 21};
Line(40) = {21, 23};
Line(41) = {21, 18};
Line(42) = {18, 16};
Line(43) = {18, 10};
Circle(44) = {12, 32, 20};
Circle(45) = {20, 32, 21};
Line(46) = {18, 19};
Line(47) = {19, 11};
Line(48) = {19, 20};
Line(49) = {20, 26};
Line(50) = {12, 13};
Curve Loop(1) = {1, -22, -28, 20};
Plane Surface(1) = {1};
Curve Loop(2) = {2, -24, -30, 22};
Plane Surface(2) = {2};
Curve Loop(3) = {3, -26, -32, 24};
Plane Surface(3) = {3};
Curve Loop(4) = {4, 5, 7, 26};
Plane Surface(4) = {4};
Curve Loop(12) = {28, -40, -39, 17};
Plane Surface(12) = {12};
Curve Loop(13) = {30, -42, -41, 40};
Plane Surface(13) = {13};
Curve Loop(14) = {32, 9, -43, 42};
Plane Surface(14) = {14};
Curve Loop(15) = {39, -45, 49, 16};
Plane Surface(15) = {15};
Curve Loop(16) = {41, 46, 48, 45};
Plane Surface(16) = {16};
Curve Loop(17) = {43, 10, -47, -46};
Plane Surface(17) = {17};
Curve Loop(18) = {47, 11, 44, -48};
Plane Surface(18) = {18};
Curve Loop(19) = {44, 49, -15, -50};
Plane Surface(19) = {19};
Curve Loop(20) = {12, 13, 14, -50};
Plane Surface(20) = {20};

//Inlet pipe:
Transfinite Curve {17,40,42,9,20,22,24,26,5} = 5 Using Progression 1.0;
Transfinite Curve {4,7} = 113 Using Progression 1.0;

//First Layer of boxes in the hemisphere
Transfinite Curve {3,25,38,43,47,44,15} = 41 Using Progression 1.0;

//Second Layer of boxes in the hemisphere
Transfinite Curve {2,23,32,30,36,41,48,11} = 41 Using Progression 1.0;
Transfinite Curve {16,45,46,10} = 29 Using Progression 1.0;

//Outlet
Transfinite Curve {1,21,28,34,39,49,50,13} = 11 Using Progression 1.0;
Transfinite Curve {12,14} = 17 Using Progression 1.0;

Transfinite Surface "*";
Recombine Surface "*";
Physical Curve("inlet", 1) = {5, 6};
//+
Physical Curve("outlet", 2) = {13};
//+
Physical Curve("wall", 3) = {7,8,9,10,11,12,14,15,16,17,18,19,20};
//+
Physical Curve("SYM", 4) = {1,2,3,4};
//+
Physical Surface("fluid", 5) = {1:20};
Mesh.ElementOrder = 2;
Mesh.MshFileVersion = 2.16;
Mesh 3;
Save "TAMU_2D_RANS_2.msh";