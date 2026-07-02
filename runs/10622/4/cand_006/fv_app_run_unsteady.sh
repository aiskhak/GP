#!/bin/bash
source ~/miniforge/bin/activate moose
stdbuf -oL -eL /homes/aiskhak/miniforge/envs/moose/bin/mpirun -np 12 /homes/aiskhak/projects/fv_app/fv_app-opt -i tamu_2d_fv_gp_unsteady.i > log.csv 2>&1
echo "Finished Executing!!!" >> log.csv
