#!/bin/bash
#source ~/miniforge/bin/activate /home/aiskhak/miniforge/envs/moose
source ~/miniforge/bin/activate moose
#mpirun -np 4 /home/aiskhak/projects/fv_app/fv_app-opt -i tamu_2d_fv_gp.i > log.csv
stdbuf -oL -eL mpirun -np 4 /homes/aiskhak/projects/fv_app/fv_app-opt -i tamu_2d_fv_gp.i > log.csv 2>&1
echo "Finished Executing!!!" >> log.csv