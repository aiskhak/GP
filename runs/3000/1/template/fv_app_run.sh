#!/bin/bash
source ~/miniforge/bin/activate moose
stdbuf -oL -eL srun --exclusive --exact --mpi=pmi2 -n 4 -c 1 /homes/aiskhak/projects/fv_app/fv_app-opt -i tamu_2d_fv_gp.i > log.csv 2>&1
echo "Finished Executing!!!" >> log.csv