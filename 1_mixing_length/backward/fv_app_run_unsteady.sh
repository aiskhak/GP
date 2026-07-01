#!/bin/bash
source ~/miniforge/bin/activate moose
stdbuf -oL -eL mpirun -np 32 /homes/aiskhak/projects/fv_app/fv_app-opt -i backward.i > log.csv 2>&1
echo "Finished Executing!!!" >> log.csv