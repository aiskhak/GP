#!/bin/bash
source ~/miniforge/bin/activate moose
stdbuf -oL -eL mpirun -np 2 /homes/aiskhak/projects/fv_app/fv_app-opt -i cervi_ml_dx025_delta0p1.i > log.csv 2>&1
echo "Finished Executing!!!" >> log.csv