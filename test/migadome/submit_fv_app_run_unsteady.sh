#!/bin/bash
#SBATCH --partition=ksu-mne-train.q
#SBATCH --job-name=fv_app
#SBATCH --output=out.%j
#SBATCH --error=err.%j
#SBATCH --nodes=1
#SBATCH --ntasks=32
##SBATCH --nodelist=warlock34
#SBATCH --time=0-20:00:00
#SBATCH --mem=400G

module purge

source ~/miniforge/bin/activate moose

export OMP_NUM_THREADS=1

stdbuf -oL -eL mpirun -np ${SLURM_NTASKS} \
  /homes/aiskhak/projects/fv_app/fv_app-opt \
  -i migadome.i > log.csv 2>&1

echo "Finished Executing!!!" >> log.csv