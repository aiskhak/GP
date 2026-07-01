#!/bin/bash
#SBATCH --job-name=migadome_proj
#SBATCH --partition=ksu-mne-train.q
#SBATCH --nodes=4
#SBATCH --ntasks=512
#SBATCH --ntasks-per-node=128
#SBATCH --nodelist=warlock34,warlock35,warlock36,warlock37
#SBATCH --exclusive
#SBATCH --mem=700G
#SBATCH --time=00:30:00
#SBATCH --output=proj512mem_%j.out
#SBATCH --error=proj512mem_%j.err

export OMP_NUM_THREADS=1

cd /homes/aiskhak/projects/GP/1_mixing_length/migadome/nek

export NEK5000_HOME=/homes/aiskhak/Nek5000
export PATH=$NEK5000_HOME/bin:$PATH
hash -r

echo "Job info:"
scontrol show job $SLURM_JOB_ID | egrep "JobId|NodeList|NumNodes|NumCPUs|NumTasks|TRES|mem|MinMemory|OverSubscribe|Exclusive"

echo "Nodes:"
scontrol show hostnames $SLURM_JOB_NODELIST

echo "Node memory:"
for n in $(scontrol show hostnames $SLURM_JOB_NODELIST); do
  echo "===== $n ====="
  scontrol show node $n | egrep "RealMemory|AllocMem|FreeMem|CPUAlloc|CPUTot|State"
done

echo "Executable size:"
size nek5000

echo "Start: $(date)"

nekmpi michigan 512

echo "End: $(date)"

echo "Output check:"
ls -lh nek_avg_on_moose.csv || true
wc -l nek_avg_on_moose.csv || true
head nek_avg_on_moose.csv || true
