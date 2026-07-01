# MiGaDome Nek-to-MOOSE velocity projection

This folder contains the lightweight files needed to reproduce the projection of
Nek mean velocity data onto the MOOSE coarse-grid cell centers.

## Tracked files

- `SIZE` — Nek5000 size configuration used for the projection.
- `michigan.usr` — user file that reads the Nek average field and projects it onto MOOSE cell centers.
- `recycle.usr` — helper user file required by `michigan.usr`.
- `michigan.par` — Nek5000-compatible parameter file.
- `moose_cell_centers.dat` — MOOSE target cell centers for interpolation.
- `run_project_warlock_512_mem.sh` — Slurm script used on Beocat warlock nodes.

## Not tracked

The following files are generated or too large for regular Git tracking:

- `michigan.re2`
- `michigan.ma2`
- `avgmichigan0.f00001`
- `nek5000`
- `obj/`
- `build.log`
- `proj*.out`
- `proj*.err`

The binary field file `avgmichigan0.f00001` must be provided externally.

## Required external files

Place these files in this folder before running the projection:

```text
avgmichigan0.f00001
michigan.re2
michigan.ma2
Regenerating michigan.ma2

If michigan.re2 is available but michigan.ma2 is missing, regenerate the map file with Nek5000 genmap:

cd /homes/aiskhak/projects/GP/0_data_les/migadome/interpolation/nek

export NEK5000_HOME=/homes/aiskhak/Nek5000
export PATH=$NEK5000_HOME/bin:$PATH

genmap

When prompted, enter:

michigan
1e-4

This creates michigan.ma2.

Building the projection executable
cd /homes/aiskhak/projects/GP/0_data_les/migadome/interpolation/nek

export NEK5000_HOME=/homes/aiskhak/Nek5000
export PATH=$NEK5000_HOME/bin:$PATH

makenek clean
rm -rf obj
makenek michigan

The case uses:

lx1 = 7
lxd = 10
lx2 = lx1 - 2
lelt = 950
ldimt = 1
lpmin = 512
Running the projection
sbatch run_project_warlock_512_mem.sh

Expected output:

Wrote nek_avg_on_moose.csv
16981 nek_avg_on_moose.csv

The output file has columns:

id,x,y,z,volume,yw,U,V,W
Copying final projected velocity file

After a successful run:

cp nek_avg_on_moose.csv /homes/aiskhak/projects/GP/0_data_les/migadome/nek_avg_on_moose.csv