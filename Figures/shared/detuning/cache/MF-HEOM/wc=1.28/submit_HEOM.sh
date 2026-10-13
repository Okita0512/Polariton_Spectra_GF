#!/bin/bash
#SBATCH --job-name=MF_HEOM
#SBATCH --partition=genoa-std-mem
#SBATCH --account=xtao1995-chirality-molecules-soli
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=120:00:00
#SBATCH --output=%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

source /etc/profile.d/Z98-lmod.sh
module load gcc/15.2.0
export LD_LIBRARY_PATH="/vast/home/w/wying3/DEOM/MF-DEOM/lib:/vast/home/w/wying3/envs/deom/lib:${LD_LIBRARY_PATH:-}"

date > timeneeded.${SLURM_JOB_ID}
../bin/rhot ./input.json
date >> timeneeded.${SLURM_JOB_ID}
