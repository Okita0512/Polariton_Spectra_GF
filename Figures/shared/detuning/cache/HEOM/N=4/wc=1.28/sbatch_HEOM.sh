#!/bin/bash
#SBATCH --job-name=HEOM
#SBATCH --partition=genoa-std-mem
#SBATCH --account=xtao1995-chirality-molecules-soli
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=120:00:00
#SBATCH --output=%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

date > timeneeded.${SLURM_JOB_ID}
../../../bin/1d-resp ./input.json
date >> timeneeded.${SLURM_JOB_ID}
