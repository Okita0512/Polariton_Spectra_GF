#!/bin/bash
#SBATCH --job-name=h_m25
#SBATCH --partition=genoa-std-mem
#SBATCH --account=xtao1995-chirality-molecules-soli
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=120:00:00
#SBATCH --output=%x_%j.out
#SBATCH --open-mode=truncate

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    CASE_DIR="$SLURM_SUBMIT_DIR"
else
    CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$CASE_DIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

HEOM_EXE="${HEOM_EXE:-$CASE_DIR/../../bin/1d-resp}"
if [[ "$HEOM_EXE" == */* ]]; then
    if [[ ! -x "$HEOM_EXE" ]]; then
        echo "HEOM executable is not available at: $HEOM_EXE" >&2
        echo "Set HEOM_EXE to the cluster 1d-resp executable." >&2
        exit 127
    fi
else
    HEOM_EXE="$(command -v "$HEOM_EXE")"
fi

date > "timeneeded.${SLURM_JOB_ID}"
time "$HEOM_EXE" ./input.json
date >> "timeneeded.${SLURM_JOB_ID}"
