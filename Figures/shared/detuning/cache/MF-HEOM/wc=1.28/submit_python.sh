#!/bin/bash
#SBATCH --job-name=mf_gen_input
#SBATCH --partition=genoa-std-mem
#SBATCH --account=xtao1995-chirality-molecules-soli
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

source /etc/profile.d/Z98-lmod.sh
module load miniconda3/25.5.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /vast/home/w/wying3/envs/deom

date > timeneeded_genInput.${SLURM_JOB_ID}
python gen_input.py
date >> timeneeded_genInput.${SLURM_JOB_ID}
