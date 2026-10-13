#!/bin/bash
#SBATCH -p action 
#SBATCH -o output.log
#SBATCH --mem=16GB
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --job-name=HEOM
#SBATCH --open-mode=append

time ../../bin/1d-resp ./input.json

