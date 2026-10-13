#!/bin/bash
#SBATCH -p debug 
#SBATCH -o output.log
#SBATCH --mem=4GB
#SBATCH --time=0-01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --job-name=HEOM
#SBATCH --open-mode=append

time ../../bin/1d-resp ./input.json

