#!/bin/bash
#SBATCH -p debug
#SBATCH -o output.log
#SBATCH --mem=2GB
#SBATCH --time=0-01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --job-name=HEOM

time ../bin/rhot ./input.json

