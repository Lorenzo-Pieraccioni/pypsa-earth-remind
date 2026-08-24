#!/bin/bash
#SBATCH --cpus-per-task 16
#SBATCH --mem 32G
#SBATCH --time 0:30:00
#SBATCH --qos=short
#SBATCH -o diagnose_nan-%j.log

python3 analysis/network/network_scripts/diagnose_nan_stripes.py
