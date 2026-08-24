#!/bin/bash
#SBATCH --cpus-per-task 16
#SBATCH --mem 32G
#SBATCH --time 1:00:00
#SBATCH --qos=short
#SBATCH -o cutout_cf_final-%j.log

python3 analysis/network/network_scripts/cutout_cf_final.py
