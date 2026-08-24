#!/bin/bash
#SBATCH --cpus-per-task 8
#SBATCH --mem 16G
#SBATCH --time 0:30:00
#SBATCH --qos=short
#SBATCH -o province_stats-%j.log

python3 analysis/network/network_scripts/province_stats_and_map.py
