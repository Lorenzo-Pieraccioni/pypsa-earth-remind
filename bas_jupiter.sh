#!/bin/bash
#SBATCH  --cpus-per-task 16
#SBATCH  --time 1:00:00
#SBATCH  --qos=short
#SBATCH  -o jupyter-%j.log

PORT=8035
echo "SSH-TUNNEL COMMAND:"
echo "ssh -v -N $USER@$(hostname) -J $USER@hpc.pik-potsdam.de -L $PORT:127.0.0.1:$PORT"
jupyter notebook --no-browser --port $PORT