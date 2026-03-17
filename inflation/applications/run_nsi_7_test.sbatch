#!/bin/bash
#SBATCH --job-name=nsi-7
#SBATCH --partition=amdq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=250G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
module load python/3.12
echo "running in directory $(pwd)"
export PYTHONUNBUFFERED=1
python -u NSI_7_test.py
