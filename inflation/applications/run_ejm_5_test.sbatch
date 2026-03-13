#!/bin/bash
#SBATCH --job-name=ejm-5
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
module load python/3.12
echo "running in directory $(pwd)"
export PYTHONUNBUFFERED=1
python -u EJM_5_test.py
