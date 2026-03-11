#!/bin/bash
#SBATCH --job-name=ejm-numba
#SBATCH --partition=debugq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# If your cluster uses modules/conda, load/activate here, e.g.:
# module load python
# source /path/to/venv/bin/activate
module load python/3.12
echo "running in directory $(pwd)"

export PYTHONUNBUFFERED=1
 python -u EJM_numba_test.py
