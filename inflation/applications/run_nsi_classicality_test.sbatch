#!/bin/bash
#SBATCH --job-name=nsi-classicality
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# If your cluster uses modules/conda, load/activate here, e.g.:
# module load python
# source /path/to/venv/bin/activate

#cd "$(dirname "$0")"
module load python/3.12
echo "running in directory $(pwd)"
export PYTHONUNBUFFERED=1
python -u NSI_classicality_test.py
