#!/bin/bash

#SBATCH --partition debugq
#SBATCH --job-name=ejm-numba
#SBATCH --nodes=1
#SBATCH --output=ejm-numba.out
#SBATCH --time=0:10:00

set -euxo pipefail

pwd
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
date

module load python
env OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE python inflation/applications/EJM_numba_test.py

date
