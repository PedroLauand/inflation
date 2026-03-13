#!/bin/bash

#SBATCH --partition debugq
#SBATCH --job-name=ejm-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --output=ejm-5.out
#SBATCH --time=0:10:00

set -euxo pipefail

pwd
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
date

module load python
env OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK python inflation/applications/EJM_5_test.py

date
