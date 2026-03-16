#!/bin/bash
#SBATCH --job-name=TE_Analysis_yy
#SBATCH --output=hpc/logs/TE_Analysis_yy_%a.out
#SBATCH --partition=gpu-single
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A40:1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=00:10:00

mkdir -p hpc/logs
 
# Load the Miniforge and CUDA modules
module load devel/miniforge/24.9.2
module load devel/cuda/12.6

# Initialize Conda for this shell session
source $(conda info --base)/etc/profile.d/conda.sh

# Activate your environment
conda activate RMDN-TE

# Ensure your local project is accessible
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run the worker script for this specific array index
python worker.py $SLURM_ARRAY_TASK_ID