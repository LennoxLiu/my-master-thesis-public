#!/bin/bash
#SBATCH --job-name=TE_reduced
#SBATCH --output=hpc/logs/TE_reduced_%a.out
#SBATCH --partition=gpu-single
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:A40:1
#SBATCH --array=0-1
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
python hpc/hyper_opt_reduced.py \
    --data_file_path "./data/event_times_data.h5" \
    --num_trials 10 \
    --history_length 128 \
    --data_time_length 300 \
    --task_id $SLURM_ARRAY_TASK_ID \
    --seed 42