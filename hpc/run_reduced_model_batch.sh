#!/bin/bash
#SBATCH --job-name=TE_reduced
#SBATCH --output=hpc/logs/TE_reduced_%a.out
#SBATCH --partition=gpu-single
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0
#SBATCH --mem-per-cpu=2gb
#SBATCH --time=00:30:00
#SBATCH --account=bw20g013

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

# Start the MPS daemon
nvidia-cuda-mps-control -d

# Define the number of concurrent tasks per GPU
TASKS_PER_JOB=8

# Launch tasks in the background
for (( i=0; i<$TASKS_PER_JOB; i++ )); do
    CURRENT_TASK_ID=$((SLURM_ARRAY_TASK_ID * TASKS_PER_JOB + i))
    
    python hpc/hyper_opt_reduced.py \
        --data_file_path "./data/event_times_data.h5" \
        --num_trials 5 \
        --history_length 128 \
        --batch_size 512 \
        --data_time_length 900 \
        --task_id $CURRENT_TASK_ID \
        --seed 42 &
done

wait # Wait for all 4 to finish

# Log GPU state right before shutdown to verify utilization
nvidia-smi >> hpc/logs/gpu_usage_reduced_${SLURM_ARRAY_TASK_ID}.log

nvidia-cuda-mps-control -q # Shut down daemon


# Multiple runs with different seeds