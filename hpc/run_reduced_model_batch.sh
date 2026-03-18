#!/bin/bash
#SBATCH --job-name=TE_reduced_runs
#SBATCH --output=hpc/logs/TE_reduced_runs_arrayjob%a.out
#SBATCH --partition=gpu-single
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --array=0
#SBATCH --mem-per-cpu=3gb
#SBATCH --time=02:00:00
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
TASKS_PER_JOB=32
#--cpus-per-task should be half of TASKS_PER_JOB

# Launch tasks in the background
for (( i=0; i<$TASKS_PER_JOB; i++ )); do
    CURRENT_TASK_ID=$((SLURM_ARRAY_TASK_ID * TASKS_PER_JOB + i))
    
    python hpc/multi_runs_reduced.py \
        --data_file_path "./data/event_times_data.h5" \
        --num_runs 50 \
        --history_length 128 \
        --data_time_length 900 \
        --task_id $CURRENT_TASK_ID \
        --seed 42 &
done

wait # Wait for all tasks to finish

# Log GPU state right before shutdown to verify utilization
nvidia-smi >> hpc/logs/gpu_usage_runs_reduced_${SLURM_ARRAY_TASK_ID}.log

nvidia-cuda-mps-control -q # Shut down daemon
