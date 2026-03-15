#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --output=gpu_test_%j.out
#SBATCH --partition=gpu          # Check your cluster's partition name
#SBATCH --gres=gpu:1             # Requests 1 GPU
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00

# 1. Load the Miniforge and CUDA modules
module load devel/miniforge/24.9.2
module load devel/cuda/12.6

# 2. Initialize Conda for this shell session
source $(conda info --base)/etc/profile.d/conda.sh

# 3. Activate your environment
conda activate RMDN-TE

# Ensure your local project is accessible
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 4. Run the diagnostic Python script
python check_env.py