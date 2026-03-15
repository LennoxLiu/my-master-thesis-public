#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --output=hpc/logs/gpu_test_%j.out
#SBATCH --partition=gpu-single
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1 #--gres=gpu:A40:1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00

mkdir -p hpc/logs
 
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
python hpc/check_env.py

python src/entropy_tpp.py