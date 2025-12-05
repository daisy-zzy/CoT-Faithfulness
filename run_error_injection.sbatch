#!/bin/bash
#SBATCH --job-name=error_inject
#SBATCH --output=logs/error_injection.out
#SBATCH --error=logs/error_injection.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=200G
#SBATCH --time=4:00:00
#SBATCH --partition=general

# Activate conda
source ~/.bashrc
conda activate base

# Set environment variables
export MKL_THREADING_LAYER=GNU
export RAY_USE_UVLOOP=0
export RAY_TMPDIR=/data/user_data/$USER/ray_tmp

cd /home/riyaza/10701-project

echo "=== Step 1: Error Injection with Qwen ==="
python run_error_injection_step1.py

echo ""
echo "=== Step 2: Model B Inference with Llama ==="
python run_error_injection_step2.py

echo ""
echo "=== Done ==="
