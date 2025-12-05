#!/bin/bash
#SBATCH --job-name=cot-faithfulness
#SBATCH --output=logs/main_run3.out
#SBATCH --error=logs/main_run3.err
#SBATCH --time=24:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:8
#SBATCH --mem=200G
#SBATCH --cpus-per-task=12

export HF_HOME=/data/user_data/riyaza/HF
mkdir -p logs

# Read HF token from cache if available
if [ -f "$HF_HOME/token" ]; then
    export HF_TOKEN=$(cat "$HF_HOME/token")
    export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
fi

# Fix uvloop compatibility with Ray - disable uvloop in Ray workers
export RAY_USE_UVLOOP=0

# Fix MKL threading incompatibility with libgomp
export MKL_THREADING_LAYER=GNU

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate 10701-env-hw4

python run_experiments.py --n-rollouts 4