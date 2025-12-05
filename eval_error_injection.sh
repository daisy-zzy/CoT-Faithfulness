#!/bin/bash
#SBATCH --job-name=error-injection
#SBATCH --output=logs/error_injection.out
#SBATCH --error=logs/error_injection.err
#SBATCH --time=4:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:8
#SBATCH --mem=200G
#SBATCH --cpus-per-task=12

# Error injection in two steps:
# Step 1: Ray + Qwen to inject errors into CoTs
# Step 2: Ray + Llama to run inference on modified CoTs

export HF_HOME=/data/user_data/riyaza/HF
mkdir -p logs

# Read HF token from cache if available
if [ -f "$HF_HOME/token" ]; then
    export HF_TOKEN=$(cat "$HF_HOME/token")
    export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
fi

# Fix MKL threading
export MKL_THREADING_LAYER=GNU
export MKL_SERVICE_FORCE_INTEL=0

# Disable uvloop for Ray compatibility  
export RAY_USE_UVLOOP=0

# Activate environment
source ~/.bashrc
conda activate 10701-env-hw4

cd /home/riyaza/10701-project

echo "=========================================="
echo "STEP 1: Injecting errors with Qwen"
echo "=========================================="
python run_error_injection_step1.py

echo ""
echo "=========================================="
echo "STEP 2: Running Model B on modified CoTs"
echo "=========================================="
python run_error_injection_step2.py

echo ""
echo "Done!"
