#!/bin/bash
#SBATCH --job-name=quick_di_test
#SBATCH --output=slurm_logs/quick_di_test_%j.out
#SBATCH --error=slurm_logs/quick_di_test_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=pi_linaresr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

echo "🚀 QUICK DOUBLE INTEGRATOR TEST (10 GRPO STEPS)"
echo "============================================================"

# Activate conda environment
source ~/.bashrc
conda activate unsloth_env

echo "📊 PHASE 1: QUICK DATA GENERATION (100 samples)"
python scripts/generate_data.py \
    --systems double_integrator \
    --total-samples 100 \
    --split-ratio 0.9 \
    --dataset-name di_quick_test \
    --output-dir datasets/

echo "🚀 PHASE 2: QUICK SFT + GRPO TRAINING (10 steps)"
python scripts/train_grpo_quick_test.py

echo "✅ QUICK TEST COMPLETED!"
echo "📁 Check models/quick_test/ for results"