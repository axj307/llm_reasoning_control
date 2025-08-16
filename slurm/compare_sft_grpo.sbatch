#!/bin/bash
#SBATCH --job-name=compare_sft_grpo
#SBATCH --output=slurm_logs/compare_sft_grpo_%j.out
#SBATCH --error=slurm_logs/compare_sft_grpo_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

echo "🔬 SFT vs GRPO MODEL COMPARISON"
echo "============================================================"

# Environment setup
echo "Environment Information:"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "=========================================="

# Activate conda environment
source ~/.bashrc
conda activate unsloth_env

# Run comparison script
echo "📊 Running SFT vs GRPO comparison..."
python scripts/evaluate_grpo_comparison.py

echo "✅ Model comparison completed!"