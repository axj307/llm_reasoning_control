#!/bin/bash
#SBATCH --job-name=extend_sft_exact_notebook
#SBATCH --output=slurm_logs/extend_sft_exact_%j.out
#SBATCH --error=slurm_logs/extend_sft_exact_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --partition=mit_normal_gpu

echo "🚀 Extended SFT Training - EXACT Notebook Approach"
echo "=================================================="

# Load modules and activate environment
module purge
module load cuda/12.4
source ~/.bashrc
conda activate unsloth_env

echo "Environment Information:"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "=========================================="

# Run the extended training script that follows exact notebook
python scripts/extend_sft_exact_notebook.py

echo "🎉 Extended SFT training completed (exact notebook approach)!"