#!/bin/bash
#SBATCH --job-name=train_diverse_sft
#SBATCH --output=slurm_logs/diverse_sft_%j.out
#SBATCH --error=slurm_logs/diverse_sft_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --partition=mit_normal_gpu

echo "🚀 Diverse SFT Training - Fixing Control Diversity Issue"
echo "========================================================"

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

# Create slurm logs directory if it doesn't exist
mkdir -p slurm_logs

echo "📊 Dataset Information:"
echo "Training samples: $(python -c 'import pickle; print(len(pickle.load(open(\"datasets/di_diverse_train.pkl\", \"rb\"))))')"
echo "Evaluation samples: $(python -c 'import pickle; print(len(pickle.load(open(\"datasets/di_diverse_eval.pkl\", \"rb\"))))')"
echo "=========================================="

# Run the diverse SFT training
python scripts/train_diverse_sft.py

echo "🎉 Diverse SFT training completed!"
echo "Next step: Evaluate model performance and control diversity"