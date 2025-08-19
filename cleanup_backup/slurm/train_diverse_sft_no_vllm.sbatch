#!/bin/bash
#SBATCH --job-name=diverse_sft_no_vllm
#SBATCH --output=slurm_logs/diverse_sft_no_vllm_%j.out
#SBATCH --error=slurm_logs/diverse_sft_no_vllm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --partition=mit_normal_gpu

echo "🚀 Diverse SFT Training (No vLLM) - Fixing Control Diversity"
echo "=========================================================="

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

# Check dataset
echo "📊 Dataset Information:"
python -c "import pickle; data=pickle.load(open('datasets/di_diverse_train.pkl', 'rb')); print(f'Training samples: {len(data)}')"
python -c "import pickle; data=pickle.load(open('datasets/di_diverse_eval.pkl', 'rb')); print(f'Evaluation samples: {len(data)}')"
echo "=========================================="

# Run the diverse SFT training (no vLLM)
python scripts/train_diverse_sft_no_vllm.py

echo "🎉 Diverse SFT training (no vLLM) completed!"
echo "Next step: Evaluate control diversity improvement"