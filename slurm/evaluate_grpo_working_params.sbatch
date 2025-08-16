#!/bin/bash
#SBATCH --job-name=eval_grpo_working
#SBATCH --output=slurm_logs/eval_grpo_working_%j.out
#SBATCH --error=slurm_logs/eval_grpo_working_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

echo "🔬 EVALUATING GRPO MODEL WITH WORKING PARAMETERS"
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

# Evaluate GRPO model with working parameters
echo "📊 Evaluating GRPO model performance..."
python scripts/evaluate_model.py \
    --model-path models/working_notebook/grpo_working_params_model \
    --model-type single_system \
    --systems double_integrator \
    --eval-dataset di_test_eval \
    --num-test-cases 5 \
    --save-plots \
    --plot-dir figures/grpo_working_evaluation

echo "✅ GRPO model evaluation completed!"
echo "📈 Check results in figures/grpo_working_evaluation/"