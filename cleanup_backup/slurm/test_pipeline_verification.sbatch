#!/bin/bash
#SBATCH --job-name=test_pipeline
#SBATCH --partition=mit_normal_gpu  # For validation jobs  
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=50G
#SBATCH --time=01:00:00
#SBATCH --output=logs/test_pipeline_%j.out
#SBATCH --error=logs/test_pipeline_%j.err

# Test job for verifying the refactored pipeline works

echo "============================================="
echo "PIPELINE VERIFICATION TEST"
echo "============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"

# Create necessary directories
mkdir -p logs
mkdir -p figures/job_${SLURM_JOB_ID}

# Load modules and environment
module purge
module load cuda/12.4.0
source ~/.bashrc
conda activate unsloth_env

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Test 1: Generate small dataset
echo "TEST 1: Generating test dataset..."
python scripts/generate_data.py \
    --systems double_integrator \
    --total-samples 20 \
    --split-ratio 0.8 \
    --dataset-name pipeline_test

if [ $? -eq 0 ]; then
    echo "✅ Dataset generation: SUCCESS"
else
    echo "❌ Dataset generation: FAILED"
    exit 1
fi

# Test 2: Create verification plots
echo "TEST 2: Creating verification plots..."
python -c "
from evaluation.simple_plotting import visualize_lqr_solution
visualize_lqr_solution(0.5, -0.3, 0.1, 50, 'figures/job_${SLURM_JOB_ID}/verification_plot.png')
print('✅ Verification plot created')
"

if [ $? -eq 0 ]; then
    echo "✅ Plot generation: SUCCESS"
else
    echo "❌ Plot generation: FAILED"
fi

# Test 3: Quick functionality test
echo "TEST 3: Running core functionality test..."
python verify_slurm_ready.py --test-only

echo ""
echo "============================================="
echo "VERIFICATION COMPLETE"
echo "============================================="
echo "Completed: $(date)"
echo "Results saved to: figures/job_${SLURM_JOB_ID}/"
