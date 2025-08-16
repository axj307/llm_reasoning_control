#!/bin/bash
#SBATCH --job-name=quick_test_pipeline   # Job name
#SBATCH --partition=mit_normal_gpu       # For jobs < 4 hours  
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=8                # Number of CPU cores
#SBATCH --gres=gpu:1                     # Request 1 GPU
#SBATCH --mem=50G                        # Memory
#SBATCH --time=03:00:00                  # Time limit (3 hours)
#SBATCH --output=logs/quick_test_%j.out  # Output path
#SBATCH --error=logs/quick_test_%j.err   # Error path

# Quick testing pipeline for evaluation/verification jobs < 4 hours
# Use this template for testing, evaluation, or verification tasks

CONTROL_SYSTEM=${CONTROL_SYSTEM:-"double_integrator"}

echo "============================================="
echo "QUICK TEST PIPELINE (< 4 hours)"
echo "============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Control System: $CONTROL_SYSTEM"
echo "Partition: mit_normal_gpu"
echo "============================================="

# Setup
mkdir -p logs
mkdir -p figures/job_${SLURM_JOB_ID}

# Load modules and environment
module purge
module load cuda/12.4.0
source ~/.bashrc
conda activate unsloth_env
export CUDA_VISIBLE_DEVICES=0
cd $SLURM_SUBMIT_DIR

# Quick test phases
echo "PHASE 1: Testing Data Generation..."
python scripts/generate_data.py --systems $CONTROL_SYSTEM --total-samples 20 --split-ratio 0.8 --dataset-name quick_test_${SLURM_JOB_ID}

echo "PHASE 2: Testing Core Functionality..."
python verify_slurm_ready.py --test-only

echo "PHASE 3: Creating Test Plots..."
python -c "
from evaluation.simple_plotting import visualize_lqr_solution
visualize_lqr_solution(0.5, -0.3, 0.1, 30, 'figures/job_${SLURM_JOB_ID}/quick_test.png')
print('✅ Quick test plot created')
"

echo "============================================="
echo "QUICK TEST COMPLETED: $(date)"
echo "Results: figures/job_${SLURM_JOB_ID}/"
echo "============================================="