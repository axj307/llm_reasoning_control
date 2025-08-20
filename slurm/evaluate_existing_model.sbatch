#!/bin/bash
#SBATCH --job-name=eval_model            # Job name
#SBATCH --partition=pi_linaresr          # pi_linaresr, mit_normal_gpu
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=10               # Number of CPU cores
#SBATCH --gres=gpu:h100:1                # Request 1 H100 GPU
#SBATCH --mem=50G                        # Memory (reduced since no training)
#SBATCH --time=02:00:00                  # Time limit (2 hours)
#SBATCH --output=logs/eval_model_%j.out  # Output path
#SBATCH --error=logs/eval_model_%j.err   # Error path

# Configuration - customize these
MODEL_PATH=${MODEL_PATH:-"models/single_system/double_integrator/sft/latest"}
MODEL_TYPE=${MODEL_TYPE:-"single_system"}
EVAL_DATASET=${EVAL_DATASET:-"di"}
NUM_TEST_CASES=${NUM_TEST_CASES:-3}
TEST_TYPE=${TEST_TYPE:-"both"}  # standard, mpc, or both

# Print job information
echo "============================================="
echo "SLURM MODEL EVALUATION JOB"
echo "============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Model Path: $MODEL_PATH"
echo "Model Type: $MODEL_TYPE"
echo "Evaluation Dataset: $EVAL_DATASET"
echo "Number of Test Cases: $NUM_TEST_CASES"
echo "Test Type: $TEST_TYPE"
echo "============================================="

# Create necessary directories
mkdir -p logs
mkdir -p figures/job_${SLURM_JOB_ID}

# Load necessary modules
module purge
module load cuda/12.4.0

# Activate conda environment
source ~/.bashrc
conda activate unsloth_env

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Run evaluation
echo ""
echo "Starting Model Evaluation..."
echo "=================================="

python scripts/evaluate_model.py \
    --model-path "$MODEL_PATH" \
    --model-type "$MODEL_TYPE" \
    --eval-dataset "$EVAL_DATASET" \
    --num-test-cases "$NUM_TEST_CASES" \
    --test-type "$TEST_TYPE" \
    --save-plots \
    --plot-dir "figures/job_${SLURM_JOB_ID}" \
    --gpu-id 0

# Check if evaluation was successful
if [ $? -eq 0 ]; then
    echo "✅ Model evaluation completed successfully!"
    
    # Create evaluation summary
    echo "Evaluation Results - Job $SLURM_JOB_ID" > figures/job_${SLURM_JOB_ID}/evaluation_summary.txt
    echo "Model: $MODEL_PATH" >> figures/job_${SLURM_JOB_ID}/evaluation_summary.txt
    echo "Dataset: $EVAL_DATASET" >> figures/job_${SLURM_JOB_ID}/evaluation_summary.txt
    echo "Test Cases: $NUM_TEST_CASES" >> figures/job_${SLURM_JOB_ID}/evaluation_summary.txt
    echo "Completed: $(date)" >> figures/job_${SLURM_JOB_ID}/evaluation_summary.txt
    
    echo ""
    echo "============================================="
    echo "JOB COMPLETION SUMMARY"
    echo "============================================="
    echo "✅ Model Evaluation: SUCCESS"
    echo "📊 Results saved to: figures/job_${SLURM_JOB_ID}/"
    echo "Completed: $(date)"
    echo "============================================="
else
    echo "❌ Model evaluation failed!"
    exit 1
fi

echo "🎉 Evaluation job finished successfully!"