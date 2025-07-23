#!/bin/bash
#SBATCH --job-name=control_sft_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/sft_eval_%j.out
#SBATCH --error=logs/sft_eval_%j.err

# Job configuration - customize these parameters
ENVIRONMENT=${ENVIRONMENT:-"double_integrator"}
SAMPLES=${SAMPLES:-300}
LORA_RANK=${LORA_RANK:-16}
LEARNING_RATE=${LEARNING_RATE:-0.0002}
NUM_EPOCHS=${NUM_EPOCHS:-4}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_TEST_CASES=${NUM_TEST_CASES:-20}
RUN_NAME=${RUN_NAME:-"slurm_sft_${SLURM_JOB_ID}"}

# Print job information
echo "============================================="
echo "SLURM SFT TRAINING + EVALUATION JOB"
echo "============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Environment: $ENVIRONMENT"
echo "Samples: $SAMPLES"
echo "LoRA Rank: $LORA_RANK"
echo "Learning Rate: $LEARNING_RATE"
echo "Epochs: $NUM_EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Test Cases: $NUM_TEST_CASES"
echo "Run Name: $RUN_NAME"
echo "============================================="

# Create necessary directories
mkdir -p logs
mkdir -p plots/job_${SLURM_JOB_ID}
mkdir -p results/job_${SLURM_JOB_ID}

# Set up environment (adjust as needed for your cluster)
# source ~/.bashrc
# conda activate your_env_name
# module load cuda/11.8 python/3.9

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Change to project directory
cd $SLURM_SUBMIT_DIR

# ===================
# PHASE 1: SFT TRAINING
# ===================
echo ""
echo "PHASE 1: Starting SFT Training..."
echo "=================================="

python scripts/train_single_system.py \
    --system $ENVIRONMENT \
    --num-samples $SAMPLES \
    --training-type sft \
    --lora-rank $LORA_RANK \
    --output-base "./slurm_output_${SLURM_JOB_ID}" \
    --run-name $RUN_NAME \
    --gpu-id 0

# Check if SFT training was successful
if [ $? -eq 0 ]; then
    echo "✅ SFT training completed successfully!"
    
    # Save training info
    echo "SFT Training Results - Job $SLURM_JOB_ID" > results/job_${SLURM_JOB_ID}/training_summary.txt
    echo "Environment: $ENVIRONMENT" >> results/job_${SLURM_JOB_ID}/training_summary.txt
    echo "Samples: $SAMPLES" >> results/job_${SLURM_JOB_ID}/training_summary.txt
    echo "LoRA Rank: $LORA_RANK" >> results/job_${SLURM_JOB_ID}/training_summary.txt
    echo "Learning Rate: $LEARNING_RATE" >> results/job_${SLURM_JOB_ID}/training_summary.txt
    echo "Completed: $(date)" >> results/job_${SLURM_JOB_ID}/training_summary.txt
    
else
    echo "❌ SFT training failed!"
    echo "SFT Training FAILED - Job $SLURM_JOB_ID" > results/job_${SLURM_JOB_ID}/training_summary.txt
    echo "Check logs/sft_eval_${SLURM_JOB_ID}.err for details" >> results/job_${SLURM_JOB_ID}/training_summary.txt
    exit 1
fi

# ===================
# PHASE 2: EVALUATION
# ===================
echo ""
echo "PHASE 2: Starting Model Evaluation..."
echo "===================================="

python scripts/evaluate_model.py \
    --model-path "models/single_system/${ENVIRONMENT}/sft/latest" \
    --model-type single_system \
    --num-test-cases $NUM_TEST_CASES \
    --test-type both \
    --save-plots \
    --plot-dir "plots/job_${SLURM_JOB_ID}" \
    --gpu-id 0

# Check if evaluation was successful
if [ $? -eq 0 ]; then
    echo "✅ Model evaluation completed successfully!"
    
    # Create evaluation summary
    echo "" >> results/job_${SLURM_JOB_ID}/training_summary.txt
    echo "=== EVALUATION RESULTS ===" >> results/job_${SLURM_JOB_ID}/training_summary.txt
    echo "Test Cases: $NUM_TEST_CASES" >> results/job_${SLURM_JOB_ID}/training_summary.txt
    echo "Evaluation completed: $(date)" >> results/job_${SLURM_JOB_ID}/training_summary.txt
    echo "Plots saved to: plots/job_${SLURM_JOB_ID}/" >> results/job_${SLURM_JOB_ID}/training_summary.txt
    
else
    echo "❌ Model evaluation failed!"
    echo "EVALUATION FAILED" >> results/job_${SLURM_JOB_ID}/training_summary.txt
    exit 1
fi

# ===================
# PHASE 3: SUMMARY
# ===================
echo ""
echo "============================================="
echo "JOB COMPLETION SUMMARY"
echo "============================================="
echo "✅ SFT Training: SUCCESS"
echo "✅ Model Evaluation: SUCCESS"
echo "📊 Results saved to: results/job_${SLURM_JOB_ID}/"
echo "📈 Plots saved to: plots/job_${SLURM_JOB_ID}/"
echo "🔗 Model path: models/single_system/${ENVIRONMENT}/sft/latest"
echo "Completed: $(date)"
echo "============================================="

# Copy important files to results directory
cp "plots/job_${SLURM_JOB_ID}"/*.png "results/job_${SLURM_JOB_ID}/" 2>/dev/null || true

echo "🎉 Complete job finished successfully!"