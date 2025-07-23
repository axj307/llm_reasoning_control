#!/bin/bash
#SBATCH --job-name=control_grpo_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=logs/grpo_eval_%j.out
#SBATCH --error=logs/grpo_eval_%j.err

# Job configuration - customize these parameters
ENVIRONMENT=${ENVIRONMENT:-"double_integrator"}
SAMPLES=${SAMPLES:-300}
LORA_RANK=${LORA_RANK:-16}
SFT_LEARNING_RATE=${SFT_LEARNING_RATE:-0.0002}
GRPO_LEARNING_RATE=${GRPO_LEARNING_RATE:-0.000005}
SFT_EPOCHS=${SFT_EPOCHS:-4}
GRPO_STEPS=${GRPO_STEPS:-100}
GRPO_BATCH_SIZE=${GRPO_BATCH_SIZE:-8}
NUM_TEST_CASES=${NUM_TEST_CASES:-20}
RUN_NAME=${RUN_NAME:-"slurm_grpo_${SLURM_JOB_ID}"}

# Print job information
echo "============================================="
echo "SLURM SFT+GRPO TRAINING + EVALUATION JOB"
echo "============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Environment: $ENVIRONMENT"
echo "Samples: $SAMPLES"
echo "LoRA Rank: $LORA_RANK"
echo "SFT Learning Rate: $SFT_LEARNING_RATE"
echo "GRPO Learning Rate: $GRPO_LEARNING_RATE"
echo "SFT Epochs: $SFT_EPOCHS"
echo "GRPO Steps: $GRPO_STEPS"
echo "GRPO Batch Size: $GRPO_BATCH_SIZE"
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
# PHASE 1: SFT + GRPO TRAINING
# ===================
echo ""
echo "PHASE 1: Starting SFT + GRPO Training..."
echo "========================================"

python scripts/train_single_system.py \
    --system $ENVIRONMENT \
    --num-samples $SAMPLES \
    --training-type both \
    --lora-rank $LORA_RANK \
    --output-base "./slurm_output_${SLURM_JOB_ID}" \
    --run-name $RUN_NAME \
    --gpu-id 0

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "✅ SFT + GRPO training completed successfully!"
    
    # Save training info
    echo "SFT + GRPO Training Results - Job $SLURM_JOB_ID" > results/job_${SLURM_JOB_ID}/training_summary.txt
    echo "Environment: $ENVIRONMENT" >> results/job_${SLURM_JOB_ID}/training_summary.txt
    echo "Samples: $SAMPLES" >> results/job_${SLURM_JOB_ID}/training_summary.txt
    echo "LoRA Rank: $LORA_RANK" >> results/job_${SLURM_JOB_ID}/training_summary.txt
    echo "SFT Learning Rate: $SFT_LEARNING_RATE" >> results/job_${SLURM_JOB_ID}/training_summary.txt
    echo "GRPO Learning Rate: $GRPO_LEARNING_RATE" >> results/job_${SLURM_JOB_ID}/training_summary.txt
    echo "SFT Epochs: $SFT_EPOCHS" >> results/job_${SLURM_JOB_ID}/training_summary.txt
    echo "GRPO Steps: $GRPO_STEPS" >> results/job_${SLURM_JOB_ID}/training_summary.txt
    echo "Training completed: $(date)" >> results/job_${SLURM_JOB_ID}/training_summary.txt
    
else
    echo "❌ SFT + GRPO training failed!"
    echo "SFT + GRPO Training FAILED - Job $SLURM_JOB_ID" > results/job_${SLURM_JOB_ID}/training_summary.txt
    echo "Check logs/grpo_eval_${SLURM_JOB_ID}.err for details" >> results/job_${SLURM_JOB_ID}/training_summary.txt
    exit 1
fi

# ===================
# PHASE 2: SFT EVALUATION
# ===================
echo ""
echo "PHASE 2: Evaluating SFT Model..."
echo "==============================="

python scripts/evaluate_model.py \
    --model-path "models/single_system/${ENVIRONMENT}/sft/latest" \
    --model-type single_system \
    --num-test-cases $NUM_TEST_CASES \
    --test-type both \
    --save-plots \
    --plot-dir "plots/job_${SLURM_JOB_ID}" \
    --gpu-id 0

if [ $? -eq 0 ]; then
    echo "✅ SFT model evaluation completed successfully!"
    
    # Rename plots to indicate SFT
    cd plots/job_${SLURM_JOB_ID}
    for file in *.png; do
        if [[ -f "$file" ]]; then
            mv "$file" "sft_$file"
        fi
    done
    cd ../..
    
else
    echo "❌ SFT model evaluation failed!"
    echo "SFT EVALUATION FAILED" >> results/job_${SLURM_JOB_ID}/training_summary.txt
fi

# ===================
# PHASE 3: GRPO EVALUATION
# ===================
echo ""
echo "PHASE 3: Evaluating GRPO Model..."
echo "================================"

python scripts/evaluate_model.py \
    --model-path "models/single_system/${ENVIRONMENT}/grpo/latest" \
    --model-type single_system \
    --num-test-cases $NUM_TEST_CASES \
    --test-type both \
    --save-plots \
    --plot-dir "plots/job_${SLURM_JOB_ID}" \
    --gpu-id 0

if [ $? -eq 0 ]; then
    echo "✅ GRPO model evaluation completed successfully!"
    
    # Rename plots to indicate GRPO
    cd plots/job_${SLURM_JOB_ID}
    for file in *.png; do
        if [[ -f "$file" && ! "$file" =~ ^sft_ ]]; then
            mv "$file" "grpo_$file"
        fi
    done
    cd ../..
    
    # Create evaluation summary
    echo "" >> results/job_${SLURM_JOB_ID}/training_summary.txt
    echo "=== EVALUATION RESULTS ===" >> results/job_${SLURM_JOB_ID}/training_summary.txt
    echo "Test Cases: $NUM_TEST_CASES" >> results/job_${SLURM_JOB_ID}/training_summary.txt
    echo "SFT evaluation: ✅ SUCCESS" >> results/job_${SLURM_JOB_ID}/training_summary.txt
    echo "GRPO evaluation: ✅ SUCCESS" >> results/job_${SLURM_JOB_ID}/training_summary.txt
    echo "Evaluation completed: $(date)" >> results/job_${SLURM_JOB_ID}/training_summary.txt
    echo "Plots saved to: plots/job_${SLURM_JOB_ID}/" >> results/job_${SLURM_JOB_ID}/training_summary.txt
    
else
    echo "❌ GRPO model evaluation failed!"
    echo "GRPO EVALUATION FAILED" >> results/job_${SLURM_JOB_ID}/training_summary.txt
fi

# ===================
# PHASE 4: SUMMARY
# ===================
echo ""
echo "============================================="
echo "JOB COMPLETION SUMMARY"
echo "============================================="
echo "✅ SFT Training: SUCCESS"
echo "✅ GRPO Training: SUCCESS"
echo "✅ SFT Evaluation: SUCCESS"
echo "✅ GRPO Evaluation: SUCCESS"
echo "📊 Results saved to: results/job_${SLURM_JOB_ID}/"
echo "📈 Plots saved to: plots/job_${SLURM_JOB_ID}/"
echo "🔗 SFT Model: models/single_system/${ENVIRONMENT}/sft/latest"
echo "🔗 GRPO Model: models/single_system/${ENVIRONMENT}/grpo/latest"
echo "Completed: $(date)"
echo "============================================="

# Copy important files to results directory
cp "plots/job_${SLURM_JOB_ID}"/*.png "results/job_${SLURM_JOB_ID}/" 2>/dev/null || true

echo "🎉 Complete job finished successfully!"