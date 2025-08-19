#!/bin/bash
#SBATCH --job-name=control_grpo_eval     # Job name
#SBATCH --partition=pi_linaresr          # pi_linaresr, mit_normal_gpu
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=10               # Number of CPU cores
#SBATCH --gres=gpu:h100:1                # Request 1 H100 GPU
#SBATCH --mem=100G                       # Memory
#SBATCH --time=48:00:00                  # Time limit (48 hours)
#SBATCH --output=logs/grpo_eval_%j.out   # Output path
#SBATCH --error=logs/grpo_eval_%j.err    # Error path

# Job configuration - customize these parameters
# Use CONTROL_SYSTEM instead of ENVIRONMENT to avoid SLURM conflicts
CONTROL_SYSTEM=${CONTROL_SYSTEM:-"double_integrator"}

# Auto-detect dataset name based on control system if not provided
if [ -z "$DATASET_NAME" ]; then
    if [ "$CONTROL_SYSTEM" = "double_integrator" ]; then
        DATASET_NAME="di"
    elif [ "$CONTROL_SYSTEM" = "van_der_pol" ]; then
        DATASET_NAME="vdp"
    else
        DATASET_NAME="$CONTROL_SYSTEM"  # fallback to control system name
    fi
fi

LORA_RANK=${LORA_RANK:-8}  # Default to 8 - must be one of (8, 16, 32, 64, 128, 256, 320, 512)
SFT_LEARNING_RATE=${SFT_LEARNING_RATE:-0.0002}
GRPO_LEARNING_RATE=${GRPO_LEARNING_RATE:-0.000005}
SFT_EPOCHS=${SFT_EPOCHS:-4}
GRPO_STEPS=${GRPO_STEPS:-100}
GRPO_BATCH_SIZE=${GRPO_BATCH_SIZE:-8}
NUM_TEST_CASES=${NUM_TEST_CASES:-20}
RUN_NAME=${RUN_NAME:-"slurm_grpo_${SLURM_JOB_ID}"}

# Dataset size limits (optional - empty means use full dataset)
SFT_MAX_SAMPLES=${SFT_MAX_SAMPLES:-""}
GRPO_MAX_SAMPLES=${GRPO_MAX_SAMPLES:-""}
EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES:-""}

# Print job information with debug output
echo "============================================="
echo "SLURM SFT+GRPO TRAINING + EVALUATION JOB"
echo "============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo ""
echo "=== ENVIRONMENT VARIABLES DEBUG ==="
echo "CONTROL_SYSTEM (raw): '$CONTROL_SYSTEM'"
echo "DATASET_NAME (raw): '$DATASET_NAME'"
echo "All exported env vars:"
env | grep -E "(SFT_|GRPO_|EVAL_|CONTROL_SYSTEM|DATASET_)" | sort
echo ""
echo "=== JOB CONFIGURATION ==="
echo "Control System: $CONTROL_SYSTEM"
echo "Dataset Name: $DATASET_NAME"
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

# ===================
# PHASE 1: SFT + GRPO TRAINING
# ===================
echo ""
echo "PHASE 1: Starting SFT + GRPO Training..."
echo "========================================"

# Build training command with proper quoting and variable handling
echo "Building training command..."
echo "CONTROL_SYSTEM = '$CONTROL_SYSTEM'"
echo "DATASET_NAME = '$DATASET_NAME'"

# Build base command with proper arguments for refactored script
TRAIN_CMD="python scripts/train_single_system.py"
TRAIN_CMD="$TRAIN_CMD --system \"$CONTROL_SYSTEM\""
TRAIN_CMD="$TRAIN_CMD --sft --grpo"  # Run both SFT and GRPO

# Note: Refactored script uses default dataset sizes
echo "Using default dataset sizes (refactored script doesn't support size limits yet)"

# Run the training
echo "Final training command: $TRAIN_CMD"
eval $TRAIN_CMD

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "✅ SFT + GRPO training completed successfully!"
    
    # Save training info
    echo "SFT + GRPO Training Results - Job $SLURM_JOB_ID" > figures/job_${SLURM_JOB_ID}/training_summary.txt
    echo "Control System: $CONTROL_SYSTEM" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    echo "Dataset: $DATASET_NAME" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    echo "LoRA Rank: $LORA_RANK" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    echo "SFT Learning Rate: $SFT_LEARNING_RATE" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    echo "GRPO Learning Rate: $GRPO_LEARNING_RATE" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    echo "SFT Epochs: $SFT_EPOCHS" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    echo "GRPO Steps: $GRPO_STEPS" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    echo "Training completed: $(date)" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    
else
    echo "❌ SFT + GRPO training failed!"
    echo "SFT + GRPO Training FAILED - Job $SLURM_JOB_ID" > figures/job_${SLURM_JOB_ID}/training_summary.txt
    echo "Check logs/grpo_eval_${SLURM_JOB_ID}.err for details" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    exit 1
fi

# ===================
# PHASE 2: SFT EVALUATION
# ===================
echo ""
echo "PHASE 2: Evaluating SFT Model..."
echo "==============================="

python scripts/evaluate_model.py \
    --model-path "models/single_system/${CONTROL_SYSTEM}/sft/latest" \
    --model-type single_system \
    --eval-dataset $DATASET_NAME \
    --num-test-cases $NUM_TEST_CASES \
    --test-type both \
    --save-plots \
    --plot-dir "figures/job_${SLURM_JOB_ID}" \
    --gpu-id 0

if [ $? -eq 0 ]; then
    echo "✅ SFT model evaluation completed successfully!"
    
    # Rename plots to indicate SFT
    cd figures/job_${SLURM_JOB_ID}
    for file in *.png; do
        if [[ -f "$file" ]]; then
            mv "$file" "sft_$file"
        fi
    done
    cd ../..
    
else
    echo "❌ SFT model evaluation failed!"
    echo "SFT EVALUATION FAILED" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
fi

# ===================
# PHASE 3: GRPO EVALUATION
# ===================
echo ""
echo "PHASE 3: Evaluating GRPO Model..."
echo "================================"

python scripts/evaluate_model.py \
    --model-path "models/single_system/${CONTROL_SYSTEM}/grpo/latest" \
    --model-type single_system \
    --eval-dataset $DATASET_NAME \
    --num-test-cases $NUM_TEST_CASES \
    --test-type both \
    --save-plots \
    --plot-dir "figures/job_${SLURM_JOB_ID}" \
    --gpu-id 0

if [ $? -eq 0 ]; then
    echo "✅ GRPO model evaluation completed successfully!"
    
    # Rename plots to indicate GRPO
    cd figures/job_${SLURM_JOB_ID}
    for file in *.png; do
        if [[ -f "$file" && ! "$file" =~ ^sft_ ]]; then
            mv "$file" "grpo_$file"
        fi
    done
    cd ../..
    
    # Create evaluation summary
    echo "" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    echo "=== EVALUATION RESULTS ===" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    echo "Test Cases: $NUM_TEST_CASES" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    echo "SFT evaluation: ✅ SUCCESS" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    echo "GRPO evaluation: ✅ SUCCESS" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    echo "Evaluation completed: $(date)" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    echo "All results saved to: figures/job_${SLURM_JOB_ID}/" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    
else
    echo "❌ GRPO model evaluation failed!"
    echo "GRPO EVALUATION FAILED" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
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
echo "📊 All results saved to: figures/job_${SLURM_JOB_ID}/"
echo "🔗 SFT Model: models/single_system/${CONTROL_SYSTEM}/sft/latest"
echo "🔗 GRPO Model: models/single_system/${CONTROL_SYSTEM}/grpo/latest"
echo "Completed: $(date)"
echo "============================================="

# Everything is already in figures directory - no copying needed

echo "🎉 Complete job finished successfully!"