#!/bin/bash
#SBATCH --job-name=control_sft_only      # Job name
#SBATCH --partition=pi_linaresr          # pi_linaresr, mit_normal_gpu
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=10               # Number of CPU cores
#SBATCH --gres=gpu:h100:1                # Request 1 H100 GPU
#SBATCH --mem=100G                       # Memory
#SBATCH --time=24:00:00                  # Time limit (24 hours)
#SBATCH --output=logs/sft_only_%j.out    # Output path
#SBATCH --error=logs/sft_only_%j.err     # Error path

# Job configuration - customize these parameters
ENVIRONMENT=${ENVIRONMENT:-"double_integrator"}

# Auto-detect dataset name based on environment if not provided
if [ -z "$DATASET_NAME" ]; then
    if [ "$ENVIRONMENT" = "double_integrator" ]; then
        DATASET_NAME="di"
    elif [ "$ENVIRONMENT" = "van_der_pol" ]; then
        DATASET_NAME="vdp"
    else
        DATASET_NAME="$ENVIRONMENT"  # fallback to environment name
    fi
fi

LORA_RANK=${LORA_RANK:-8}  # Default to 8 - must be one of (8, 16, 32, 64, 128, 256, 320, 512)
SFT_LEARNING_RATE=${SFT_LEARNING_RATE:-0.0002}
SFT_EPOCHS=${SFT_EPOCHS:-4}
NUM_TEST_CASES=${NUM_TEST_CASES:-20}
RUN_NAME=${RUN_NAME:-"slurm_sft_${SLURM_JOB_ID}"}

# Dataset size limits (optional - empty means use full dataset)
SFT_MAX_SAMPLES=${SFT_MAX_SAMPLES:-""}
EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES:-""}

# Print job information with debug output
echo "============================================="
echo "SLURM SFT TRAINING + EVALUATION JOB"
echo "============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo ""
echo "=== ENVIRONMENT VARIABLES DEBUG ==="
echo "ENVIRONMENT (raw): '$ENVIRONMENT'"
echo "DATASET_NAME (raw): '$DATASET_NAME'"
echo "All exported env vars:"
env | grep -E "(SFT_|EVAL_|ENVIRONMENT|DATASET_)" | sort
echo ""
echo "=== JOB CONFIGURATION ==="
echo "Environment: $ENVIRONMENT"
echo "Dataset Name: $DATASET_NAME"
echo "LoRA Rank: $LORA_RANK"
echo "SFT Learning Rate: $SFT_LEARNING_RATE"
echo "SFT Epochs: $SFT_EPOCHS"
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
# PHASE 1: SFT TRAINING
# ===================
echo ""
echo "PHASE 1: Starting SFT Training..."
echo "================================="

# Build training command with proper quoting and variable handling
echo "Building training command..."
echo "ENVIRONMENT = '$ENVIRONMENT'"
echo "DATASET_NAME = '$DATASET_NAME'"

# Build base command with proper quoting
TRAIN_CMD="python scripts/train_single_system.py"
TRAIN_CMD="$TRAIN_CMD --system \"$ENVIRONMENT\""
TRAIN_CMD="$TRAIN_CMD --dataset-name \"$DATASET_NAME\""
TRAIN_CMD="$TRAIN_CMD --training-type sft"
TRAIN_CMD="$TRAIN_CMD --lora-rank $LORA_RANK"
TRAIN_CMD="$TRAIN_CMD --output-base ./slurm_output_${SLURM_JOB_ID}"
TRAIN_CMD="$TRAIN_CMD --sft-run-name \"${RUN_NAME}_sft\""
TRAIN_CMD="$TRAIN_CMD --gpu-id 0"

# Add dataset size limits if provided
if [ ! -z "$SFT_MAX_SAMPLES" ]; then
    TRAIN_CMD="$TRAIN_CMD --sft-max-samples $SFT_MAX_SAMPLES"
    echo "🔽 Limiting SFT training to $SFT_MAX_SAMPLES samples"
fi

if [ ! -z "$EVAL_MAX_SAMPLES" ]; then
    TRAIN_CMD="$TRAIN_CMD --eval-max-samples $EVAL_MAX_SAMPLES"
    echo "🔽 Limiting evaluation to $EVAL_MAX_SAMPLES samples"
fi

# Run the training
echo "Final training command: $TRAIN_CMD"
eval $TRAIN_CMD

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "✅ SFT training completed successfully!"
    
    # Save training info
    echo "SFT Training Results - Job $SLURM_JOB_ID" > figures/job_${SLURM_JOB_ID}/training_summary.txt
    echo "Environment: $ENVIRONMENT" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    echo "Dataset: $DATASET_NAME" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    echo "LoRA Rank: $LORA_RANK" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    echo "SFT Learning Rate: $SFT_LEARNING_RATE" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    echo "SFT Epochs: $SFT_EPOCHS" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    echo "Training completed: $(date)" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    
else
    echo "❌ SFT training failed!"
    echo "SFT Training FAILED - Job $SLURM_JOB_ID" > figures/job_${SLURM_JOB_ID}/training_summary.txt
    echo "Check logs/sft_only_${SLURM_JOB_ID}.err for details" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
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
    
    # Create evaluation summary
    echo "" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    echo "=== EVALUATION RESULTS ===" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    echo "Test Cases: $NUM_TEST_CASES" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    echo "SFT evaluation: ✅ SUCCESS" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    echo "Evaluation completed: $(date)" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    echo "All results saved to: figures/job_${SLURM_JOB_ID}/" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    
else
    echo "❌ SFT model evaluation failed!"
    echo "SFT EVALUATION FAILED" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
fi

# ===================
# PHASE 3: SUMMARY
# ===================
echo ""
echo "============================================="
echo "JOB COMPLETION SUMMARY"
echo "============================================="
echo "✅ SFT Training: SUCCESS"
echo "✅ SFT Evaluation: SUCCESS"
echo "📊 All results saved to: figures/job_${SLURM_JOB_ID}/"
echo "🔗 SFT Model: models/single_system/${ENVIRONMENT}/sft/latest"
echo "Completed: $(date)"
echo "============================================="

echo "🎉 SFT-only job finished successfully!"