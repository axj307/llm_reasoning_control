#!/bin/bash
#SBATCH --job-name=control_sft_eval      # Job name
#SBATCH --partition=pi_linaresr          # pi_linaresr, mit_normal_gpu
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=10               # Number of CPU cores
#SBATCH --gres=gpu:h100:1                # Request 1 H100 GPU
#SBATCH --mem=100G                       # Memory
#SBATCH --time=24:00:00                  # Time limit (24 hours)
#SBATCH --output=logs/sft_eval_%j.out    # Output path
#SBATCH --error=logs/sft_eval_%j.err     # Error path

# Job configuration - customize these parameters
# Use SLURM_EXPORT_ENV to check if variables were exported
if [ ! -z "${SLURM_EXPORT_ENV}" ]; then
    echo "SLURM_EXPORT_ENV: ${SLURM_EXPORT_ENV}"
fi

# Set default values if not provided
SYSTEM=${SYSTEM:-"double_integrator"}

# Auto-detect dataset name based on system if not provided
if [ -z "$DATASET_NAME" ]; then
    if [ "$SYSTEM" = "double_integrator" ]; then
        DATASET_NAME="di"
    elif [ "$SYSTEM" = "van_der_pol" ]; then
        DATASET_NAME="vdp"
    else
        DATASET_NAME="$SYSTEM"  # fallback to system name
    fi
fi

LORA_RANK=${LORA_RANK:-8}  # Default to 8 as commonly used
LEARNING_RATE=${LEARNING_RATE:-0.0002}
NUM_EPOCHS=${NUM_EPOCHS:-4}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_TEST_CASES=${NUM_TEST_CASES:-5}
NUM_SAMPLES=${NUM_SAMPLES:-500}
RUN_NAME=${RUN_NAME:-"slurm_sft_${SLURM_JOB_ID}"}

# Dataset size limits (optional - empty means use full dataset)
SFT_MAX_SAMPLES=${SFT_MAX_SAMPLES:-""}
EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES:-""}

# Print job information
echo "============================================="
echo "SLURM SFT TRAINING + EVALUATION JOB"
echo "============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "System: $SYSTEM"
echo "Dataset Name: $DATASET_NAME"
echo "LoRA Rank: $LORA_RANK"
echo "Learning Rate: $LEARNING_RATE"
echo "Epochs: $NUM_EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Test Cases: $NUM_TEST_CASES"
echo "Number of Samples: $NUM_SAMPLES"
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
echo "=================================="

# Debug: Print the actual command
echo "Running command:"
echo "python scripts/train_single_system.py \\"
echo "    --system $SYSTEM \\"
echo "    --dataset-name $DATASET_NAME \\"
echo "    --training-type sft \\"
echo "    --lora-rank $LORA_RANK \\"
echo "    --output-base ./slurm_output_${SLURM_JOB_ID} \\"
echo "    --run-name $RUN_NAME \\"
echo "    --gpu-id 0"
echo ""

# First, list available datasets
echo "Checking available datasets..."
python scripts/train_single_system.py --list-datasets

# Build training command with optional dataset limits
TRAIN_CMD="python scripts/train_single_system.py \
    --system $SYSTEM \
    --dataset-name $DATASET_NAME \
    --training-type sft \
    --lora-rank $LORA_RANK \
    --num-samples $NUM_SAMPLES \
    --output-base ./slurm_output_${SLURM_JOB_ID} \
    --sft-run-name $RUN_NAME \
    --gpu-id 0"

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

# Check if SFT training was successful
if [ $? -eq 0 ]; then
    echo "✅ SFT training completed successfully!"
    
    # Save training info
    echo "SFT Training Results - Job $SLURM_JOB_ID" > figures/job_${SLURM_JOB_ID}/training_summary.txt
    echo "System: $SYSTEM" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    echo "Dataset: $DATASET_NAME" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    echo "LoRA Rank: $LORA_RANK" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    echo "Learning Rate: $LEARNING_RATE" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    echo "Completed: $(date)" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    
else
    echo "❌ SFT training failed!"
    echo "SFT Training FAILED - Job $SLURM_JOB_ID" > figures/job_${SLURM_JOB_ID}/training_summary.txt
    echo "Check logs/sft_eval_${SLURM_JOB_ID}.err for details" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    exit 1
fi

# ===================
# PHASE 2: EVALUATION
# ===================
echo ""
echo "PHASE 2: Starting Model Evaluation..."
echo "===================================="

python scripts/evaluate_model.py \
    --model-path "models/single_system/${SYSTEM}/sft/latest" \
    --model-type single_system \
    --eval-dataset $DATASET_NAME \
    --num-test-cases $NUM_TEST_CASES \
    --test-type both \
    --save-plots \
    --plot-dir "figures/job_${SLURM_JOB_ID}" \
    --gpu-id 0

# Check if evaluation was successful
if [ $? -eq 0 ]; then
    echo "✅ Model evaluation completed successfully!"
    
    # Create evaluation summary
    echo "" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    echo "=== EVALUATION RESULTS ===" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    echo "Test Cases: $NUM_TEST_CASES" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    echo "Evaluation completed: $(date)" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    echo "Plots saved to: figures/job_${SLURM_JOB_ID}/" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    
else
    echo "❌ Model evaluation failed!"
    echo "EVALUATION FAILED" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
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
echo "📊 All results saved to: figures/job_${SLURM_JOB_ID}/"
echo "🔗 Model path: models/single_system/${SYSTEM}/sft/latest"
echo "Completed: $(date)"
echo "============================================="

# Everything is already in figures directory - no copying needed

echo "🎉 Complete job finished successfully!"