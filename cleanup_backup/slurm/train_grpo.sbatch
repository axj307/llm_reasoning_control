#!/bin/bash
#SBATCH --job-name=control_grpo_fixed    # Job name
#SBATCH --partition=pi_linaresr          # pi_linaresr, mit_normal_gpu
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=10               # Number of CPU cores
#SBATCH --gres=gpu:h100:1                # Request 1 H100 GPU
#SBATCH --mem=100G                       # Memory
#SBATCH --time=48:00:00                  # Time limit (48 hours for GRPO)
#SBATCH --output=logs/grpo_fixed_%j.out  # Output path
#SBATCH --error=logs/grpo_fixed_%j.err   # Error path

# Job configuration - customize these parameters
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

LORA_RANK=${LORA_RANK:-8}
TRAINING_TYPE=${TRAINING_TYPE:-"both"}  # sft, grpo, or both
RUN_NAME=${RUN_NAME:-"slurm_grpo_fixed_${SLURM_JOB_ID}"}

# Dataset size limits (optional - for testing/debugging)
SFT_MAX_SAMPLES=${SFT_MAX_SAMPLES:-""}        # Empty = full dataset
GRPO_MAX_SAMPLES=${GRPO_MAX_SAMPLES:-""}      # Empty = full dataset  
SFT_MAX_STEPS=${SFT_MAX_STEPS:-""}            # Empty = use config default
GRPO_MAX_STEPS=${GRPO_MAX_STEPS:-""}          # Empty = use config default

NUM_TEST_CASES=${NUM_TEST_CASES:-10}

# Print job information
echo "============================================="
echo "SLURM GRPO TRAINING JOB (FIXED VERSION)"
echo "============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "System: $SYSTEM"
echo "Dataset Name: $DATASET_NAME"
echo "LoRA Rank: $LORA_RANK"
echo "Training Type: $TRAINING_TYPE"
echo "Run Name: $RUN_NAME"
if [ ! -z "$SFT_MAX_SAMPLES" ]; then
    echo "SFT Max Samples: $SFT_MAX_SAMPLES (limited for testing)"
fi
if [ ! -z "$GRPO_MAX_SAMPLES" ]; then
    echo "GRPO Max Samples: $GRPO_MAX_SAMPLES (limited for testing)"
fi
if [ ! -z "$SFT_MAX_STEPS" ]; then
    echo "SFT Max Steps: $SFT_MAX_STEPS (limited for testing)"
fi
if [ ! -z "$GRPO_MAX_STEPS" ]; then
    echo "GRPO Max Steps: $GRPO_MAX_STEPS (limited for testing)"
fi
echo "Test Cases: $NUM_TEST_CASES"
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
# PHASE 1: TRAINING
# ===================
echo ""
echo "PHASE 1: Starting Training (Fixed GRPO Implementation)..."
echo "========================================================"

# First, list available datasets
echo "Checking available datasets..."
python scripts/train_single_system.py --list-datasets

# Build training command
TRAIN_CMD="python scripts/train_single_system.py \\
    --system $SYSTEM \\
    --dataset-name $DATASET_NAME \\
    --training-type $TRAINING_TYPE \\
    --lora-rank $LORA_RANK \\
    --output-base ./slurm_output_${SLURM_JOB_ID} \\
    --run-name $RUN_NAME \\
    --gpu-id 0"

# Add dataset size limits if provided
if [ ! -z "$SFT_MAX_SAMPLES" ]; then
    TRAIN_CMD="$TRAIN_CMD --sft-max-samples $SFT_MAX_SAMPLES"
    echo "🔽 Limiting SFT training to $SFT_MAX_SAMPLES samples"
fi

if [ ! -z "$GRPO_MAX_SAMPLES" ]; then
    TRAIN_CMD="$TRAIN_CMD --grpo-max-samples $GRPO_MAX_SAMPLES"
    echo "🔽 Limiting GRPO training to $GRPO_MAX_SAMPLES samples"
fi

if [ ! -z "$SFT_MAX_STEPS" ]; then
    TRAIN_CMD="$TRAIN_CMD --sft-max-steps $SFT_MAX_STEPS"
    echo "🔽 Limiting SFT to $SFT_MAX_STEPS steps"
fi

if [ ! -z "$GRPO_MAX_STEPS" ]; then
    TRAIN_CMD="$TRAIN_CMD --grpo-max-steps $GRPO_MAX_STEPS"
    echo "🔽 Limiting GRPO to $GRPO_MAX_STEPS steps"
fi

# Debug: Print the actual command
echo ""
echo "Running training command:"
echo "$TRAIN_CMD"
echo ""

# Run the training
eval $TRAIN_CMD

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully!"
    
    # Save training info
    echo "GRPO Training Results (Fixed) - Job $SLURM_JOB_ID" > figures/job_${SLURM_JOB_ID}/training_summary.txt
    echo "System: $SYSTEM" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    echo "Dataset: $DATASET_NAME" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    echo "LoRA Rank: $LORA_RANK" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    echo "Training Type: $TRAINING_TYPE" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    echo "Completed: $(date)" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    
else
    echo "❌ Training failed!"
    echo "Training FAILED - Job $SLURM_JOB_ID" > figures/job_${SLURM_JOB_ID}/training_summary.txt
    echo "Check logs/grpo_fixed_${SLURM_JOB_ID}.err for details" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    exit 1
fi

# ===================
# PHASE 2: EVALUATION
# ===================
echo ""
echo "PHASE 2: Starting Model Evaluation..."
echo "===================================="

# Determine which model to evaluate based on training type
if [ "$TRAINING_TYPE" = "both" ] || [ "$TRAINING_TYPE" = "grpo" ]; then
    MODEL_PATH="models/single_system/${SYSTEM}/grpo/latest"
    echo "Evaluating GRPO model at: $MODEL_PATH"
else
    MODEL_PATH="models/single_system/${SYSTEM}/sft/latest"
    echo "Evaluating SFT model at: $MODEL_PATH"
fi

python scripts/evaluate_model.py \\
    --model-path "$MODEL_PATH" \\
    --model-type single_system \\
    --eval-dataset $DATASET_NAME \\
    --num-test-cases $NUM_TEST_CASES \\
    --test-type both \\
    --save-plots \\
    --plot-dir "figures/job_${SLURM_JOB_ID}" \\
    --gpu-id 0

# Check if evaluation was successful
if [ $? -eq 0 ]; then
    echo "✅ Model evaluation completed successfully!"
    
    # Create evaluation summary
    echo "" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    echo "=== EVALUATION RESULTS ===" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
    echo "Model Path: $MODEL_PATH" >> figures/job_${SLURM_JOB_ID}/training_summary.txt
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
echo "✅ Training ($TRAINING_TYPE): SUCCESS"
echo "✅ Model Evaluation: SUCCESS"
echo "📊 All results saved to: figures/job_${SLURM_JOB_ID}/"
if [ "$TRAINING_TYPE" = "both" ] || [ "$TRAINING_TYPE" = "grpo" ]; then
    echo "🔗 GRPO Model path: models/single_system/${SYSTEM}/grpo/latest"
fi
if [ "$TRAINING_TYPE" = "both" ] || [ "$TRAINING_TYPE" = "sft" ]; then
    echo "🔗 SFT Model path: models/single_system/${SYSTEM}/sft/latest"
fi
echo "Completed: $(date)"
echo "============================================="

echo "🎉 GRPO training job (FIXED) finished successfully!"