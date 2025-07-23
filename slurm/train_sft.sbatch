#!/bin/bash
#SBATCH --job-name=control_sft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/sft_%j.out
#SBATCH --error=logs/sft_%j.err

# Job configuration - customize these parameters
ENVIRONMENT=${ENVIRONMENT:-"double_integrator"}
SAMPLES=${SAMPLES:-300}
LORA_RANK=${LORA_RANK:-16}
LEARNING_RATE=${LEARNING_RATE:-0.0002}
NUM_EPOCHS=${NUM_EPOCHS:-4}
BATCH_SIZE=${BATCH_SIZE:-4}
RUN_NAME=${RUN_NAME:-"slurm_sft_${SLURM_JOB_ID}"}

# Set up environment
echo "Starting SFT training job: $SLURM_JOB_ID"
echo "Environment: $ENVIRONMENT"
echo "Samples: $SAMPLES"
echo "LoRA Rank: $LORA_RANK"
echo "Learning Rate: $LEARNING_RATE"

# Create logs directory
mkdir -p logs

# Activate conda environment (adjust as needed)
# source ~/.bashrc
# conda activate your_env_name

# Or load modules (adjust as needed)
# module load cuda/11.8
# module load python/3.9

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$SLURM_GPUS_ON_NODE

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Run SFT training
python scripts/train_single_system.py \
    --system $ENVIRONMENT \
    --num-samples $SAMPLES \
    --training-type sft \
    --lora-rank $LORA_RANK \
    --output-base "./slurm_output_${SLURM_JOB_ID}" \
    --run-name $RUN_NAME \
    --gpu-id 0

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "SFT training completed successfully!"
    
    # Optional: Run quick evaluation
    echo "Running quick evaluation..."
    python scripts/evaluate_model.py \
        --model-path "models/single_system/${ENVIRONMENT}/sft/latest" \
        --model-type single_system \
        --num-test-cases 5 \
        --save-plots \
        --plot-dir "plots/sft_${SLURM_JOB_ID}"
else
    echo "SFT training failed!"
    exit 1
fi

echo "Job completed: $(date)"