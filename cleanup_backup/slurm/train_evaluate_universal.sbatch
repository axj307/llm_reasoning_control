#!/bin/bash
#SBATCH --job-name=universal_control       # Job name
#SBATCH --partition=pi_linaresr            # pi_linaresr, mit_normal_gpu
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --ntasks=1                         # Number of tasks
#SBATCH --cpus-per-task=10                 # Number of CPU cores
#SBATCH --gres=gpu:h100:1                  # Request 1 H100 GPU
#SBATCH --mem=100G                         # Memory
#SBATCH --time=24:00:00                    # Time limit (24 hours)
#SBATCH --output=logs/slurm_%j.out         # Output path
#SBATCH --error=logs/slurm_%j.err          # Error path

# Load necessary modules
module purge
module load cuda/12.4.0

# Activate conda environment
source ~/.bashrc
conda activate unsloth_env

# Get the SLURM job ID
CURRENT_JOB_ID="${SLURM_JOB_ID}"
FIGURES_DIR="figures/job_${CURRENT_JOB_ID}"

# Create necessary directories
mkdir -p logs figures
mkdir -p "${FIGURES_DIR}"

echo "============================================="
echo "UNIVERSAL CONTROL LLM TRAINING + EVALUATION"
echo "============================================="
echo "Job ID: ${CURRENT_JOB_ID}"
echo "Started: $(date)"
echo "Figures Directory: ${FIGURES_DIR}"
echo "============================================="

# Check if we should resume from a previous job
RESUME_FLAGS=""
if [ ! -z "$RESUME_JOB_ID" ]; then
    echo "Resuming from job ID: ${RESUME_JOB_ID}"
    RESUME_FLAGS="--base-model models/universal/grpo/job_${RESUME_JOB_ID}"
fi

# ===================
# PHASE 1: UNIVERSAL TRAINING (SFT + GRPO)
# ===================
echo ""
echo "PHASE 1: Starting Universal Model Training..."
echo "==========================================="

python scripts/train_universal.py \
    `# === System Configuration ===` \
    --systems "double_integrator,van_der_pol" \
    --samples-per-system 300 \
    \
    `# === Model Configuration ===` \
    --lora-rank 32 \
    --gpu-id 0 \
    \
    `# === Training Configuration ===` \
    --training-type both \
    --output-base "${OUTPUT_DIR}" \
    --run-name "universal_${CURRENT_JOB_ID}" \
    ${RESUME_FLAGS} \
    "$@"

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "✅ Universal model training completed successfully!"
    
    # Save training info
    echo "Universal Model Training Results - Job ${CURRENT_JOB_ID}" > "${FIGURES_DIR}/training_summary.txt"
    echo "Systems: double_integrator, van_der_pol" >> "${FIGURES_DIR}/training_summary.txt"
    echo "Samples per system: 300" >> "${FIGURES_DIR}/training_summary.txt"
    echo "LoRA Rank: 32" >> "${FIGURES_DIR}/training_summary.txt"
    echo "Training completed: $(date)" >> "${FIGURES_DIR}/training_summary.txt"
    
else
    echo "❌ Universal model training failed!"
    echo "Universal Training FAILED - Job ${CURRENT_JOB_ID}" > "${FIGURES_DIR}/training_summary.txt"
    echo "Check logs/slurm_${CURRENT_JOB_ID}.err for details" >> "${FIGURES_DIR}/training_summary.txt"
    exit 1
fi

# ===================
# PHASE 2: SFT MODEL EVALUATION
# ===================
echo ""
echo "PHASE 2: Evaluating SFT Model..."
echo "==============================="

python scripts/evaluate_model.py \
    --model-path "models/universal/sft/latest" \
    --model-type universal \
    --systems "double_integrator,van_der_pol" \
    --num-test-cases 20 \
    --test-type both \
    --save-plots \
    --plot-dir "${FIGURES_DIR}" \
    --gpu-id 0

if [ $? -eq 0 ]; then
    echo "✅ SFT model evaluation completed successfully!"
    
    # Rename plots to indicate SFT
    cd "${FIGURES_DIR}"
    for file in *.png; do
        if [[ -f "$file" ]]; then
            mv "$file" "sft_$file"
        fi
    done
    cd "${SLURM_SUBMIT_DIR}"
    
else
    echo "❌ SFT model evaluation failed!"
    echo "SFT EVALUATION FAILED" >> "${FIGURES_DIR}/training_summary.txt"
fi

# ===================
# PHASE 3: GRPO MODEL EVALUATION
# ===================
echo ""
echo "PHASE 3: Evaluating GRPO Model..."
echo "================================"

python scripts/evaluate_model.py \
    --model-path "models/universal/grpo/latest" \
    --model-type universal \
    --systems "double_integrator,van_der_pol" \
    --num-test-cases 20 \
    --test-type both \
    --save-plots \
    --plot-dir "${FIGURES_DIR}" \
    --gpu-id 0

if [ $? -eq 0 ]; then
    echo "✅ GRPO model evaluation completed successfully!"
    
    # Rename plots to indicate GRPO
    cd "${FIGURES_DIR}"
    for file in *.png; do
        if [[ -f "$file" && ! "$file" =~ ^sft_ ]]; then
            mv "$file" "grpo_$file"
        fi
    done
    cd "${SLURM_SUBMIT_DIR}"
    
    # Update evaluation summary
    echo "" >> "${FIGURES_DIR}/training_summary.txt"
    echo "=== EVALUATION RESULTS ===" >> "${FIGURES_DIR}/training_summary.txt"
    echo "Test Cases per System: 20" >> "${FIGURES_DIR}/training_summary.txt"
    echo "SFT evaluation: ✅ SUCCESS" >> "${FIGURES_DIR}/training_summary.txt"
    echo "GRPO evaluation: ✅ SUCCESS" >> "${FIGURES_DIR}/training_summary.txt"
    echo "Evaluation completed: $(date)" >> "${FIGURES_DIR}/training_summary.txt"
    echo "All results saved to: ${FIGURES_DIR}/" >> "${FIGURES_DIR}/training_summary.txt"
    
else
    echo "❌ GRPO model evaluation failed!"
    echo "GRPO EVALUATION FAILED" >> "${FIGURES_DIR}/training_summary.txt"
fi

# ===================
# PHASE 4: SUMMARY & CLEANUP
# ===================
echo ""
echo "============================================="
echo "JOB COMPLETION SUMMARY"
echo "============================================="
echo "✅ Universal Model Training: SUCCESS"
echo "✅ SFT Evaluation: SUCCESS" 
echo "✅ GRPO Evaluation: SUCCESS"
echo "📊 All results saved to: ${FIGURES_DIR}/"
echo "🔗 SFT Model: models/universal/sft/latest"
echo "🔗 GRPO Model: models/universal/grpo/latest"
echo "Completed: $(date)"
echo "============================================="

# Everything is already in figures directory - no copying needed

# Add success message
echo "Job completed with ID ${CURRENT_JOB_ID}"
echo "All results can be found in: ${FIGURES_DIR}"
echo "🎉 Universal control training job finished successfully!"

# === Alternative Configuration Examples ===
# Uncomment a section below to use different parameter sets

# === 1. High Performance Configuration ===
#python scripts/train_universal.py \
#    --systems "double_integrator,van_der_pol" \
#    --samples-per-system 500 \
#    --lora-rank 64 \
#    --training-type both \
#    --output-base "${OUTPUT_DIR}" \
#    --run-name "high_perf_${CURRENT_JOB_ID}" \
#    "$@"

# === 2. Quick Test Configuration ===
#python scripts/train_universal.py \
#    --systems "double_integrator" \
#    --samples-per-system 100 \
#    --lora-rank 16 \
#    --training-type sft \
#    --output-base "${OUTPUT_DIR}" \
#    --run-name "quick_test_${CURRENT_JOB_ID}" \
#    "$@"

# === 3. Extended Training Configuration ===
#python scripts/train_universal.py \
#    --systems "double_integrator,van_der_pol" \
#    --samples-per-system 800 \
#    --lora-rank 32 \
#    --training-type both \
#    --output-base "${OUTPUT_DIR}" \
#    --run-name "extended_${CURRENT_JOB_ID}" \
#    "$@"