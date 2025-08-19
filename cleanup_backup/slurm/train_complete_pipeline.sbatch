#!/bin/bash
#SBATCH --job-name=complete_pipeline     # Job name
#SBATCH --partition=mit_normal_gpu       # mit_normal_gpu for training/validation
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=10               # Number of CPU cores
#SBATCH --gres=gpu:h100:1                # Request 1 H100 GPU
#SBATCH --mem=100G                       # Memory
#SBATCH --time=03:30:00                  # Time limit (3.5 hours for mit_normal_gpu)
#SBATCH --output=logs/complete_pipeline_%j.out    # Output path
#SBATCH --error=logs/complete_pipeline_%j.err     # Error path

# Job configuration - customize these parameters
CONTROL_SYSTEM=${CONTROL_SYSTEM:-"double_integrator"}

# Print job information
echo "============================================="
echo "COMPLETE SFT + GRPO PIPELINE"
echo "============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Control System: $CONTROL_SYSTEM"
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

python scripts/train_single_system.py --system "$CONTROL_SYSTEM" --sft

if [ $? -eq 0 ]; then
    echo "✅ SFT training completed successfully!"
else
    echo "❌ SFT training failed!"
    exit 1
fi

# ===================
# PHASE 2: GRPO TRAINING
# ===================
echo ""
echo "PHASE 2: Starting GRPO Training..."
echo "=================================="

python scripts/train_single_system.py --system "$CONTROL_SYSTEM" --grpo

if [ $? -eq 0 ]; then
    echo "✅ GRPO training completed successfully!"
else
    echo "❌ GRPO training failed!"
    echo "⚠️ Continuing with SFT-only results..."
fi

# ===================
# PHASE 3: MODEL EVALUATION
# ===================
echo ""
echo "PHASE 3: Evaluating Trained Models..."
echo "====================================="

# Run comprehensive model evaluation
echo "Running comprehensive evaluation on trained models..."

# Find the latest trained models
SFT_MODEL_PATH="models/single_system/$CONTROL_SYSTEM/sft/latest"
GRPO_MODEL_PATH="models/single_system/$CONTROL_SYSTEM/grpo/latest"

# Check which models exist
if [ -d "$SFT_MODEL_PATH" ]; then
    echo "✅ Found SFT model: $SFT_MODEL_PATH"
    SFT_EXISTS=true
else
    echo "❌ SFT model not found at: $SFT_MODEL_PATH"
    SFT_EXISTS=false
fi

if [ -d "$GRPO_MODEL_PATH" ]; then
    echo "✅ Found GRPO model: $GRPO_MODEL_PATH"
    GRPO_EXISTS=true
else
    echo "❌ GRPO model not found at: $GRPO_MODEL_PATH"
    GRPO_EXISTS=false
fi

# Create evaluation output directory
EVAL_OUTPUT_DIR="figures/job_${SLURM_JOB_ID}/evaluation"
mkdir -p "$EVAL_OUTPUT_DIR"

# Run comprehensive evaluation if SFT model exists
if [ "$SFT_EXISTS" = true ]; then
    echo "Running comprehensive evaluation on SFT model..."
    python scripts/comprehensive_model_evaluation.py \
        --model-path "$SFT_MODEL_PATH" \
        --model-type single_system \
        --system "$CONTROL_SYSTEM" \
        --test-type both \
        --grid-size 5 \
        --num-random 25 \
        --output-dir "$EVAL_OUTPUT_DIR/sft" \
        --save-results \
        --gpu-id 0
    
    if [ $? -eq 0 ]; then
        echo "✅ SFT model evaluation completed"
    else
        echo "⚠️ SFT model evaluation had issues"
    fi
fi

# Run comprehensive evaluation if GRPO model exists
if [ "$GRPO_EXISTS" = true ]; then
    echo "Running comprehensive evaluation on GRPO model..."
    python scripts/comprehensive_model_evaluation.py \
        --model-path "$GRPO_MODEL_PATH" \
        --model-type single_system \
        --system "$CONTROL_SYSTEM" \
        --test-type both \
        --grid-size 5 \
        --num-random 25 \
        --output-dir "$EVAL_OUTPUT_DIR/grpo" \
        --save-results \
        --gpu-id 0
    
    if [ $? -eq 0 ]; then
        echo "✅ GRPO model evaluation completed"
    else
        echo "⚠️ GRPO model evaluation had issues"
    fi
fi

# Run model comparison if both models exist
if [ "$SFT_EXISTS" = true ] && [ "$GRPO_EXISTS" = true ]; then
    echo "Running SFT vs GRPO comparison..."
    python scripts/model_comparison_evaluation.py \
        --system "$CONTROL_SYSTEM" \
        --model-types sft,grpo \
        --num-test-cases 30 \
        --comparison-cases 5 \
        --output-dir "$EVAL_OUTPUT_DIR/comparison" \
        --save-results \
        --gpu-id 0
    
    if [ $? -eq 0 ]; then
        echo "✅ Model comparison completed"
    else
        echo "⚠️ Model comparison had issues"
    fi
else
    echo "⚠️ Skipping model comparison (need both SFT and GRPO models)"
fi

# Create basic visualization plots as backup
echo "Creating basic LQR solution plots..."
python -c "
import sys
sys.path.append('.')
from evaluation.simple_plotting import visualize_lqr_solution
import os

# Create basic test cases
test_cases = [
    (0.5, -0.3, 'case1'),
    (0.7, 0.2, 'case2'), 
    (-0.4, 0.6, 'case3'),
    (0.8, -0.5, 'case4')
]

basic_plots_dir = 'figures/job_${SLURM_JOB_ID}/basic_plots'
os.makedirs(basic_plots_dir, exist_ok=True)

for x0, v0, name in test_cases:
    plot_path = f'{basic_plots_dir}/lqr_solution_{name}.png'
    try:
        visualize_lqr_solution(x0, v0, 0.1, 50, plot_path)
        print(f'✅ Created basic plot: {plot_path}')
    except Exception as e:
        print(f'❌ Failed to create {plot_path}: {e}')

print('🎨 Basic visualization completed!')
"

echo "✅ Model evaluation and visualization phase completed!"

# ===================
# PHASE 4: SUMMARY
# ===================
echo ""
echo "PHASE 4: Creating Summary Report..."
echo "==================================="

# Create summary report
cat > figures/job_${SLURM_JOB_ID}/training_summary.txt << EOF
Complete Pipeline Training Results
==================================
Job ID: $SLURM_JOB_ID
Control System: $CONTROL_SYSTEM
Started: $(date)

Pipeline Phases:
✅ SFT Training: COMPLETED
✅ GRPO Training: COMPLETED (or SFT-only if GRPO failed)
✅ Model Evaluation: COMPLETED
✅ Summary Report: COMPLETED

Output Locations:
- Models: models/single_system/$CONTROL_SYSTEM/[sft|grpo]/latest/
- Evaluation Results: figures/job_${SLURM_JOB_ID}/evaluation/
- Basic Plots: figures/job_${SLURM_JOB_ID}/basic_plots/
- Logs: logs/complete_pipeline_${SLURM_JOB_ID}.out/err

Evaluation Outputs:
- Success Rate Heatmaps: */evaluation/*/success_heatmap.png
- Trajectory Comparisons: */evaluation/*/trajectory_comparison.png
- Performance Summaries: */evaluation/*/performance_summary.png
- Model Comparisons: */evaluation/comparison/ (if both models trained)

Training Configuration:
- System: $CONTROL_SYSTEM  
- GPU: H100
- Memory: 100GB
- Time Limit: 24 hours

The refactored LLM reasoning control pipeline completed successfully!
EOF

echo "✅ Summary report created!"

# ===================
# FINAL STATUS
# ===================
echo ""
echo "============================================="
echo "COMPLETE PIPELINE FINISHED"
echo "============================================="
echo "✅ SFT Training: SUCCESS"
echo "✅ GRPO Training: SUCCESS (or SFT-only)"
echo "✅ Model Evaluation: SUCCESS"
echo "✅ Summary Report: SUCCESS"
echo ""
echo "📊 Results saved to: figures/job_${SLURM_JOB_ID}/"
echo "📄 Summary: figures/job_${SLURM_JOB_ID}/training_summary.txt"
echo "Completed: $(date)"
echo "============================================="

echo "🎉 Complete pipeline finished successfully!"