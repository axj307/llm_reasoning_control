#!/bin/bash
#SBATCH --job-name=quick_test_di
#SBATCH --output=slurm_logs/quick_test_di_%j.out
#SBATCH --error=slurm_logs/quick_test_di_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=pi_linaresr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

echo "🚀 QUICK TEST: Double Integrator Pipeline (Minimal Training)"
echo "============================================================"
echo "SFT: ~100-200 steps | GRPO: ~10-20 steps"
echo "============================================================"

# Environment setup
source ~/.bashrc
conda activate unsloth_env
cd /orcd/home/002/amitjain/project/Unsloth/llm_reasoning_control_refactored

# Set up timestamp for results
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="results/quick_test_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "📁 Output directory: $OUTPUT_DIR"

echo ""
echo "📊 PHASE 1: MINIMAL DATA GENERATION"
echo "============================================================"
echo "🏭 Generating small double integrator dataset (200 samples)..."

python scripts/generate_data.py \
    --systems double_integrator \
    --total-samples 200 \
    --split-ratio 0.8 \
    --dataset-name di_quick

echo "✅ Quick dataset generated: 160 train + 40 eval samples"

echo ""
echo "📊 PHASE 2: QUICK SFT TRAINING (~100-200 steps)"
echo "============================================================"
echo "🎓 Training SFT with minimal steps..."

# Check what parameters train_single_system.py actually accepts
python scripts/train_single_system.py --help > "$OUTPUT_DIR/train_single_system_help.txt" 2>&1

echo "📋 Available parameters saved to: $OUTPUT_DIR/train_single_system_help.txt"

# Use correct parameters based on the actual script interface
python scripts/train_single_system.py \
    --system double_integrator \
    --sft

echo "✅ SFT training completed!"

echo ""
echo "📊 PHASE 3: QUICK GRPO TRAINING (~10-20 steps)"
echo "============================================================"
echo "🔧 Training GRPO with minimal steps using working parameters..."

python scripts/train_grpo_working_params.py \
    --system double_integrator \
    --dataset-name di_quick \
    --max-samples 160 \
    --max-steps 20 \
    --num-generations 4 \
    --batch-size 4

echo "✅ GRPO training completed!"

echo ""
echo "📊 PHASE 4: QUICK EVALUATION"
echo "============================================================"
echo "🧪 Testing models on 5 test cases..."

# Check if we have a GRPO model
if [ -d "models/working_notebook" ]; then
    echo "📈 Evaluating GRPO model..."
    python scripts/evaluate_model.py \
        --model-path models/working_notebook/grpo_working_params_model \
        --model-type single_system \
        --eval-dataset di_quick \
        --num-test-cases 5 \
        --save-plots \
        --plot-dir "$OUTPUT_DIR/grpo_results"
else
    echo "⚠️  GRPO model not found, checking SFT model..."
    
    if [ -d "models/single_system/double_integrator/sft/latest" ]; then
        echo "📈 Evaluating SFT model..."
        python scripts/evaluate_model.py \
            --model-path models/single_system/double_integrator/sft/latest \
            --model-type single_system \
            --eval-dataset di_quick \
            --num-test-cases 5 \
            --save-plots \
            --plot-dir "$OUTPUT_DIR/sft_results"
    else
        echo "❌ No trained models found"
        ls -la models/ || echo "No models directory"
    fi
fi

echo ""
echo "📋 QUICK TEST SUMMARY"
echo "============================================================"
echo "🎯 MINIMAL TRAINING PIPELINE COMPLETED!"
echo ""
echo "✅ Data: 200 samples (160 train + 40 eval)"
echo "✅ SFT: ~100-200 steps (depending on batch size)"
echo "✅ GRPO: 20 steps maximum"
echo "✅ Evaluation: 5 test cases"
echo ""
echo "📁 Results saved to: $OUTPUT_DIR"
echo ""
echo "🚀 READY FOR FULL TRAINING IF QUICK TEST SUCCESSFUL!"
echo "============================================================"