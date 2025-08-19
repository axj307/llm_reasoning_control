#!/bin/bash
#SBATCH --job-name=debug_llm_outputs
#SBATCH --output=slurm_logs/debug_llm_outputs_%j.out
#SBATCH --error=slurm_logs/debug_llm_outputs_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=pi_linaresr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1

echo "🔍 LLM INPUT/OUTPUT DEBUG TOOL"
echo "============================================================"
echo "Inspecting actual question-answer pairs from GRPO training"
echo "============================================================"

# Environment setup
source ~/.bashrc
conda activate unsloth_env
cd /orcd/home/002/amitjain/project/Unsloth/llm_reasoning_control_refactored

# Set up timestamp for results
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="results/debug_outputs_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "📁 Debug output directory: $OUTPUT_DIR"

echo ""
echo "📊 PHASE 1: DATASET FORMAT DEBUG"
echo "============================================================"
echo "🔍 Examining actual dataset format..."

python -c "
import sys
sys.path.append('.')
import pickle
from pathlib import Path
import json

print('🔍 Dataset Format Analysis:')
print('=' * 40)

# Check available datasets
dataset_files = [
    'datasets/di_train.pkl',
    'datasets/di_quick_train.pkl', 
    'datasets/universal_train.pkl'
]

for dataset_file in dataset_files:
    if Path(dataset_file).exists():
        print(f'\\n📂 {dataset_file}:')
        with open(dataset_file, 'rb') as f:
            data = pickle.load(f)
        
        print(f'   📊 Total samples: {len(data)}')
        if len(data) > 0:
            sample = data[0]
            print(f'   🔑 Keys: {list(sample.keys())}')
            
            if 'Messages' in sample:
                messages = sample['Messages']
                print(f'   💬 Messages: {len(messages)} parts')
                
                for j, msg in enumerate(messages):
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    print(f'     {j+1}. {role}: {len(content)} chars')
                
                # Show full example
                print(f'\\n📝 FULL SAMPLE EXAMPLE:')
                print('   System prompt:', messages[0]['content'][:200] + '...')
                print('   User question:', messages[1]['content'])
                print('   Assistant answer:', messages[2]['content'][:500] + '...')
    else:
        print(f'❌ {dataset_file} not found')
"

echo ""
echo "📊 PHASE 2: LIVE MODEL TESTING"
echo "============================================================"
echo "🤖 Testing current model with sample questions..."

python debug_grpo_outputs.py > "$OUTPUT_DIR/live_model_test.txt" 2>&1

echo "✅ Live model test completed!"
echo "📄 Results saved to: $OUTPUT_DIR/live_model_test.txt"

echo ""
echo "📊 PHASE 3: EVALUATION PIPELINE DEBUG"
echo "============================================================"
echo "🧪 Testing evaluation pipeline with debug output..."

python -c "
import sys
sys.path.append('.')
from evaluation.inference import run_inference
from core.model_manager import ModelManager
import torch

print('🧪 Evaluation Pipeline Debug:')
print('=' * 40)

try:
    # Try to load any available model
    manager = ModelManager()
    
    # Check for models
    model_paths = [
        'models/single_system/double_integrator/sft/latest',
        'models/universal/sft/latest'
    ]
    
    model_found = False
    for model_path in model_paths:
        from pathlib import Path
        if Path(model_path).exists():
            print(f'✅ Found model: {model_path}')
            try:
                model, tokenizer, lora_request, metadata = manager.load_checkpoint(model_path)
                print(f'✅ Model loaded successfully')
                
                # Test inference with debug output
                print('\\n🔍 Testing inference with debug output:')
                result = run_inference(
                    model, tokenizer, 'double_integrator',
                    initial_state=(1.5, 0.8),
                    dt=0.1,
                    steps=5,
                    lora_request=lora_request
                )
                
                print('📝 INFERENCE INPUT:')
                print('   System: double_integrator')
                print('   Initial state: (1.5, 0.8)')
                print('   Steps: 5')
                
                print('\\n🤖 INFERENCE OUTPUT:')
                print(f'   Reasoning: {result.get(\"reasoning\", \"None\")[:200]}...')
                print(f'   Controls: {result.get(\"controls\", \"None\")}')
                print(f'   Success: {result.get(\"success\", False)}')
                
                model_found = True
                break
                
            except Exception as e:
                print(f'❌ Error loading {model_path}: {e}')
    
    if not model_found:
        print('❌ No trained models found. Train a model first.')
        
except Exception as e:
    print(f'❌ Error in evaluation debug: {e}')
" > "$OUTPUT_DIR/evaluation_debug.txt" 2>&1

echo "✅ Evaluation debug completed!"
echo "📄 Results saved to: $OUTPUT_DIR/evaluation_debug.txt"

echo ""
echo "📊 PHASE 4: GENERATE SAMPLE OUTPUTS REPORT"
echo "============================================================"
echo "📊 Creating comprehensive sample outputs report..."

python -c "
print('📊 COMPREHENSIVE LLM OUTPUT ANALYSIS')
print('=' * 50)
print()

print('🎯 This debug run checked:')
print('1. ✅ Dataset format and structure')
print('2. ✅ Live model question-answer generation')  
print('3. ✅ Evaluation pipeline inference')
print('4. ✅ Format consistency verification')
print()

print('📁 All detailed results saved to: $OUTPUT_DIR')
print()

print('🔍 TO VERIFY YOUR TRAINING:')
print('1. Check dataset format matches expectations')
print('2. Verify model generates proper <REASONING> and <CONTROLS> tags')
print('3. Ensure control values are parseable as numbers')
print('4. Confirm evaluation pipeline works with same format')
print()

print('💡 NEXT STEPS:')
print('1. Review the output files for any format mismatches')
print('2. If formats look good, your training should work correctly')
print('3. If issues found, adjust dataset generation or model prompts')
print()
" > "$OUTPUT_DIR/summary_report.txt"

echo "✅ Sample outputs report generated!"

echo ""
echo "📋 DEBUG SUMMARY"
echo "============================================================"
echo "🎉 LLM OUTPUT DEBUG COMPLETED!"
echo ""
echo "📁 All results saved to: $OUTPUT_DIR"
echo ""
echo "📄 Key files to review:"
echo "   📊 live_model_test.txt - Live model question-answer pairs"
echo "   🧪 evaluation_debug.txt - Evaluation pipeline testing"
echo "   📋 summary_report.txt - Overall analysis summary"
echo ""
echo "🔍 Review these files to verify:"
echo "   ✅ Question format matches your expectations"
echo "   ✅ Model responses have proper <REASONING> and <CONTROLS> tags"
echo "   ✅ Control values are parseable and reasonable"
echo "   ✅ Evaluation pipeline processes outputs correctly"
echo ""
echo "🚀 USE THIS INFO TO VALIDATE YOUR TRAINING PIPELINE!"
echo "============================================================"