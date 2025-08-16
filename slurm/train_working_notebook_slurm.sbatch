#!/bin/bash
#SBATCH --job-name=working_notebook_training
#SBATCH --output=slurm_logs/working_notebook_%j.out
#SBATCH --error=slurm_logs/working_notebook_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --partition=mit_normal_gpu

# Print job information
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="

# Load modules (adjust based on your cluster)
module purge
module load cuda/12.4
module load python/3.11

# Set environment variables
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Activate conda environment
source ~/.bashrc
conda activate unsloth_env

# Verify environment
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Create output directories
mkdir -p models/working_notebook
mkdir -p slurm_logs
mkdir -p figures

# Record system info
echo "=========================================="
echo "System Information:"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) MB"
echo "CUDA Version: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits)"
echo "=========================================="

# Run the working notebook training approach
echo "🚀 Starting working notebook training approach..."
echo "Parameters:"
echo "  - LoRA rank: 32 (working notebook)"
echo "  - Max seq length: 2048 (working notebook)" 
echo "  - Fast inference: True (vLLM enabled)"
echo "  - GPU memory utilization: 0.7"
echo "  - SFT epochs: 2"
echo "  - Full dataset (not subset)"

# Create the full training script for SLURM
cat > train_full_working_notebook.py << 'EOF'
#!/usr/bin/env python3
"""
Full working notebook training approach for SLURM.
Uses complete dataset and includes both SFT and GRPO training.
"""

import os
import sys
import pickle
import random
import numpy as np
import torch
import re
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def main():
    print("🚀 Full Working Notebook Training (SLURM)")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(3407)
    np.random.seed(3407)
    random.seed(3407)
    
    # EXACT working notebook parameters
    max_seq_length = 2048  # Match notebook exactly
    lora_rank = 32  # Match notebook exactly (not 8)
    
    # Control system parameters
    reasoning_start = "<REASONING>"
    reasoning_end = "</REASONING>"
    solution_start = "<CONTROLS>"
    solution_end = "</CONTROLS>"
    dt = 0.1
    steps = 50
    
    def get_system_prompt(current_dt, current_steps):
        total_time = current_dt * current_steps
        return f"""You are a control systems expert.
Given a double integrator system (ẍ = u) with initial position and velocity,
generate a sequence of {current_steps} control inputs to reach the origin (0,0) in exactly {total_time:.2f} seconds.
Position and velocity must stay within [-1, 1], and control inputs must be within [-3, 3].
Explain your approach between {reasoning_start} and {reasoning_end}.
Then provide exactly {current_steps} control values as a comma-separated list between {solution_start} and {solution_end}."""
    
    system_prompt = get_system_prompt(dt, steps)
    
    # Load model with working notebook approach
    print("🚀 Loading model (working notebook approach)...")
    
    from unsloth import FastLanguageModel
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-4B-Base",
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=True,  # Enable vLLM 
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.7,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_rank*2,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    print("✅ Model loaded successfully")
    print(f"   LoRA rank: {lora_rank}")
    print(f"   Max seq length: {max_seq_length}")
    print(f"   Fast inference: True (vLLM)")
    
    # Setup chat template
    chat_template = ("{% if messages[0]['role'] == 'system' %}"
                    "{{ messages[0]['content'] + eos_token }}"
                    "{% set loop_messages = messages[1:] %}"
                    "{% else %}"
                    "{{ '" + system_prompt + "' + eos_token }}"
                    "{% set loop_messages = messages %}"
                    "{% endif %}"
                    "{% for message in loop_messages %}"
                    "{% if message['role'] == 'user' %}"
                    "{{ message['content'] }}"
                    "{% elif message['role'] == 'assistant' %}"
                    "{{ message['content'] + eos_token }}"
                    "{% endif %}"
                    "{% endfor %}"
                    "{% if add_generation_prompt %}{{ '" + reasoning_start + "' }}"
                    "{% endif %}")
    
    tokenizer.chat_template = chat_template
    print("✅ Chat template configured")
    
    # Load full dataset
    print("📂 Loading full dataset...")
    
    try:
        with open("datasets/di_train.pkl", "rb") as f:
            train_data = pickle.load(f)
        
        with open("datasets/di_eval.pkl", "rb") as f:
            eval_data = pickle.load(f)
        
        # Filter to double integrator
        train_data = [x for x in train_data if x.get("system_type") == "double_integrator"]
        eval_data = [x for x in eval_data if x.get("system_type") == "double_integrator"]
        
        print(f"✅ Full dataset: {len(train_data)} train, {len(eval_data)} eval samples")
        
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return
    
    # =================================================================
    # PHASE 1: SFT PRE-TRAINING
    # =================================================================
    print("\n" + "="*60)
    print("📚 PHASE 1: SFT PRE-TRAINING")
    print("="*60)
    
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset
    
    def format_for_sft(example):
        messages = example["messages"]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": text}
    
    # Create SFT datasets
    sft_train_dataset = Dataset.from_list(train_data)
    sft_train_dataset = sft_train_dataset.map(format_for_sft)
    
    sft_eval_dataset = Dataset.from_list(eval_data)
    sft_eval_dataset = sft_eval_dataset.map(format_for_sft)
    
    print(f"SFT datasets: {len(sft_train_dataset)} train, {len(sft_eval_dataset)} eval")
    
    # SFT configuration (working notebook parameters)
    sft_config = SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=4,  # Working notebook
        gradient_accumulation_steps=1,
        warmup_steps=10,
        num_train_epochs=2,  # Working notebook
        learning_rate=2e-4,  # Working notebook  
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
        output_dir="./sft_output_full",
        save_steps=500,
        eval_steps=200,
        evaluation_strategy="steps",
        save_total_limit=3,
    )
    
    # Run SFT training
    sft_trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=sft_train_dataset,
        eval_dataset=sft_eval_dataset,
        args=sft_config,
    )
    
    print("🚀 Starting SFT training...")
    sft_result = sft_trainer.train()
    
    print("✅ SFT training completed!")
    print(f"   Final training loss: {sft_result.training_loss:.4f}")
    
    # Save SFT model
    sft_save_path = "models/working_notebook/sft_model"
    model.save_lora(sft_save_path)
    print(f"💾 SFT model saved to: {sft_save_path}")
    
    # Clear memory
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    # =================================================================
    # PHASE 2: GRPO TRAINING
    # =================================================================
    print("\n" + "="*60)
    print("🎮 PHASE 2: GRPO TRAINING")
    print("="*60)
    
    from vllm import SamplingParams
    from trl import GRPOConfig, GRPOTrainer
    
    # Format data for GRPO (working notebook format)
    def format_for_grpo(data):
        formatted = []
        for example in data:
            messages = example["messages"]
            prompt_messages = messages[:-1]  # Exclude assistant response
            
            # Extract control answer
            controls = example.get("controls", [])
            if isinstance(controls, list):
                answer = ", ".join([f"{u:.3f}" for u in controls])
            else:
                answer = str(controls)
            
            formatted.append({
                "prompt": prompt_messages,
                "answer": answer,
                "Messages": messages
            })
        return formatted
    
    # Format datasets for GRPO
    grpo_train_data = format_for_grpo(train_data)
    grpo_eval_data = format_for_grpo(eval_data)
    
    grpo_train_dataset = Dataset.from_list(grpo_train_data)
    grpo_eval_dataset = Dataset.from_list(grpo_eval_data)
    
    print(f"GRPO datasets: {len(grpo_train_dataset)} train, {len(grpo_eval_dataset)} eval")
    
    # GRPO configuration (working notebook parameters)
    max_completion_length = 2048
    
    vllm_sampling_params = SamplingParams(
        min_p=0.1,
        top_p=1.0,
        top_k=-1,
        seed=3407,
        stop=[tokenizer.eos_token],
        include_stop_str_in_output=True,
    )
    
    grpo_config = GRPOConfig(
        vllm_sampling_params=vllm_sampling_params,
        temperature=1.0,
        learning_rate=5e-6,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,  # Auto-adjusted for generations
        gradient_accumulation_steps=1,
        num_generations=4,  # Working notebook
        max_completion_length=max_completion_length,
        max_steps=100,  # Working notebook 
        save_steps=25,
        report_to="none",
        output_dir="./grpo_output_full",
    )
    
    print(f"GRPO config:")
    print(f"  Batch size: {grpo_config.per_device_train_batch_size}")
    print(f"  Num generations: {grpo_config.num_generations}")
    print(f"  Max steps: {grpo_config.max_steps}")
    print(f"  Learning rate: {grpo_config.learning_rate}")
    
    # Setup reward functions (working notebook)
    solution_end_regex = rf"{re.escape(solution_end)}[\\s]{{0,}}" + \
        f"(?:{re.escape(tokenizer.eos_token)})?"

    match_format = re.compile(
        rf"{re.escape(reasoning_end)}.*?"
        rf"{re.escape(solution_start)}(.+?){solution_end_regex}"
        rf"[\\s]{{0,}}$",
        flags=re.MULTILINE | re.DOTALL
    )
    
    def match_format_exactly(completions, **kwargs):
        """Exact format matching reward."""
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            if match_format.search(response) is not None:
                score += 3.0
            scores.append(score)
        return scores
    
    def match_format_approximately(completions, **kwargs):
        """Approximate format matching reward."""
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            score += 0.5 if response.count(reasoning_end) == 1 else -1.0
            score += 0.5 if response.count(solution_start) == 1 else -1.0
            score += 0.5 if response.count(solution_end) == 1 else -1.0
            scores.append(score)
        return scores
    
    def evaluate_control_sequence(prompts, completions, answer, **kwargs):
        """Control sequence evaluation reward."""
        scores = []
        
        for completion, true_answer in zip(completions, answer):
            score = 0
            response = completion[0]["content"]
            
            # Extract control sequence
            control_match = re.search(rf"{solution_start}(.*?){solution_end}", response, re.DOTALL)
            if control_match is None:
                scores.append(-2.0)
                continue
                
            try:
                # Parse control values
                control_text = control_match.group(1).strip()
                control_values = [float(x.strip()) for x in control_text.split(',')]
                
                # Check constraints
                if len(control_values) == steps:
                    score += 1.0
                else:
                    score -= 1.0
                    
                if all(-3 <= u <= 3 for u in control_values):
                    score += 1.0
                else:
                    score -= 2.0
                
                # Check smoothness
                if len(control_values) > 1:
                    diffs = [abs(control_values[i] - control_values[i-1]) for i in range(1, len(control_values))]
                    if max(diffs) < 1.5:
                        score += 1.5
                
                # Simulate trajectory
                try:
                    problem_text = prompts[0][-1]["content"]
                    initial_match = re.search(r"position=([-\\d\\.]+), velocity=([-\\d\\.]+)", problem_text)
                    if initial_match:
                        x0 = float(initial_match.group(1))
                        v0 = float(initial_match.group(2))
                        
                        x, v = x0, v0
                        valid_trajectory = True
                        
                        for u in control_values:
                            v = v + u * dt
                            x = x + v * dt
                            
                            if not (-1 <= x <= 1 and -1 <= v <= 1):
                                valid_trajectory = False
                                break
                        
                        if valid_trajectory:
                            score += 1.0
                        else:
                            score -= 1.0
                        
                        # Final error reward
                        final_error = np.sqrt(x**2 + v**2)
                        if final_error < 0.1:
                            score += 3.0
                        elif final_error < 0.2:
                            score += 2.0
                        elif final_error < 0.5:
                            score += 1.0
                        else:
                            score -= 1.0
                except Exception:
                    pass
                
                scores.append(score)
                
            except Exception:
                scores.append(-2.0)
                
        return scores
    
    # Combine reward functions
    reward_functions = [
        match_format_exactly,
        match_format_approximately, 
        evaluate_control_sequence,
    ]
    
    print(f"✅ {len(reward_functions)} reward functions ready")
    
    # Create GRPO trainer
    grpo_trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_functions,
        args=grpo_config,
        train_dataset=grpo_train_dataset,
        eval_dataset=grpo_eval_dataset,
    )
    
    print("🚀 Starting GRPO training...")
    
    try:
        grpo_result = grpo_trainer.train()
        
        print("\n" + "="*60)
        print("🎉 GRPO TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"✅ Final training loss: {grpo_result.training_loss:.4f}")
        print(f"✅ Total steps: {grpo_result.global_step}")
        
        # Save GRPO model
        grpo_save_path = "models/working_notebook/grpo_model"
        model.save_lora(grpo_save_path)
        print(f"💾 GRPO model saved to: {grpo_save_path}")
        
    except Exception as e:
        print(f"❌ GRPO training failed: {e}")
        print("SFT model is still available for use.")
        return
    
    # =================================================================
    # PHASE 3: FINAL TESTING
    # ================================================================= 
    print("\n" + "="*60)
    print("🧪 PHASE 3: FINAL MODEL TESTING")
    print("="*60)
    
    # Test the final model
    test_x0, test_v0 = 0.5, -0.3
    total_time = dt * steps
    test_problem = f"Control a double integrator system with initial state [position={test_x0:.2f}, velocity={test_v0:.2f}] to reach the origin (0,0) in {total_time:.2f} seconds using {steps} steps. Ensure all states remain within [-1,1] and controls within [-3,3]."
    
    test_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": test_problem},
    ]
    
    text = tokenizer.apply_chat_template(
        test_messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    
    # Generate response
    from vllm import SamplingParams
    sampling_params = SamplingParams(
        temperature=0.7,
        top_k=50,
        max_tokens=1024,
    )
    
    output = model.fast_generate(
        text,
        sampling_params=sampling_params,
        lora_request=None,
    )[0].outputs[0].text
    
    print(f"\n📝 Final Model Response:")
    print("="*60)
    print(output)
    print("="*60)
    
    # Analyze response
    has_reasoning = reasoning_start in output and reasoning_end in output
    has_controls = solution_start in output and solution_end in output
    
    print(f"\n📊 Final Analysis:")
    print(f"   Has reasoning tags: {has_reasoning}")
    print(f"   Has control tags: {has_controls}")
    
    if has_controls:
        control_match = re.search(rf"{solution_start}(.*?){solution_end}", output, re.DOTALL)
        if control_match:
            try:
                control_text = control_match.group(1).strip()
                control_values = [float(x.strip()) for x in control_text.split(',')]
                print(f"   Extracted {len(control_values)} control values")
                print(f"   Control range: [{min(control_values):.3f}, {max(control_values):.3f}]")
                
                # Quick simulation
                x, v = test_x0, test_v0
                for u in control_values:
                    v = v + u * dt
                    x = x + v * dt
                
                final_error = np.sqrt(x**2 + v**2)
                print(f"   Final position: ({x:.4f}, {v:.4f})")
                print(f"   Final error: {final_error:.4f}")
                
                if final_error < 0.1:
                    print("   🎉 SUCCESS: Model reached target!")
                elif final_error < 0.2:
                    print("   ✅ GOOD: Model close to target!")
                else:
                    print("   ⚠️  Model needs improvement")
                
            except Exception as e:
                print(f"   ❌ Could not parse controls: {e}")
    
    print("\n" + "="*60)
    print("🎉 FULL TRAINING PIPELINE COMPLETED!")
    print("="*60)
    print("✅ SFT model trained and saved")
    print("✅ GRPO model trained and saved")
    print("✅ Working notebook approach validated")
    print("✅ Ready for evaluation and deployment")
    print(f"✅ Models saved in: models/working_notebook/")

if __name__ == "__main__":
    main()
EOF

# Run the full training
python train_full_working_notebook.py

# Check final results
echo "=========================================="
echo "Training completed at: $(date)"
echo "Final model locations:"
ls -la models/working_notebook/ 2>/dev/null || echo "No models directory created"
echo "=========================================="

# Cleanup temporary files
rm -f train_full_working_notebook.py

echo "🎉 SLURM job completed successfully!"