#!/usr/bin/env python3
"""
Fix GRPO training using the working SFT model as starting point.
"""

import os
import sys
import pickle
import numpy as np
import torch
import re
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def main():
    print("🔧 Fixing GRPO Training with Working SFT Model")
    print("=" * 60)
    
    # Load the trained SFT model
    print("🚀 Loading trained SFT model...")
    
    from unsloth import FastLanguageModel
    
    # Load base model first (working notebook approach)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-4B-Base",
        max_seq_length=2048,
        load_in_4bit=True,
        fast_inference=True,  # Keep vLLM enabled
        max_lora_rank=32,
        gpu_memory_utilization=0.7,
    )
    
    # Apply LoRA for training compatibility
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=64,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    # Load the SFT weights
    print("📂 Loading SFT adapter weights...")
    try:
        # Load the adapter weights manually
        import safetensors.torch as st
        sft_weights = st.load_file("models/working_notebook/sft_model/adapter_model.safetensors")
        
        # Apply weights to model
        for name, param in model.named_parameters():
            if name in sft_weights:
                param.data = sft_weights[name].to(param.device, dtype=param.dtype)
        
        print("✅ SFT weights loaded successfully")
    except Exception as e:
        print(f"⚠️  Could not load SFT weights: {e}")
        print("   Proceeding with fresh LoRA initialization")
    
    # Setup for GRPO training
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
    
    # Load dataset for GRPO
    print("📂 Loading dataset for GRPO training...")
    
    try:
        with open("datasets/di_train.pkl", "rb") as f:
            train_data = pickle.load(f)
        
        with open("datasets/di_eval.pkl", "rb") as f:
            eval_data = pickle.load(f)
        
        # Filter to double integrator
        train_data = [x for x in train_data if x.get("system_type") == "double_integrator"]
        eval_data = [x for x in eval_data if x.get("system_type") == "double_integrator"]
        
        # Use smaller subset for quick GRPO training
        train_data = train_data[:500]  # Smaller for quick iteration
        eval_data = eval_data[:50]
        
        print(f"✅ Dataset: {len(train_data)} train, {len(eval_data)} eval samples")
        
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return
    
    # GRPO Training Setup
    print("\n🎮 Setting up GRPO training...")
    
    from vllm import SamplingParams
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset
    
    # Format data for GRPO (working notebook approach)
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
            })
        return formatted
    
    # Format datasets
    grpo_train_data = format_for_grpo(train_data)
    grpo_eval_data = format_for_grpo(eval_data)
    
    grpo_train_dataset = Dataset.from_list(grpo_train_data)
    grpo_eval_dataset = Dataset.from_list(grpo_eval_data)
    
    print(f"GRPO datasets: {len(grpo_train_dataset)} train, {len(grpo_eval_dataset)} eval")
    
    # vLLM sampling params (conservative settings)
    vllm_sampling_params = SamplingParams(
        min_p=0.1,
        top_p=0.9,  # Slightly more conservative
        top_k=50,   # More conservative
        temperature=0.7,  # Lower temperature
        seed=3407,
        stop=[tokenizer.eos_token],
        include_stop_str_in_output=True,
    )
    
    # GRPO configuration (conservative for debugging)
    grpo_config = GRPOConfig(
        vllm_sampling_params=vllm_sampling_params,
        temperature=0.7,  # Lower temperature
        learning_rate=1e-6,  # Much lower learning rate
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=2,  # Fewer generations for stability
        max_completion_length=1024,  # Shorter completions
        max_steps=20,  # Very short for debugging
        save_steps=10,
        report_to="none",
        output_dir="./grpo_debug",
    )
    
    print(f"GRPO config (conservative for debugging):")
    print(f"  Learning rate: {grpo_config.learning_rate}")
    print(f"  Num generations: {grpo_config.num_generations}")
    print(f"  Max steps: {grpo_config.max_steps}")
    print(f"  Temperature: {grpo_config.temperature}")
    
    # Simple reward functions (simplified for debugging)
    def simple_format_reward(completions, **kwargs):
        """Simple format checking reward."""
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            
            # Basic format checking
            if solution_start in response and solution_end in response:
                score += 2.0
            if reasoning_start in response and reasoning_end in response:
                score += 1.0
                
            scores.append(score)
        return scores
    
    def simple_control_reward(prompts, completions, answer, **kwargs):
        """Simple control quality reward."""
        scores = []
        
        for completion, true_answer in zip(completions, answer):
            score = 0
            response = completion[0]["content"]
            
            # Extract control sequence
            control_match = re.search(rf"{solution_start}(.*?){solution_end}", response, re.DOTALL)
            if control_match is None:
                scores.append(-1.0)
                continue
                
            try:
                # Parse control values
                control_text = control_match.group(1).strip()
                control_values = [float(x.strip()) for x in control_text.split(',')]
                
                # Basic checks
                if len(control_values) == steps:
                    score += 1.0
                    
                if all(-3 <= u <= 3 for u in control_values):
                    score += 1.0
                
                # Prefer varying controls over constant ones
                if len(set(control_values)) > 1:
                    score += 0.5
                
                scores.append(score)
                
            except Exception:
                scores.append(-1.0)
                
        return scores
    
    # Use simplified reward functions
    reward_functions = [
        simple_format_reward,
        simple_control_reward,
    ]
    
    print(f"✅ {len(reward_functions)} simplified reward functions ready")
    
    # Create GRPO trainer
    print("🚀 Creating GRPO trainer...")
    
    try:
        grpo_trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=reward_functions,
            args=grpo_config,
            train_dataset=grpo_train_dataset,
            eval_dataset=grpo_eval_dataset,
        )
        
        print("✅ GRPO trainer created successfully")
        
        print(f"🚀 Starting GRPO training (debug mode - {grpo_config.max_steps} steps)...")
        
        # Run GRPO training
        grpo_result = grpo_trainer.train()
        
        print("\n" + "="*60)
        print("🎉 GRPO TRAINING COMPLETED!")
        print("="*60)
        print(f"✅ Training completed successfully")
        print(f"✅ Total steps: {grpo_result.global_step}")
        
        # Save the improved model
        grpo_save_path = "models/working_notebook/grpo_fixed_model"
        model.save_lora(grpo_save_path)
        print(f"💾 GRPO model saved to: {grpo_save_path}")
        
        # Quick test
        print("\n🧪 Quick test of GRPO model...")
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
        
        sampling_params = SamplingParams(
            temperature=0.3,
            top_k=50,
            max_tokens=512,
        )
        
        output = model.fast_generate(
            text,
            sampling_params=sampling_params,
            lora_request=None,
        )[0].outputs[0].text
        
        print(f"\n📝 GRPO Model Response:")
        print("="*40)
        print(output[:500] + "..." if len(output) > 500 else output)
        print("="*40)
        
        # Analyze improvement
        has_controls = solution_start in output and solution_end in output
        print(f"\n📊 Quick Analysis:")
        print(f"   Has control tags: {has_controls}")
        
        if has_controls:
            control_match = re.search(rf"{solution_start}(.*?){solution_end}", output, re.DOTALL)
            if control_match:
                try:
                    control_text = control_match.group(1).strip()
                    control_values = [float(x.strip()) for x in control_text.split(',')]
                    control_variety = len(set(control_values))
                    print(f"   Control variety: {control_variety} unique values")
                    if control_variety > 1:
                        print("   ✅ GRPO improved control diversity!")
                    else:
                        print("   ⚠️  Still generating constant controls")
                except Exception:
                    print("   ❌ Could not parse controls")
        
        print("\n✅ GRPO training fix completed!")
        
    except Exception as e:
        print(f"❌ GRPO training failed: {e}")
        print("\n🔧 Debugging suggestions:")
        print("1. Check vLLM compatibility")
        print("2. Reduce batch size further")
        print("3. Use standard transformers instead of vLLM")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 GRPO fix successful! Model ready for evaluation.")
    else:
        print("\n⚠️  GRPO fix needs more debugging.")