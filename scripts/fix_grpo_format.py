#!/usr/bin/env python3
"""
Fix GRPO model format adherence with stronger format rewards.
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
    print("🔧 Fixing GRPO Model Format Adherence")
    print("=" * 50)
    
    # Load the GRPO-trained model
    print("🚀 Loading GRPO-trained model...")
    
    from unsloth import FastLanguageModel
    
    # Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-4B-Base",
        max_seq_length=2048,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=32,
        gpu_memory_utilization=0.7,
    )
    
    # Apply LoRA
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
    
    # Load the GRPO weights
    print("📂 Loading GRPO adapter weights...")
    try:
        import safetensors.torch as st
        grpo_weights = st.load_file("models/working_notebook/grpo_fixed_model/adapter_model.safetensors")
        
        for name, param in model.named_parameters():
            if name in grpo_weights:
                param.data = grpo_weights[name].to(param.device, dtype=param.dtype)
        
        print("✅ GRPO weights loaded successfully")
    except Exception as e:
        print(f"❌ Could not load GRPO weights: {e}")
        return
    
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
    
    # Load smaller dataset for focused format training
    print("📂 Loading dataset for format-focused GRPO...")
    
    try:
        with open("datasets/di_train.pkl", "rb") as f:
            train_data = pickle.load(f)
        
        # Use smaller subset for focused training
        train_data = [x for x in train_data if x.get("system_type") == "double_integrator"]
        train_data = train_data[:100]  # Very small for focused format training
        
        print(f"✅ Dataset: {len(train_data)} samples (format-focused)")
        
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return
    
    # Format-focused GRPO training
    print("\n🎯 Setting up format-focused GRPO training...")
    
    from vllm import SamplingParams
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset
    
    # Format data for GRPO
    def format_for_grpo(data):
        formatted = []
        for example in data:
            messages = example["messages"]
            prompt_messages = messages[:-1]
            
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
    
    grpo_train_data = format_for_grpo(train_data)
    grpo_train_dataset = Dataset.from_list(grpo_train_data)
    
    print(f"Format-focused GRPO dataset: {len(grpo_train_dataset)} samples")
    
    # Very conservative vLLM settings for format focus
    vllm_sampling_params = SamplingParams(
        min_p=0.1,
        top_p=0.8,
        top_k=40,
        temperature=0.5,  # Lower temperature for more consistent format
        seed=3407,
        stop=[tokenizer.eos_token],
        include_stop_str_in_output=True,
        max_tokens=512,  # Shorter responses to encourage format compliance
    )
    
    # Format-focused GRPO configuration
    grpo_config = GRPOConfig(
        vllm_sampling_params=vllm_sampling_params,
        temperature=0.5,  # Lower temperature
        learning_rate=5e-7,  # Very conservative learning rate
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=2,
        max_completion_length=512,  # Shorter completions
        max_steps=15,  # Short focused training
        save_steps=5,
        report_to="none",
        output_dir="./grpo_format_fix",
    )
    
    print(f"Format-focused GRPO config:")
    print(f"  Max tokens: {vllm_sampling_params.max_tokens}")
    print(f"  Temperature: {grpo_config.temperature}")
    print(f"  Max steps: {grpo_config.max_steps}")
    
    # Strict format reward functions
    def strict_format_reward(completions, **kwargs):
        """Very strict format checking with high penalties/rewards."""
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            
            # Must have both reasoning and control tags
            has_reasoning = reasoning_start in response and reasoning_end in response
            has_controls = solution_start in response and solution_end in response
            
            if has_reasoning and has_controls:
                score += 10.0  # High reward for correct format
            elif has_controls:
                score += 5.0   # Partial reward for controls only
            else:
                score -= 5.0   # Penalty for missing format
            
            # Penalty for excessive length (encourage conciseness)
            if len(response) > 1000:
                score -= 2.0
            
            scores.append(score)
        return scores
    
    def strict_control_parsing_reward(prompts, completions, answer, **kwargs):
        """Reward for parseable control sequences."""
        scores = []
        
        for completion, true_answer in zip(completions, answer):
            score = 0
            response = completion[0]["content"]
            
            # Extract control sequence
            control_match = re.search(rf"{solution_start}(.*?){solution_end}", response, re.DOTALL)
            if control_match is None:
                scores.append(-3.0)  # High penalty for missing controls
                continue
                
            try:
                # Parse control values
                control_text = control_match.group(1).strip()
                control_values = [float(x.strip()) for x in control_text.split(',')]
                
                # High reward for correct number of controls
                if len(control_values) == steps:
                    score += 5.0
                else:
                    score -= 2.0
                    
                # Reward for valid control constraints
                if all(-3 <= u <= 3 for u in control_values):
                    score += 3.0
                else:
                    score -= 2.0
                
                # Reward for control diversity (not constant)
                if len(set(control_values)) > 1:
                    score += 2.0
                
                scores.append(score)
                
            except Exception:
                scores.append(-3.0)  # High penalty for unparseable controls
                
        return scores
    
    # Use strict format-focused rewards
    reward_functions = [
        strict_format_reward,
        strict_control_parsing_reward,
    ]
    
    print(f"✅ {len(reward_functions)} strict format reward functions ready")
    
    # Create format-focused GRPO trainer
    print("🚀 Creating format-focused GRPO trainer...")
    
    try:
        grpo_trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=reward_functions,
            args=grpo_config,
            train_dataset=grpo_train_dataset,
        )
        
        print("✅ Format-focused GRPO trainer created successfully")
        
        print(f"🚀 Starting format-focused GRPO training ({grpo_config.max_steps} steps)...")
        
        # Run format-focused GRPO training
        grpo_result = grpo_trainer.train()
        
        print("\n" + "="*50)
        print("🎉 FORMAT-FOCUSED GRPO COMPLETED!")
        print("="*50)
        print(f"✅ Training completed successfully")
        
        # Save the format-improved model
        format_save_path = "models/working_notebook/grpo_format_fixed_model"
        model.save_lora(format_save_path)
        print(f"💾 Format-fixed GRPO model saved to: {format_save_path}")
        
        # Test the format-improved model
        print("\n🧪 Testing format-improved model...")
        
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
        
        print(f"\n📝 Format-Improved Model Response:")
        print("="*60)
        print(output)
        print("="*60)
        
        # Detailed format analysis
        has_reasoning = reasoning_start in output and reasoning_end in output
        has_controls = solution_start in output and solution_end in output
        
        print(f"\n📊 Format Analysis:")
        print(f"   ✅ Has reasoning tags: {has_reasoning}")
        print(f"   ✅ Has control tags: {has_controls}")
        print(f"   📏 Response length: {len(output)} characters")
        
        if has_controls:
            control_match = re.search(rf"{solution_start}(.*?){solution_end}", output, re.DOTALL)
            if control_match:
                try:
                    control_text = control_match.group(1).strip()
                    control_values = [float(x.strip()) for x in control_text.split(',')]
                    control_variety = len(set(control_values))
                    
                    print(f"   📊 Control values parsed: {len(control_values)}")
                    print(f"   📊 Control variety: {control_variety} unique values")
                    print(f"   📊 Control range: [{min(control_values):.3f}, {max(control_values):.3f}]")
                    
                    if len(control_values) == steps:
                        print("   ✅ Correct number of controls!")
                    if control_variety > 1:
                        print("   ✅ Diverse control values!")
                    if all(-3 <= u <= 3 for u in control_values):
                        print("   ✅ Control constraints satisfied!")
                        
                except Exception as e:
                    print(f"   ❌ Control parsing failed: {e}")
        
        success = has_reasoning and has_controls
        print(f"\n🎯 FORMAT SUCCESS: {success}")
        
        return success
        
    except Exception as e:
        print(f"❌ Format-focused GRPO training failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Format fix successful! Model now adheres to required format.")
    else:
        print("\n⚠️  Format fix needs more work.")