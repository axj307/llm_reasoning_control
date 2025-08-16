#!/usr/bin/env python3
"""
Train SFT model with diverse dataset and diversity-focused approach.
Addresses the constant control generation issue.
"""

import torch
import numpy as np
import random
import pickle
from pathlib import Path
import sys
import os

sys.path.append('.')

def main():
    print("🚀 DIVERSE SFT TRAINING - FIXING CONTROL DIVERSITY")
    print("=" * 70)
    
    # Set seeds (exact from notebook)
    torch.manual_seed(3407)
    np.random.seed(3407)
    random.seed(3407)
    
    # GPU selection (exact from notebook)
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        chosen_gpu = random.randint(0, num_gpus - 1)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(chosen_gpu)
        print(f"Randomly selected GPU: {chosen_gpu}")
    else:
        print("No GPUs available.")
    
    # Model parameters (EXACT from working notebook)
    max_seq_length = 2048
    lora_rank = 32
    
    # Load model (EXACT from working notebook)
    from unsloth import FastLanguageModel
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-4B-Base",
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=True,  # Enable vLLM fast inference
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
    
    print("✅ Model loaded successfully (exact notebook approach)")
    print(f"   LoRA rank: {lora_rank}")
    print(f"   Max seq length: {max_seq_length}")
    print(f"   Fast inference: True (vLLM enabled)")
    
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
    
    # Chat template (EXACT from working notebook)
    chat_template = \
        "{% if messages[0]['role'] == 'system' %}"\
            "{{ messages[0]['content'] + eos_token }}"\
            "{% set loop_messages = messages[1:] %}"\
        "{% else %}"\
            "{{ '{system_prompt}' + eos_token }}"\
            "{% set loop_messages = messages %}"\
        "{% endif %}"\
        "{% for message in loop_messages %}"\
            "{% if message['role'] == 'user' %}"\
                "{{ message['content'] }}"\
            "{% elif message['role'] == 'assistant' %}"\
                "{{ message['content'] + eos_token }}"\
            "{% endif %}"\
        "{% endfor %}"\
        "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"\
        "{% endif %}"
    
    chat_template = chat_template\
        .replace("'{system_prompt}'", f"'{system_prompt}'")\
        .replace("'{reasoning_start}'", f"'{reasoning_start}'")
    tokenizer.chat_template = chat_template
    
    print("✅ Chat template configured")
    
    # Load DIVERSE dataset
    print("📂 Loading diverse dataset...")
    with open("datasets/di_diverse_train.pkl", "rb") as f:
        train_data = pickle.load(f)
    
    with open("datasets/di_diverse_eval.pkl", "rb") as f:
        eval_data = pickle.load(f)
    
    train_data = [x for x in train_data if x.get("system_type") == "double_integrator"]
    eval_data = [x for x in eval_data if x.get("system_type") == "double_integrator"]
    
    print(f"Diverse dataset: {len(train_data)} train, {len(eval_data)} eval")
    
    # Analyze dataset diversity
    train_controls = [item["controls"] for item in train_data]
    train_varieties = [len(set(controls)) for controls in train_controls]
    avg_variety = np.mean(train_varieties)
    
    print(f"📊 Dataset diversity: {avg_variety:.1f} unique controls per sequence (vs 1-2 in previous)")
    
    # SFT training with diversity focus
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
    
    sft_train_dataset = Dataset.from_list(train_data)
    sft_train_dataset = sft_train_dataset.map(format_for_sft)
    
    sft_eval_dataset = Dataset.from_list(eval_data)
    sft_eval_dataset = sft_eval_dataset.map(format_for_sft)
    
    print(f"SFT datasets: {len(sft_train_dataset)} train, {len(sft_eval_dataset)} eval")
    
    # Enhanced SFT configuration for diversity
    sft_config = SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=4,  # From working notebook
        gradient_accumulation_steps=1,
        warmup_steps=50,  # More warmup for diverse data
        num_train_epochs=4,  # Good balance
        learning_rate=8e-5,  # Slightly lower for diverse data
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",  # Cosine for smooth learning
        seed=3407,
        report_to="none",  # Disable wandb for now
        output_dir="./sft_diverse_output",
        save_steps=200,
        save_total_limit=3,
        max_seq_length=max_seq_length,  # Ensure consistency
    )
    
    # Create SFT trainer
    sft_trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=sft_train_dataset,
        eval_dataset=sft_eval_dataset,
        args=sft_config,
    )
    
    print("🚀 Starting diverse SFT training...")
    print(f"   Training epochs: {sft_config.num_train_epochs}")
    print(f"   Learning rate: {sft_config.learning_rate}")
    print(f"   Expected benefit: Diverse control generation")
    
    # Run SFT training
    sft_result = sft_trainer.train()
    
    print("✅ Diverse SFT training completed!")
    print(f"   Final training loss: {sft_result.training_loss:.4f}")
    
    # Save the diverse model
    diverse_save_path = "models/working_notebook/sft_diverse_model"
    model.save_lora(diverse_save_path)
    print(f"💾 Diverse SFT model saved to: {diverse_save_path}")
    
    # Test the model immediately
    print("🧪 Testing diverse SFT model...")
    
    test_cases = [
        (0.5, -0.3),   # Original problematic case
        (0.7, 0.2),    # Different case
        (-0.4, 0.6),   # Negative position
    ]
    
    from vllm import SamplingParams
    sampling_params = SamplingParams(
        temperature=0.3,
        top_k=50,
        max_tokens=1024,
    )
    
    print(f"\n📊 DIVERSITY TEST RESULTS:")
    print("-" * 50)
    
    for i, (x0, v0) in enumerate(test_cases):
        total_time = dt * steps
        test_problem = f"Control a double integrator system with initial state [position={x0:.2f}, velocity={v0:.2f}] to reach the origin (0,0) in {total_time:.2f} seconds using {steps} steps. Ensure all states remain within [-1,1] and controls within [-3,3]."
        
        test_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": test_problem},
        ]
        
        text = tokenizer.apply_chat_template(
            test_messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        
        try:
            output = model.fast_generate(
                text,
                sampling_params=sampling_params,
                lora_request=None,
            )[0].outputs[0].text
            
            # Extract controls for diversity analysis
            import re
            control_match = re.search(rf"{solution_start}(.*?){solution_end}", output, re.DOTALL)
            if control_match:
                control_text = control_match.group(1).strip()
                control_values = [float(x.strip()) for x in control_text.split(',')]
                
                if len(control_values) == steps:
                    control_variety = len(set(control_values))
                    control_range = f"[{min(control_values):.3f}, {max(control_values):.3f}]"
                    is_constant = control_variety <= 3  # Very low diversity
                    
                    print(f"Test {i+1} - Init: ({x0:.1f}, {v0:.1f})")
                    print(f"  Diversity: {control_variety}/50 unique values")
                    print(f"  Range: {control_range}")
                    print(f"  Status: {'❌ STILL CONSTANT' if is_constant else '✅ DIVERSE'}")
                    
                    # Show first few controls
                    first_controls = ", ".join([f"{u:.3f}" for u in control_values[:10]])
                    print(f"  First 10: {first_controls}...")
                else:
                    print(f"Test {i+1} - ❌ Wrong number of controls: {len(control_values)}")
            else:
                print(f"Test {i+1} - ❌ No controls found")
                
        except Exception as e:
            print(f"Test {i+1} - ❌ Generation failed: {e}")
        
        print()
    
    print(f"✅ Diverse SFT training and testing completed!")
    print(f"📈 Expected improvement: Model should now generate varied controls")
    print(f"🎯 Next step: If diversity improved, proceed to GRPO training")

if __name__ == "__main__":
    main()