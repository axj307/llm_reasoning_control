#!/usr/bin/env python3
"""
Training script using EXACT working notebook approach.
Based on: notebooks/Qwen3_(4B)-GRPO_control.ipynb
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
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config


def main():
    print("🚀 Training using EXACT working notebook approach")
    print("=" * 50)
    
    # Set random seeds for reproducibility (exact from notebook)
    torch.manual_seed(3407)
    np.random.seed(3407)
    random.seed(3407)
    
    # GPU selection (exact from notebook)
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        chosen_gpu = random.randint(0, num_gpus - 1)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(chosen_gpu)
        print(f"🖥️  Selected GPU: {chosen_gpu}")
    else:
        print("❌ No GPUs available.")
        return
    
    # EXACT working notebook parameters
    max_seq_length = 2048  # Match notebook exactly
    lora_rank = 32  # Match notebook exactly (not 8)
    
    # Control system parameters (exact from notebook)
    reasoning_start = "<REASONING>"
    reasoning_end = "</REASONING>"
    solution_start = "<CONTROLS>"
    solution_end = "</CONTROLS>"
    dt = 0.1  # Default time step duration
    steps = 50  # Default number of steps
    
    def get_system_prompt(current_dt, current_steps):
        total_time = current_dt * current_steps
        return f"""You are a control systems expert.
Given a double integrator system (ẍ = u) with initial position and velocity,
generate a sequence of {current_steps} control inputs to reach the origin (0,0) in exactly {total_time:.2f} seconds.
Position and velocity must stay within [-1, 1], and control inputs must be within [-3, 3].
Explain your approach between {reasoning_start} and {reasoning_end}.
Then provide exactly {current_steps} control values as a comma-separated list between {solution_start} and {solution_end}."""
    
    system_prompt = get_system_prompt(dt, steps)
    
    # Load model (EXACT from working notebook)
    print("🚀 Loading model (working notebook approach)...")
    
    from unsloth import FastLanguageModel
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-4B-Base",
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=True,  # Enable vLLM fast inference (KEY DIFFERENCE!)
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.7,  # Reduce if out of memory (KEY DIFFERENCE!)
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,  # Higher rank than current pipeline
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_rank*2,  # *2 speeds up training
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    print("✅ Model loaded successfully (working notebook approach)")
    print(f"   Fast inference: True (vLLM enabled)")
    print(f"   LoRA rank: {lora_rank}")
    print(f"   Max seq length: {max_seq_length}")
    
    # Setup chat template (EXACT from notebook)
    print("🔧 Setting up chat template...")
    
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
    
    # Load dataset
    print("📂 Loading dataset...")
    
    try:
        with open("datasets/di_train.pkl", "rb") as f:
            train_data = pickle.load(f)
        
        with open("datasets/di_eval.pkl", "rb") as f:
            eval_data = pickle.load(f)
        
        print(f"✅ Loaded {len(train_data)} train and {len(eval_data)} eval samples")
        
        # Filter to double integrator
        train_data = [x for x in train_data if x.get("system_type") == "double_integrator"]
        eval_data = [x for x in eval_data if x.get("system_type") == "double_integrator"]
        
        print(f"   Filtered: {len(train_data)} train, {len(eval_data)} eval for double_integrator")
        
        # Use subset for quick demonstration (like clean notebook)
        train_data = train_data[:100]  # Quick demo
        eval_data = eval_data[:20]     # Quick demo
        print(f"   Using subset: {len(train_data)} train, {len(eval_data)} eval")
        
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return
    
    # Format dataset for SFT (exact from notebook)
    print("📚 Starting SFT pre-training...")
    
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
    
    # Create datasets
    sft_train_dataset = Dataset.from_list(train_data)
    sft_train_dataset = sft_train_dataset.map(format_for_sft)
    
    sft_eval_dataset = Dataset.from_list(eval_data)
    sft_eval_dataset = sft_eval_dataset.map(format_for_sft)
    
    print(f"   SFT datasets: {len(sft_train_dataset)} train, {len(sft_eval_dataset)} eval")
    
    # SFT configuration (exact from notebook)
    sft_config = SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=4,  # Match notebook
        gradient_accumulation_steps=1,
        warmup_steps=5,
        num_train_epochs=2,  # Match notebook
        learning_rate=2e-4,  # Match notebook
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",  # Change to "wandb" if desired
        output_dir="./sft_output_working",
        save_steps=1000,
    )
    
    # Create SFT trainer
    sft_trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=sft_train_dataset,
        eval_dataset=sft_eval_dataset,
        args=sft_config,
    )
    
    print("   Running SFT training...")
    sft_result = sft_trainer.train()
    
    print("✅ SFT pre-training completed!")
    print(f"   Final loss: {sft_result.training_loss:.4f}")
    
    # Clear memory
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    # Test the trained model first before GRPO
    print("🧪 Testing SFT model...")
    
    test_x0, test_v0 = 0.5, -0.3
    total_time = dt * steps
    test_problem = f"Control a double integrator system with initial state [position={test_x0:.2f}, velocity={test_v0:.2f}] to reach the origin (0,0) in {total_time:.2f} seconds using {steps} steps. Ensure all states remain within [-1,1] and controls within [-3,3]."
    
    test_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": test_problem},
    ]
    
    # Format for generation
    text = tokenizer.apply_chat_template(
        test_messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    
    print(f"   Prompt length: {len(text)} characters")
    
    # Generate response using vLLM (exact from notebook)
    from vllm import SamplingParams
    sampling_params = SamplingParams(
        temperature=0.7,
        top_k=50,
        max_tokens=1024,
    )
    
    print("   Generating SFT response using vLLM...")
    output = model.fast_generate(
        text,
        sampling_params=sampling_params,
        lora_request=None,
    )[0].outputs[0].text
    
    print(f"\n📝 SFT Model Response:")
    print("="*60)
    print(output)
    print("="*60)
    
    # Check format
    has_reasoning = reasoning_start in output and reasoning_end in output
    has_controls = solution_start in output and solution_end in output
    
    print(f"\n📊 SFT Response Analysis:")
    print(f"   Has reasoning tags: {has_reasoning}")
    print(f"   Has control tags: {has_controls}")
    
    print("\n✅ SFT training and testing completed using working notebook approach!")
    print("\n🎯 Key success factors demonstrated:")
    print("   ✅ fast_inference=True (vLLM enabled)")
    print("   ✅ gpu_memory_utilization=0.7")
    print("   ✅ Higher LoRA rank (32 vs 8)")
    print("   ✅ Larger max_seq_length (2048)")
    print("   ✅ Proper SFT pre-training (2 epochs)")
    print("   ✅ vLLM generation working successfully")
    print("\n📌 SFT baseline established - ready for GRPO training!")


if __name__ == "__main__":
    main()