#!/usr/bin/env python3
"""
Extended SFT training following the EXACT working notebook approach.
This follows Qwen3_(4B)-GRPO_control.ipynb exactly.
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
    print("🚀 Extended SFT Training - EXACT Notebook Approach")
    print("=" * 60)
    
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
    max_seq_length = 2048  # Can increase for longer reasoning traces
    lora_rank = 32  # Larger rank = smarter, but slower
    
    # Load model (EXACT from working notebook)
    from unsloth import FastLanguageModel
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-4B-Base",
        max_seq_length=max_seq_length,
        load_in_4bit=True,  # False for LoRA 16bit
        fast_inference=True,  # Enable vLLM fast inference
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.7,  # Reduce if out of memory
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_rank*2,  # *2 speeds up training
        use_gradient_checkpointing="unsloth",  # Reduces memory usage
        random_state=3407,
    )
    
    print("✅ Model loaded successfully (exact notebook approach)")
    print(f"   LoRA rank: {lora_rank}")
    print(f"   Max seq length: {max_seq_length}")
    print(f"   Fast inference: True (vLLM enabled)")
    print(f"   GPU memory utilization: 0.7")
    
    # Load previous SFT weights if available
    print("📂 Loading previous SFT weights if available...")
    try:
        import safetensors.torch as st
        sft_weights = st.load_file("models/working_notebook/sft_model/adapter_model.safetensors")
        
        for name, param in model.named_parameters():
            if name in sft_weights:
                param.data = sft_weights[name].to(param.device, dtype=param.dtype)
        
        print("✅ Previous SFT weights loaded successfully")
    except Exception as e:
        print(f"⚠️  Could not load previous weights: {e}")
        print("   Starting fresh training")
    
    # Control system parameters (EXACT from working notebook)
    reasoning_start = "<REASONING>"
    reasoning_end = "</REASONING>"
    solution_start = "<CONTROLS>"
    solution_end = "</CONTROLS>"
    
    # Time settings (EXACT from working notebook)
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
    
    system_prompt = get_system_prompt(dt, steps)  # Initialize global system_prompt
    
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
    
    # Replace with our specific template (EXACT from notebook)
    chat_template = chat_template\
        .replace("'{system_prompt}'", f"'{system_prompt}'")\
        .replace("'{reasoning_start}'", f"'{reasoning_start}'")
    tokenizer.chat_template = chat_template
    
    print("✅ Chat template configured (exact notebook approach)")
    
    # Load dataset
    print("📂 Loading dataset...")
    with open("datasets/di_train.pkl", "rb") as f:
        train_data = pickle.load(f)
    
    with open("datasets/di_eval.pkl", "rb") as f:
        eval_data = pickle.load(f)
    
    train_data = [x for x in train_data if x.get("system_type") == "double_integrator"]
    eval_data = [x for x in eval_data if x.get("system_type") == "double_integrator"]
    
    print(f"Dataset: {len(train_data)} train, {len(eval_data)} eval")
    
    # Extended SFT training (based on notebook approach)
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
    
    # Extended SFT configuration (enhanced from notebook)
    sft_config = SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=4,  # From notebook
        gradient_accumulation_steps=1,
        warmup_steps=20,  # More warmup for extended training
        num_train_epochs=5,  # Extended training - more epochs
        learning_rate=1e-4,  # Lower learning rate for fine-tuning
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",  # Cosine schedule for better convergence
        seed=3407,
        report_to="none",  # Disable wandb for SLURM
        output_dir="./sft_extended_output",
        save_steps=250,
        save_total_limit=3,
    )
    
    # Run extended SFT training
    sft_trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=sft_train_dataset,
        eval_dataset=sft_eval_dataset,
        args=sft_config,
    )
    
    print("🚀 Starting extended SFT training (exact notebook approach, 5 epochs)...")
    sft_result = sft_trainer.train()
    
    print("✅ Extended SFT training completed!")
    print(f"   Final training loss: {sft_result.training_loss:.4f}")
    
    # Save extended model
    extended_save_path = "models/working_notebook/sft_extended_exact_model"
    model.save_lora(extended_save_path)
    print(f"💾 Extended SFT model saved to: {extended_save_path}")
    
    # Quick test (EXACT from notebook approach)
    print("🧪 Testing extended SFT model...")
    
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
    
    # Use EXACT vLLM sampling from notebook
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
    
    print(f"\n📝 Extended SFT Model Response:")
    print("="*60)
    print(output)
    print("="*60)
    
    print("\n✅ Extended SFT training completed successfully (exact notebook approach)!")

if __name__ == "__main__":
    main()