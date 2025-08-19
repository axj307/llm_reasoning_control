#!/usr/bin/env python3
"""
Train GRPO model with working notebook parameters.
Based on successful Qwen3_(4B)-GRPO_control.ipynb approach.
"""

import torch
import numpy as np
import random
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def main():
    print("🚀 GRPO TRAINING WITH WORKING NOTEBOOK PARAMETERS")
    print("=" * 60)
    
    # Set seeds for reproducibility
    torch.manual_seed(3407)
    np.random.seed(3407)
    random.seed(3407)
    
    # GPU selection (same as notebook)
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        chosen_gpu = random.randint(0, num_gpus - 1)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(chosen_gpu)
        print(f"Randomly selected GPU: {chosen_gpu}")
    else:
        print("No GPUs available.")
    
    # Working notebook parameters
    max_seq_length = 2048  # From notebook
    lora_rank = 32         # From notebook
    
    # Load model with exact notebook parameters
    from unsloth import FastLanguageModel
    
    print("📥 Loading model with working notebook parameters...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-4B-Base",
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=True,  # Enable vLLM (from notebook)
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.7,  # From notebook
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_rank*2,  # *2 speeds up training (from notebook)
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    print("✅ Model loaded successfully")
    print(f"   Max seq length: {max_seq_length}")
    print(f"   LoRA rank: {lora_rank}")
    print(f"   GPU memory utilization: 0.7")
    
    # Control system parameters (from notebook)
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
    
    # Chat template (exact from notebook)
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
    
    # Generate dataset using notebook's approach
    print("📊 Generating control dataset...")
    control_data = generate_simple_control_dataset(num_samples=500, target_dt=dt, target_steps=steps)
    
    from datasets import Dataset
    control_dataset = Dataset.from_list(control_data)
    
    # Format dataset for SFT pre-training (from notebook)
    def format_for_sft(example):
        full_text = tokenizer.apply_chat_template(
            example["Messages"],
            tokenize=False
        )
        return {"text": full_text}
    
    control_dataset = control_dataset.map(format_for_sft)
    print(f"Dataset created: {len(control_dataset)} samples")
    
    # Pre fine-tune for formatting (exact from notebook)
    print("🔧 Pre fine-tuning for format...")
    from trl import SFTTrainer, SFTConfig
    
    sft_trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=control_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=4,  # From notebook
            gradient_accumulation_steps=1,
            warmup_steps=5,
            num_train_epochs=2,
            learning_rate=2e-4,
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="none",  # Disable wandb for now
            output_dir="./sft_pretraining_output",
            dataloader_num_workers=0,  # Disable multiprocessing
            max_seq_length=2048,  # Set max length
        ),
    )
    
    # Run pre-training
    sft_result = sft_trainer.train()
    print(f"✅ SFT pre-training completed. Loss: {sft_result.training_loss:.4f}")
    
    # Clear cache
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    # GRPO training with exact notebook parameters
    print("🚀 Starting GRPO training...")
    
    max_completion_length = 2048
    
    from vllm import SamplingParams
    vllm_sampling_params = SamplingParams(
        min_p=0.1,
        top_p=1.0,
        top_k=-1,
        seed=3407,
        stop=[tokenizer.eos_token],
        include_stop_str_in_output=True,
    )
    
    from trl import GRPOConfig, GRPOTrainer
    training_args = GRPOConfig(
        vllm_sampling_params=vllm_sampling_params,
        temperature=1.0,
        learning_rate=5e-6,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,  # Will be changed to num_generations
        gradient_accumulation_steps=1,
        num_generations=4,
        max_completion_length=max_completion_length,
        max_steps=100,
        save_steps=500,
        report_to="none",  # Disable wandb for now
        output_dir="./grpo_working_output",
    )
    
    # Setup reward functions (from notebook)
    import re
    
    # Match format functions (from notebook)
    solution_end_regex = r"</CONTROLS>[\s]{0,}" + \
        "(?:" + re.escape(tokenizer.eos_token) + ")?"
    
    match_format = re.compile(
        rf"{reasoning_end}.*?"\
        rf"{solution_start}(.+?){solution_end_regex}"\
        rf"[\s]{{0,}}$",
        flags=re.MULTILINE | re.DOTALL
    )
    
    def match_format_exactly(completions, **kwargs):
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            if match_format.search(response) is not None: 
                score += 3.0
            scores.append(score)
        return scores
    
    def match_format_approximately(completions, **kwargs):
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
        """Enhanced evaluation from notebook."""
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
                control_text = control_match.group(1).strip()
                control_values = [float(x.strip()) for x in control_text.split(',')]
                
                # Basic checks
                if len(control_values) == steps:
                    score += 1.0
                else:
                    score -= 1.0
                    
                if all(-3 <= u <= 3 for u in control_values):
                    score += 1.0
                else:
                    score -= 2.0
                
                # LQR smoothness check
                if len(control_values) > 1:
                    diffs = [abs(control_values[i] - control_values[i-1]) for i in range(1, len(control_values))]
                    if max(diffs) < 1.5:
                        score += 1.5
                        
                # Simulate system
                problem_text = prompts[0][-1]["content"]
                initial_match = re.search(r"position=([-\d\.]+), velocity=([-\d\.]+)", problem_text)
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
                
                scores.append(score)
                
            except Exception as e:
                scores.append(-2.0)
                
        return scores
    
    # Create GRPO trainer
    grpo_trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            match_format_exactly,
            match_format_approximately,
            evaluate_control_sequence,
        ],
        args=training_args,
        train_dataset=control_dataset,
    )
    
    print("🚀 Starting GRPO training...")
    print(f"   Max steps: {training_args.max_steps}")
    print(f"   Learning rate: {training_args.learning_rate}")
    print(f"   Num generations: {training_args.num_generations}")
    
    # Run GRPO training
    grpo_result = grpo_trainer.train()
    
    print("✅ GRPO training completed!")
    print(f"   Training loss: {grpo_result.training_loss:.4f}")
    
    # Save the model
    save_path = "models/working_notebook/grpo_working_params_model"
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Create metadata.json file for evaluation script
    import json
    metadata = {
        "model_type": "grpo",
        "system_name": "double_integrator", 
        "training_type": "grpo",
        "base_model": "unsloth/Qwen3-4B-Base",
        "max_seq_length": max_seq_length,
        "lora_rank": lora_rank,
        "sft_loss": float(sft_result.training_loss),
        "grpo_loss": float(grpo_result.training_loss),
        "grpo_max_steps": training_args.max_steps,
        "timestamp": str(np.datetime64('now')),
        "num_generations": training_args.num_generations
    }
    
    with open(os.path.join(save_path, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 GRPO model saved to: {save_path}")
    print(f"   📋 Metadata saved with SFT loss: {sft_result.training_loss:.4f}, GRPO loss: {grpo_result.training_loss:.4f}")
    
    # Quick test
    print("🧪 Testing GRPO model...")
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
    
    from vllm import SamplingParams
    test_sampling_params = SamplingParams(
        temperature=0.7,
        top_k=50,
        max_tokens=1024,
    )
    
    output = model.fast_generate(
        text,
        sampling_params=test_sampling_params,
        lora_request=None,
    )[0].outputs[0].text
    
    print(f"\n📝 GRPO Model Test Response:")
    print("=" * 50)
    print(output[:500] + "..." if len(output) > 500 else output)
    print("=" * 50)
    
    print(f"\n✅ GRPO training completed successfully!")
    print(f"📈 Model saved and ready for evaluation")
    
    return save_path


def generate_simple_control_dataset(num_samples=500, target_dt=0.1, target_steps=50):
    """Generate control dataset - exact copy from notebook."""
    import numpy as np
    import scipy.linalg as la
    
    reasoning_start = "<REASONING>"
    reasoning_end = "</REASONING>"
    solution_start = "<CONTROLS>"
    solution_end = "</CONTROLS>"
    
    def get_system_prompt(current_dt, current_steps):
        total_time = current_dt * current_steps
        return f"""You are a control systems expert.
Given a double integrator system (ẍ = u) with initial position and velocity,
generate a sequence of {current_steps} control inputs to reach the origin (0,0) in exactly {total_time:.2f} seconds.
Position and velocity must stay within [-1, 1], and control inputs must be within [-3, 3].
Explain your approach between {reasoning_start} and {reasoning_end}.
Then provide exactly {current_steps} control values as a comma-separated list between {solution_start} and {solution_end}."""
    
    def solve_double_integrator(x0, v0, dt, steps):
        """LQR solver from notebook."""
        A = np.array([[1, dt], [0, 1]])
        B = np.array([[0.5*dt**2], [dt]])
        Q = np.diag([10.0, 10.0])
        R = np.array([[0.1]])
        
        P = la.solve_discrete_are(A, B, Q, R)
        K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        
        x = np.array([[x0], [v0]])
        controls = []
        
        for i in range(steps):
            u = -K @ x
            u_clamped = max(-3.0, min(3.0, float(u[0])))
            x = A @ x + B * u_clamped
            x[0,0] = max(-1.0, min(1.0, x[0,0]))
            x[1,0] = max(-1.0, min(1.0, x[1,0]))
            controls.append(u_clamped)
        
        return controls
    
    data = []
    total_time_sec = target_dt * target_steps
    sample_system_prompt = get_system_prompt(target_dt, target_steps)
    
    for i in range(num_samples):
        x0 = np.random.uniform(-0.8, 0.8)
        v0 = np.random.uniform(-0.8, 0.8)
        
        problem = f"Control a double integrator system with initial state [position={x0:.2f}, velocity={v0:.2f}] to reach the origin (0,0) in {total_time_sec:.2f} seconds using {target_steps} steps. Ensure all states remain within [-1,1] and controls within [-3,3]."
        
        control_inputs = solve_double_integrator(x0, v0, target_dt, target_steps)
        
        reasoning = f"""For the double integrator system starting at position {x0:.2f} and velocity {v0:.2f}, I'll apply Linear Quadratic Regulator (LQR) control to reach the origin optimally in {total_time_sec:.2f} seconds using {target_steps} steps.

        The LQR approach provides an optimal feedback control law by minimizing a quadratic cost function that balances:
        1. The error in state (position and velocity)
        2. The control effort used

        For a double integrator with dynamics:
        - ẋ = v
        - v̇ = u

        The discrete-time state-space representation is:
        - x(k+1) = Ax(k) + Bu(k)

        Where:
        - A = [[1, Δt], [0, 1]]
        - B = [[0.5(Δt)², Δt]]
        - Δt = {target_dt:.2f} seconds

        Computing the optimal gain matrix K through the Riccati equation gives a feedback law u = -Kx.
        This produces a smooth control sequence that brings the system to the origin while respecting constraints.

        The resulting {target_steps} control inputs applied over {total_time_sec:.2f} seconds will optimally control the system to the target state."""
        
        control_str = ", ".join([f"{u:.3f}" for u in control_inputs])
        complete_output = f"{reasoning_start}{reasoning}{reasoning_end}{solution_start}{control_str}{solution_end}"
        
        data.append({
            "prompt": [
                {"role": "system", "content": sample_system_prompt},
                {"role": "user", "content": problem}
            ],
            "answer": control_str,
            "Messages": [
                {"role": "system", "content": sample_system_prompt},
                {"role": "user", "content": problem},
                {"role": "assistant", "content": complete_output}
            ]
        })
    
    return data


if __name__ == "__main__":
    main()