#!/usr/bin/env python3
"""
Train GRPO model for Van der Pol oscillator with working notebook parameters.
Adapted from successful double integrator approach.
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
    print("🚀 VAN DER POL GRPO TRAINING WITH WORKING PARAMETERS")
    print("=" * 60)
    
    # Set seeds for reproducibility
    torch.manual_seed(3407)
    np.random.seed(3407)
    random.seed(3407)
    
    # GPU selection
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        chosen_gpu = random.randint(0, num_gpus - 1)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(chosen_gpu)
        print(f"Randomly selected GPU: {chosen_gpu}")
    else:
        print("No GPUs available.")
    
    # Working notebook parameters
    max_seq_length = 2048
    lora_rank = 32
    
    # Load model with exact working parameters
    from unsloth import FastLanguageModel
    
    print("📥 Loading model with working parameters...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-4B-Base",
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=True,
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
    
    # Van der Pol control system parameters
    reasoning_start = "<REASONING>"
    reasoning_end = "</REASONING>"
    solution_start = "<CONTROLS>"
    solution_end = "</CONTROLS>"
    dt = 0.1
    steps = 50
    
    def get_system_prompt(current_dt, current_steps):
        total_time = current_dt * current_steps
        return f"""You are a control systems expert.
Given a Van der Pol oscillator system (ẍ - μ(1-x²)ẋ + x = u) with initial position and velocity,
generate a sequence of {current_steps} control inputs to reach the origin (0,0) in exactly {total_time:.2f} seconds.
Position and velocity must stay within [-2, 2], and control inputs must be within [-5, 5].
The Van der Pol oscillator has nonlinear damping that creates limit cycle behavior without control.
Explain your approach between {reasoning_start} and {reasoning_end}.
Then provide exactly {current_steps} control values as a comma-separated list between {solution_start} and {solution_end}."""
    
    system_prompt = get_system_prompt(dt, steps)
    
    # Chat template (same as working approach)
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
    
    print("✅ Chat template configured for Van der Pol")
    
    # Generate dataset using Van der Pol specific approach
    print("📊 Generating Van der Pol control dataset...")
    vdp_data = generate_vdp_control_dataset(num_samples=400, target_dt=dt, target_steps=steps)
    
    from datasets import Dataset
    vdp_dataset = Dataset.from_list(vdp_data)
    
    # Format dataset for SFT pre-training
    def format_for_sft(example):
        full_text = tokenizer.apply_chat_template(
            example["Messages"],
            tokenize=False
        )
        return {"text": full_text}
    
    vdp_dataset = vdp_dataset.map(format_for_sft)
    print(f"Van der Pol dataset created: {len(vdp_dataset)} samples")
    
    # Pre fine-tune for formatting
    print("🔧 Pre fine-tuning for Van der Pol format...")
    from trl import SFTTrainer, SFTConfig
    
    sft_trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=vdp_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            warmup_steps=5,
            num_train_epochs=2,
            learning_rate=2e-4,
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="none",
            output_dir="./vdp_sft_pretraining_output",
        ),
    )
    
    # Run pre-training
    sft_result = sft_trainer.train()
    print(f"✅ VDP SFT pre-training completed. Loss: {sft_result.training_loss:.4f}")
    
    # Clear cache
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    # GRPO training with Van der Pol specific rewards
    print("🚀 Starting Van der Pol GRPO training...")
    
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
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=4,
        max_completion_length=max_completion_length,
        max_steps=100,
        save_steps=500,
        report_to="none",
        output_dir="./vdp_grpo_working_output",
    )
    
    # Setup Van der Pol specific reward functions
    import re
    
    # Format matching functions (same as DI)
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
    
    def evaluate_vdp_control_sequence(prompts, completions, answer, **kwargs):
        """Van der Pol specific control evaluation."""
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
                    
                if all(-5 <= u <= 5 for u in control_values):
                    score += 1.0
                else:
                    score -= 2.0
                
                # Van der Pol specific: check for control smoothness (important for nonlinear system)
                if len(control_values) > 1:
                    diffs = [abs(control_values[i] - control_values[i-1]) for i in range(1, len(control_values))]
                    if max(diffs) < 2.0:  # Slightly higher tolerance than DI
                        score += 1.5
                        
                # Simulate Van der Pol system
                problem_text = prompts[0][-1]["content"]
                initial_match = re.search(r"position=([-\d\.]+), velocity=([-\d\.]+)", problem_text)
                if initial_match:
                    x0 = float(initial_match.group(1))
                    v0 = float(initial_match.group(2))
                    
                    x, v = x0, v0
                    valid_trajectory = True
                    mu = 1.0  # Van der Pol parameter
                    
                    for u in control_values:
                        # Van der Pol dynamics
                        dvdt = mu * (1 - x**2) * v - x + u
                        dxdt = v
                        
                        v = v + dvdt * dt
                        x = x + dxdt * dt
                        
                        if not (-2 <= x <= 2 and -2 <= v <= 2):
                            valid_trajectory = False
                            break
                    
                    if valid_trajectory:
                        score += 1.0
                    else:
                        score -= 1.0
                    
                    # Final error reward
                    final_error = np.sqrt(x**2 + v**2)
                    if final_error < 0.2:
                        score += 3.0
                    elif final_error < 0.4:
                        score += 2.0
                    elif final_error < 0.8:
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
            evaluate_vdp_control_sequence,
        ],
        args=training_args,
        train_dataset=vdp_dataset,
    )
    
    print("🚀 Starting Van der Pol GRPO training...")
    
    # Run GRPO training
    grpo_result = grpo_trainer.train()
    
    print("✅ Van der Pol GRPO training completed!")
    print(f"   Training loss: {grpo_result.training_loss:.4f}")
    
    # Save the model
    save_path = "models/working_notebook/vdp_grpo_working_params_model"
    model.save_lora(save_path)
    print(f"💾 Van der Pol GRPO model saved to: {save_path}")
    
    print(f"\n✅ Van der Pol GRPO training completed successfully!")
    return save_path


def generate_vdp_control_dataset(num_samples=400, target_dt=0.1, target_steps=50):
    """Generate Van der Pol control dataset using fast solver."""
    import numpy as np
    from core.solvers.vdp_solver_fast import solve_van_der_pol_fast
    
    reasoning_start = "<REASONING>"
    reasoning_end = "</REASONING>"
    solution_start = "<CONTROLS>"
    solution_end = "</CONTROLS>"
    
    def get_system_prompt(current_dt, current_steps):
        total_time = current_dt * current_steps
        return f"""You are a control systems expert.
Given a Van der Pol oscillator system (ẍ - μ(1-x²)ẋ + x = u) with initial position and velocity,
generate a sequence of {current_steps} control inputs to reach the origin (0,0) in exactly {total_time:.2f} seconds.
Position and velocity must stay within [-2, 2], and control inputs must be within [-5, 5].
The Van der Pol oscillator has nonlinear damping that creates limit cycle behavior without control.
Explain your approach between {reasoning_start} and {reasoning_end}.
Then provide exactly {current_steps} control values as a comma-separated list between {solution_start} and {solution_end}."""
    
    data = []
    total_time_sec = target_dt * target_steps
    sample_system_prompt = get_system_prompt(target_dt, target_steps)
    
    for i in range(num_samples):
        # Random initial states within Van der Pol bounds
        x0 = np.random.uniform(-1.5, 1.5)
        v0 = np.random.uniform(-1.5, 1.5)
        
        problem = f"Control a Van der Pol oscillator system with initial state [position={x0:.2f}, velocity={v0:.2f}] to reach the origin (0,0) in {total_time_sec:.2f} seconds using {target_steps} steps. Ensure all states remain within [-2,2] and controls within [-5,5]."
        
        # Solve using fast Van der Pol solver
        control_inputs = solve_van_der_pol_fast(x0, v0, mu=1.0, dt=target_dt, steps=target_steps)
        
        # Generate Van der Pol specific reasoning text
        reasoning = f"""For the Van der Pol oscillator starting at position {x0:.2f} and velocity {v0:.2f}, I'll apply optimal control to reach the origin in {total_time_sec:.2f} seconds using {target_steps} steps.

        The Van der Pol oscillator has dynamics: ẍ - μ(1-x²)ẋ + x = u where μ = 1.0

        Key characteristics:
        1. Nonlinear damping: -μ(1-x²)ẋ provides negative damping for |x| < 1 and positive damping for |x| > 1
        2. This creates a limit cycle behavior without control
        3. The restoring force is linear: -x
        4. Control input u directly affects acceleration

        Control strategy:
        1. Use feedback control to counteract the nonlinear dynamics
        2. Apply stronger control when |x| < 1 to overcome negative damping
        3. Use smoother control when |x| > 1 to work with positive damping
        4. Gradually reduce control magnitude as state approaches origin

        The resulting {target_steps} control inputs will stabilize the system to the origin while respecting state and control constraints."""
        
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
