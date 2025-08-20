"""Fixed GRPO training implementation based on working notebook.

Updated with LQR-aligned quadratic reward function that directly mirrors
the optimal control cost function for better training consistency.
"""

import re
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from vllm import SamplingParams
from datasets import Dataset

# Import TRL components
from trl import GRPOConfig, GRPOTrainer

from core.model_manager import UniversalModelManager
from environments import get_system


def train_grpo_model(model_manager: UniversalModelManager,
                    train_data: List[Dict[str, Any]],
                    eval_data: Optional[List[Dict[str, Any]]] = None,
                    training_config: Optional[Dict] = None,
                    reasoning_start: str = "<REASONING>",
                    reasoning_end: str = "</REASONING>",
                    solution_start: str = "<CONTROLS>",
                    solution_end: str = "</CONTROLS>") -> Dict[str, Any]:
    """
    Train GRPO model on control data using EXACT working notebook approach.
    
    Args:
        model_manager: Universal model manager with loaded model
        train_data: Training data from data pipeline
        eval_data: Optional evaluation data
        training_config: Training configuration
        reasoning_start/end: Tags for reasoning
        solution_start/end: Tags for solutions
    
    Returns:
        Training results and metrics
    """
    print("🎮 Setting up GRPO training (working notebook approach)...")
    
    # Clear memory first
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    # EXACT working notebook GRPO configuration
    max_completion_length = 2048  # Match notebook exactly
    model_max_length = model_manager.model.config.max_position_embeddings or 2048
    
    # vLLM sampling params (EXACT from notebook)
    vllm_sampling_params = SamplingParams(
        min_p=0.1,
        top_p=1.0,
        top_k=-1,
        seed=3407,
        stop=[model_manager.tokenizer.eos_token],
        include_stop_str_in_output=True,
    )
    
    # Match working notebook GRPO config EXACTLY
    default_config = {
        "vllm_sampling_params": vllm_sampling_params,
        "temperature": 1.0,  # Exact from notebook
        "learning_rate": 5e-6,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "linear",
        "optim": "adamw_8bit",
        "logging_steps": 1,
        "per_device_train_batch_size": 4,  # Notebook adjusts to 4 for num_generations
        "gradient_accumulation_steps": 1,  
        "num_generations": 4,  # Working notebook uses 4 generations
        "max_completion_length": max_completion_length,
        "max_steps": 50,  # Working notebook uses 50 steps
        "save_steps": 500,
        "report_to": "wandb",
        "output_dir": "./temp_grpo_output",
        "temperature": 1.0,  # Working notebook uses temperature 1.0
        "min_p": 0.1,
        "top_p": 1.0,
        "top_k": -1,
        "seed": 3407,
    }
    
    if training_config:
        default_config.update(training_config)
    
    # Store training config
    model_manager.training_config = default_config
    
    # Log sequence length configuration
    print(f"🔧 GRPO sequence length configuration:")
    print(f"   Model max length: {model_max_length}")
    print(f"   Max completion length: {max_completion_length}")
    
    # Set up VLLM sampling parameters
    vllm_sampling_params = SamplingParams(
        min_p=default_config["min_p"],
        top_p=default_config["top_p"],
        top_k=default_config["top_k"],
        seed=default_config["seed"],
        stop=[model_manager.tokenizer.eos_token],
        include_stop_str_in_output=True,
    )
    
    # Create GRPO config
    grpo_config = GRPOConfig(
        vllm_sampling_params=vllm_sampling_params,
        temperature=default_config["temperature"],
        learning_rate=default_config["learning_rate"],
        weight_decay=default_config["weight_decay"],
        warmup_ratio=default_config["warmup_ratio"],
        lr_scheduler_type=default_config["lr_scheduler_type"],
        optim=default_config["optim"],
        logging_steps=default_config["logging_steps"],
        per_device_train_batch_size=default_config["per_device_train_batch_size"],
        gradient_accumulation_steps=default_config["gradient_accumulation_steps"],
        num_generations=default_config["num_generations"],
        max_completion_length=default_config["max_completion_length"],
        max_steps=default_config["max_steps"],
        save_steps=default_config["save_steps"],
        report_to=default_config["report_to"],
        output_dir=default_config["output_dir"],
    )
    
    # Get the system type from data
    system_types = list(set(d.get("system_type", "double_integrator") for d in train_data))
    if len(system_types) == 1:
        system_type = system_types[0]
        system = get_system(system_type)()
        system_prompt = system.get_system_prompt(
            reasoning_start, reasoning_end, solution_start, solution_end
        )
    else:
        # Universal model - use generic prompt
        system_prompt = f"""You are a control systems expert.
Given a control system with initial state, generate an optimal control sequence.
Explain your approach between {reasoning_start} and {reasoning_end}.
Then provide control values as a comma-separated list between {solution_start} and {solution_end}."""
    
    # Prepare datasets for GRPO training - match working notebook format exactly
    def format_for_grpo(example):
        """Format a single example for GRPO training - EXACT match to working notebook."""
        # Get the messages - match working notebook format
        messages = example.get("messages", example.get("Messages", []))
        
        # Extract prompt (system + user messages only)
        prompt_messages = []
        for msg in messages:
            if msg["role"] != "assistant":
                prompt_messages.append(msg)
        
        # Extract answer from controls field (as done in working notebook)
        controls = example.get("controls", [])
        if isinstance(controls, list):
            answer = ", ".join([f"{u:.3f}" for u in controls])
        else:
            answer = str(controls)
        
        # Return EXACT format expected by GRPO trainer from working notebook
        return {
            "prompt": prompt_messages,  # List of message dicts (system + user only)
            "answer": answer,           # Control string
            "Messages": messages        # Full conversation for reference (working notebook has this)
        }
    
    # Convert to HuggingFace format
    train_dataset = Dataset.from_list(train_data)
    train_dataset = train_dataset.map(format_for_grpo, remove_columns=train_dataset.column_names)
    
    eval_dataset = None
    if eval_data:
        eval_dataset = Dataset.from_list(eval_data)
        eval_dataset = eval_dataset.map(format_for_grpo, remove_columns=eval_dataset.column_names)
    
    print(f"Training dataset size: {len(train_dataset)}")
    if eval_dataset:
        print(f"Evaluation dataset size: {len(eval_dataset)}")
    
    # Setup chat template to match working notebook
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
    
    # Replace with specific values
    chat_template = chat_template\
        .replace("'{system_prompt}'", f"'{system_prompt}'")\
        .replace("'{reasoning_start}'", f"'{reasoning_start}'")
    
    model_manager.tokenizer.chat_template = chat_template
    
    # Get reward functions with fixed parameters
    reward_functions = get_reward_functions_fixed(
        reasoning_end, solution_start, solution_end,
        model_manager.tokenizer, system_type if len(system_types) == 1 else None
    )
    
    print(f"Using {len(reward_functions)} reward functions")
    
    # Now run GRPO training
    trainer = GRPOTrainer(
        model=model_manager.model,
        processing_class=model_manager.tokenizer,
        reward_funcs=reward_functions,
        args=grpo_config,
        train_dataset=train_dataset,
    )
    trainer.train()


def get_reward_functions_fixed(reasoning_end: str, solution_start: str, solution_end: str,
                              tokenizer, system_type: Optional[str] = None) -> List[Callable]:
    """Get reward functions for GRPO training - EXACT match to working notebook."""
    
    # Define regex pattern to match control sequence - EXACT from notebook
    solution_end_regex = rf"{re.escape(solution_end)}[\s]{{0,}}" + \
        f"(?:{re.escape(tokenizer.eos_token)})?"

    match_format = re.compile(
        rf"{re.escape(reasoning_end)}.*?"
        rf"{re.escape(solution_start)}(.+?){solution_end_regex}"
        rf"[\s]{{0,}}$",
        flags=re.MULTILINE | re.DOTALL
    )
    
    def match_format_exactly(completions, **kwargs):
        """Reward function: exact format matching."""
        scores = []
        for completion in completions:
            score = 0.0
            # Match working notebook format
            response = completion[0]["content"]
            if match_format.search(response) is not None:
                score += 3.0
            scores.append(score)
        return scores
    
    def match_format_approximately(completions, **kwargs):
        """Reward function: approximate format matching."""
        scores = []
        for completion in completions:
            score = 0.0
            # Match working notebook format
            response = completion[0]["content"]
            score += 0.5 if response.count(reasoning_end) == 1 else -1.0
            score += 0.5 if response.count(solution_start) == 1 else -1.0
            score += 0.5 if response.count(solution_end) == 1 else -1.0
            scores.append(score)
        return scores
    
    def evaluate_control_sequence_lqr_aligned(prompts, completions, answer, **kwargs):
        """LQR-aligned quadratic reward function matching optimal control cost."""
        scores = []
        
        # LQR cost parameters (matching lqr_solver.py)
        Q = np.array([[10.0, 0.0], [0.0, 10.0]])  # State cost matrix
        R = 0.1  # Control cost scalar
        Q_terminal = Q * 5.0  # Terminal cost multiplier for final state bonus
        
        # Use system-specific parameters if available
        if system_type == "double_integrator":
            dt = 0.1
            steps = 50
        elif system_type == "van_der_pol":
            dt = 0.1
            steps = 50
        else:
            # Default values
            dt = 0.1
            steps = 50
        
        for completion, true_answer in zip(completions, answer):
            score = 0.0
            response = completion[0]["content"]
            
            # Extract control sequence
            control_match = re.search(rf"{solution_start}(.*?){solution_end}", response, re.DOTALL)
            if control_match is None:
                scores.append(-10.0)  # Large penalty for format failure
                continue
                
            try:
                # Parse control values
                control_text = control_match.group(1).strip()
                control_values = [float(x.strip()) for x in control_text.split(',')]
                
                # Basic format check
                if len(control_values) != steps:
                    scores.append(-5.0)
                    continue
                
                # Extract initial state
                problem_text = prompts[0][-1]["content"]
                initial_match = re.search(r"position=([-\d\.]+), velocity=([-\d\.]+)", problem_text)
                if not initial_match:
                    scores.append(-5.0)
                    continue
                    
                x0 = float(initial_match.group(1))
                v0 = float(initial_match.group(2))
                
                # Simulate trajectory and compute LQR-style cost
                states = [(x0, v0)]  # Store as tuples for easier handling
                x, v = x0, v0
                valid_trajectory = True
                constraint_violations = 0
                
                # Simulate system with generated controls
                for u in control_values:
                    # Apply control bounds check
                    if not (-3.0 <= u <= 3.0):
                        constraint_violations += 1
                        u = max(-3.0, min(3.0, u))  # Clamp control
                    
                    # Update state
                    v = v + u * dt
                    x = x + v * dt
                    
                    # Check state bounds
                    if not (-1.0 <= x <= 1.0 and -1.0 <= v <= 1.0):
                        constraint_violations += 1
                        valid_trajectory = False
                        # Clamp states for continued simulation
                        x = max(-1.0, min(1.0, x))
                        v = max(-1.0, min(1.0, v))
                    
                    states.append((x, v))
                
                # Compute LQR-style quadratic cost (converted to reward)
                total_cost = 0.0
                
                # State costs for all intermediate states
                for i, (pos, vel) in enumerate(states[:-1]):
                    state_vec = np.array([pos, vel])
                    state_cost = state_vec.T @ Q @ state_vec
                    total_cost += state_cost
                
                # Control costs
                for u in control_values:
                    control_cost = R * u**2
                    total_cost += control_cost
                
                # Terminal state cost (final state)
                final_pos, final_vel = states[-1]
                final_state = np.array([final_pos, final_vel])
                terminal_cost = final_state.T @ Q_terminal @ final_state
                total_cost += terminal_cost
                
                # Convert cost to reward (negative cost = positive reward)
                # Scale down the cost to reasonable reward range
                lqr_reward = -total_cost / 100.0  # Scale factor to normalize
                
                # Terminal bonus for reaching near origin
                final_error = np.sqrt(final_pos**2 + final_vel**2)
                terminal_bonus = 0.0
                if final_error < 0.05:
                    terminal_bonus = 10.0  # Large bonus for excellent performance
                elif final_error < 0.1:
                    terminal_bonus = 5.0   # Good performance
                elif final_error < 0.2:
                    terminal_bonus = 2.0   # Decent performance
                elif final_error < 0.5:
                    terminal_bonus = 0.5   # Acceptable performance
                
                # Constraint violation penalties
                constraint_penalty = -constraint_violations * 2.0
                
                # Trajectory validity bonus
                validity_bonus = 3.0 if valid_trajectory else -5.0
                
                # Combine all reward components
                score = lqr_reward + terminal_bonus + constraint_penalty + validity_bonus
                
                scores.append(score)
                
            except Exception as e:
                scores.append(-10.0)  # Large penalty for parsing errors
                
        return scores
    
    return [
        match_format_exactly,
        match_format_approximately,
        evaluate_control_sequence_lqr_aligned,
    ]


def save_grpo_model(model_manager: UniversalModelManager,
                   systems: List[str],
                   metrics: Dict[str, Any],
                   run_name: Optional[str] = None) -> str:
    """Save the trained GRPO model."""
    if len(systems) == 1:
        # Single system model
        save_path = model_manager.save_single_system_checkpoint(
            system_name=systems[0],
            training_type="grpo",
            run_name=run_name,
            metrics=metrics
        )
    else:
        # Universal model
        save_path = model_manager.save_universal_checkpoint(
            systems_list=systems,
            training_type="grpo",
            run_name=run_name,
            metrics=metrics
        )
    
    return save_path