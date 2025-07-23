"""GRPO training implementation."""

import re
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from vllm import SamplingParams
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

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
    Train GRPO model on control data.
    
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
    # Default GRPO training config
    default_config = {
        "learning_rate": 5e-6,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "linear",
        "optim": "adamw_8bit",
        "logging_steps": 1,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "num_generations": 8,
        "max_completion_length": 2048,
        "max_steps": 100,
        "save_steps": 500,
        "report_to": "wandb",
        "output_dir": "./temp_grpo_output",
        "temperature": 1.0,
        "min_p": 0.1,
        "top_p": 1.0,
        "top_k": -1,
        "seed": 3407,
    }
    
    if training_config:
        default_config.update(training_config)
    
    # Store training config
    model_manager.training_config = default_config
    
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
    
    # Prepare datasets for GRPO training
    def format_for_grpo(example):
        """Format a single example for GRPO training."""
        messages = example["messages"]
        
        # For GRPO, we need the conversation up to the user message
        # The assistant response will be generated during training
        train_messages = messages[:-1]  # Remove assistant response
        
        return {
            "messages": train_messages,
            "system_type": example.get("system_type", "unknown"),
            "initial_state": example["initial_state"],
            "target_controls": example["controls"]
        }
    
    # Convert to HuggingFace format
    train_dataset = Dataset.from_list(train_data)
    train_dataset = train_dataset.map(format_for_grpo)
    
    eval_dataset = None
    if eval_data:
        eval_dataset = Dataset.from_list(eval_data)
        eval_dataset = eval_dataset.map(format_for_grpo)
    
    print(f"Training dataset size: {len(train_dataset)}")
    if eval_dataset:
        print(f"Evaluation dataset size: {len(eval_dataset)}")
    
    # Get reward functions
    reward_functions = get_reward_functions(
        reasoning_end, solution_start, solution_end,
        model_manager.tokenizer
    )
    
    print(f"Using {len(reward_functions)} reward functions")
    
    # Create trainer
    trainer = GRPOTrainer(
        model=model_manager.model,
        processing_class=model_manager.tokenizer,
        reward_funcs=reward_functions,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    print("Starting GRPO training...")
    
    # Train the model
    train_result = trainer.train()
    
    print("GRPO training completed!")
    
    # Extract metrics
    metrics = train_result.metrics if hasattr(train_result, 'metrics') else {}
    
    return {
        "trainer": trainer,
        "train_result": train_result,
        "metrics": metrics,
        "training_config": default_config
    }


def get_reward_functions(reasoning_end: str, solution_start: str, solution_end: str,
                        tokenizer) -> List[Callable]:
    """Get reward functions for GRPO training."""
    
    # Define regex pattern to match control sequence
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
            response = completion[0]["content"]
            score += 0.5 if response.count(reasoning_end) == 1 else -1.0
            score += 0.5 if response.count(solution_start) == 1 else -1.0
            score += 0.5 if response.count(solution_end) == 1 else -1.0
            scores.append(score)
        return scores
    
    def evaluate_control_performance(completions, prompts, **kwargs):
        """Reward function: control performance evaluation."""
        scores = []
        
        # Performance parameters (aggressive rewards for good performance)
        Q = np.diag([25.0, 5.0])  # State cost weights
        R = 0.05  # Control cost weight
        position_terminal_weight = 50.0
        velocity_terminal_weight = 25.0
        terminal_precision = 50.0
        
        for completion, prompt in zip(completions, prompts):
            score = 0.0
            response = completion[0]["content"]
            
            # Extract control sequence
            control_match = re.search(
                rf"{re.escape(solution_start)}(.*?){re.escape(solution_end)}", 
                response, re.DOTALL
            )
            
            if control_match is None:
                scores.append(-5.0)
                continue
            
            try:
                # Parse control values
                control_text = control_match.group(1).strip()
                control_values = [float(x.strip()) for x in control_text.split(',')]
                
                # Determine system type from prompt
                problem_text = prompt[-1]["content"].lower()
                if "double integrator" in problem_text:
                    system_type = "double_integrator"
                    dt = 0.1
                    steps = 50
                    control_bounds = (-3.0, 3.0)
                    state_bounds = (-1.0, 1.0)
                elif "van der pol" in problem_text:
                    system_type = "van_der_pol"
                    dt = 0.1
                    steps = 50
                    control_bounds = (-5.0, 5.0)
                    state_bounds = (-2.0, 2.0)
                else:
                    scores.append(-2.0)
                    continue
                
                # Basic validation
                if len(control_values) == steps:
                    score += 1.0
                else:
                    score -= 2.0
                
                control_min, control_max = control_bounds
                if all(control_min <= u <= control_max for u in control_values):
                    score += 0.5
                else:
                    score -= 3.0
                
                # Extract initial state from prompt
                initial_match = re.search(r"position=([-\d\.]+), velocity=([-\d\.]+)", 
                                        prompt[-1]["content"])
                
                if initial_match:
                    x0 = float(initial_match.group(1))
                    v0 = float(initial_match.group(2))
                    
                    # Simulate trajectory based on system type
                    system = get_system(system_type)(dt=dt, steps=steps)
                    initial_state = np.array([x0, v0])
                    
                    try:
                        trajectory = system.simulate_trajectory(initial_state, control_values)
                        
                        if trajectory['valid_trajectory']:
                            score += 1.0
                        else:
                            score -= 5.0
                        
                        # Terminal state rewards
                        final_state = trajectory['final_state']
                        final_x, final_v = final_state
                        
                        # Exponential rewards for reaching origin
                        position_reward = position_terminal_weight * np.exp(-terminal_precision * final_x**2)
                        velocity_reward = velocity_terminal_weight * np.exp(-terminal_precision * final_v**2)
                        score += position_reward + velocity_reward
                        
                        # Tiered rewards for precision
                        if abs(final_x) < 0.005 and abs(final_v) < 0.005:
                            score += 30.0
                        elif abs(final_x) < 0.01 and abs(final_v) < 0.01:
                            score += 20.0
                        elif abs(final_x) < 0.05 and abs(final_v) < 0.05:
                            score += 10.0
                        elif abs(final_x) < 0.1 and abs(final_v) < 0.1:
                            score += 5.0
                    
                    except Exception as e:
                        score -= 3.0
                
                scores.append(score)
                
            except Exception as e:
                scores.append(-3.0)
        
        return scores
    
    return [
        match_format_exactly,
        match_format_approximately,
        evaluate_control_performance,
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