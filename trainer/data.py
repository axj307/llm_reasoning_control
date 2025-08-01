"""
Data generation and preprocessing module.
"""

import numpy as np
from control import solve_double_integrator
from config import *
from logger import logger

# Import HuggingFace datasets
import datasets as hf_datasets

# Import our custom datasets module
from datasets import dataset_manager
from datasets.systems import DoubleIntegratorDataset


def get_system_prompt(dt=DT, steps=STEPS):
    """Generate system prompt with specific time parameters."""
    total_time = dt * steps
    return f"""You are a control systems expert.
Given a double integrator system (ẍ = u) with initial position and velocity,
generate a sequence of {steps} control inputs to reach the origin (0,0) in exactly {total_time:.2f} seconds.
Position and velocity must stay within [-1, 1], and control inputs must be within [-3, 3].
Explain your approach between {REASONING_START} and {REASONING_END}.
Then provide exactly {steps} control values as a comma-separated list between {SOLUTION_START} and {SOLUTION_END}."""


def generate_control_dataset(num_samples=NUM_SAMPLES, dt=DT, steps=STEPS, use_cache=True):
    """
    Generate or load double integrator control problems with LQR solutions.
    
    Args:
        num_samples: Number of samples to generate
        dt: Time step
        steps: Number of control steps
        use_cache: Whether to use cached dataset if available
        
    Returns:
        List of dataset samples
    """
    # Register dataset class if not already registered
    dataset_manager.register_dataset_class("double_integrator", DoubleIntegratorDataset)
    
    # Get or create dataset
    data = dataset_manager.get_or_create(
        system="double_integrator",
        num_samples=num_samples,
        dt=dt,
        steps=steps,
        force_regenerate=not use_cache
    )
    
    logger.info(f"Dataset ready with {len(data)} samples")
    return data


def format_dataset_for_sft(dataset, tokenizer):
    """Format dataset for SFT training."""
    def format_example(example):
        # Create prompt without assistant's response
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": example["Messages"][1]["content"]}
            ],
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Full conversation including assistant's response
        full_text = tokenizer.apply_chat_template(
            example["Messages"],
            tokenize=False
        )
        
        return {"text": full_text}
    
    return dataset.map(format_example)


def create_dataset(num_samples=NUM_SAMPLES, use_cache=True):
    """Create and return a HuggingFace dataset."""
    data = generate_control_dataset(num_samples, use_cache=use_cache)
    return hf_datasets.Dataset.from_list(data)