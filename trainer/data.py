"""
Data generation and preprocessing module.
"""

import numpy as np
from datasets import Dataset
from control import solve_double_integrator
from config import *


def get_system_prompt(dt=DT, steps=STEPS):
    """Generate system prompt with specific time parameters."""
    total_time = dt * steps
    return f"""You are a control systems expert.
Given a double integrator system (ẍ = u) with initial position and velocity,
generate a sequence of {steps} control inputs to reach the origin (0,0) in exactly {total_time:.2f} seconds.
Position and velocity must stay within [-1, 1], and control inputs must be within [-3, 3].
Explain your approach between {REASONING_START} and {REASONING_END}.
Then provide exactly {steps} control values as a comma-separated list between {SOLUTION_START} and {SOLUTION_END}."""


def generate_control_dataset(num_samples=NUM_SAMPLES, dt=DT, steps=STEPS):
    """Generate double integrator control problems with LQR solutions."""
    data = []
    total_time = dt * steps
    system_prompt = get_system_prompt(dt, steps)
    
    for i in range(num_samples):
        # Random initial states
        x0 = np.random.uniform(-0.8, 0.8)
        v0 = np.random.uniform(-0.8, 0.8)
        
        # Problem statement
        problem = f"Control a double integrator system with initial state [position={x0:.2f}, velocity={v0:.2f}] to reach the origin (0,0) in {total_time:.2f} seconds using {steps} steps. Ensure all states remain within [-1,1] and controls within [-3,3]."
        
        # Solve for optimal control
        control_inputs = solve_double_integrator(x0, v0, dt, steps)
        
        # Generate reasoning text
        reasoning = f"""For the double integrator system starting at position {x0:.2f} and velocity {v0:.2f}, I'll apply Linear Quadratic Regulator (LQR) control to reach the origin optimally in {total_time:.2f} seconds using {steps} steps.

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
        - Δt = {dt:.2f} seconds

        Computing the optimal gain matrix K through the Riccati equation gives a feedback law u = -Kx.
        This produces a smooth control sequence that brings the system to the origin while respecting constraints.

        The resulting {steps} control inputs applied over {total_time:.2f} seconds will optimally control the system to the target state."""
        
        # Format control values
        control_str = ", ".join([f"{u:.3f}" for u in control_inputs])
        
        # Create output
        complete_output = f"{REASONING_START}{reasoning}{REASONING_END}{SOLUTION_START}{control_str}{SOLUTION_END}"
        
        # Add to dataset
        data.append({
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": problem}
            ],
            "answer": control_str,
            "Messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": problem},
                {"role": "assistant", "content": complete_output}
            ]
        })
    
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


def create_dataset(num_samples=NUM_SAMPLES):
    """Create and return a HuggingFace dataset."""
    data = generate_control_dataset(num_samples)
    return Dataset.from_list(data)