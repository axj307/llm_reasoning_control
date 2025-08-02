"""
Data generation and preprocessing module.
"""

import numpy as np
from control import solve_double_integrator
from logger import logger

# Simple constants - no need for complex imports
DT = 0.1
STEPS = 50
NUM_SAMPLES = 500
REASONING_START = "<REASONING>"
REASONING_END = "</REASONING>"
SOLUTION_START = "<CONTROLS>"
SOLUTION_END = "</CONTROLS>"


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
    """
    Generate double integrator control problems with LQR solutions.
    
    Args:
        num_samples: Number of samples to generate
        dt: Time step
        steps: Number of control steps
        
    Returns:
        List of dataset samples
    """
    logger.info(f"Generating {num_samples} control samples...")
    
    samples = []
    for i in range(num_samples):
        # Random initial conditions
        x0 = np.random.uniform(-0.8, 0.8)
        v0 = np.random.uniform(-0.8, 0.8)
        
        # Generate optimal control sequence using LQR
        controls = solve_double_integrator(x0, v0, dt, steps)
        
        # Create user query
        query = f"Control a double integrator system with initial state [position={x0:.3f}, velocity={v0:.3f}] to reach the origin (0,0) in {dt*steps:.2f} seconds using {steps} steps. Ensure all states remain within [-1,1] and controls within [-3,3]."
        
        # Create optimal response
        controls_str = ", ".join([f"{u:.4f}" for u in controls])
        response = f"{REASONING_START}\nTo control the double integrator from initial state [x={x0:.3f}, v={v0:.3f}] to origin [0,0], I'll use LQR optimal control. The system dynamics are ẍ = u, so I need to carefully balance position and velocity control to reach the target efficiently while respecting constraints.\n{REASONING_END}\n{SOLUTION_START}\n{controls_str}\n{SOLUTION_END}"
        
        # Format as conversation
        sample = {
            "Messages": [
                {"role": "system", "content": get_system_prompt(dt, steps)},
                {"role": "user", "content": query},
                {"role": "assistant", "content": response}
            ]
        }
        samples.append(sample)
    
    logger.info(f"Generated {len(samples)} samples")
    return samples


def format_dataset_for_sft(dataset, tokenizer):
    """Format dataset for SFT training."""
    from datasets import Dataset
    
    formatted_samples = []
    
    for example in dataset:
        # Full conversation including assistant's response
        full_text = tokenizer.apply_chat_template(
            example["Messages"],
            tokenize=False
        )
        formatted_samples.append({"text": full_text})
    
    # Convert to HuggingFace Dataset
    return Dataset.from_list(formatted_samples)


def create_dataset(num_samples=NUM_SAMPLES, dt=DT, steps=STEPS):
    """Create and return a dataset."""
    return generate_control_dataset(num_samples, dt, steps)