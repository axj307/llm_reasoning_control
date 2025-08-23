#!/usr/bin/env python3
"""
Simplified evaluation script that matches the Jupyter notebook approach.
Uses direct model loading, fast_generate(), and simple trajectory plotting.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from unsloth import FastLanguageModel
from vllm import SamplingParams

# Import optimal control solver
sys.path.append(str(Path(__file__).parent.parent / 'core' / 'solvers'))
from lqr_solver import solve_double_integrator_lqr

def get_system_prompt(dt, steps):
    """Get system prompt for given dt and steps - matches notebook exactly."""
    total_time = dt * steps
    reasoning_start = "<REASONING>"
    reasoning_end = "</REASONING>"
    solution_start = "<CONTROLS>"
    solution_end = "</CONTROLS>"
    
    return f"""You are a control systems expert.
Given a double integrator system (ẍ = u) with initial position and velocity,
generate a sequence of {steps} control inputs to reach the origin (0,0) in exactly {total_time:.2f} seconds.
Position and velocity must stay within [-1, 1], and control inputs must be within [-3, 3].
Explain your approach between {reasoning_start} and {reasoning_end}.
Then provide exactly {steps} control values as a comma-separated list between {solution_start} and {solution_end}."""

def load_model_notebook_style(model_path):
    """Load model exactly like the notebook does."""
    print(f"🚀 Loading model from: {model_path}")
    
    # Load base model with same parameters as notebook
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-4B-Base",
        max_seq_length=2048,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=32,
        gpu_memory_utilization=0.7,
    )
    
    # Enable PEFT model for LoRA (required for load_lora to work)
    model = FastLanguageModel.get_peft_model(
        model,
        r=32, 
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=64,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=True,
        loftq_config=None,
    )

    # Set up chat template (EXACT match with training script)
    reasoning_start = "<REASONING>"
    reasoning_end = "</REASONING>"
    solution_start = "<CONTROLS>"
    solution_end = "</CONTROLS>"
    system_prompt = get_system_prompt(0.1, 50)  # Default values

    # Chat template (exact from training script)
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

    # Replace placeholders (exact from training script)  
    chat_template = chat_template\
        .replace("'{system_prompt}'", f"'{system_prompt}'")\
        .replace("'{reasoning_start}'", f"'{reasoning_start}'")
    tokenizer.chat_template = chat_template
    
    # Load LoRA weights
    try:
        lora_request = model.load_lora(model_path)
        print(f"✅ Model loaded with LoRA from: {model_path}")
    except Exception as e:
        print(f"❌ Failed to load LoRA weights from {model_path}: {e}")
        # Fallback to base model if LoRA fails
        lora_request = None
    
    return model, tokenizer, lora_request

def extract_controls_from_response(response, steps=50):
    """Extract control values from model response with improved error handling."""
    print(f"🔍 DEBUG: Looking for <CONTROLS>...</CONTROLS> pattern...")
    
    # Look for <CONTROLS>...</CONTROLS> pattern
    control_match = re.search(r"<CONTROLS>(.*?)</CONTROLS>", response, re.DOTALL)
    if not control_match:
        print(f"❌ DEBUG: No <CONTROLS> tags found, trying fallback...")
        # Fallback for models that don't use the CONTROLS tag
        # Try to find a comma-separated list of floats at the end
        fallback_match = re.search(r"([\d\.\s,-]+)$", response)
        if fallback_match:
            control_text = fallback_match.group(1).strip()
            print(f"✅ DEBUG: Found fallback pattern: '{control_text[:50]}...'")
        else:
            print("❌ DEBUG: No control sequence found in response (no fallback pattern either)")
            return None
    else:
        control_text = control_match.group(1).strip()
        print(f"✅ DEBUG: Found <CONTROLS> tags with content: '{control_text[:50]}...'")
    
    try:
        # Clean up the text: remove brackets, newlines, and extra spaces
        control_text = control_text.replace("[", "").replace("]", "").replace("\n", " ")
        controls = [float(x.strip()) for x in control_text.split(",") if x.strip()]
        
        if len(controls) == steps:
            return controls
        else:
            print(f"⚠️  Warning: Expected {steps} controls, but found {len(controls)}.")
            # Pad with zeros or truncate if necessary
            if len(controls) > steps:
                return controls[:steps]
            else:
                return controls + [0.0] * (steps - len(controls))
                
    except ValueError as e:
        print(f"❌ Failed to parse controls due to ValueError: {e}")
        print(f"   Raw control text: '{control_text}'")
        return None
    except Exception as e:
        print(f"❌ An unexpected error occurred during control parsing: {e}")
        return None

def simulate_trajectory(x0, v0, controls, dt=0.1):
    """Simulate double integrator trajectory."""
    x, v = x0, v0
    positions = [x]
    velocities = [v]
    times = [0]
    
    for i, u in enumerate(controls):
        # Apply control with bounds checking
        u = max(-3.0, min(3.0, u))
        v = v + u * dt
        x = x + v * dt
        
        # State bounds checking
        v = max(-1.0, min(1.0, v))
        x = max(-1.0, min(1.0, x))
        
        positions.append(x)
        velocities.append(v)
        times.append((i+1) * dt)
    
    return times, positions, velocities

def evaluate_single_case_mpc(model, tokenizer, lora_request, x0, v0, dt=0.1, total_steps=50, mpc_horizon=10):
    """
    Evaluate model using MPC-style control - step by step horizon planning.
    
    Args:
        model, tokenizer, lora_request: Model components
        x0, v0: Initial state
        dt: Time step
        total_steps: Total control steps
        mpc_horizon: Planning horizon for each MPC step
        
    Returns:
        Dictionary with MPC trajectory results
    """
    print(f"🎯 MPC Evaluation: Initial ({x0:.4f}, {v0:.4f}), Horizon={mpc_horizon}")
    
    # Initialize state and trajectory
    current_state = [x0, v0]
    mpc_trajectory = [current_state.copy()]
    mpc_controls = []
    mpc_step_details = []
    
    for step in range(total_steps):
        remaining_steps = min(mpc_horizon, total_steps - step)
        remaining_time = remaining_steps * dt
        
        print(f"  Step {step}: State=[{current_state[0]:.4f}, {current_state[1]:.4f}], Planning {remaining_steps} steps ahead")
        
        # Create MPC prompt for current state and remaining horizon
        system_prompt = get_system_prompt(dt, remaining_steps)
        test_prompt = (f"Control a double integrator system with initial state "
                      f"[position={current_state[0]:.4f}, velocity={current_state[1]:.4f}] "
                      f"to reach the origin (0,0) in {remaining_time:.2f} seconds using {remaining_steps} steps. "
                      f"Ensure all states remain within [-1,1] and controls within [-3,3].")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": test_prompt},
        ]
        
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        
        # Generate controls for this horizon
        sampling_params = SamplingParams(
            temperature=0.7,
            top_k=50,
            max_tokens=1024,
        )
        
        try:
            output = model.fast_generate(
                text,
                sampling_params=sampling_params,
                lora_request=lora_request,
            )[0].outputs[0].text
            
            # Extract controls for this horizon
            horizon_controls = extract_controls_from_response(output, remaining_steps)
            
            if horizon_controls is None:
                print(f"    ❌ Failed to extract controls at step {step}")
                return {
                    "success": False,
                    "error": f"MPC control extraction failed at step {step}",
                    "mpc_trajectory": mpc_trajectory,
                    "mpc_controls": mpc_controls,
                    "step_details": mpc_step_details
                }
            
            # Apply only the first control (MPC principle)
            first_control = horizon_controls[0]
            mpc_controls.append(first_control)
            
            # Store step details for analysis
            step_detail = {
                "step": step,
                "state_before": current_state.copy(),
                "planned_controls": horizon_controls,
                "applied_control": first_control,
                "remaining_steps": remaining_steps
            }
            mpc_step_details.append(step_detail)
            
            print(f"    ✅ Planned {len(horizon_controls)} controls, applied first: {first_control:.3f}")
            
            # Simulate one step forward
            x, v = current_state
            v = v + first_control * dt
            x = x + v * dt
            
            # Apply bounds
            x = max(-1.0, min(1.0, x))
            v = max(-1.0, min(1.0, v))
            
            current_state = [x, v]
            mpc_trajectory.append(current_state.copy())
                
        except Exception as e:
            print(f"    ❌ MPC step {step} failed: {e}")
            return {
                "success": False,
                "error": f"MPC step {step} failed: {str(e)}",
                "mpc_trajectory": mpc_trajectory,
                "mpc_controls": mpc_controls,
                "step_details": mpc_step_details
            }
    
    # Calculate final metrics
    final_state = mpc_trajectory[-1]
    final_error = np.sqrt(final_state[0]**2 + final_state[1]**2)
    
    print(f"  🏁 MPC Final: [{final_state[0]:.4f}, {final_state[1]:.4f}], Error: {final_error:.4f}")
    
    return {
        "success": True,
        "mpc_trajectory": mpc_trajectory,
        "mpc_controls": mpc_controls,
        "final_error": final_error,
        "step_details": mpc_step_details,
        "mpc_horizon": mpc_horizon,
        "total_steps": len(mpc_controls)
    }

def evaluate_single_case(model, tokenizer, lora_request, x0, v0, dt=0.1, steps=50):
    """Evaluate model on a single test case - notebook style."""
    total_time = dt * steps
    system_prompt = get_system_prompt(dt, steps)
    
    # Create test prompt exactly like notebook
    test_prompt = f"Control a double integrator system with initial state [position={x0:.4f}, velocity={v0:.4f}] to reach the origin (0,0) in {total_time:.2f} seconds using {steps} steps. Ensure all states remain within [-1,1] and controls within [-3,3]."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": test_prompt},
    ]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    
    # Generate using fast_generate (notebook style)
    sampling_params = SamplingParams(
        temperature=0.7,
        top_k=50,
        max_tokens=1024,
    )
    
    try:
        output = model.fast_generate(
            text,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )[0].outputs[0].text
        
        # Debug output: Show complete model response
        print(f"\n📝 DEBUG: Complete model output for case ({x0:.4f}, {v0:.4f})")
        print("=" * 80)
        print(output)
        print("=" * 80)
        
        # Extract controls with debug info
        print(f"🔍 DEBUG: Extracting controls from response...")
        controls = extract_controls_from_response(output, steps)
        
        if controls is None:
            print(f"❌ DEBUG: Control extraction failed!")
            return {
                "success": False,
                "error": "Failed to extract controls",
                "response": output[:200] + "..." if len(output) > 200 else output
            }
        else:
            print(f"✅ DEBUG: Successfully extracted {len(controls)} controls")
            print(f"🎯 DEBUG: First 5 controls: {controls[:5]}")
            print(f"🎯 DEBUG: Last 5 controls: {controls[-5:]}")
        
        # Simulate trajectory
        times, positions, velocities = simulate_trajectory(x0, v0, controls, dt)
        
        # Calculate final error
        final_pos = positions[-1]
        final_vel = velocities[-1]
        final_error = np.sqrt(final_pos**2 + final_vel**2)
        
        # Debug trajectory results
        print(f"🚀 DEBUG: Trajectory simulation results:")
        print(f"   Initial state: pos={x0:.4f}, vel={v0:.4f}")
        print(f"   Final state: pos={final_pos:.4f}, vel={final_vel:.4f}")
        print(f"   Final error: {final_error:.4f}")
        
        # Check constraints
        valid_controls = all(-3 <= u <= 3 for u in controls)
        valid_states = all(-1 <= p <= 1 and -1 <= v <= 1 for p, v in zip(positions, velocities))
        
        print(f"✅ DEBUG: Constraint validation:")
        print(f"   Valid controls ([-3,3]): {valid_controls}")
        print(f"   Valid states ([-1,1]): {valid_states}")
        if not valid_controls:
            invalid_controls = [u for u in controls if not (-3 <= u <= 3)]
            print(f"   ⚠️ Invalid controls found: {invalid_controls[:3]}...")
        if not valid_states:
            print(f"   ⚠️ State constraint violations detected")
        
        return {
            "success": True,
            "controls": controls,
            "times": times,
            "positions": positions,
            "velocities": velocities,
            "final_error": final_error,
            "valid_controls": valid_controls,
            "valid_states": valid_states,
            "response": output
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "response": ""
        }

def plot_mpc_trajectories(mpc_results, save_path=None, trajectory_color='#2E86AB'):
    """Plot MPC trajectories in professional 2x2 layout with phase space."""
    # Set professional style with larger fonts and tick sizes
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 20,
        'axes.labelsize': 24,
        'axes.titlesize': 26,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'xtick.major.size': 8,
        'ytick.major.size': 8,
        'xtick.major.width': 2,
        'ytick.major.width': 2,
        'legend.fontsize': 18,
        'figure.titlesize': 28,
        'lines.linewidth': 3.0,
        'lines.markersize': 7,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.axisbelow': True
    })
    
    successful_mpc = [r for r in mpc_results if r["success"]]
    if not successful_mpc:
        print("No successful MPC results to plot")
        return 0
    
    # Create 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for i, result in enumerate(successful_mpc):
        trajectory = result["mpc_trajectory"]
        controls = result["mpc_controls"]
        
        positions = [state[0] for state in trajectory]
        velocities = [state[1] for state in trajectory]
        
        # Times for states (n+1 points) and controls (n points)
        state_times = np.arange(len(positions)) * 0.1
        control_times = np.arange(len(controls)) * 0.1
        
        # Use consistent color for all trajectories
        color = trajectory_color
        
        # Phase Space Plot (top-left)
        axes[0,0].plot(positions, velocities, 'o-', color=color, 
                      alpha=0.7, linewidth=3, markersize=5)
        axes[0,0].plot(positions[0], velocities[0], 'o', color=color, markersize=10, 
                      markeredgecolor='black', markeredgewidth=2)  # Start point
        axes[0,0].plot(positions[-1], velocities[-1], 's', color=color, markersize=10, 
                      markeredgecolor='black', markeredgewidth=2)  # End point
        
        # Position vs Time (top-right)
        axes[0,1].plot(state_times, positions, 'o-', color=color, 
                      alpha=0.7, linewidth=3, markersize=5)
        
        # Velocity vs Time (bottom-left)
        axes[1,0].plot(state_times, velocities, 'o-', color=color, 
                      alpha=0.7, linewidth=3, markersize=5)
        
        # Control vs Time (bottom-right)
        axes[1,1].step(control_times, controls, where='post', color=color, 
                      alpha=0.7, linewidth=3)

    # Configure Phase Space Plot (top-left) - NO TITLE, NO LEGEND
    axes[0,0].set_xlabel('Position', fontweight='bold')
    axes[0,0].set_ylabel('Velocity', fontweight='bold')
    axes[0,0].axhline(y=0, color='red', linestyle='--', alpha=0.6, linewidth=2)
    axes[0,0].axvline(x=0, color='red', linestyle='--', alpha=0.6, linewidth=2)
    axes[0,0].plot(0, 0, 'r*', markersize=15, markeredgecolor='darkred', markeredgewidth=2)  # Target
    # Constraint boundaries
    axes[0,0].axhspan(-1, 1, alpha=0.1, color='gray')
    axes[0,0].axvspan(-1, 1, alpha=0.1, color='gray')
    axes[0,0].set_xlim(-1.1, 1.1)
    axes[0,0].set_ylim(-1.1, 1.1)
    axes[0,0].grid(True, alpha=0.3)
    
    # Configure Position Plot (top-right) - NO TITLE, NO LEGEND
    axes[0,1].set_xlabel('Time (s)', fontweight='bold')
    axes[0,1].set_ylabel('Position', fontweight='bold')
    axes[0,1].axhline(y=0, color='red', linestyle='--', alpha=0.6, linewidth=2)
    axes[0,1].axhspan(-1, 1, alpha=0.1, color='gray')
    axes[0,1].set_ylim(-1.1, 1.1)
    axes[0,1].grid(True, alpha=0.3)
    
    # Configure Velocity Plot (bottom-left) - NO TITLE, NO LEGEND
    axes[1,0].set_xlabel('Time (s)', fontweight='bold')
    axes[1,0].set_ylabel('Velocity', fontweight='bold')
    axes[1,0].axhline(y=0, color='red', linestyle='--', alpha=0.6, linewidth=2)
    axes[1,0].axhspan(-1, 1, alpha=0.1, color='gray')
    axes[1,0].set_ylim(-1.1, 1.1)
    axes[1,0].grid(True, alpha=0.3)
    
    # Configure Control Plot (bottom-right) - NO TITLE, NO LEGEND
    axes[1,1].set_xlabel('Time (s)', fontweight='bold')
    axes[1,1].set_ylabel('Control Input', fontweight='bold')
    axes[1,1].axhline(y=0, color='red', linestyle='--', alpha=0.6, linewidth=2)
    # Control constraint bounds
    axes[1,1].axhline(y=3, color='red', linestyle=':', alpha=0.8, linewidth=2)
    axes[1,1].axhline(y=-3, color='red', linestyle=':', alpha=0.8, linewidth=2)
    axes[1,1].axhspan(-3, 3, alpha=0.1, color='gray')
    axes[1,1].set_ylim(-3.2, 3.2)
    axes[1,1].grid(True, alpha=0.3)
    
    # Adjust layout for clean appearance
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', format='png')
        print(f"✅ Clean MPC trajectory plot saved: {save_path}")
    
    plt.close()
    return len(successful_mpc)

def plot_trajectories_notebook_style(results, save_path=None, trajectory_color='#A23B72'):
    """Plot standard trajectories in professional 2x2 layout with phase space."""
    # Set professional style with larger fonts
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 20,
        'axes.labelsize': 24,
        'axes.titlesize': 26,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'xtick.major.size': 8,
        'ytick.major.size': 8,
        'xtick.major.width': 2,
        'ytick.major.width': 2,
        'legend.fontsize': 18,
        'figure.titlesize': 28,
        'lines.linewidth': 3.0,
        'lines.markersize': 7,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.axisbelow': True
    })
    
    # Filter successful results
    successful_results = [r for r in results if r["success"]]
    if not successful_results:
        print("No successful results to plot")
        return 0
    
    # Create 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    success_count = 0
    for idx, result in enumerate(successful_results):
        success_count += 1
        
        # Use consistent color for all trajectories
        color = trajectory_color
        
        positions = result["positions"]
        velocities = result["velocities"]
        times = result["times"]
        controls = result["controls"]
        
        # Phase Space Plot (top-left)
        axes[0,0].plot(positions, velocities, 'o-', color=color, 
                      alpha=0.7, linewidth=3, markersize=5)
        axes[0,0].plot(positions[0], velocities[0], 'o', color=color, markersize=10, 
                      markeredgecolor='black', markeredgewidth=2)  # Start point
        axes[0,0].plot(positions[-1], velocities[-1], 's', color=color, markersize=10, 
                      markeredgecolor='black', markeredgewidth=2)  # End point
        
        # Position vs Time (top-right)
        axes[0,1].plot(times, positions, 'o-', color=color, 
                      alpha=0.7, linewidth=3, markersize=5)
        
        # Velocity vs Time (bottom-left)
        axes[1,0].plot(times, velocities, 'o-', color=color, 
                      alpha=0.7, linewidth=3, markersize=5)
        
        # Control vs Time (bottom-right)
        axes[1,1].step(times[:-1], controls, where='post', color=color, 
                      alpha=0.7, linewidth=3)

    # Configure Phase Space Plot (top-left) - NO TITLE, NO LEGEND
    axes[0,0].set_xlabel('Position', fontweight='bold')
    axes[0,0].set_ylabel('Velocity', fontweight='bold')
    axes[0,0].axhline(y=0, color='red', linestyle='--', alpha=0.6, linewidth=2)
    axes[0,0].axvline(x=0, color='red', linestyle='--', alpha=0.6, linewidth=2)
    axes[0,0].plot(0, 0, 'r*', markersize=15, markeredgecolor='darkred', markeredgewidth=2)  # Target
    # Constraint boundaries
    axes[0,0].axhspan(-1, 1, alpha=0.1, color='gray')
    axes[0,0].axvspan(-1, 1, alpha=0.1, color='gray')
    axes[0,0].set_xlim(-1.1, 1.1)
    axes[0,0].set_ylim(-1.1, 1.1)
    axes[0,0].grid(True, alpha=0.3)
    
    # Configure Position Plot (top-right) - NO TITLE, NO LEGEND
    axes[0,1].set_xlabel('Time (s)', fontweight='bold')
    axes[0,1].set_ylabel('Position', fontweight='bold')
    axes[0,1].axhline(y=0, color='red', linestyle='--', alpha=0.6, linewidth=2)
    axes[0,1].axhspan(-1, 1, alpha=0.1, color='gray')
    axes[0,1].set_ylim(-1.1, 1.1)
    axes[0,1].grid(True, alpha=0.3)
    
    # Configure Velocity Plot (bottom-left) - NO TITLE, NO LEGEND
    axes[1,0].set_xlabel('Time (s)', fontweight='bold')
    axes[1,0].set_ylabel('Velocity', fontweight='bold')
    axes[1,0].axhline(y=0, color='red', linestyle='--', alpha=0.6, linewidth=2)
    axes[1,0].axhspan(-1, 1, alpha=0.1, color='gray')
    axes[1,0].set_ylim(-1.1, 1.1)
    axes[1,0].grid(True, alpha=0.3)
    
    # Configure Control Plot (bottom-right) - NO TITLE, NO LEGEND
    axes[1,1].set_xlabel('Time (s)', fontweight='bold')
    axes[1,1].set_ylabel('Control Input', fontweight='bold')
    axes[1,1].axhline(y=0, color='red', linestyle='--', alpha=0.6, linewidth=2)
    # Control constraint bounds
    axes[1,1].axhline(y=3, color='red', linestyle=':', alpha=0.8, linewidth=2)
    axes[1,1].axhline(y=-3, color='red', linestyle=':', alpha=0.8, linewidth=2)
    axes[1,1].axhspan(-3, 3, alpha=0.1, color='gray')
    axes[1,1].set_ylim(-3.2, 3.2)
    axes[1,1].grid(True, alpha=0.3)
    
    # Adjust layout for clean appearance
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', format='png')
        print(f"✅ Clean trajectory plot saved: {save_path}")
    
    plt.close()
    return success_count

def generate_optimal_trajectories(test_cases, dt=0.1, steps=50):
    """Generate optimal LQR trajectories for given test cases."""
    optimal_results = []
    
    for x0, v0 in test_cases:
        # Generate optimal controls using LQR
        optimal_controls = solve_double_integrator_lqr([x0, v0], dt, steps)
        
        # Simulate the trajectory with optimal controls
        times, positions, velocities = simulate_trajectory(x0, v0, optimal_controls, dt)
        
        # Calculate final error
        final_pos = positions[-1]
        final_vel = velocities[-1]
        final_error = np.sqrt(final_pos**2 + final_vel**2)
        
        optimal_results.append({
            "success": True,
            "controls": optimal_controls,
            "times": times,
            "positions": positions,
            "velocities": velocities,
            "final_error": final_error,
            "initial_state": [x0, v0]
        })
    
    return optimal_results

def plot_optimal_vs_grpo_comparison(grpo_results, optimal_results, save_path=None):
    """Plot comparison between optimal LQR and GRPO trajectories in professional 2x2 layout."""
    # Set professional style with larger fonts
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 20,
        'axes.labelsize': 24,
        'axes.titlesize': 26,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'xtick.major.size': 8,
        'ytick.major.size': 8,
        'xtick.major.width': 2,
        'ytick.major.width': 2,
        'legend.fontsize': 18,
        'figure.titlesize': 28,
        'lines.linewidth': 3.0,
        'lines.markersize': 7,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.axisbelow': True
    })
    
    # Filter successful results
    successful_grpo = [r for r in grpo_results if r["success"]]
    successful_optimal = [r for r in optimal_results if r["success"]]
    
    if not successful_grpo or not successful_optimal:
        print("No successful results to compare")
        return 0
    
    # Create 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Define colors
    optimal_color = '#2E86AB'  # Blue for optimal solutions
    grpo_color = '#A23B72'     # Magenta for GRPO trajectories
    
    # Plot optimal trajectories (blue)
    for idx, result in enumerate(successful_optimal):
        positions = result["positions"]
        velocities = result["velocities"] 
        times = result["times"]
        controls = result["controls"]
        
        # Phase Space Plot (top-left)
        axes[0,0].plot(positions, velocities, 'o-', color=optimal_color, 
                      alpha=0.7, linewidth=3, markersize=5)
        axes[0,0].plot(positions[0], velocities[0], 'o', color=optimal_color, markersize=10, 
                      markeredgecolor='black', markeredgewidth=2)  # Start point
        axes[0,0].plot(positions[-1], velocities[-1], 's', color=optimal_color, markersize=10, 
                      markeredgecolor='black', markeredgewidth=2)  # End point
        
        # Position vs Time (top-right)
        axes[0,1].plot(times, positions, 'o-', color=optimal_color, 
                      alpha=0.7, linewidth=3, markersize=5)
        
        # Velocity vs Time (bottom-left)
        axes[1,0].plot(times, velocities, 'o-', color=optimal_color, 
                      alpha=0.7, linewidth=3, markersize=5)
        
        # Control vs Time (bottom-right)
        axes[1,1].step(times[:-1], controls, where='post', color=optimal_color, 
                      alpha=0.7, linewidth=3)

    # Plot GRPO trajectories (magenta)
    for idx, result in enumerate(successful_grpo):
        positions = result["positions"]
        velocities = result["velocities"]
        times = result["times"]
        controls = result["controls"]
        
        # Phase Space Plot (top-left)
        axes[0,0].plot(positions, velocities, 'o-', color=grpo_color, 
                      alpha=0.7, linewidth=3, markersize=5)
        axes[0,0].plot(positions[0], velocities[0], 'o', color=grpo_color, markersize=10, 
                      markeredgecolor='black', markeredgewidth=2)  # Start point
        axes[0,0].plot(positions[-1], velocities[-1], 's', color=grpo_color, markersize=10, 
                      markeredgecolor='black', markeredgewidth=2)  # End point
        
        # Position vs Time (top-right)
        axes[0,1].plot(times, positions, 'o-', color=grpo_color, 
                      alpha=0.7, linewidth=3, markersize=5)
        
        # Velocity vs Time (bottom-left)
        axes[1,0].plot(times, velocities, 'o-', color=grpo_color, 
                      alpha=0.7, linewidth=3, markersize=5)
        
        # Control vs Time (bottom-right)
        axes[1,1].step(times[:-1], controls, where='post', color=grpo_color, 
                      alpha=0.7, linewidth=3)

    # Configure Phase Space Plot (top-left)
    axes[0,0].set_xlabel('Position', fontweight='bold')
    axes[0,0].set_ylabel('Velocity', fontweight='bold')
    axes[0,0].axhline(y=0, color='red', linestyle='--', alpha=0.6, linewidth=2)
    axes[0,0].axvline(x=0, color='red', linestyle='--', alpha=0.6, linewidth=2)
    axes[0,0].plot(0, 0, 'r*', markersize=15, markeredgecolor='darkred', markeredgewidth=2)  # Target
    # Constraint boundaries
    axes[0,0].axhspan(-1, 1, alpha=0.1, color='gray')
    axes[0,0].axvspan(-1, 1, alpha=0.1, color='gray')
    axes[0,0].set_xlim(-1.1, 1.1)
    axes[0,0].set_ylim(-1.1, 1.1)
    axes[0,0].grid(True, alpha=0.3)
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=optimal_color, linewidth=3, label='Optimal (LQR)'),
                      Line2D([0], [0], color=grpo_color, linewidth=3, label='GRPO Model')]
    axes[0,0].legend(handles=legend_elements, loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Configure Position Plot (top-right)
    axes[0,1].set_xlabel('Time (s)', fontweight='bold')
    axes[0,1].set_ylabel('Position', fontweight='bold')
    axes[0,1].axhline(y=0, color='red', linestyle='--', alpha=0.6, linewidth=2)
    axes[0,1].axhspan(-1, 1, alpha=0.1, color='gray')
    axes[0,1].set_ylim(-1.1, 1.1)
    axes[0,1].grid(True, alpha=0.3)
    
    # Configure Velocity Plot (bottom-left)
    axes[1,0].set_xlabel('Time (s)', fontweight='bold')
    axes[1,0].set_ylabel('Velocity', fontweight='bold')
    axes[1,0].axhline(y=0, color='red', linestyle='--', alpha=0.6, linewidth=2)
    axes[1,0].axhspan(-1, 1, alpha=0.1, color='gray')
    axes[1,0].set_ylim(-1.1, 1.1)
    axes[1,0].grid(True, alpha=0.3)
    
    # Configure Control Plot (bottom-right)
    axes[1,1].set_xlabel('Time (s)', fontweight='bold')
    axes[1,1].set_ylabel('Control Input', fontweight='bold')
    axes[1,1].axhline(y=0, color='red', linestyle='--', alpha=0.6, linewidth=2)
    # Control constraint bounds
    axes[1,1].axhline(y=3, color='red', linestyle=':', alpha=0.8, linewidth=2)
    axes[1,1].axhline(y=-3, color='red', linestyle=':', alpha=0.8, linewidth=2)
    axes[1,1].axhspan(-3, 3, alpha=0.1, color='gray')
    axes[1,1].set_ylim(-3.2, 3.2)
    axes[1,1].grid(True, alpha=0.3)
    
    # Adjust layout for clean appearance
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', format='png')
        print(f"✅ Optimal vs GRPO comparison plot saved: {save_path}")
    
    plt.close()
    return len(successful_grpo)

def main():
    parser = argparse.ArgumentParser(description="Evaluate model using notebook-style approach")
    parser.add_argument("--model-path", type=str, required=True, help="Path to LoRA model")
    parser.add_argument("--num-cases", type=int, default=10, help="Number of test cases")
    parser.add_argument("--save-dir", type=str, default="figures/notebook_eval", help="Save directory")
    parser.add_argument("--dt", type=float, default=0.1, help="Time step")
    parser.add_argument("--steps", type=int, default=50, help="Number of steps")
    parser.add_argument("--mpc-horizon", type=int, default=10, help="MPC planning horizon")
    parser.add_argument("--skip-mpc", action="store_true", help="Skip MPC evaluation for faster testing")
    
    args = parser.parse_args()
    
    print(f"🧪 Notebook-Style Evaluation")
    print(f"📁 Model: {args.model_path}")
    print(f"🎯 Test cases: {args.num_cases}")
    print(f"⏱️  Time step: {args.dt}, Steps: {args.steps}")
    
    # Debug MPC settings
    if args.skip_mpc:
        print(f"⏩ MPC Evaluation: SKIPPED (--skip-mpc flag enabled)")
        print(f"💪 Focusing on one-shot trajectory generation for faster evaluation")
    else:
        print(f"🎯 MPC Evaluation: ENABLED (Horizon: {args.mpc_horizon})")
        print(f"⚠️  This will be 50x slower - use --skip-mcp to disable")
    
    # Load model
    model, tokenizer, lora_request = load_model_notebook_style(args.model_path)
    
    # Use a fixed set of 10 test cases for consistent evaluation
    print(f"\n🎲 Using a fixed set of 10 test cases for evaluation...")
    test_cases = [
        ( 0.7,  0.3),
        (-0.5, -0.5),
        ( 0.6, -0.2),
        (-0.8,  0.4),
        ( 0.4,  0.6),
        (-0.3, -0.7),
        ( 0.9,  0.1),
        (-0.2,  0.8),
        ( 0.1, -0.9),
        (-0.6,  0.6),
    ]
    
    # If num-cases is specified and different from 10, notify user
    if args.num_cases != 10:
        print(f"⚠️  Warning: --num-cases is set to {args.num_cases}, but we are using a fixed set of 10 cases.")
    
    for i, (x0, v0) in enumerate(test_cases):
        print(f"   Case {i+1}: pos={x0:.4f}, vel={v0:.4f}")

    # Generate optimal trajectories for comparison
    print(f"\n🎯 Generating optimal LQR trajectories for comparison...")
    optimal_results = generate_optimal_trajectories(test_cases, args.dt, args.steps)
    print(f"✅ Generated {len(optimal_results)} optimal trajectories")

    # Evaluate each case with both standard and MPC approaches
    print(f"\n🚀 Running evaluation...")
    results = []
    mpc_results = []
    
    for i, (x0, v0) in enumerate(test_cases):
        print(f"\n   📊 Case {i+1}/{args.num_cases}: ({x0:.4f}, {v0:.4f})")
        
        # Standard evaluation (full horizon)
        print(f"      🔄 Standard evaluation...")
        result = evaluate_single_case(model, tokenizer, lora_request, x0, v0, args.dt, args.steps)
        results.append(result)
        
        if result["success"]:
            print(f"      ✅ Standard - Final error: {result['final_error']:.4f}")
        else:
            print(f"      ❌ Standard failed - {result['error']}")
        
        # MPC evaluation with configurable horizon (skip if --skip-mpc flag is used)
        if not args.skip_mpc:
            print(f"      🎯 MPC evaluation (horizon={args.mpc_horizon})...")
            mpc_result = evaluate_single_case_mpc(model, tokenizer, lora_request, x0, v0, args.dt, args.steps, mpc_horizon=args.mpc_horizon)
            mpc_results.append(mpc_result)
            
            if mpc_result["success"]:
                print(f"      ✅ MPC - Final error: {mpc_result['final_error']:.4f}")
            else:
                print(f"      ❌ MPC failed - {mpc_result['error']}")
        else:
            print(f"      ⏩ Skipping MPC evaluation (--skip-mpc flag enabled)")
            mpc_result = {"success": False, "skipped": True, "error": "MPC evaluation skipped"}
            mpc_results.append(mpc_result)
    
    # Ensure num_cases is correctly set for statistics
    num_cases = len(test_cases)
    
    # Calculate statistics for both approaches
    successful_results = [r for r in results if r["success"]]
    # Only count successful MPC results if MPC was actually run (not skipped)
    successful_mpc_results = [r for r in mpc_results if r["success"] and not r.get("skipped", False)]
    
    success_rate = len(successful_results) / num_cases * 100 if num_cases > 0 else 0
    mpc_success_rate = len(successful_mpc_results) / num_cases * 100 if num_cases > 0 else 0
    
    print(f"\n📊 Evaluation Results Comparison:")
    print(f"   📈 Standard Approach:")
    if successful_results:
        final_errors = [r["final_error"] for r in successful_results]
        mean_error = np.mean(final_errors)
        print(f"      Success rate: {success_rate:.1f}% ({len(successful_results)}/{num_cases})")
        print(f"      Mean final error: {mean_error:.4f}")
        print(f"      Error range: {min(final_errors):.4f} - {max(final_errors):.4f}")
    else:
        print(f"      Success rate: {success_rate:.1f}% ({len(successful_results)}/{num_cases})")
    
    # MPC results summary (check if MPC was actually run)
    mpc_skipped = any(r.get("skipped", False) for r in mpc_results)
    if not mpc_skipped:
        print(f"   🎯 MPC Approach (Horizon={args.mpc_horizon}):")
        if successful_mpc_results:
            mpc_final_errors = [r["final_error"] for r in successful_mpc_results]
            mpc_mean_error = np.mean(mpc_final_errors)
            print(f"      Success rate: {mpc_success_rate:.1f}% ({len(successful_mpc_results)}/{num_cases})")
            print(f"      Mean final error: {mpc_mean_error:.4f}")
            print(f"      Error range: {min(mpc_final_errors):.4f} - {max(mpc_final_errors):.4f}")
        else:
            print(f"      Success rate: {mpc_success_rate:.1f}% ({len(successful_mpc_results)}/{num_cases})")
    else:
        print(f"   ⏩ MPC Approach: Skipped (--skip-mpc flag enabled)")
        print(f"      To enable MPC evaluation, remove --skip-mpc flag from command")
    
    if successful_results and successful_mpc_results:
        # Safe division for comparison
        if mpc_mean_error > 0:
            print(f"   📊 Performance Comparison:")
            print(f"      Standard vs MPC error ratio: {mean_error/mpc_mean_error:.2f}")
            if mpc_mean_error < mean_error:
                print(f"      🎯 MPC performs better by {((mean_error-mpc_mean_error)/mean_error*100):.1f}%")
            else:
                print(f"      📈 Standard performs better by {((mpc_mean_error-mean_error)/mpc_mean_error*100):.1f}%")
        else:
            print(f"      📊 MPC Mean Error is zero, comparison not applicable.")
    
    # Plot trajectories for both approaches with standardized colors
    if successful_results:
        save_path = os.path.join(args.save_dir, "trajectories_standard.png")
        plot_count = plot_trajectories_notebook_style(results, save_path, trajectory_color='#A23B72')
        print(f"   📈 Plotted {plot_count} standard trajectories (RL - magenta)")
    
    if successful_mpc_results:
        save_path_mpc = os.path.join(args.save_dir, "trajectories_mpc.png")
        plot_count_mpc = plot_mpc_trajectories(mpc_results, save_path_mpc, trajectory_color='#2E86AB')
        print(f"   🎯 Plotted {plot_count_mpc} MPC trajectories (optimal - blue)")
    
    # Generate optimal vs GRPO comparison plot
    if successful_results and optimal_results:
        save_path_comparison = os.path.join(args.save_dir, "trajectories_optimal_vs_grpo.png")
        plot_count_comparison = plot_optimal_vs_grpo_comparison(results, optimal_results, save_path_comparison)
        print(f"   📊 Plotted comparison: {plot_count_comparison} GRPO vs {len(optimal_results)} optimal trajectories")
        
        # Calculate comparison statistics
        optimal_errors = [r["final_error"] for r in optimal_results]
        grpo_errors = [r["final_error"] for r in successful_results]
        
        mean_optimal_error = np.mean(optimal_errors)
        mean_grpo_error = np.mean(grpo_errors)
        
        print(f"   📈 Performance Comparison:")
        print(f"      Optimal LQR mean error: {mean_optimal_error:.4f}")
        print(f"      GRPO model mean error: {mean_grpo_error:.4f}")
        if mean_optimal_error > 0:
            error_ratio = mean_grpo_error / mean_optimal_error
            print(f"      GRPO/Optimal error ratio: {error_ratio:.2f}x")
            if error_ratio < 1.5:
                print(f"      🎉 GRPO performs very well (within 1.5x of optimal)")
            elif error_ratio < 3.0:
                print(f"      ✅ GRPO performs reasonably (within 3x of optimal)")
            else:
                print(f"      ⚠️  GRPO has room for improvement (>3x optimal error)")
        
    # Save detailed results for all approaches
    results_path = os.path.join(args.save_dir, "evaluation_results.json")
    mpc_results_path = os.path.join(args.save_dir, "evaluation_results_mpc.json")
    optimal_results_path = os.path.join(args.save_dir, "evaluation_results_optimal.json")
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_for_json(results_list):
        json_results = []
        for r in results_list:
            json_result = r.copy()
            for key in ["times", "positions", "velocities", "controls", "mpc_trajectory", "mpc_controls"]:
                if key in json_result and json_result[key] is not None:
                    if isinstance(json_result[key], list):
                        json_result[key] = [float(x) if isinstance(x, (int, float, np.number)) else x for x in json_result[key]]
                    elif isinstance(json_result[key], (int, float, np.number)):
                        json_result[key] = float(json_result[key])
            
            # Handle step_details for MPC results
            if "step_details" in json_result and json_result["step_details"] is not None:
                for detail in json_result["step_details"]:
                    for detail_key in ["state_before", "planned_controls", "applied_control"]:
                        if detail_key in detail and detail[detail_key] is not None:
                            if isinstance(detail[detail_key], list):
                                detail[detail_key] = [float(x) for x in detail[detail_key]]
                            elif isinstance(detail[detail_key], (int, float, np.number)):
                                detail[detail_key] = float(detail[detail_key])
                                
            json_results.append(json_result)
        return json_results
    
    # Save standard evaluation results
    json_results = convert_for_json(results)
    
    if successful_results:
        with open(results_path, 'w') as f:
            json.dump({
                "summary": {
                    "evaluation_type": "standard",
                    "success_rate": success_rate,
                    "mean_error": mean_error,
                    "num_cases": num_cases,
                    "successful_cases": len(successful_results)
                },
                "results": json_results
            }, f, indent=2)
        print(f"   📄 Standard results saved: {results_path}")
    
    # Save MPC evaluation results
    json_mpc_results = convert_for_json(mpc_results)
    
    if successful_mpc_results:
        with open(mpc_results_path, 'w') as f:
            json.dump({
                "summary": {
                    "evaluation_type": "mpc",
                    "mpc_horizon": args.mpc_horizon,
                    "success_rate": mpc_success_rate,
                    "mean_error": mpc_mean_error,
                    "num_cases": num_cases,
                    "successful_cases": len(successful_mpc_results)
                },
                "results": json_mpc_results
            }, f, indent=2)
        print(f"   📄 MPC results saved: {mpc_results_path}")
    
    # Save optimal results
    json_optimal_results = convert_for_json(optimal_results)
    with open(optimal_results_path, 'w') as f:
        json.dump({
            "summary": {
                "evaluation_type": "optimal_lqr", 
                "mean_error": mean_optimal_error if optimal_results else 0.0,
                "num_cases": len(optimal_results),
                "successful_cases": len(optimal_results)
            },
            "results": json_optimal_results
        }, f, indent=2)
    print(f"   📄 Optimal LQR results saved: {optimal_results_path}")
    
    if not successful_results:
        print(f"\n❌ No successful results - evaluation failed!")
    else:
        print(f"\n🎉 Evaluation completed!")
        print(f"📁 Results saved in: {args.save_dir}")
        print(f"📊 Generated plots:")
        print(f"   - trajectories_standard.png (GRPO trajectories)")
        print(f"   - trajectories_mpc.png (MPC trajectories)")
        print(f"   - trajectories_optimal_vs_grpo.png (Comparison plot)")
        
    print(f"\n🎉 All evaluations completed!")

if __name__ == "__main__":
    main()