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
    # Look for <CONTROLS>...</CONTROLS> pattern
    control_match = re.search(r"<CONTROLS>(.*?)</CONTROLS>", response, re.DOTALL)
    if not control_match:
        # Fallback for models that don't use the CONTROLS tag
        # Try to find a comma-separated list of floats at the end
        fallback_match = re.search(r"([\d\.\s,-]+)$", response)
        if fallback_match:
            control_text = fallback_match.group(1).strip()
        else:
            print("❌ No control sequence found in response.")
            return None
    else:
        control_text = control_match.group(1).strip()
    
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
        
        # Extract controls
        controls = extract_controls_from_response(output, steps)
        
        if controls is None:
            return {
                "success": False,
                "error": "Failed to extract controls",
                "response": output[:200] + "..." if len(output) > 200 else output
            }
        
        # Simulate trajectory
        times, positions, velocities = simulate_trajectory(x0, v0, controls, dt)
        
        # Calculate final error
        final_pos = positions[-1]
        final_vel = velocities[-1]
        final_error = np.sqrt(final_pos**2 + final_vel**2)
        
        # Check constraints
        valid_controls = all(-3 <= u <= 3 for u in controls)
        valid_states = all(-1 <= p <= 1 and -1 <= v <= 1 for p, v in zip(positions, velocities))
        
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

def plot_trajectories_notebook_style(results, save_path=None):
    """Plot trajectories exactly like the notebook."""
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 10,
        'figure.titlesize': 18
    })
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    fig.suptitle('Double Integrator: Model Generated Trajectories', fontsize=16)
    
    success_count = 0
    for idx, result in enumerate(results):
        if not result["success"]:
            continue
            
        success_count += 1
        label = f'Case {idx+1} (err={result["final_error"]:.3f})'
        
        # Position plot
        axes[0].plot(result["times"], result["positions"], 'o-', label=label, alpha=0.7)
        
        # Velocity plot  
        axes[1].plot(result["times"], result["velocities"], 'o-', label=label, alpha=0.7)
        
        # Control plot
        axes[2].step(result["times"][:-1], result["controls"], where='post', label=label, alpha=0.7)
    
    # Configure plots
    for ax in axes:
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    axes[0].set_ylabel('Position')
    axes[0].set_ylim(-1.1, 1.1)
    
    axes[1].set_ylabel('Velocity')  
    axes[1].set_ylim(-1.1, 1.1)
    
    axes[2].set_ylabel('Control Input')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylim(-3.1, 3.1)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Trajectory plot saved: {save_path}")
    
    plt.close()
    return success_count

def main():
    parser = argparse.ArgumentParser(description="Evaluate model using notebook-style approach")
    parser.add_argument("--model-path", type=str, required=True, help="Path to LoRA model")
    parser.add_argument("--num-cases", type=int, default=10, help="Number of test cases")
    parser.add_argument("--save-dir", type=str, default="figures/notebook_eval", help="Save directory")
    parser.add_argument("--dt", type=float, default=0.1, help="Time step")
    parser.add_argument("--steps", type=int, default=50, help="Number of steps")
    
    args = parser.parse_args()
    
    print(f"🧪 Notebook-Style Evaluation")
    print(f"📁 Model: {args.model_path}")
    print(f"🎯 Test cases: {args.num_cases}")
    print(f"⏱️  Time step: {args.dt}, Steps: {args.steps}")
    
    # Load model
    model, tokenizer, lora_request = load_model_notebook_style(args.model_path)
    
    # Generate test cases (random initial states)
    print(f"\n🎲 Generating {args.num_cases} random test cases...")
    np.random.seed(42)  # For reproducibility
    test_cases = []
    for i in range(args.num_cases):
        x0 = np.random.uniform(-0.8, 0.8)
        v0 = np.random.uniform(-0.8, 0.8)
        test_cases.append((x0, v0))
        print(f"   Case {i+1}: pos={x0:.4f}, vel={v0:.4f}")
    
    # Evaluate each case
    print(f"\n🚀 Running evaluation...")
    results = []
    for i, (x0, v0) in enumerate(test_cases):
        print(f"   Processing case {i+1}/{args.num_cases}: ({x0:.4f}, {v0:.4f})")
        result = evaluate_single_case(model, tokenizer, lora_request, x0, v0, args.dt, args.steps)
        results.append(result)
        
        if result["success"]:
            print(f"      ✅ Success - Final error: {result['final_error']:.4f}")
        else:
            print(f"      ❌ Failed - {result['error']}")
    
    # Calculate statistics
    successful_results = [r for r in results if r["success"]]
    success_rate = len(successful_results) / len(results) * 100
    
    if successful_results:
        final_errors = [r["final_error"] for r in successful_results]
        mean_error = np.mean(final_errors)
        print(f"\n📊 Results:")
        print(f"   Success rate: {success_rate:.1f}% ({len(successful_results)}/{len(results)})")
        print(f"   Mean final error: {mean_error:.4f}")
        print(f"   Error range: {min(final_errors):.4f} - {max(final_errors):.4f}")
        
        # Plot trajectories
        save_path = os.path.join(args.save_dir, "trajectories_notebook_style.png")
        plot_count = plot_trajectories_notebook_style(results, save_path)
        print(f"   Plotted {plot_count} successful trajectories")
        
        # Save detailed results
        results_path = os.path.join(args.save_dir, "evaluation_results.json")
        os.makedirs(args.save_dir, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = []
        for r in results:
            json_result = r.copy()
            for key in ["times", "positions", "velocities", "controls"]:
                if key in json_result and json_result[key] is not None:
                    json_result[key] = [float(x) for x in json_result[key]]
            json_results.append(json_result)
        
        with open(results_path, 'w') as f:
            json.dump({
                "summary": {
                    "success_rate": success_rate,
                    "mean_error": mean_error,
                    "num_cases": len(results),
                    "successful_cases": len(successful_results)
                },
                "results": json_results
            }, f, indent=2)
        
        print(f"   📄 Detailed results saved: {results_path}")
        
    else:
        print(f"\n❌ No successful results - all cases failed!")
        
    print(f"\n🎉 Evaluation completed!")

if __name__ == "__main__":
    main()