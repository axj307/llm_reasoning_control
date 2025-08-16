#!/usr/bin/env python3
"""
Evaluate trained SFT and GRPO models and compare with optimal LQR control.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Any
import json
import re

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config
from environments.double_integrator import DoubleIntegrator
from core.solvers.lqr_solver import solve_double_integrator_lqr


def load_and_test_sft_model(test_cases: List[Tuple[float, float]]) -> Dict[str, Any]:
    """Load SFT model and test on initial conditions."""
    print("🤖 Loading and testing SFT model...")
    
    try:
        from unsloth import FastLanguageModel
        from transformers import AutoTokenizer
        from peft import PeftModel
        
        # Find the SFT model
        possible_paths = [
            "models/single_system/double_integrator/sft/latest",
            "temp_sft_output/checkpoint-3",
            "sft_trainer_lora_model"
        ]
        
        sft_model_path = None
        for path in possible_paths:
            if os.path.exists(path) and (os.path.exists(f"{path}/adapter_model.safetensors") or 
                                        os.path.exists(f"{path}/adapter_config.json")):
                sft_model_path = path
                break
        
        if sft_model_path is None:
            print(f"❌ SFT model not found in any of: {possible_paths}")
            return None
        
        print(f"📂 Loading SFT model from: {sft_model_path}")
        
        # Load base model and apply LoRA adapter
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen3-4B-Base",
            max_seq_length=2048,
            load_in_4bit=True,
            dtype=None,
        )
        
        # Load the adapter using PeftModel
        model = PeftModel.from_pretrained(model, sft_model_path)
        
        # Enable inference mode
        FastLanguageModel.for_inference(model)
        
        print("✅ SFT model loaded successfully")
        
        # Test the model
        results = test_model_on_cases(model, tokenizer, test_cases, "SFT")
        return results
        
    except Exception as e:
        print(f"❌ Error loading SFT model: {e}")
        return None


def load_and_test_grpo_model(test_cases: List[Tuple[float, float]]) -> Dict[str, Any]:
    """Load GRPO model and test on initial conditions."""
    print("🤖 Loading and testing GRPO model...")
    
    try:
        from unsloth import FastLanguageModel
        from transformers import AutoTokenizer
        from peft import PeftModel
        
        # Find the GRPO model
        possible_paths = [
            "models/single_system/double_integrator/grpo/latest",
            "temp_grpo_output/checkpoint-3",
            "grpo_trainer_lora_model"
        ]
        
        grpo_model_path = None
        for path in possible_paths:
            if os.path.exists(path) and (os.path.exists(f"{path}/adapter_model.safetensors") or 
                                        os.path.exists(f"{path}/adapter_config.json")):
                grpo_model_path = path
                break
        
        if grpo_model_path is None:
            print(f"❌ GRPO model not found in any of: {possible_paths}")
            return None
        
        print(f"📂 Loading GRPO model from: {grpo_model_path}")
        
        # Load base model and apply LoRA adapter
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen3-4B-Base",
            max_seq_length=2048,
            load_in_4bit=True,
            dtype=None,
        )
        
        # Load the adapter using PeftModel
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, grpo_model_path)
        
        # Enable inference mode
        FastLanguageModel.for_inference(model)
        
        print("✅ GRPO model loaded successfully")
        
        # Test the model
        results = test_model_on_cases(model, tokenizer, test_cases, "GRPO")
        return results
        
    except Exception as e:
        print(f"❌ Error loading GRPO model: {e}")
        return None


def test_model_on_cases(model, tokenizer, test_cases: List[Tuple[float, float]], 
                       model_name: str) -> Dict[str, Any]:
    """Test a model on given test cases."""
    print(f"🧪 Testing {model_name} model on {len(test_cases)} cases...")
    
    config = get_config()
    system = DoubleIntegrator()
    dt = config['system']['dt']
    steps = config['system']['steps']
    
    results = {
        'model_name': model_name,
        'trajectories': [],
        'success_count': 0,
        'total_count': len(test_cases)
    }
    
    for i, (x0, v0) in enumerate(test_cases):
        print(f"  Case {i+1}/{len(test_cases)}: ({x0:.2f}, {v0:.2f})")
        
        # Create prompt for the model
        prompt = f"""You are a control systems expert.
Given a double integrator system (ẍ = u) with initial position and velocity,
generate a sequence of {steps} control inputs to reach the origin (0,0) in exactly {dt * steps:.2f} seconds.
Position and velocity must stay within [-1, 1], and control inputs must be within [-3, 3].
Explain your approach between <REASONING> and </REASONING>.
Then provide exactly {steps} control values as a comma-separated list between <CONTROLS> and </CONTROLS>.

Initial state: position={x0:.2f}, velocity={v0:.2f}"""
        
        try:
            # Generate response
            inputs = tokenizer([prompt], return_tensors="pt")
            
            import torch
            # Move inputs to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.1,
                    do_sample=True,
                    top_k=50,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Extract controls from response
            controls = extract_controls_from_response(response)
            
            if controls is not None and len(controls) == steps:
                # Simulate trajectory
                states = simulate_trajectory([x0, v0], controls, system)
                final_error = np.linalg.norm(states[-1])
                success = final_error < 0.1
                
                trajectory_data = {
                    'initial_state': [x0, v0],
                    'states': states,
                    'controls': controls,
                    'final_error': final_error,
                    'success': success,
                    'valid_format': True
                }
                
                if success:
                    results['success_count'] += 1
                
                print(f"    ✅ Valid response, final error: {final_error:.4f}")
            else:
                print(f"    ❌ Invalid response format")
                trajectory_data = {
                    'initial_state': [x0, v0],
                    'states': None,
                    'controls': None,
                    'final_error': float('inf'),
                    'success': False,
                    'valid_format': False
                }
            
            results['trajectories'].append(trajectory_data)
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
            trajectory_data = {
                'initial_state': [x0, v0],
                'states': None,
                'controls': None,
                'final_error': float('inf'),
                'success': False,
                'valid_format': False
            }
            results['trajectories'].append(trajectory_data)
    
    success_rate = results['success_count'] / results['total_count']
    print(f"📊 {model_name} Results: {success_rate:.1%} success rate ({results['success_count']}/{results['total_count']})")
    
    return results


def extract_controls_from_response(response: str) -> List[float]:
    """Extract control sequence from model response."""
    try:
        # Look for content between <CONTROLS> and </CONTROLS>
        match = re.search(r'<CONTROLS>(.*?)</CONTROLS>', response, re.DOTALL)
        if not match:
            return None
        
        controls_text = match.group(1).strip()
        
        # Parse comma-separated values
        controls = []
        for value in controls_text.split(','):
            try:
                controls.append(float(value.strip()))
            except:
                continue
        
        return controls if len(controls) > 0 else None
        
    except Exception as e:
        return None


def simulate_trajectory(initial_state: List[float], controls: List[float], 
                       system: DoubleIntegrator) -> List[List[float]]:
    """Simulate trajectory given initial state and controls."""
    states = [initial_state]
    current_state = np.array(initial_state)
    
    for control in controls:
        # Clip control to bounds
        control = np.clip(control, -3.0, 3.0)
        next_state = system.simulate_step(current_state, control)
        states.append(next_state.tolist())
        current_state = next_state
    
    return states


def get_optimal_baseline(test_cases: List[Tuple[float, float]]) -> Dict[str, Any]:
    """Get optimal LQR baseline for comparison."""
    print("📐 Computing optimal LQR baseline...")
    
    config = get_config()
    system = DoubleIntegrator()
    dt = config['system']['dt']
    steps = config['system']['steps']
    
    results = {
        'model_name': 'LQR Optimal',
        'trajectories': [],
        'success_count': 0,
        'total_count': len(test_cases)
    }
    
    for x0, v0 in test_cases:
        # Solve optimal control
        optimal_controls = solve_double_integrator_lqr([x0, v0], dt, steps)
        
        # Simulate optimal trajectory
        states = simulate_trajectory([x0, v0], optimal_controls, system)
        final_error = np.linalg.norm(states[-1])
        success = final_error < 0.1
        
        if success:
            results['success_count'] += 1
        
        trajectory_data = {
            'initial_state': [x0, v0],
            'states': states,
            'controls': optimal_controls if isinstance(optimal_controls, list) else optimal_controls.tolist(),
            'final_error': final_error,
            'success': success,
            'valid_format': True
        }
        
        results['trajectories'].append(trajectory_data)
    
    success_rate = results['success_count'] / results['total_count']
    print(f"📊 LQR Optimal Results: {success_rate:.1%} success rate ({results['success_count']}/{results['total_count']})")
    
    return results


def create_model_comparison_plots(model_results: Dict[str, Dict], save_dir: str = "model_comparison_results"):
    """Create comprehensive comparison plots."""
    
    os.makedirs(save_dir, exist_ok=True)
    print(f"🎨 Creating model comparison plots in {save_dir}...")
    
    model_names = list(model_results.keys())
    colors = {'GRPO': 'green', 'LQR Optimal': 'red', 'SFT': 'blue'}
    
    # 1. Success Rate Comparison Bar Chart
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    success_rates = []
    names = []
    bar_colors = []
    
    for name, results in model_results.items():
        if results is not None:
            success_rate = results['success_count'] / results['total_count']
            success_rates.append(success_rate)
            names.append(name)
            bar_colors.append(colors.get(name, 'gray'))
    
    bars = ax1.bar(names, success_rates, color=bar_colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_ylabel('Success Rate')
    ax1.set_title('Model Performance Comparison\nSuccess Rate on Diverse Initial Conditions')
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig1.savefig(f"{save_dir}/success_rate_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print("✅ Created success rate comparison")
    
    # 2. Time-Series Trajectory Comparison (Position, Velocity, Control vs Time)
    valid_models = {name: results for name, results in model_results.items() 
                   if results is not None}
    
    if len(valid_models) > 0:
        # Create figure with 4 subplots: position vs time, velocity vs time, control vs time, and phase plot
        fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Get configuration for time steps
        from config import get_config
        config = get_config()
        dt = config['system']['dt']
        
        legend_entries = set()
        
        # Select a representative test case for detailed plotting (first valid trajectory)
        selected_case = None
        for model_name, results in valid_models.items():
            for traj in results['trajectories']:
                if traj['valid_format'] and traj['states'] is not None:
                    selected_case = traj
                    break
            if selected_case:
                break
        
        if selected_case:
            case_info = f"Initial: ({selected_case['initial_state'][0]:.2f}, {selected_case['initial_state'][1]:.2f})"
        else:
            case_info = "No valid trajectories"
        
        for model_name, results in valid_models.items():
            color = colors.get(model_name, 'gray')
            linestyle = '--' if model_name == 'LQR Optimal' else '-'
            alpha = 0.8 if model_name != 'LQR Optimal' else 0.6
            linewidth = 3 if model_name == 'LQR Optimal' else 2
            
            # Plot only the first valid trajectory for detailed time-series analysis
            for case_idx, traj in enumerate(results['trajectories']):
                if traj['valid_format'] and traj['states'] is not None:
                    states = np.array(traj['states'])
                    controls = traj['controls']
                    initial_state = traj['initial_state']
                    
                    # Create time array
                    time_steps = np.arange(len(states)) * dt
                    control_time = np.arange(len(controls)) * dt
                    
                    # Label only once per model
                    label = model_name if model_name not in legend_entries else ""
                    if label:
                        legend_entries.add(model_name)
                    
                    # Plot position vs time
                    ax1.plot(time_steps, states[:, 0], color=color, 
                            linestyle=linestyle, linewidth=linewidth,
                            alpha=alpha, label=label)
                    
                    # Plot velocity vs time
                    ax2.plot(time_steps, states[:, 1], color=color, 
                            linestyle=linestyle, linewidth=linewidth,
                            alpha=alpha, label=label)
                    
                    # Plot control vs time
                    ax3.plot(control_time, controls, color=color, 
                            linestyle=linestyle, linewidth=linewidth,
                            alpha=alpha, label=label)
                    
                    # Plot phase portrait (position vs velocity)
                    ax4.plot(states[:, 0], states[:, 1], color=color, 
                            linestyle=linestyle, linewidth=linewidth,
                            alpha=alpha, label=label)
                    
                    # Mark start point on phase plot
                    ax4.scatter(initial_state[0], initial_state[1], c='orange', s=80, 
                               marker='o', alpha=0.8, zorder=3)
                    
                    # Only plot first valid trajectory for clarity in time-series
                    break
        
        # Configure position vs time plot
        ax1.set_title('Position vs Time', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time (s)', fontsize=10)
        ax1.set_ylabel('Position', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle=':', alpha=0.5)
        ax1.legend(fontsize=9)
        
        # Configure velocity vs time plot
        ax2.set_title('Velocity vs Time', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Time (s)', fontsize=10)
        ax2.set_ylabel('Velocity', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle=':', alpha=0.5)
        ax2.legend(fontsize=9)
        
        # Configure control vs time plot
        ax3.set_title('Control Input vs Time', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Time (s)', fontsize=10)
        ax3.set_ylabel('Control Input', fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle=':', alpha=0.5)
        ax3.axhline(y=3, color='red', linestyle='--', alpha=0.5, label='Control Limits')
        ax3.axhline(y=-3, color='red', linestyle='--', alpha=0.5)
        ax3.legend(fontsize=9)
        
        # Configure phase portrait
        ax4.scatter(0, 0, c='black', s=120, marker='*', label='Target', zorder=5)
        circle = plt.Circle((0, 0), 0.1, fill=False, color='black', 
                           linewidth=2, linestyle=':', alpha=0.7, label='Success Region')
        ax4.add_patch(circle)
        ax4.set_title('Phase Portrait (Position vs Velocity)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Position', fontsize=10)
        ax4.set_ylabel('Velocity', fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=9)
        ax4.set_aspect('equal', adjustable='box')
        
        plt.suptitle(f'Trajectory Analysis - {case_info}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig2.savefig(f"{save_dir}/trajectory_comparison.png", dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print("✅ Created time-series trajectory comparison")
        
        # 2b. RL-Only Trajectory Analysis (without LQR comparison)
        rl_models = {name: results for name, results in valid_models.items() 
                    if name != 'LQR Optimal'}
        
        if len(rl_models) > 0:
            fig2b, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            rl_legend_entries = set()
            rl_colors = {'SFT': 'blue', 'GRPO': 'green'}
            
            # Select first valid RL trajectory
            selected_rl_case = None
            for model_name, results in rl_models.items():
                for traj in results['trajectories']:
                    if traj['valid_format'] and traj['states'] is not None:
                        selected_rl_case = traj
                        break
                if selected_rl_case:
                    break
            
            if selected_rl_case:
                rl_case_info = f"Initial: ({selected_rl_case['initial_state'][0]:.2f}, {selected_rl_case['initial_state'][1]:.2f})"
            else:
                rl_case_info = "No valid RL trajectories"
            
            for model_name, results in rl_models.items():
                color = rl_colors.get(model_name, 'gray')
                linestyle = '-'
                alpha = 0.8
                linewidth = 2
                
                # Plot first valid trajectory for each RL model
                for case_idx, traj in enumerate(results['trajectories']):
                    if traj['valid_format'] and traj['states'] is not None:
                        states = np.array(traj['states'])
                        controls = traj['controls']
                        initial_state = traj['initial_state']
                        
                        # Create time array
                        time_steps = np.arange(len(states)) * dt
                        control_time = np.arange(len(controls)) * dt
                        
                        # Label only once per model
                        label = model_name if model_name not in rl_legend_entries else ""
                        if label:
                            rl_legend_entries.add(model_name)
                        
                        # Plot position vs time
                        ax1.plot(time_steps, states[:, 0], color=color, 
                                linestyle=linestyle, linewidth=linewidth,
                                alpha=alpha, label=label)
                        
                        # Plot velocity vs time
                        ax2.plot(time_steps, states[:, 1], color=color, 
                                linestyle=linestyle, linewidth=linewidth,
                                alpha=alpha, label=label)
                        
                        # Plot control vs time
                        ax3.plot(control_time, controls, color=color, 
                                linestyle=linestyle, linewidth=linewidth,
                                alpha=alpha, label=label)
                        
                        # Plot phase portrait
                        ax4.plot(states[:, 0], states[:, 1], color=color, 
                                linestyle=linestyle, linewidth=linewidth,
                                alpha=alpha, label=label)
                        
                        # Mark start point on phase plot
                        ax4.scatter(initial_state[0], initial_state[1], c='orange', s=80, 
                                   marker='o', alpha=0.8, zorder=3)
                        
                        # Only plot first valid trajectory
                        break
            
            # Configure RL-only plots
            ax1.set_title('RL Model: Position vs Time', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Time (s)', fontsize=10)
            ax1.set_ylabel('Position', fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0, color='black', linestyle=':', alpha=0.5)
            ax1.legend(fontsize=9)
            
            ax2.set_title('RL Model: Velocity vs Time', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Time (s)', fontsize=10)
            ax2.set_ylabel('Velocity', fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='black', linestyle=':', alpha=0.5)
            ax2.legend(fontsize=9)
            
            ax3.set_title('RL Model: Control Input vs Time', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Time (s)', fontsize=10)
            ax3.set_ylabel('Control Input', fontsize=10)
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0, color='black', linestyle=':', alpha=0.5)
            ax3.axhline(y=3, color='red', linestyle='--', alpha=0.5, label='Control Limits')
            ax3.axhline(y=-3, color='red', linestyle='--', alpha=0.5)
            ax3.legend(fontsize=9)
            
            ax4.scatter(0, 0, c='black', s=120, marker='*', label='Target', zorder=5)
            circle_rl = plt.Circle((0, 0), 0.1, fill=False, color='black', 
                               linewidth=2, linestyle=':', alpha=0.7, label='Success Region')
            ax4.add_patch(circle_rl)
            ax4.set_title('RL Model: Phase Portrait', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Position', fontsize=10)
            ax4.set_ylabel('Velocity', fontsize=10)
            ax4.grid(True, alpha=0.3)
            ax4.legend(fontsize=9)
            ax4.set_aspect('equal', adjustable='box')
            
            plt.suptitle(f'RL Model Analysis (SFT & GRPO) - {rl_case_info}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            fig2b.savefig(f"{save_dir}/rl_trajectory_analysis.png", dpi=300, bbox_inches='tight')
            plt.close(fig2b)
            print("✅ Created RL-only trajectory analysis")
    
    # 3. Performance Analysis
    fig3, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Final error distributions
    for model_name, results in valid_models.items():
        final_errors = [traj['final_error'] for traj in results['trajectories'] 
                       if traj['valid_format'] and traj['final_error'] != float('inf')]
        
        if final_errors:
            ax1.hist(final_errors, bins=15, alpha=0.6, label=model_name, 
                    color=colors.get(model_name, 'gray'), density=True)
    
    ax1.axvline(0.1, color='red', linestyle='--', linewidth=2, label='Success Threshold')
    ax1.set_xlabel('Final Error')
    ax1.set_ylabel('Density')
    ax1.set_title('Final Error Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Valid prediction rates
    valid_rates = []
    model_names_clean = []
    for model_name, results in valid_models.items():
        valid_count = sum(1 for traj in results['trajectories'] if traj['valid_format'])
        valid_rate = valid_count / len(results['trajectories'])
        valid_rates.append(valid_rate)
        model_names_clean.append(model_name)
    
    bars2 = ax2.bar(model_names_clean, valid_rates, 
                   color=[colors.get(name, 'gray') for name in model_names_clean], 
                   alpha=0.7, edgecolor='black')
    
    for bar, rate in zip(bars2, valid_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_ylabel('Valid Prediction Rate')
    ax2.set_title('Model Reliability\n(Valid Format Predictions)')
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Success vs Initial Distance
    for model_name, results in valid_models.items():
        initial_distances = []
        successes = []
        
        for traj in results['trajectories']:
            if traj['valid_format']:
                dist = np.linalg.norm(traj['initial_state'])
                initial_distances.append(dist)
                successes.append(1 if traj['success'] else 0)
        
        if initial_distances:
            ax3.scatter(initial_distances, successes, alpha=0.6, 
                       color=colors.get(model_name, 'gray'), label=model_name, s=30)
    
    ax3.set_xlabel('Initial Distance from Target')
    ax3.set_ylabel('Success (1) / Failure (0)')
    ax3.set_title('Success vs Initial Distance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Summary statistics
    ax4.axis('off')
    summary_text = "MODEL PERFORMANCE SUMMARY\n\n"
    
    for model_name, results in valid_models.items():
        valid_count = sum(1 for traj in results['trajectories'] if traj['valid_format'])
        success_rate = results['success_count'] / results['total_count']
        valid_rate = valid_count / len(results['trajectories'])
        
        final_errors = [traj['final_error'] for traj in results['trajectories'] 
                       if traj['valid_format'] and traj['final_error'] != float('inf')]
        mean_error = np.mean(final_errors) if final_errors else float('inf')
        
        summary_text += f"{model_name}:\n"
        summary_text += f"  Success Rate: {success_rate:.1%}\n"
        summary_text += f"  Valid Predictions: {valid_rate:.1%}\n"
        summary_text += f"  Mean Final Error: {mean_error:.4f}\n\n"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('Comprehensive Model Performance Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig3.savefig(f"{save_dir}/performance_analysis.png", dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print("✅ Created performance analysis")
    
    plot_files = [
        f"{save_dir}/success_rate_comparison.png",
        f"{save_dir}/trajectory_comparison.png",
        f"{save_dir}/performance_analysis.png"
    ]
    
    # Add RL-only plot if it was created
    rl_plot_path = f"{save_dir}/rl_trajectory_analysis.png"
    if os.path.exists(rl_plot_path):
        plot_files.append(rl_plot_path)
    
    return plot_files


def main():
    print("🚀 Evaluating Trained SFT and GRPO Models")
    print("=" * 50)
    
    # Import torch here to avoid issues if not available
    try:
        import torch
        print("✅ PyTorch available")
    except ImportError:
        print("❌ PyTorch not available - needed for model inference")
        return
    
    # Generate test cases (reduced for faster testing)
    print("\n🎯 Generating test cases...")
    test_cases = [
        (-0.7, -0.7), (-0.7, 0.7), (0.7, -0.7), (0.7, 0.7),
        (-0.5, 0.0), (0.5, 0.0), (0.0, -0.5), (0.0, 0.5)
    ]
    
    print(f"📊 Testing on {len(test_cases)} initial conditions")
    
    # Test all models
    all_results = {}
    
    # 1. Test SFT model
    sft_results = load_and_test_sft_model(test_cases)
    if sft_results:
        all_results['SFT'] = sft_results
    
    # 2. Test GRPO model
    grpo_results = load_and_test_grpo_model(test_cases)
    if grpo_results:
        all_results['GRPO'] = grpo_results
    
    # 3. Get optimal baseline
    optimal_results = get_optimal_baseline(test_cases)
    all_results['LQR Optimal'] = optimal_results
    
    # 3. Create comparison plots
    if len(all_results) > 0:
        plot_files = create_model_comparison_plots(all_results, "model_comparison_results")
        
        # Save results
        results_file = "model_comparison_results/evaluation_results.json"
        json_results = {}
        
        for model_name, results in all_results.items():
            if results:
                json_results[model_name] = {
                    'success_count': int(results['success_count']),
                    'total_count': int(results['total_count']),
                    'success_rate': float(results['success_count'] / results['total_count']),
                    'valid_predictions': int(sum(1 for t in results['trajectories'] if t['valid_format']))
                }
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n🎉 Evaluation complete!")
        print(f"📁 Results saved to: model_comparison_results/")
        print(f"📊 Generated plots:")
        for plot_file in plot_files:
            print(f"   - {plot_file}")
        print(f"   - {results_file}")
        
        # Print summary
        print(f"\n📈 RESULTS SUMMARY:")
        for model_name, results in all_results.items():
            if results:
                success_rate = results['success_count'] / results['total_count']
                valid_count = sum(1 for t in results['trajectories'] if t['valid_format'])
                valid_rate = valid_count / len(results['trajectories'])
                print(f"   {model_name}: {success_rate:.1%} success, {valid_rate:.1%} valid predictions")
    
    else:
        print("❌ No models could be evaluated")


if __name__ == "__main__":
    main()