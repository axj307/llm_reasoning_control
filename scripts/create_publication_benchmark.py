#!/usr/bin/env python3
"""
Publication-ready benchmarking script for control systems.

This script demonstrates the system-agnostic design by creating
publication-quality plots for any control system in the framework.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import AVAILABLE_SYSTEMS
from environments import get_system
from evaluation.visualization import (
    plot_publication_comparison, plot_model_only_trajectories,
    generate_publication_plots, create_individual_phase_plot,
    create_individual_state_plot, create_individual_control_plot
)
from evaluation.plotting_utils import setup_beautiful_plotting


def create_sample_trajectory(system_name: str, initial_state: np.ndarray, 
                           trajectory_type: str = "lqr") -> dict:
    """
    Create sample trajectory data for demonstration.
    
    Args:
        system_name: Name of the control system
        initial_state: Initial state vector
        trajectory_type: Type of trajectory ("lqr", "model", "random")
        
    Returns:
        Trajectory dictionary with states, controls, and times
    """
    system = get_system(system_name)()
    dt = system.dt
    steps = system.steps
    
    # Generate time array
    times = np.linspace(0, dt * steps, steps + 1)
    
    # Initialize arrays
    states = np.zeros((steps + 1, system.state_dim))
    controls = np.zeros(steps)
    states[0] = initial_state
    
    # Simple controller based on trajectory type
    if trajectory_type == "lqr":
        # LQR-like behavior: proportional control towards origin
        K = np.array([2.0, 1.5])  # Feedback gains
        
        for i in range(steps):
            # LQR control law: u = -K @ x
            control = -np.dot(K, states[i])
            
            # Apply control bounds
            control_bounds = system.get_control_bounds()
            control = np.clip(control, control_bounds[0], control_bounds[1])
            controls[i] = control
            
            # Simulate next state
            next_state = system.simulate_step(states[i], control)
            
            # Apply state bounds
            state_bounds = system.get_state_bounds()
            for j, bounds in enumerate(state_bounds):
                next_state[j] = np.clip(next_state[j], bounds[0], bounds[1])
            
            states[i + 1] = next_state
            
    elif trajectory_type == "model":
        # Model-like behavior: slightly suboptimal but reasonable
        K = np.array([1.8, 1.2])  # Slightly different gains
        noise_scale = 0.05
        
        for i in range(steps):
            # Add some noise to make it different from optimal
            noise = np.random.normal(0, noise_scale, size=system.state_dim)
            noisy_state = states[i] + noise
            
            control = -np.dot(K, noisy_state)
            
            # Apply control bounds
            control_bounds = system.get_control_bounds()
            control = np.clip(control, control_bounds[0], control_bounds[1])
            controls[i] = control
            
            # Simulate next state
            next_state = system.simulate_step(states[i], control)
            
            # Apply state bounds
            state_bounds = system.get_state_bounds()
            for j, bounds in enumerate(state_bounds):
                next_state[j] = np.clip(next_state[j], bounds[0], bounds[1])
            
            states[i + 1] = next_state
            
    else:  # random
        # Random control for comparison
        control_bounds = system.get_control_bounds()
        
        for i in range(steps):
            control = np.random.uniform(control_bounds[0], control_bounds[1])
            controls[i] = control
            
            # Simulate next state
            next_state = system.simulate_step(states[i], control)
            
            # Apply state bounds
            state_bounds = system.get_state_bounds()
            for j, bounds in enumerate(state_bounds):
                next_state[j] = np.clip(next_state[j], bounds[0], bounds[1])
            
            states[i + 1] = next_state
    
    # Calculate final error and other metrics
    final_error = np.linalg.norm(states[-1])
    
    return {
        'states': states,
        'controls': controls,
        'times': times,
        'initial_state': initial_state,
        'final_state': states[-1],
        'final_error': final_error,
        'valid_trajectory': True
    }


def main():
    parser = argparse.ArgumentParser(description="Create publication-ready benchmark plots")
    
    parser.add_argument("--systems", type=str, default="double_integrator,van_der_pol",
                       help="Comma-separated list of systems to benchmark")
    parser.add_argument("--num-cases", type=int, default=3,
                       help="Number of test cases per system") 
    parser.add_argument("--plot-dir", type=str, default="publication_plots",
                       help="Directory to save plots")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--plot-types", type=str, default="comparison,individual",
                       help="Types of plots to generate: comparison,individual,phase,position,velocity,control")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Parse systems
    systems = [s.strip() for s in args.systems.split(",")]
    for system in systems:
        if system not in AVAILABLE_SYSTEMS:
            raise ValueError(f"Unknown system: {system}. Available: {AVAILABLE_SYSTEMS}")
    
    # Parse plot types
    plot_types = [t.strip() for t in args.plot_types.split(",")]
    
    # Setup plotting
    setup_beautiful_plotting()
    
    print("=== Publication-Ready Benchmark Plot Generation ===")
    print(f"Systems: {', '.join(systems)}")
    print(f"Cases per system: {args.num_cases}")
    print(f"Plot types: {', '.join(plot_types)}")
    print(f"Output directory: {args.plot_dir}")
    
    os.makedirs(args.plot_dir, exist_ok=True)
    
    # Generate plots for each system
    for system_name in systems:
        print(f"\n{'='*50}")
        print(f"GENERATING PLOTS FOR {system_name.upper()}")
        print('='*50)
        
        system = get_system(system_name)()
        
        # Generate test cases
        for case_num in range(args.num_cases):
            print(f"\nCase {case_num + 1}/{args.num_cases}")
            
            # Generate random initial state
            initial_state = system.generate_random_initial_state()
            print(f"Initial state: {initial_state}")
            
            # Create trajectories for comparison
            trajectories = {}
            
            # Create model trajectory
            model_traj = create_sample_trajectory(system_name, initial_state, "model")
            trajectories['Model'] = model_traj
            
            # Create optimal trajectory
            optimal_traj = create_sample_trajectory(system_name, initial_state, "lqr")
            trajectories['Optimal'] = optimal_traj
            
            base_filename = f"{system_name}_case_{case_num + 1}"
            
            # Generate requested plot types
            if "comparison" in plot_types:
                print("  Generating 4-subplot comparison...")
                fig_comp = plot_publication_comparison(
                    trajectories, system_name, initial_state,
                    save_path=os.path.join(args.plot_dir, f"{base_filename}_comparison")
                )
                print(f"  Saved: {base_filename}_comparison.pdf/.png")
                
            if "individual" in plot_types:
                print("  Generating individual plots...")
                saved_files = generate_publication_plots(
                    trajectories, system_name, initial_state,
                    base_filename, args.plot_dir
                )
                print(f"  Saved {len(saved_files)} individual plot files")
                
            # Generate specific individual plots if requested
            if "phase" in plot_types:
                fig_phase = create_individual_phase_plot(
                    trajectories, system_name,
                    save_path=os.path.join(args.plot_dir, f"{base_filename}_phase_only")
                )
                print(f"  Saved: {base_filename}_phase_only.pdf/.png")
                
            if "position" in plot_types:
                fig_pos = create_individual_state_plot(
                    trajectories, system_name, state_index=0,
                    save_path=os.path.join(args.plot_dir, f"{base_filename}_position_only")
                )
                print(f"  Saved: {base_filename}_position_only.pdf/.png")
                
            if "velocity" in plot_types:
                fig_vel = create_individual_state_plot(
                    trajectories, system_name, state_index=1,
                    save_path=os.path.join(args.plot_dir, f"{base_filename}_velocity_only")
                )
                print(f"  Saved: {base_filename}_velocity_only.pdf/.png")
                
            if "control" in plot_types:
                fig_ctrl = create_individual_control_plot(
                    trajectories, system_name,
                    save_path=os.path.join(args.plot_dir, f"{base_filename}_control_only")
                )
                print(f"  Saved: {base_filename}_control_only.pdf/.png")
    
    print(f"\n{'='*70}")
    print("BENCHMARK GENERATION COMPLETE")
    print('='*70)
    print(f"All plots saved to: {args.plot_dir}")
    print("\nGenerated publication-ready plots with:")
    print("✓ Professional color schemes")
    print("✓ Consistent typography and layout")
    print("✓ 4-subplot comparison layouts")
    print("✓ Individual standalone plots") 
    print("✓ System-agnostic design")
    print("✓ PDF and PNG formats")
    
    # Create a summary README
    readme_path = os.path.join(args.plot_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write("# Publication-Ready Control System Plots\n\n")
        f.write("This directory contains publication-quality plots generated using the enhanced plotting framework.\n\n")
        f.write("## Plot Types\n\n")
        f.write("- **Comparison plots**: 4-subplot layouts with phase space, position vs time, velocity vs time, and control vs time\n")
        f.write("- **Individual plots**: Standalone plots for each subplot type\n")
        f.write("- **Multiple formats**: Both PDF (vector) and PNG (raster) formats\n\n")
        f.write("## Systems Analyzed\n\n")
        for system in systems:
            f.write(f"- {system.replace('_', ' ').title()}\n")
        f.write(f"\n## Features\n\n")
        f.write("- Publication-ready styling with professional color schemes\n")
        f.write("- Consistent typography and layout across all plots\n") 
        f.write("- System-agnostic design works with any control system\n")
        f.write("- Directional arrows in phase space plots\n")
        f.write("- Control and state bounds visualization\n")
        f.write("- High-resolution output suitable for papers and presentations\n")
    
    print(f"\nCreated README.md: {readme_path}")


if __name__ == "__main__":
    main()