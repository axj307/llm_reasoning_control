#!/usr/bin/env python3
"""
Main script with benchmark-style visualization.
"""

import argparse
import os
import sys
import torch

# Add current directory and parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

# Import modules
from control import solve_double_integrator
from trainer_module import ControlTrainer
from utils import parse_control_output
from utils_enhanced_v3 import create_benchmark_style_comparison, create_clean_comparison_plot


def main():
    parser = argparse.ArgumentParser(description="Evaluate control models with benchmark-style visualization")
    
    # Model settings
    parser.add_argument("--model-name", default="unsloth/Qwen2.5-3B-Instruct",
                        help="Model to evaluate")
    
    # Test settings
    parser.add_argument("--num-tests", type=int, default=5,
                        help="Number of test cases")
    parser.add_argument("--dt", type=float, default=0.1,
                        help="Time step")
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of control steps")
    
    # Visualization settings
    parser.add_argument("--style", choices=["benchmark", "clean"], default="benchmark",
                        help="Visualization style")
    parser.add_argument("--no-optimal", action="store_true",
                        help="Don't show optimal trajectories")
    
    # Output
    parser.add_argument("--output-dir", default="outputs",
                        help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create trainer
    print(f"Loading model: {args.model_name}")
    trainer = ControlTrainer(args.model_name)
    trainer.setup_model()
    
    # Test cases - diverse initial conditions
    test_cases = [
        (0.5, -0.3),
        (0.7, 0.2),
        (-0.6, -0.4),
        (0.3, 0.6),
        (-0.4, 0.5),
    ][:args.num_tests]
    
    # Generate trajectories
    print("\nGenerating trajectories...")
    all_trajectories = []
    
    for x0, v0 in test_cases:
        # Create prompt
        prompt = f"Control a double integrator system with initial state [position={x0:.2f}, velocity={v0:.2f}] to reach the origin (0,0) in 5.00 seconds using 50 steps. Ensure all states remain within [-1,1] and controls within [-3,3]."
        
        # Generate response
        output = trainer.generate(prompt)
        
        # Parse controls
        controls = parse_control_output(output)
        if controls:
            all_trajectories.append((x0, v0, controls))
            print(f"✓ Generated controls for IC: [{x0:.2f}, {v0:.2f}]")
        else:
            # Use zeros as fallback
            all_trajectories.append((x0, v0, [0.0] * args.steps))
            print(f"✗ Failed to parse controls for IC: [{x0:.2f}, {v0:.2f}]")
    
    # Create visualization
    print(f"\nCreating {args.style} style visualization...")
    
    if args.style == "benchmark":
        # Create benchmark-style plot
        save_path = os.path.join(args.output_dir, "benchmark_comparison")
        create_benchmark_style_comparison(
            test_cases=all_trajectories,
            dt=args.dt,
            save_path=save_path,
            show_optimal=not args.no_optimal
        )
        print(f"Saved to: {save_path}.png and {save_path}.pdf")
        
    else:  # clean style
        # Create clean comparison plot
        save_path = os.path.join(args.output_dir, "clean_comparison")
        create_clean_comparison_plot(
            test_cases=all_trajectories,
            dt=args.dt,
            save_path=save_path,
            show_optimal=not args.no_optimal
        )
        print(f"Saved to: {save_path}.png and {save_path}.pdf")
    
    # Print summary
    print("\n=== Summary ===")
    from control import simulate_trajectory
    
    errors = []
    for x0, v0, controls in all_trajectories:
        positions, velocities, _ = simulate_trajectory(x0, v0, controls, args.dt)
        final_error = (positions[-1]**2 + velocities[-1]**2)**0.5
        errors.append(final_error)
    
    print(f"Test cases: {len(all_trajectories)}")
    print(f"Average final error: {sum(errors)/len(errors):.4f}")
    print(f"Best final error: {min(errors):.4f}")
    print(f"Worst final error: {max(errors):.4f}")


if __name__ == "__main__":
    main()