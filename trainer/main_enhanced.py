#!/usr/bin/env python3
"""
Main entry point with enhanced publication-quality visualizations.
"""

import argparse
import os
import sys
import random
import torch

# Add current directory and parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

# Import modules directly
import config as cfg
from control import solve_double_integrator
from data import create_dataset, get_system_prompt
from trainer_module import ControlTrainer
from utils import parse_control_output
from utils_enhanced_v2 import (
    create_publication_comparison_plot,
    create_model_trajectories_plot, 
    create_control_dashboard
)

# Import config variables
from config import *


def main():
    parser = argparse.ArgumentParser(description="Train control system models with enhanced visualization")
    
    # Training mode
    parser.add_argument("--mode", choices=["train", "eval", "both"], default="both",
                        help="Training mode: train only, eval only, or both")
    
    # Model settings
    parser.add_argument("--model-name", default=MODEL_NAME,
                        help="Base model name to fine-tune")
    parser.add_argument("--lora-rank", type=int, default=LORA_RANK,
                        help="LoRA rank for fine-tuning")
    
    # Dataset settings
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES,
                        help="Number of training samples to generate")
    parser.add_argument("--dt", type=float, default=DT,
                        help="Time step for control system")
    parser.add_argument("--steps", type=int, default=STEPS,
                        help="Number of control steps")
    
    # Training settings
    parser.add_argument("--sft-epochs", type=int, default=SFT_EPOCHS,
                        help="Number of SFT epochs")
    parser.add_argument("--grpo-steps", type=int, default=GRPO_MAX_STEPS,
                        help="Number of GRPO steps")
    parser.add_argument("--skip-sft", action="store_true",
                        help="Skip SFT training")
    parser.add_argument("--skip-grpo", action="store_true",
                        help="Skip GRPO training")
    
    # Visualization settings
    parser.add_argument("--viz-type", choices=["comparison", "model-only", "dashboard", "all"], 
                        default="comparison",
                        help="Type of visualization to create")
    parser.add_argument("--no-optimal", action="store_true",
                        help="Don't show optimal trajectories in comparison plots")
    
    # GPU selection
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU to use (default: random)")
    
    # Output
    parser.add_argument("--save-name", default=MODEL_SAVE_NAME,
                        help="Name for saved model")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                        help="Output directory for models and logs")
    
    args = parser.parse_args()
    
    # GPU selection
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        if args.gpu is not None:
            chosen_gpu = args.gpu
        else:
            chosen_gpu = random.randint(0, num_gpus - 1)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(chosen_gpu)
        print(f"Using GPU: {chosen_gpu}")
    else:
        print("No GPUs available.")
    
    # Update config values
    cfg.DT = args.dt
    cfg.STEPS = args.steps
    cfg.NUM_SAMPLES = args.num_samples
    cfg.SFT_EPOCHS = args.sft_epochs
    cfg.GRPO_MAX_STEPS = args.grpo_steps
    cfg.MODEL_SAVE_NAME = args.save_name
    cfg.OUTPUT_DIR = args.output_dir
    
    # Create trainer
    trainer = ControlTrainer(args.model_name)
    
    if args.mode in ["train", "both"]:
        print(f"\n=== Training Mode ===")
        print(f"Generating {args.num_samples} samples...")
        
        # Create dataset
        dataset = create_dataset(args.num_samples)
        
        # Setup model
        trainer.setup_model()
        
        # Train
        trainer.train(
            dataset,
            do_sft=not args.skip_sft,
            do_grpo=not args.skip_grpo
        )
        
        print("\nTraining complete!")
    
    if args.mode in ["eval", "both"]:
        print(f"\n=== Evaluation Mode ===")
        
        # Test problems - more diverse set
        test_problems = [
            (0.5, -0.3),
            (0.7, 0.2),
            (-0.6, -0.4),
            (0.3, 0.6),
            (-0.4, 0.5),
            (0.8, -0.2),
            (-0.5, -0.5)
        ]
        
        # If not already loaded, setup model
        if trainer.model is None:
            trainer.setup_model()
        
        # Collect all test cases for visualization
        all_test_cases = []
        
        print("\nGenerating trajectories...")
        for x0, v0 in test_problems:
            # Create prompt
            problem = f"Control a double integrator system with initial state [position={x0}, velocity={v0}] to reach the origin (0,0) in 5.00 seconds using 50 steps. Ensure all states remain within [-1,1] and controls within [-3,3]."
            
            # Generate response
            output = trainer.generate(problem)
            
            # Parse controls
            controls = parse_control_output(output)
            if controls:
                all_test_cases.append((x0, v0, controls))
                print(f"✓ Generated controls for IC: [{x0:.2f}, {v0:.2f}]")
            else:
                # Use zeros as fallback
                all_test_cases.append((x0, v0, [0.0] * args.steps))
                print(f"✗ Failed to parse controls for IC: [{x0:.2f}, {v0:.2f}]")
        
        # Create visualizations based on selected type
        os.makedirs(args.output_dir, exist_ok=True)
        
        if args.viz_type in ["comparison", "all"]:
            print("\nCreating publication-quality comparison plot...")
            save_path = os.path.join(args.output_dir, "publication_comparison")
            create_publication_comparison_plot(
                test_cases=all_test_cases[:5],  # Limit to 5 for clarity
                dt=args.dt,
                save_path=save_path,
                show_optimal=not args.no_optimal,
                title="Double Integrator Control: Model vs Optimal Comparison"
            )
        
        if args.viz_type in ["model-only", "all"]:
            print("\nCreating model-only trajectories plot...")
            save_path = os.path.join(args.output_dir, "model_trajectories")
            create_model_trajectories_plot(
                test_cases=all_test_cases[:5],
                dt=args.dt,
                save_path=save_path,
                title="Double Integrator Model Predictions"
            )
        
        if args.viz_type in ["dashboard", "all"]:
            print("\nCreating comprehensive control dashboard...")
            save_path = os.path.join(args.output_dir, "control_dashboard")
            create_control_dashboard(
                test_cases=all_test_cases,
                dt=args.dt,
                save_path=save_path
            )
        
        # Print summary statistics
        print("\n=== Evaluation Summary ===")
        from control import simulate_trajectory
        
        errors = []
        efforts = []
        for x0, v0, controls in all_test_cases:
            positions, velocities, _ = simulate_trajectory(x0, v0, controls, args.dt)
            final_error = (positions[-1]**2 + velocities[-1]**2)**0.5
            control_effort = sum(c**2 for c in controls) * args.dt
            errors.append(final_error)
            efforts.append(control_effort)
        
        print(f"Number of test cases: {len(all_test_cases)}")
        print(f"Average final error: {sum(errors)/len(errors):.4f}")
        print(f"Best final error: {min(errors):.4f}")
        print(f"Worst final error: {max(errors):.4f}")
        print(f"Average control effort: {sum(efforts)/len(efforts):.2f}")
        
        print(f"\nVisualizations saved to {args.output_dir}/")


if __name__ == "__main__":
    main()