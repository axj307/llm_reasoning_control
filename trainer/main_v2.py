#!/usr/bin/env python3
"""
Main entry point v2 with combined visualization for all test cases.
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
from utils_enhanced import visualize_all_trajectories_with_controls

# Import config variables
from config import *


def main():
    parser = argparse.ArgumentParser(description="Train control system models with SFT and GRPO")
    
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
        
        # Test problems
        test_problems = [
            (0.5, -0.3, "Control a double integrator system with initial state [position=0.5, velocity=-0.3] to reach the origin (0,0) in 5.00 seconds using 50 steps. Ensure all states remain within [-1,1] and controls within [-3,3]."),
            (0.7, 0.2, "Control a double integrator system with initial state [position=0.7, velocity=0.2] to reach the origin (0,0) in 5.00 seconds using 50 steps. Ensure all states remain within [-1,1] and controls within [-3,3]."),
            (-0.6, -0.4, "Control a double integrator system with initial state [position=-0.6, velocity=-0.4] to reach the origin (0,0) in 5.00 seconds using 50 steps. Ensure all states remain within [-1,1] and controls within [-3,3]."),
        ]
        
        # If not already loaded, setup model
        if trainer.model is None:
            trainer.setup_model()
        
        # Collect all test cases for combined visualization
        all_test_cases = []
        
        for x0, v0, problem in test_problems:
            print(f"\nTesting: x0={x0}, v0={v0}")
            
            # Generate response
            output = trainer.generate(problem)
            print("Generated output:")
            print(output[:200] + "..." if len(output) > 200 else output)
            
            # Parse controls
            controls = parse_control_output(output)
            if controls:
                print(f"Parsed {len(controls)} control values")
                
                # Add to test cases for combined plot
                all_test_cases.append((x0, v0, controls))
            else:
                print("Failed to parse control values")
                # Add empty controls for failed case
                all_test_cases.append((x0, v0, [0.0] * args.steps))
        
        # Create combined visualization showing all trajectories
        if all_test_cases:
            print("\nCreating combined visualization of all trajectories...")
            
            # Create comprehensive plot with all trajectories
            save_path = f"{args.output_dir}/combined_evaluation_full.png"
            visualize_all_trajectories_with_controls(
                test_cases=all_test_cases,
                dt=args.dt,
                save_path=save_path,
                show_optimal=True
            )
            
            # Also create a simpler version without controls subplot
            save_path_simple = f"{args.output_dir}/combined_evaluation.png"
            from utils_enhanced import visualize_all_trajectories
            visualize_all_trajectories(
                test_cases=all_test_cases,
                dt=args.dt,
                save_path=save_path_simple,
                show_optimal=True
            )
            
            print(f"\nVisualization saved to:")
            print(f"  - Full version: {save_path}")
            print(f"  - Simple version: {save_path_simple}")
            
            # Print final states for all trajectories
            print("\n=== Final State Summary ===")
            for i, (x0, v0, controls) in enumerate(all_test_cases):
                from control import simulate_trajectory
                positions, velocities, _ = simulate_trajectory(x0, v0, controls, args.dt)
                final_x = positions[-1]
                final_v = velocities[-1]
                final_error = (final_x**2 + final_v**2)**0.5
                print(f"Test {i+1} - Initial: [{x0:.2f}, {v0:.2f}] → Final: [{final_x:.4f}, {final_v:.4f}] (Error: {final_error:.4f})")


if __name__ == "__main__":
    main()