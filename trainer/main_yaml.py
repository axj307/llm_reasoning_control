#!/usr/bin/env python3
"""
Main entry point for control trainer with YAML configuration support.
"""

import argparse
import os
import sys
import random
import torch
from pathlib import Path

# Add current directory and parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

# Import configuration system
from config import ConfigManager, create_config_parser

# Import modules
from control import solve_double_integrator
from data import create_dataset, get_system_prompt
from trainer_module_yaml import ControlTrainer
from utils import visualize_solution, parse_control_output


def main():
    # Create argument parser with config support
    parent_parser = create_config_parser()
    parser = argparse.ArgumentParser(
        description="Train control system models with YAML configuration",
        parents=[parent_parser]
    )
    
    # Additional command-line arguments
    parser.add_argument("--mode", choices=["train", "eval", "both"], default="both",
                        help="Training mode: train only, eval only, or both")
    parser.add_argument("--skip-sft", action="store_true",
                        help="Skip SFT training")
    parser.add_argument("--skip-grpo", action="store_true",
                        help="Skip GRPO training")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU to use (default: auto-select)")
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="Experiment name for output directory")
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config_manager = ConfigManager.from_args(args)
    config = config_manager.get()
    
    # GPU selection
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        if args.gpu is not None:
            chosen_gpu = args.gpu
        else:
            # Use seed for reproducible GPU selection
            random.seed(config.training.seed)
            chosen_gpu = random.randint(0, num_gpus - 1)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(chosen_gpu)
        print(f"Using GPU: {chosen_gpu}")
    else:
        print("No GPUs available.")
    
    # Set random seeds
    random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.training.seed)
    
    # Create output directory
    output_dir = config_manager.create_output_dir(suffix=args.experiment_name)
    print(f"Output directory: {output_dir}")
    
    # Create trainer with configuration
    trainer = ControlTrainer(config.model.name)
    
    if args.mode in ["train", "both"]:
        print(f"\n=== Training Mode ===")
        print(f"Configuration: {args.config or 'base/base.yaml'}")
        print(f"Model: {config.model.name}")
        print(f"LoRA rank: {config.lora.rank}")
        print(f"Dataset size: {config.dataset.num_samples}")
        print(f"Control steps: {config.control.steps}")
        print(f"Time step: {config.control.dt}")
        
        # Create dataset
        print(f"\nGenerating {config.dataset.num_samples} samples...")
        dataset = create_dataset(
            num_samples=config.dataset.num_samples,
            dt=config.control.dt,
            steps=config.control.steps
        )
        
        # Setup model with configuration
        trainer.setup_model(
            lora_rank=config.lora.rank,
            lora_alpha=config.lora.lora_alpha,
            lora_dropout=config.lora.lora_dropout,
            target_modules=config.lora.target_modules,
            max_seq_length=config.model.max_seq_length
        )
        
        # Prepare training arguments
        sft_config = config_manager.get_training_config("sft")
        grpo_config = config_manager.get_training_config("grpo")
        
        # Train
        trainer.train(
            dataset,
            do_sft=not args.skip_sft,
            do_grpo=not args.skip_grpo,
            sft_epochs=sft_config.num_train_epochs,
            sft_batch_size=sft_config.per_device_train_batch_size,
            sft_learning_rate=sft_config.learning_rate,
            grpo_max_steps=grpo_config.max_steps if hasattr(grpo_config, 'max_steps') else 50,
            grpo_batch_size=grpo_config.per_device_train_batch_size,
            grpo_learning_rate=grpo_config.learning_rate,
            output_dir=str(output_dir)
        )
        
        print("\nTraining complete!")
        
        # Save final model path
        model_save_path = output_dir / f"{config.system.name if config.system else 'model'}_final"
        trainer.save_model(str(model_save_path))
        print(f"Model saved to: {model_save_path}")
    
    if args.mode in ["eval", "both"]:
        print(f"\n=== Evaluation Mode ===")
        
        # If not already loaded, setup model
        if trainer.model is None:
            trainer.setup_model(
                lora_rank=config.lora.rank,
                lora_alpha=config.lora.lora_alpha,
                lora_dropout=config.lora.lora_dropout,
                target_modules=config.lora.target_modules,
                max_seq_length=config.model.max_seq_length
            )
        
        # Use test cases from configuration
        for i, (x0, v0) in enumerate(config.evaluation.test_initial_conditions):
            print(f"\nTest case {i+1}: x0={x0}, v0={v0}")
            
            # Create problem description
            problem = f"Control a double integrator system with initial state [position={x0}, velocity={v0}] to reach the origin (0,0) in {config.control.dt * config.control.steps:.2f} seconds using {config.control.steps} steps. Ensure all states remain within [-1,1] and controls within [{config.control.control_bounds[0]},{config.control.control_bounds[1]}]."
            
            # Generate response
            output = trainer.generate(problem)
            print("Generated output:")
            print(output[:200] + "..." if len(output) > 200 else output)
            
            # Parse controls
            controls = parse_control_output(output)
            if controls:
                print(f"Parsed {len(controls)} control values")
                
                # Compare with LQR solution
                lqr_controls = solve_double_integrator(x0, v0, config.control.dt, config.control.steps)
                
                # Visualize if enabled
                if config.evaluation.visualize:
                    plot_dir = output_dir / config.evaluation.plot_dir
                    plot_dir.mkdir(exist_ok=True)
                    save_path = plot_dir / f"eval_{x0}_{v0}.png"
                    final_x, final_v = visualize_solution(
                        x0, v0, controls, config.control.dt, str(save_path)
                    )
                    print(f"Final state: x={final_x:.4f}, v={final_v:.4f}")
                    print(f"Visualization saved to {save_path}")
                else:
                    # Just compute final state
                    from control import simulate_trajectory
                    positions, velocities, _ = simulate_trajectory(x0, v0, controls, config.control.dt)
                    final_x, final_v = positions[-1], velocities[-1]
                    print(f"Final state: x={final_x:.4f}, v={final_v:.4f}")
                
                # Compute error
                error = (final_x**2 + final_v**2)**0.5
                print(f"Final state error: {error:.4f}")
            else:
                print("Failed to parse control values")
    
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()