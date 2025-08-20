#!/usr/bin/env python3
"""Universal training script for multiple systems."""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import ALL_CONFIG, AVAILABLE_SYSTEMS
from core.model_manager import UniversalModelManager
from core.data_pipeline import UniversalDataGenerator
from training.sft_training import train_sft_model, setup_universal_chat_template, save_sft_model
from training.grpo_training import train_grpo_model, save_grpo_model


def main():
    parser = argparse.ArgumentParser(description="Train universal control model")
    
    # System selection
    parser.add_argument("--systems", type=str, default="double_integrator,van_der_pol",
                       help=f"Comma-separated list of systems. Available: {','.join(AVAILABLE_SYSTEMS)}")
    
    # Training type
    parser.add_argument("--training-type", choices=["sft", "grpo", "both"], default="both",
                       help="Type of training to perform")
    
    # Data configuration
    parser.add_argument("--samples-per-system", type=int, 
                       default=ALL_CONFIG["data"]["default_samples_per_system"],
                       help="Number of training samples per system")
    parser.add_argument("--use-saved-data", type=str,
                       help="Path to saved dataset file")
    
    # Training configuration
    parser.add_argument("--sft-run-name", type=str,
                       help="Run name for SFT model (auto-generated if not provided)")
    parser.add_argument("--grpo-run-name", type=str, 
                       help="Run name for GRPO model (auto-generated if not provided)")
    parser.add_argument("--load-sft-model", type=str,
                       help="Path to SFT model for GRPO training")
    
    # Model configuration
    parser.add_argument("--lora-rank", type=int, default=ALL_CONFIG["model"]["lora_rank"],
                       help="LoRA rank")
    parser.add_argument("--max-seq-length", type=int, default=ALL_CONFIG["model"]["max_seq_length"],
                       help="Maximum sequence length")
    
    # Hardware configuration
    parser.add_argument("--gpu-id", type=str, default="0",
                       help="GPU ID to use")
    
    # Output configuration
    parser.add_argument("--output-base", type=str, default="./temp_training",
                       help="Base output directory for training")
    
    args = parser.parse_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    print(f"Using GPU: {args.gpu_id}")
    
    # Parse systems
    systems = [s.strip() for s in args.systems.split(",")]
    
    # Validate systems
    for system in systems:
        if system not in AVAILABLE_SYSTEMS:
            raise ValueError(f"Unknown system: {system}. Available: {AVAILABLE_SYSTEMS}")
    
    print(f"Training on systems: {', '.join(systems)}")
    
    # Create model manager
    manager = UniversalModelManager(ALL_CONFIG["model"]["base_model_name"])
    
    # Set up model
    print("Setting up model...")
    model, tokenizer = manager.setup_model(
        max_seq_length=args.max_seq_length,
        lora_rank=args.lora_rank
    )
    
    # Generate or load data
    print("Preparing training data...")
    if args.use_saved_data:
        generator = UniversalDataGenerator(systems)
        data = generator.load_dataset(args.use_saved_data)
    else:
        generator = UniversalDataGenerator(
            systems=systems,
            dt=ALL_CONFIG["system"]["dt"],
            steps=ALL_CONFIG["system"]["steps"],
            reasoning_start=ALL_CONFIG["system"]["reasoning_start"],
            reasoning_end=ALL_CONFIG["system"]["reasoning_end"],
            solution_start=ALL_CONFIG["system"]["solution_start"],
            solution_end=ALL_CONFIG["system"]["solution_end"]
        )
        
        data = generator.generate_universal_dataset(args.samples_per_system)
        
        # Save dataset for future use
        if ALL_CONFIG["data"]["save_datasets"]:
            dataset_name = f"universal_{'_'.join(systems)}_{len(data)}_samples.pkl"
            generator.save_dataset(data, dataset_name)
    
    # Split data
    train_data, eval_data = generator.split_dataset(
        data, ALL_CONFIG["data"]["train_eval_split"]
    )
    
    # Set up chat template
    setup_universal_chat_template(
        manager, systems,
        ALL_CONFIG["system"]["reasoning_start"],
        ALL_CONFIG["system"]["reasoning_end"],
        ALL_CONFIG["system"]["solution_start"],
        ALL_CONFIG["system"]["solution_end"]
    )
    
    sft_save_path = None
    grpo_save_path = None
    
    # SFT Training
    if args.training_type in ["sft", "both"]:
        print("\n" + "="*70)
        print("STARTING SFT TRAINING")
        print("="*70)
        
        # Update SFT config with output directory
        sft_config = ALL_CONFIG["sft"].copy()
        sft_config["output_dir"] = f"{args.output_base}/sft"
        
        # Train SFT
        sft_result = train_sft_model(
            manager, train_data, eval_data, sft_config
        )
        
        # Save SFT model
        sft_save_path = save_sft_model(
            manager, systems, sft_result["metrics"], args.sft_run_name
        )
        
        print(f"SFT model saved to: {sft_save_path}")
    
    # GRPO Training
    if args.training_type in ["grpo", "both"]:
        print("\n" + "="*70)
        print("STARTING GRPO TRAINING")
        print("="*70)
        
        # Load SFT model if specified
        if args.load_sft_model:
            print(f"Loading SFT model from: {args.load_sft_model}")
            # Implementation depends on how the path is structured
            # For now, assume it's been handled
        elif sft_save_path:
            print(f"Using SFT model from current training: {sft_save_path}")
        
        # Update GRPO config with output directory
        grpo_config = ALL_CONFIG["grpo"].copy()
        grpo_config["output_dir"] = f"{args.output_base}/grpo"
        
        # Train GRPO
        grpo_result = train_grpo_model(
            manager, train_data, eval_data, grpo_config,
            ALL_CONFIG["system"]["reasoning_start"],
            ALL_CONFIG["system"]["reasoning_end"],
            ALL_CONFIG["system"]["solution_start"],
            ALL_CONFIG["system"]["solution_end"]
        )
        
        # Save GRPO model
        grpo_save_path = save_grpo_model(
            manager, systems, grpo_result["metrics"], args.grpo_run_name
        )
        
        print(f"GRPO model saved to: {grpo_save_path}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    
    if sft_save_path:
        print(f"SFT model: {sft_save_path}")
    if grpo_save_path:
        print(f"GRPO model: {grpo_save_path}")
    
    print(f"Trained on systems: {', '.join(systems)}")
    print(f"Total training samples: {len(train_data)}")


if __name__ == "__main__":
    main()