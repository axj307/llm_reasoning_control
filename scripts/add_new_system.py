#!/usr/bin/env python3
"""Script to add a new system to an existing universal model."""

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
    parser = argparse.ArgumentParser(description="Add a new system to existing universal model")
    
    # System configuration
    parser.add_argument("--new-system", type=str, required=True,
                       choices=AVAILABLE_SYSTEMS,
                       help=f"New system to add. Available: {AVAILABLE_SYSTEMS}")
    parser.add_argument("--base-model", type=str, default="latest",
                       help="Base universal model to extend (run name or 'latest')")
    parser.add_argument("--base-training-type", choices=["sft", "grpo"], default="grpo",
                       help="Training type of base model")
    
    # Data configuration
    parser.add_argument("--num-samples", type=int, default=200,
                       help="Number of samples for new system")
    
    # Training configuration
    parser.add_argument("--training-type", choices=["sft", "grpo", "both"], default="both",
                       help="Type of training to perform")
    parser.add_argument("--incremental", action="store_true",
                       help="Use incremental learning (start from existing model)")
    
    # Model configuration
    parser.add_argument("--lora-rank", type=int, default=ALL_CONFIG["model"]["lora_rank"],
                       help="LoRA rank")
    
    # Hardware configuration
    parser.add_argument("--gpu-id", type=str, default="0",
                       help="GPU ID to use")
    
    # Output configuration
    parser.add_argument("--output-base", type=str, default="./temp_extension",
                       help="Base output directory for training")
    
    args = parser.parse_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    print(f"Using GPU: {args.gpu_id}")
    
    new_system = args.new_system
    print(f"Adding new system: {new_system}")
    
    # Load existing universal model
    manager = UniversalModelManager()
    
    if args.incremental:
        print(f"Loading existing universal model ({args.base_training_type}/{args.base_model})...")
        model, tokenizer, lora_request, metadata = manager.load_universal_model(
            run_name=args.base_model,
            training_type=args.base_training_type
        )
        existing_systems = metadata.get("trained_systems", [])"
        
        if new_system in existing_systems:
            print(f"Warning: System {new_system} already in trained systems: {existing_systems}")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return
    else:
        print("Setting up new model...")
        model, tokenizer = manager.setup_model(lora_rank=args.lora_rank)
        existing_systems = []
    
    # Generate data for new system
    print(f"Generating {args.num_samples} samples for {new_system}...")
    generator = UniversalDataGenerator(
        systems=[new_system],
        dt=ALL_CONFIG["system"]["dt"],
        steps=ALL_CONFIG["system"]["steps"],
        reasoning_start=ALL_CONFIG["system"]["reasoning_start"],
        reasoning_end=ALL_CONFIG["system"]["reasoning_end"],
        solution_start=ALL_CONFIG["system"]["solution_start"],
        solution_end=ALL_CONFIG["system"]["solution_end"]
    )
    
    new_data = generator.generate_single_system_dataset(new_system, args.num_samples)
    
    # If incremental, we might want to mix with some existing system data
    if args.incremental and existing_systems:
        print("For incremental learning, loading some existing system data...")
        
        # Load smaller amount of existing system data to prevent forgetting
        existing_data = []
        for system in existing_systems[:2]:  # Limit to first 2 systems to save memory
            try:
                system_data = generator.generate_single_system_dataset(system, 50)  # Small amount
                existing_data.extend(system_data)
            except Exception as e:
                print(f"Could not generate data for {system}: {e}")
        
        if existing_data:
            print(f"Mixing {len(new_data)} new samples with {len(existing_data)} existing samples")
            all_data = new_data + existing_data
        else:
            all_data = new_data
    else:
        all_data = new_data
    
    # Split data
    train_data, eval_data = generator.split_dataset(
        all_data, ALL_CONFIG["data"]["train_eval_split"]
    )
    
    # Updated system list
    updated_systems = list(set(existing_systems + [new_system]))
    print(f"Updated system list: {updated_systems}")
    
    # Set up chat template for updated systems
    setup_universal_chat_template(
        manager, updated_systems,
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
        print(f"EXTENDING MODEL WITH SFT FOR {new_system.upper()}")
        print("="*70)
        
        # Use smaller learning rate for incremental learning
        sft_config = ALL_CONFIG["sft"].copy()
        if args.incremental:
            sft_config["learning_rate"] = sft_config["learning_rate"] * 0.5  # Reduce LR
            sft_config["num_train_epochs"] = 2  # Fewer epochs
        
        sft_config["output_dir"] = f"{args.output_base}/sft_extended"
        
        # Train SFT
        sft_result = train_sft_model(
            manager, train_data, eval_data, sft_config
        )
        
        # Save extended SFT model
        sft_save_path = save_sft_model(
            manager, updated_systems, sft_result["metrics"], 
            f"extended_with_{new_system}"
        )
        
        print(f"Extended SFT model saved to: {sft_save_path}")
    
    # GRPO Training
    if args.training_type in ["grpo", "both"]:
        print("\n" + "="*70)
        print(f"EXTENDING MODEL WITH GRPO FOR {new_system.upper()}")
        print("="*70)
        
        # Use smaller learning rate for incremental learning
        grpo_config = ALL_CONFIG["grpo"].copy()
        if args.incremental:
            grpo_config["learning_rate"] = grpo_config["learning_rate"] * 0.5  # Reduce LR
            grpo_config["max_steps"] = grpo_config["max_steps"] // 2  # Fewer steps
        
        grpo_config["output_dir"] = f"{args.output_base}/grpo_extended"
        
        # Train GRPO
        grpo_result = train_grpo_model(
            manager, train_data, eval_data, grpo_config,
            ALL_CONFIG["system"]["reasoning_start"],
            ALL_CONFIG["system"]["reasoning_end"],
            ALL_CONFIG["system"]["solution_start"],
            ALL_CONFIG["system"]["solution_end"]
        )
        
        # Save extended GRPO model
        grpo_save_path = save_grpo_model(
            manager, updated_systems, grpo_result["metrics"], 
            f"extended_with_{new_system}"
        )
        
        print(f"Extended GRPO model saved to: {grpo_save_path}")
    
    print("\n" + "="*70)
    print("SYSTEM EXTENSION COMPLETED")
    print("="*70)
    
    print(f"Added system: {new_system}")
    print(f"Original systems: {existing_systems}")
    print(f"Updated systems: {updated_systems}")
    
    if sft_save_path:
        print(f"Extended SFT model: {sft_save_path}")
    if grpo_save_path:
        print(f"Extended GRPO model: {grpo_save_path}")
    
    print(f"Training samples used: {len(train_data)}")
    
    # Suggestion for testing
    print("\nNext steps:")
    print(f"1. Test the extended model on {new_system} using evaluate_model.py")
    print(f"2. Test the extended model on original systems to check for forgetting")
    print(f"3. Consider running a full comparison with the original model")


if __name__ == "__main__":
    main()