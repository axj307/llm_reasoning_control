#!/usr/bin/env python3
"""Training script for single-system specialist models."""

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
from gpu_utils import auto_gpu_config, set_gpu_device
from data_utils import load_dataset, filter_dataset_by_system, load_train_eval_datasets, list_available_datasets


def main():
    parser = argparse.ArgumentParser(description="Train single-system specialist model")
    
    # System selection
    parser.add_argument("--system", type=str, required=True,
                       choices=AVAILABLE_SYSTEMS,
                       help=f"System to train on. Available: {AVAILABLE_SYSTEMS}")
    
    # Training type
    parser.add_argument("--training-type", choices=["sft", "grpo", "both"], default="both",
                       help="Type of training to perform")
    
    # Data configuration
    parser.add_argument("--num-samples", type=int, default=500,
                       help="Number of training samples (only used if generating new data)")
    parser.add_argument("--dataset-name", type=str,
                       help="Name of pre-generated dataset to use")
    parser.add_argument("--dataset-dir", type=str, default="datasets",
                       help="Directory containing pre-generated datasets")
    parser.add_argument("--use-saved-data", type=str,
                       help="Direct path to saved dataset file (legacy option)")
    parser.add_argument("--list-datasets", action="store_true",
                       help="List available datasets and exit")
    parser.add_argument("--generate-data", action="store_true",
                       help="Force generation of new data (ignore pre-generated datasets)")
    
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
    parser.add_argument("--gpu-id", type=int, default=None,
                       help="GPU ID to use (auto-select if not specified)")
    parser.add_argument("--no-auto-gpu", action="store_true",
                       help="Disable automatic GPU selection")
    
    # Output configuration
    parser.add_argument("--output-base", type=str, default="./temp_training",
                       help="Base output directory for training")
    
    args = parser.parse_args()
    
    # Handle special commands
    if args.list_datasets:
        print("📂 Available datasets:")
        datasets = list_available_datasets(args.dataset_dir)
        if datasets:
            for dataset in datasets:
                print(f"   • {dataset}")
        else:
            print(f"   No datasets found in '{args.dataset_dir}'")
        print(f"\nTo use a dataset, add: --dataset-name DATASET_NAME")
        return
    
    # Handle GPU selection
    if args.gpu_id is not None:
        print(f"📌 Using specified GPU {args.gpu_id}")
        set_gpu_device(args.gpu_id, auto_select=False)
        gpu_id = args.gpu_id
    elif not args.no_auto_gpu:
        print("🎯 Auto-selecting best available GPU...")
        gpu_config = auto_gpu_config()
        gpu_id = gpu_config['gpu_id']
    else:
        print("⚠️  Using default GPU (no auto-selection)")
        gpu_id = 0
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    system_name = args.system
    print(f"Training specialist model for system: {system_name}")
    
    # Create model manager
    manager = UniversalModelManager(ALL_CONFIG["model"]["base_model_name"])
    
    # Set up model with GPU configuration
    print("Setting up model...")
    model, tokenizer = manager.setup_model(
        max_seq_length=args.max_seq_length,
        lora_rank=args.lora_rank,
        gpu_id=gpu_id,
        auto_select_gpu=False  # We already did GPU selection above
    )
    
    # Load or generate training data
    print("📊 Preparing training data...")
    
    # Determine data loading strategy
    if args.dataset_name:
        # Load pre-generated dataset
        print(f"Loading pre-generated dataset: {args.dataset_name}")
        train_data, eval_data, dataset_info = load_train_eval_datasets(
            args.dataset_name, args.dataset_dir, system_name
        )
        print(f"   📈 Dataset info: {dataset_info.get('config', {})}")
        
    elif args.use_saved_data:
        # Legacy: load from direct file path
        print(f"Loading dataset from file: {args.use_saved_data}")
        data = load_dataset(args.use_saved_data)
        data = filter_dataset_by_system(data, system_name)
        
        # Split data manually since we don't have separate train/eval files
        split_ratio = ALL_CONFIG["data"]["train_eval_split"]
        split_idx = int(len(data) * split_ratio)
        train_data = data[:split_idx]
        eval_data = data[split_idx:]
        
    elif args.generate_data or True:  # Default to generating data
        # Generate new data
        print(f"Generating new data for {system_name} ({args.num_samples} samples)")
        generator = UniversalDataGenerator(
            systems=[system_name],
            dt=ALL_CONFIG["system"]["dt"],
            steps=ALL_CONFIG["system"]["steps"],
            reasoning_start=ALL_CONFIG["system"]["reasoning_start"],
            reasoning_end=ALL_CONFIG["system"]["reasoning_end"],
            solution_start=ALL_CONFIG["system"]["solution_start"],
            solution_end=ALL_CONFIG["system"]["solution_end"]
        )
        
        data = generator.generate_single_system_dataset(system_name, args.num_samples)
        
        # Split data
        train_data, eval_data = generator.split_dataset(
            data, ALL_CONFIG["data"]["train_eval_split"]
        )
        
        # Save dataset for future use
        if ALL_CONFIG["data"]["save_datasets"]:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_name = f"{system_name}_{len(data)}samples_{timestamp}"
            
            import pickle
            import os
            os.makedirs(args.dataset_dir, exist_ok=True)
            
            with open(f"{args.dataset_dir}/{dataset_name}_train.pkl", 'wb') as f:
                pickle.dump(train_data, f)
            with open(f"{args.dataset_dir}/{dataset_name}_eval.pkl", 'wb') as f:
                pickle.dump(eval_data, f)
            
            print(f"   💾 Saved dataset as: {dataset_name}")
    
    if not train_data:
        raise ValueError(f"No training data found for system: {system_name}")
    
    print(f"✅ Using {len(train_data)} train + {len(eval_data)} eval samples for {system_name}")
    
    # Note: train_data and eval_data are now ready for training
    
    # Set up chat template (single system)
    setup_universal_chat_template(
        manager, [system_name],
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
        print(f"STARTING SFT TRAINING FOR {system_name.upper()}")
        print("="*70)
        
        # Update SFT config with output directory
        sft_config = ALL_CONFIG["sft"].copy()
        sft_config["output_dir"] = f"{args.output_base}/{system_name}/sft"
        
        # Train SFT
        sft_result = train_sft_model(
            manager, train_data, eval_data, sft_config
        )
        
        # Save SFT model
        sft_save_path = save_sft_model(
            manager, [system_name], sft_result["metrics"], args.sft_run_name
        )
        
        print(f"SFT model saved to: {sft_save_path}")
    
    # GRPO Training
    if args.training_type in ["grpo", "both"]:
        print("\n" + "="*70)
        print(f"STARTING GRPO TRAINING FOR {system_name.upper()}")
        print("="*70)
        
        # Load SFT model if specified
        if args.load_sft_model:
            print(f"Loading SFT model from: {args.load_sft_model}")
            # Implementation depends on how the path is structured
        elif sft_save_path:
            print(f"Using SFT model from current training: {sft_save_path}")
        
        # Update GRPO config with output directory
        grpo_config = ALL_CONFIG["grpo"].copy()
        grpo_config["output_dir"] = f"{args.output_base}/{system_name}/grpo"
        
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
            manager, [system_name], grpo_result["metrics"], args.grpo_run_name
        )
        
        print(f"GRPO model saved to: {grpo_save_path}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    
    if sft_save_path:
        print(f"SFT model: {sft_save_path}")
    if grpo_save_path:
        print(f"GRPO model: {grpo_save_path}")
    
    print(f"Specialist model trained for: {system_name}")
    print(f"Total training samples: {len(train_data)}")


if __name__ == "__main__":
    main()