#!/usr/bin/env python3
"""Standalone data generation script for creating training and evaluation datasets."""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config, get_available_environments
from core.data_pipeline import UniversalDataGenerator
import pickle

def main():
    parser = argparse.ArgumentParser(description="Generate training and evaluation datasets")
    
    # System selection
    parser.add_argument("--systems", type=str, nargs="+",
                       default=None,
                       help="Systems to generate data for (default: all available)")
    
    # Data configuration
    parser.add_argument("--train-samples", type=int, default=500,
                       help="Number of training samples per system")
    parser.add_argument("--eval-samples", type=int, default=100,
                       help="Number of evaluation samples per system")
    parser.add_argument("--total-samples", type=int, default=None,
                       help="Total samples per system (auto-split train/eval)")
    
    # Dataset configuration
    parser.add_argument("--dataset-name", type=str, default=None,
                       help="Custom dataset name (auto-generated if not provided)")
    parser.add_argument("--output-dir", type=str, default="datasets",
                       help="Output directory for datasets")
    parser.add_argument("--format", choices=["pickle", "json", "both"], default="both",
                       help="Dataset output format")
    
    # Generation parameters
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible generation")
    parser.add_argument("--split-ratio", type=float, default=0.9,
                       help="Train/eval split ratio (if using total-samples)")
    
    # System parameters (override config)
    parser.add_argument("--dt", type=float, default=None,
                       help="Time step (override config)")
    parser.add_argument("--steps", type=int, default=None,
                       help="Number of control steps (override config)")
    
    # Quality control
    parser.add_argument("--validate", action="store_true",
                       help="Validate generated data quality")
    parser.add_argument("--preview", action="store_true",
                       help="Generate and show a few samples without saving")
    
    args = parser.parse_args()
    
    # Set random seed
    import numpy as np
    import random
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Determine systems to generate data for
    available_systems = get_available_environments()
    if args.systems is None:
        systems = available_systems
        print(f"🌍 Generating data for all available systems: {systems}")
    else:
        systems = args.systems
        # Validate system names
        for system in systems:
            if system not in available_systems:
                raise ValueError(f"Unknown system: {system}. Available: {available_systems}")
        print(f"🌍 Generating data for specified systems: {systems}")
    
    # Load configuration
    config = get_config()
    
    # Override config parameters if provided
    dt = args.dt if args.dt is not None else config["system"]["dt"]
    steps = args.steps if args.steps is not None else config["system"]["steps"]
    
    print(f"⚙️  Configuration:")
    print(f"   Time step (dt): {dt}")
    print(f"   Control steps: {steps}")
    print(f"   Random seed: {args.seed}")
    
    # Calculate samples per system
    if args.total_samples is not None:
        total_per_system = args.total_samples
        train_samples = int(total_per_system * args.split_ratio)
        eval_samples = total_per_system - train_samples
        print(f"📊 Auto-split {total_per_system} samples: {train_samples} train + {eval_samples} eval")
    else:
        train_samples = args.train_samples
        eval_samples = args.eval_samples
        print(f"📊 Manual split: {train_samples} train + {eval_samples} eval")
    
    # Create data generator
    generator = UniversalDataGenerator(
        systems=systems,
        dt=dt,
        steps=steps,
        reasoning_start=config["system"]["reasoning_start"],
        reasoning_end=config["system"]["reasoning_end"],
        solution_start=config["system"]["solution_start"],
        solution_end=config["system"]["solution_end"]
    )
    
    if args.preview:
        print("\\n🔍 PREVIEW MODE - Generating sample data...")
        for system in systems:
            print(f"\\n--- {system.upper()} SAMPLE ---")
            sample_data = generator.generate_single_system_dataset(system, 1)
            if sample_data:
                sample = sample_data[0]
                print(f"Input: {sample['input'][:100]}...")
                print(f"Output: {sample['output']}")
                print(f"System: {sample.get('system_type', 'Unknown')}")
        print("\\n✅ Preview completed!")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate dataset name
    if args.dataset_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        systems_str = "_".join(systems)
        dataset_name = f"{systems_str}_{train_samples}train_{eval_samples}eval_{timestamp}"
    else:
        dataset_name = args.dataset_name
    
    print(f"\\n🏭 GENERATING DATASETS: {dataset_name}")
    print("=" * 60)
    
    all_train_data = []
    all_eval_data = []
    dataset_info = {
        "name": dataset_name,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "systems": systems,
            "train_samples_per_system": train_samples,
            "eval_samples_per_system": eval_samples,
            "dt": dt,
            "steps": steps,
            "seed": args.seed,
            "split_ratio": args.split_ratio if args.total_samples else None
        },
        "system_stats": {}
    }
    
    # Generate data for each system
    for i, system in enumerate(systems):
        print(f"\\n📊 Generating data for {system} ({i+1}/{len(systems)})...")
        
        # Generate training data
        print(f"   🏋️  Training data: {train_samples} samples")
        train_data = generator.generate_single_system_dataset(system, train_samples)
        
        # Generate evaluation data
        print(f"   📈 Evaluation data: {eval_samples} samples")
        eval_data = generator.generate_single_system_dataset(system, eval_samples)
        
        # Add to combined datasets
        all_train_data.extend(train_data)
        all_eval_data.extend(eval_data)
        
        # Collect statistics
        dataset_info["system_stats"][system] = {
            "train_samples": len(train_data),
            "eval_samples": len(eval_data),
            "total_samples": len(train_data) + len(eval_data)
        }
        
        print(f"   ✅ Generated {len(train_data)} train + {len(eval_data)} eval samples")
    
    print(f"\\n💾 SAVING DATASETS...")
    
    # Save datasets in requested formats
    if args.format in ["pickle", "both"]:
        # Save as pickle files
        train_file = output_dir / f"{dataset_name}_train.pkl"
        eval_file = output_dir / f"{dataset_name}_eval.pkl"
        info_file = output_dir / f"{dataset_name}_info.pkl"
        
        with open(train_file, 'wb') as f:
            pickle.dump(all_train_data, f)
        with open(eval_file, 'wb') as f:
            pickle.dump(all_eval_data, f)
        with open(info_file, 'wb') as f:
            pickle.dump(dataset_info, f)
        
        print(f"   📦 Pickle format:")
        print(f"      Train: {train_file}")
        print(f"      Eval:  {eval_file}")
        print(f"      Info:  {info_file}")
    
    if args.format in ["json", "both"]:
        # Save as JSON files
        train_file = output_dir / f"{dataset_name}_train.json"
        eval_file = output_dir / f"{dataset_name}_eval.json"
        info_file = output_dir / f"{dataset_name}_info.json"
        
        with open(train_file, 'w') as f:
            json.dump(all_train_data, f, indent=2)
        with open(eval_file, 'w') as f:
            json.dump(all_eval_data, f, indent=2)
        with open(info_file, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"   📄 JSON format:")
        print(f"      Train: {train_file}")
        print(f"      Eval:  {eval_file}")
        print(f"      Info:  {info_file}")
    
    # Validation
    if args.validate:
        print(f"\\n🔍 VALIDATING DATASETS...")
        
        # Check data quality
        total_train = len(all_train_data)
        total_eval = len(all_eval_data)
        
        print(f"   Total samples: {total_train + total_eval}")
        print(f"   Train samples: {total_train}")
        print(f"   Eval samples: {total_eval}")
        
        # Check system distribution
        train_systems = {}
        eval_systems = {}
        
        for sample in all_train_data:
            system = sample.get('system_type', 'unknown')
            train_systems[system] = train_systems.get(system, 0) + 1
        
        for sample in all_eval_data:
            system = sample.get('system_type', 'unknown')
            eval_systems[system] = eval_systems.get(system, 0) + 1
        
        print(f"   Train distribution: {train_systems}")
        print(f"   Eval distribution: {eval_systems}")
        
        # Check sample completeness
        incomplete_samples = 0
        for sample in all_train_data + all_eval_data:
            if not all(key in sample for key in ['input', 'output', 'system_type']):
                incomplete_samples += 1
        
        if incomplete_samples > 0:
            print(f"   ⚠️  Found {incomplete_samples} incomplete samples")
        else:
            print(f"   ✅ All samples complete")
    
    # Summary
    print(f"\\n" + "=" * 60)
    print(f"🎉 DATASET GENERATION COMPLETED!")
    print(f"=" * 60)
    print(f"Dataset name: {dataset_name}")
    print(f"Total systems: {len(systems)}")
    print(f"Total train samples: {len(all_train_data)}")
    print(f"Total eval samples: {len(all_eval_data)}")
    print(f"Output directory: {output_dir.absolute()}")
    
    # Usage instructions
    print(f"\\n📋 USAGE INSTRUCTIONS:")
    print(f"\\n1. For training:")
    if args.format in ["pickle", "both"]:
        print(f"   python scripts/train_single_system.py \\\\")
        print(f"       --system {systems[0]} \\\\")
        print(f"       --use-saved-data {output_dir}/{dataset_name}_train.pkl")
    
    print(f"\\n2. For evaluation:")
    if args.format in ["pickle", "both"]:
        print(f"   python scripts/evaluate_model.py \\\\")
        print(f"       --model-path models/single_system/{systems[0]}/sft/latest \\\\")
        print(f"       --model-type single_system \\\\")
        print(f"       --eval-data {output_dir}/{dataset_name}_eval.pkl")
    
    print(f"\\n3. Dataset info:")
    print(f"   Dataset details saved in {dataset_name}_info file")

if __name__ == "__main__":
    main()