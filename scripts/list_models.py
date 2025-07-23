#!/usr/bin/env python3
"""Script to list all saved models."""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.model_manager import UniversalModelManager
import json
from datetime import datetime


def format_timestamp(timestamp_str):
    """Format timestamp string to readable format."""
    try:
        dt = datetime.fromisoformat(timestamp_str)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return timestamp_str


def print_model_info(model_path, metadata):
    """Print formatted model information."""
    print(f"    Path: {model_path}")
    print(f"    Timestamp: {format_timestamp(metadata.get('timestamp', 'Unknown'))}")
    
    if 'trained_systems' in metadata:
        systems = metadata['trained_systems']
        print(f"    Systems: {', '.join(systems)} ({len(systems)} systems)")
    elif 'system' in metadata:
        print(f"    System: {metadata['system']}")
    
    if 'metrics' in metadata:
        metrics = metadata['metrics']
        if 'train_loss' in metrics:
            print(f"    Train Loss: {metrics['train_loss']:.6f}")
        if 'eval_loss' in metrics:
            print(f"    Eval Loss: {metrics['eval_loss']:.6f}")
    
    print(f"    Model Type: {metadata.get('model_type', 'Unknown')}")
    print(f"    Training Type: {metadata.get('training_type', 'Unknown')}")
    print(f"    LoRA Rank: {metadata.get('lora_rank', 'Unknown')}")
    print()


def main():
    parser = argparse.ArgumentParser(description="List all saved models")
    
    parser.add_argument("--model-type", choices=["all", "universal", "single_system"], 
                       default="all", help="Filter by model type")
    parser.add_argument("--training-type", choices=["all", "sft", "grpo"], 
                       default="all", help="Filter by training type")
    parser.add_argument("--system", type=str, 
                       help="Filter single-system models by system name")
    parser.add_argument("--detailed", action="store_true",
                       help="Show detailed model information")
    parser.add_argument("--sort-by", choices=["name", "timestamp"], default="timestamp",
                       help="Sort models by name or timestamp")
    
    args = parser.parse_args()
    
    # Create model manager
    manager = UniversalModelManager()
    
    # Get all saved models
    saved_models = manager.list_saved_models()
    
    print("=" * 80)
    print("SAVED MODELS SUMMARY")
    print("=" * 80)
    
    total_models = 0
    
    # Universal models
    if args.model_type in ["all", "universal"]:
        print("\nUNIVERSAL MODELS:")
        print("-" * 40)
        
        for training_type in ["sft", "grpo"]:
            if args.training_type not in ["all", training_type]:
                continue
            
            models = saved_models["universal"][training_type]
            
            if models:
                print(f"\n  {training_type.upper()} Models:")
                
                # Sort models
                model_info_list = []
                for model_name in models:
                    model_path = f"models/universal/{training_type}/{model_name}"
                    try:
                        metadata = manager.get_model_info(model_path)
                        model_info_list.append((model_name, model_path, metadata))
                    except Exception as e:
                        print(f"    {model_name} (Error reading metadata: {e})")
                
                # Sort by chosen criteria
                if args.sort_by == "timestamp":
                    model_info_list.sort(key=lambda x: x[2].get('timestamp', ''), reverse=True)
                else:
                    model_info_list.sort(key=lambda x: x[0])
                
                for model_name, model_path, metadata in model_info_list:
                    total_models += 1
                    if args.detailed:
                        print(f"  • {model_name}:")
                        print_model_info(model_path, metadata)
                    else:
                        systems = metadata.get('trained_systems', ['Unknown'])
                        timestamp = format_timestamp(metadata.get('timestamp', 'Unknown'))
                        print(f"    • {model_name} - Systems: {', '.join(systems)} - {timestamp}")
            else:
                print(f"    No {training_type.upper()} models found")
    
    # Single-system models
    if args.model_type in ["all", "single_system"]:
        print("\n\nSINGLE-SYSTEM MODELS:")
        print("-" * 40)
        
        single_system_models = saved_models["single_system"]
        
        if single_system_models:
            for system_name, system_models in single_system_models.items():
                if args.system and args.system != system_name:
                    continue
                
                print(f"\n  {system_name.upper()} System:")
                
                for training_type in ["sft", "grpo"]:
                    if args.training_type not in ["all", training_type]:
                        continue
                    
                    models = system_models[training_type]
                    
                    if models:
                        print(f"    {training_type.upper()} Models:")
                        
                        # Sort models
                        model_info_list = []
                        for model_name in models:
                            model_path = f"models/single_system/{system_name}/{training_type}/{model_name}"
                            try:
                                metadata = manager.get_model_info(model_path)
                                model_info_list.append((model_name, model_path, metadata))
                            except Exception as e:
                                print(f"      {model_name} (Error reading metadata: {e})")
                        
                        # Sort by chosen criteria
                        if args.sort_by == "timestamp":
                            model_info_list.sort(key=lambda x: x[2].get('timestamp', ''), reverse=True)
                        else:
                            model_info_list.sort(key=lambda x: x[0])
                        
                        for model_name, model_path, metadata in model_info_list:
                            total_models += 1
                            if args.detailed:
                                print(f"      • {model_name}:")
                                print_model_info(model_path, metadata)
                            else:
                                timestamp = format_timestamp(metadata.get('timestamp', 'Unknown'))
                                print(f"        • {model_name} - {timestamp}")
                    else:
                        print(f"      No {training_type.upper()} models")
        else:
            print("    No single-system models found")
    
    # Summary
    print("\n" + "=" * 80)
    print(f"TOTAL MODELS FOUND: {total_models}")
    
    if total_models == 0:
        print("\nNo models found matching your criteria.")
        print("Try running training scripts to create models first.")
    else:
        print(f"\nFilters applied:")
        print(f"  Model type: {args.model_type}")
        print(f"  Training type: {args.training_type}")
        if args.system:
            print(f"  System: {args.system}")
        print(f"  Sorted by: {args.sort_by}")
    
    print("=" * 80)
    
    # Show usage examples
    if total_models > 0:
        print("\nExample usage:")
        print("  # Evaluate latest universal GRPO model:")
        print("  python scripts/evaluate_model.py --model-path models/universal/grpo/latest --model-type universal")
        print("  ")
        print("  # Evaluate specific single-system model:")
        print("  python scripts/evaluate_model.py --model-path models/single_system/double_integrator/grpo/latest --model-type single_system")


if __name__ == "__main__":
    main()