#!/usr/bin/env python3
"""
Generate and cache datasets for control systems.
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import dataset_manager
from datasets.systems import DoubleIntegratorDataset
from logger import logger


def main():
    parser = argparse.ArgumentParser(description="Generate control system datasets")
    
    # System selection
    parser.add_argument("--system", "-s", 
                       choices=["di", "double_integrator"],
                       default="di",
                       help="Control system to generate data for")
    
    # Dataset parameters
    parser.add_argument("--samples", "-n", type=int, default=1000,
                       help="Number of samples to generate")
    parser.add_argument("--dt", type=float, default=0.1,
                       help="Time step")
    parser.add_argument("--steps", type=int, default=50,
                       help="Number of control steps")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Control options
    parser.add_argument("--force", "-f", action="store_true",
                       help="Force regeneration even if cached")
    parser.add_argument("--data-dir", type=str, default="datasets",
                       help="Directory to store datasets")
    parser.add_argument("--format", choices=["pickle", "json"], default="pickle",
                       help="Dataset format (pickle is faster)")
    
    # Additional options
    parser.add_argument("--list", "-l", action="store_true",
                       help="List cached datasets")
    parser.add_argument("--clear-cache", action="store_true",
                       help="Clear cached datasets")
    
    args = parser.parse_args()
    
    # Create dataset manager with specified directory
    manager = dataset_manager
    manager.data_dir = Path(args.data_dir)
    manager.data_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle special operations
    if args.list:
        print("\n📋 Cached datasets:")
        print("=" * 60)
        datasets = manager.list_cached_datasets()
        if not datasets:
            print("No cached datasets found.")
        else:
            for ds in datasets:
                print(f"\n📦 {ds['filename']}")
                print(f"   System: {ds.get('system', 'Unknown')}")
                print(f"   Samples: {ds.get('num_samples', 'Unknown')}")
                print(f"   Size: {ds['size_mb']:.2f} MB")
                print(f"   Created: {ds.get('timestamp', ds['modified'])}")
        return
        
    if args.clear_cache:
        confirm = input("⚠️  Clear all cached datasets? (y/N): ")
        if confirm.lower() == 'y':
            manager.clear_cache()
            print("✅ Cache cleared")
        else:
            print("❌ Cancelled")
        return
    
    # Map system names
    system_map = {
        "di": "double_integrator",
        "double_integrator": "double_integrator"
    }
    system = system_map[args.system]
    
    # Register dataset classes
    if system == "double_integrator":
        manager.register_dataset_class("double_integrator", DoubleIntegratorDataset)
    
    print(f"\n🏭 Generating dataset for {system}")
    print("=" * 60)
    print(f"📊 Parameters:")
    print(f"   Samples: {args.samples}")
    print(f"   Time step: {args.dt}")
    print(f"   Control steps: {args.steps}")
    print(f"   Random seed: {args.seed}")
    print(f"   Data directory: {manager.data_dir}")
    
    # Generate or load dataset
    try:
        data = manager.get_or_create(
            system=system,
            num_samples=args.samples,
            dt=args.dt,
            steps=args.steps,
            force_regenerate=args.force,
            seed=args.seed
        )
        
        print(f"\n✅ Dataset ready with {len(data)} samples")
        
        # Show sample
        if data:
            print("\n📄 Sample data:")
            sample = data[0]
            print(f"   System: {sample.get('system_type', 'Unknown')}")
            if 'prompt' in sample and sample['prompt']:
                print(f"   User prompt: {sample['prompt'][1]['content'][:100]}...")
            if 'answer' in sample:
                print(f"   Answer preview: {sample['answer'][:50]}...")
                
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())