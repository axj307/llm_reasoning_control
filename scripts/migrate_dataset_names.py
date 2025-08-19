#!/usr/bin/env python3
"""
Migrate existing datasets to new consistent naming scheme.
"""

import os
import shutil
from pathlib import Path

def main():
    print("🔄 MIGRATING DATASET NAMES TO CONSISTENT SCHEME")
    print("=" * 60)
    
    datasets_dir = Path("datasets")
    if not datasets_dir.exists():
        print("❌ No datasets directory found")
        return
    
    # Define migration mapping
    migrations = {
        # Old confusing names -> New clear names
        "di_complete_train.json": "double_integrator_train.json",
        "di_complete_train.pkl": "double_integrator_train.pkl", 
        "di_complete_eval.json": "double_integrator_eval.json",
        "di_complete_eval.pkl": "double_integrator_eval.pkl",
        "di_complete_info.json": "double_integrator_info.json", 
        "di_complete_info.pkl": "double_integrator_info.pkl",
        
        # Handle the confusing di_eval_complete -> use the eval files
        "di_eval_complete_eval.json": "double_integrator_eval.json",
        "di_eval_complete_eval.pkl": "double_integrator_eval.pkl",
        "di_eval_complete_info.json": "double_integrator_eval_info.json",
        "di_eval_complete_info.pkl": "double_integrator_eval_info.pkl",
    }
    
    print("📋 Planned migrations:")
    successful_migrations = 0
    skipped_migrations = 0
    
    for old_name, new_name in migrations.items():
        old_path = datasets_dir / old_name
        new_path = datasets_dir / new_name
        
        if old_path.exists():
            if new_path.exists():
                print(f"⚠️  Skipping {old_name} -> {new_name} (target exists)")
                skipped_migrations += 1
            else:
                # Check if this is a duplicate (di_eval_complete vs di_complete)
                if "di_eval_complete_eval" in old_name and (datasets_dir / "di_complete_eval.json").exists():
                    print(f"🔍 Comparing {old_name} with di_complete_eval.json...")
                    # Use the larger file (likely more complete)
                    old_size = old_path.stat().st_size
                    alt_path = datasets_dir / "di_complete_eval.json"
                    alt_size = alt_path.stat().st_size
                    
                    if old_size > alt_size:
                        print(f"✅ Using {old_name} (larger: {old_size} vs {alt_size})")
                        shutil.move(str(old_path), str(new_path))
                    else:
                        print(f"🗑️  Removing smaller duplicate {old_name}")
                        old_path.unlink()
                        continue
                else:
                    print(f"✅ {old_name} -> {new_name}")
                    shutil.move(str(old_path), str(new_path))
                
                successful_migrations += 1
        else:
            print(f"⚠️  {old_name} not found")
    
    print(f"\n📊 Migration Summary:")
    print(f"   ✅ Successful migrations: {successful_migrations}")
    print(f"   ⚠️  Skipped migrations: {skipped_migrations}")
    
    # Clean up empty/duplicate files
    cleanup_files = [
        "di_eval_complete_train.json",  # This is likely empty
        "di_eval_complete_train.pkl",
    ]
    
    print(f"\n🧹 Cleaning up duplicate/empty files:")
    for cleanup_file in cleanup_files:
        cleanup_path = datasets_dir / cleanup_file
        if cleanup_path.exists():
            size = cleanup_path.stat().st_size
            if size < 100:  # Likely empty or nearly empty
                print(f"🗑️  Removing small/empty file: {cleanup_file} ({size} bytes)")
                cleanup_path.unlink()
            else:
                print(f"⚠️  {cleanup_file} is not empty ({size} bytes), keeping it")
    
    print(f"\n📁 Final dataset structure:")
    if datasets_dir.exists():
        for file_path in sorted(datasets_dir.glob("*.json")):
            size_mb = file_path.stat().st_size / 1024 / 1024
            print(f"   📄 {file_path.name} ({size_mb:.1f} MB)")
    
    print(f"\n✨ Dataset naming migration completed!")
    print(f"🎯 New consistent naming scheme:")
    print(f"   - double_integrator_train.json (training data)")
    print(f"   - double_integrator_eval.json (evaluation data)")

if __name__ == "__main__":
    main()