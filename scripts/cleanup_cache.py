#!/usr/bin/env python3
"""
Cleanup script to remove redundant SLURM scripts and datasets.
Keeps only essential files for the working pipeline.
"""

import os
import shutil
from pathlib import Path

def main():
    print("🧹 CODEBASE CLEANUP")
    print("=" * 50)
    
    # Define what to keep
    essential_slurm_scripts = {
        "complete_di_pipeline.sbatch",           # Main working pipeline
        "quick_pipeline_test.sbatch",            # Quick test (10 steps)
        "standard_pipeline.sbatch",              # Future universal model
        "evaluate_existing_model.sbatch",       # Evaluation script
    }
    
    essential_datasets = {
        "di_complete_train.pkl",
        "di_complete_eval.pkl", 
        "di_complete_info.pkl",
        "di_complete_train.json",
        "di_complete_eval.json",
        "di_complete_info.json",
        # Keep the latest eval dataset too
        "di_eval_complete_eval.pkl",
        "di_eval_complete_eval.json",
        "di_eval_complete_info.pkl",
        "di_eval_complete_info.json",
    }
    
    # Create backup directory
    backup_dir = Path("cleanup_backup")
    backup_dir.mkdir(exist_ok=True)
    
    print("📁 Creating backup directory...")
    
    # Cleanup SLURM scripts
    slurm_dir = Path("slurm")
    if slurm_dir.exists():
        print(f"\n🗂️  SLURM Scripts Cleanup")
        print(f"   Current: {len(list(slurm_dir.glob('*.sbatch')))} files")
        
        slurm_backup = backup_dir / "slurm"
        slurm_backup.mkdir(exist_ok=True)
        
        removed_count = 0
        for script in slurm_dir.glob("*.sbatch"):
            if script.name not in essential_slurm_scripts:
                # Move to backup
                shutil.move(str(script), str(slurm_backup / script.name))
                removed_count += 1
                print(f"   📦 Moved to backup: {script.name}")
        
        print(f"   ✅ Removed: {removed_count} files")
        print(f"   ✅ Kept: {len(essential_slurm_scripts)} essential files")
        print(f"   📋 Essential files:")
        for script in essential_slurm_scripts:
            print(f"      - {script}")
    
    # Cleanup datasets
    datasets_dir = Path("datasets")
    if datasets_dir.exists():
        print(f"\n📊 Datasets Cleanup")
        all_datasets = list(datasets_dir.glob("*"))
        print(f"   Current: {len(all_datasets)} files")
        
        datasets_backup = backup_dir / "datasets"
        datasets_backup.mkdir(exist_ok=True)
        
        removed_count = 0
        for dataset in all_datasets:
            if dataset.name not in essential_datasets:
                # Move to backup
                shutil.move(str(dataset), str(datasets_backup / dataset.name))
                removed_count += 1
                print(f"   📦 Moved to backup: {dataset.name}")
        
        print(f"   ✅ Removed: {removed_count} files")
        print(f"   ✅ Kept: {len(essential_datasets)} essential files")
        print(f"   📋 Essential datasets:")
        for dataset in sorted(essential_datasets):
            print(f"      - {dataset}")
    
    # Show disk space saved
    try:
        backup_size = sum(f.stat().st_size for f in backup_dir.rglob('*') if f.is_file())
        backup_size_mb = backup_size / (1024 * 1024)
        print(f"\n💾 Backup created: {backup_size_mb:.1f}MB moved to cleanup_backup/")
    except:
        print(f"\n💾 Backup created in cleanup_backup/")
    
    print(f"\n🎉 CLEANUP COMPLETED!")
    print(f"   📁 Backups stored in: cleanup_backup/")
    print(f"   🗂️  SLURM: {len(essential_slurm_scripts)} essential scripts")
    print(f"   📊 Datasets: {len([d for d in essential_datasets if d.endswith('.pkl')])} essential datasets")
    print(f"\n💡 To restore files: mv cleanup_backup/slurm/* slurm/ && mv cleanup_backup/datasets/* datasets/")

if __name__ == "__main__":
    main()