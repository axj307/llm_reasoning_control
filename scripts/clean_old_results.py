#!/usr/bin/env python3
"""
Clean up old messy result directories and organize them properly.
"""

import os
import shutil
from pathlib import Path

def main():
    print("🧹 CLEANING OLD RESULT DIRECTORIES")
    print("=" * 50)
    
    results_dir = Path("results")
    if not results_dir.exists():
        print("✅ No results directory to clean")
        return
    
    # List all old result directories
    old_dirs = list(results_dir.glob("di_complete_*"))
    
    if not old_dirs:
        print("✅ No old result directories found")
        return
    
    print(f"📋 Found {len(old_dirs)} old result directories:")
    for old_dir in old_dirs:
        print(f"   - {old_dir.name}")
    
    # Ask for confirmation (in script, we'll just proceed)
    print("\n🗑️  Removing old messy result directories...")
    
    total_size = 0
    for old_dir in old_dirs:
        # Calculate size
        dir_size = sum(f.stat().st_size for f in old_dir.rglob('*') if f.is_file())
        total_size += dir_size
        
        # Remove directory
        shutil.rmtree(old_dir)
        print(f"   ✅ Removed {old_dir.name} ({dir_size / 1024 / 1024:.1f} MB)")
    
    print(f"\n📊 Summary:")
    print(f"   Directories removed: {len(old_dirs)}")
    print(f"   Space freed: {total_size / 1024 / 1024:.1f} MB")
    print("\n✨ Results will now be organized in figures/job_XXXXX/ directories")

if __name__ == "__main__":
    main()