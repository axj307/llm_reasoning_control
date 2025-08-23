#!/usr/bin/env python3
"""
Workspace Cleanup Script for DI Research

This script removes temporary files, cache directories, and training outputs
while preserving important files like models, datasets, configurations, and logs.

Usage:
    python scripts/cleanup_workspace.py --dry-run     # Preview what will be deleted
    python scripts/cleanup_workspace.py --verbose     # Normal cleanup with details
    python scripts/cleanup_workspace.py --keep-wandb  # Keep wandb logs
    python scripts/cleanup_workspace.py --quiet       # Silent cleanup
"""

import os
import shutil
import argparse
import glob
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple


class WorkspaceCleanup:
    """Safely clean workspace while preserving important files."""
    
    # Directories that should NEVER be deleted
    PROTECTED_DIRS = [
        'models/',          # Trained models
        'datasets/',        # Training data  
        'results/',         # Experiment results
        'figures/',         # Generated figures and plots (PERMANENTLY PROTECTED)
        'scripts/',         # Core scripts
        'core/',           # Core modules
        'environments/',    # Environment definitions
        'training/',       # Training modules
        'evaluation/',     # Evaluation modules
        '.claude/',        # Claude configuration
        'configs/',        # Configuration files
        'slurm/',          # SLURM job scripts
        'slurm_logs/',     # SLURM job logs
        '.git/',           # Git repository
        'docs/',           # Documentation
        'notebooks/',      # Jupyter notebooks
        'reports/',        # Analysis reports
    ]
    
    # Patterns for files/directories to remove
    CLEANUP_PATTERNS = [
        # Training outputs
        'sft_pretraining_output/',
        'grpo_working_output/',
        'grpo_trainer_lora_model/',
        'sft_training_output/',
        'training_output*/',
        
        # Cache directories
        '__pycache__/',
        '*/__pycache__/',
        '**/__pycache__/',
        'unsloth_compiled_cache*/',
        '.cache/',
        'cache/',
        
        # Temporary directories
        'temp_*/',
        'tmp/',
        'temp/',
        'temporary*/',
        
        # Checkpoint directories
        'checkpoint*/',
        '*_lora/',
        '*.tmp',
        '*.temp',
        
        # Compiled files
        '*.pyc',
        '*.pyo',
        '*.pyd',
        
        # OS generated files
        '.DS_Store',
        'Thumbs.db',
        '*.swp',
        '*.swo',
        '*~',
    ]
    
    # Optional cleanup patterns (removed only with specific flags)
    OPTIONAL_PATTERNS = {
        'wandb': ['wandb/'],
        'models_temp': ['*_model_temp/', 'temp_model*/'],
    }
    
    def __init__(self, dry_run=False, verbose=False, quiet=False):
        self.dry_run = dry_run
        self.verbose = verbose
        self.quiet = quiet
        self.deleted_items = []
        self.total_size = 0
        self.start_time = time.time()
        self.log_file = f"cleanup_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
    def log(self, message: str, force=False):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        full_message = f"[{timestamp}] {message}"
        
        if not self.quiet or force:
            print(full_message)
            
        # Always write to log file
        with open(self.log_file, 'a') as f:
            f.write(full_message + '\n')
    
    def get_dir_size(self, path: str) -> int:
        """Calculate directory size in bytes."""
        total = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total += os.path.getsize(filepath)
                    except (OSError, FileNotFoundError):
                        pass
        except (OSError, FileNotFoundError):
            pass
        return total
    
    def format_size(self, size_bytes: int) -> str:
        """Format size in human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def is_protected(self, path: str) -> bool:
        """Check if path is protected from deletion."""
        path = path.rstrip('/')
        for protected in self.PROTECTED_DIRS:
            protected = protected.rstrip('/')
            if path == protected or path.startswith(protected + '/'):
                return True
        return False
    
    def find_cleanup_targets(self, include_optional: List[str] = None) -> List[Tuple[str, int]]:
        """Find all files/directories to be cleaned up."""
        if include_optional is None:
            include_optional = []
            
        targets = []
        all_patterns = self.CLEANUP_PATTERNS.copy()
        
        # Add optional patterns if requested
        for optional_type in include_optional:
            if optional_type in self.OPTIONAL_PATTERNS:
                all_patterns.extend(self.OPTIONAL_PATTERNS[optional_type])
        
        for pattern in all_patterns:
            matches = glob.glob(pattern, recursive=True)
            for match in matches:
                if not self.is_protected(match) and os.path.exists(match):
                    if os.path.isdir(match):
                        size = self.get_dir_size(match)
                    else:
                        try:
                            size = os.path.getsize(match)
                        except (OSError, FileNotFoundError):
                            size = 0
                    targets.append((match, size))
        
        # Remove duplicates and sort by size (largest first)
        targets = list(set(targets))
        targets.sort(key=lambda x: x[1], reverse=True)
        
        return targets
    
    def remove_item(self, path: str) -> bool:
        """Safely remove file or directory."""
        try:
            if os.path.isdir(path):
                if self.dry_run:
                    self.log(f"Would remove directory: {path}")
                else:
                    shutil.rmtree(path)
                    self.log(f"Removed directory: {path}")
            else:
                if self.dry_run:
                    self.log(f"Would remove file: {path}")
                else:
                    os.remove(path)
                    self.log(f"Removed file: {path}")
            return True
        except Exception as e:
            self.log(f"Failed to remove {path}: {e}")
            return False
    
    def cleanup(self, keep_wandb=False, keep_models_temp=False):
        """Perform the cleanup operation."""
        self.log("=" * 60, force=True)
        self.log("🧹 WORKSPACE CLEANUP STARTING", force=True)
        self.log("=" * 60, force=True)
        
        if self.dry_run:
            self.log("🔍 DRY RUN MODE - No files will actually be deleted", force=True)
        
        # Determine what to include
        include_optional = []
        if not keep_wandb:
            include_optional.append('wandb')
        if not keep_models_temp:
            include_optional.append('models_temp')
        
        # Find targets
        self.log("🔍 Scanning for cleanup targets...")
        targets = self.find_cleanup_targets(include_optional)
        
        if not targets:
            self.log("✅ No cleanup targets found - workspace is already clean!", force=True)
            return
        
        # Calculate total size
        total_size = sum(size for _, size in targets)
        self.log(f"📊 Found {len(targets)} items totaling {self.format_size(total_size)}", force=True)
        
        # Show what will be cleaned
        if self.verbose or self.dry_run:
            self.log("\n📋 Cleanup targets:")
            for path, size in targets[:10]:  # Show top 10
                self.log(f"   {path:<40} ({self.format_size(size)})")
            if len(targets) > 10:
                self.log(f"   ... and {len(targets) - 10} more items")
        
        # Confirm in interactive mode (not SLURM)
        if not self.dry_run and not os.getenv('SLURM_JOB_ID') and not self.quiet:
            response = input(f"\n⚠️  Delete {len(targets)} items ({self.format_size(total_size)})? [y/N]: ")
            if response.lower() != 'y':
                self.log("❌ Cleanup cancelled by user", force=True)
                return
        
        # Perform cleanup
        self.log(f"\n🗑️  {'Simulating' if self.dry_run else 'Performing'} cleanup...")
        success_count = 0
        
        for path, size in targets:
            if self.remove_item(path):
                success_count += 1
                self.total_size += size
                self.deleted_items.append(path)
        
        # Report results
        self.log("=" * 60, force=True)
        self.log("🎉 CLEANUP COMPLETED", force=True)
        self.log("=" * 60, force=True)
        self.log(f"✅ Successfully {'would remove' if self.dry_run else 'removed'}: {success_count}/{len(targets)} items", force=True)
        self.log(f"💾 Space {'would be' if self.dry_run else ''} freed: {self.format_size(self.total_size)}", force=True)
        
        if success_count < len(targets):
            failed_count = len(targets) - success_count
            self.log(f"⚠️  Failed to remove: {failed_count} items", force=True)
        
        elapsed = time.time() - self.start_time
        self.log(f"⏱️  Cleanup completed in {elapsed:.1f} seconds", force=True)
        
        if not self.dry_run:
            self.log(f"📄 Detailed log saved to: {self.log_file}", force=True)
        
        # Show protected directories that were preserved
        if self.verbose:
            self.log("\n🛡️  Protected directories preserved:")
            for protected in self.PROTECTED_DIRS:
                if os.path.exists(protected.rstrip('/')):
                    self.log(f"   ✅ {protected}")


def main():
    parser = argparse.ArgumentParser(
        description="Clean workspace by removing temporary files and cache directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/cleanup_workspace.py --dry-run     # Preview what will be deleted
  python scripts/cleanup_workspace.py --verbose     # Normal cleanup with details  
  python scripts/cleanup_workspace.py --keep-wandb  # Keep wandb logs
  python scripts/cleanup_workspace.py --quiet       # Silent cleanup

Protected directories (never deleted):
  .claude/, configs/, slurm/, slurm_logs/, models/, datasets/, figures/,
  scripts/, core/, environments/, training/, evaluation/, etc.
        """
    )
    
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview what would be deleted without actually deleting')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed progress information')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimize output (only show final summary)')
    parser.add_argument('--keep-wandb', action='store_true',
                       help='Preserve wandb experiment logs')
    parser.add_argument('--keep-models-temp', action='store_true',
                       help='Preserve temporary model directories')
    
    args = parser.parse_args()
    
    # Create cleanup instance
    cleaner = WorkspaceCleanup(
        dry_run=args.dry_run,
        verbose=args.verbose,
        quiet=args.quiet
    )
    
    # Perform cleanup
    cleaner.cleanup(
        keep_wandb=args.keep_wandb,
        keep_models_temp=args.keep_models_temp
    )


if __name__ == "__main__":
    main()