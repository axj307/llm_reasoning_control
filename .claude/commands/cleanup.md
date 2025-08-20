# Project Cleanup Manager

You are tasked with performing intelligent cleanup of cache files, temporary directories, and unnecessary files from the Universal Control LLM Framework project.

## Instructions

1. **Use the cleanup-manager agent** for all cleanup operations to ensure safe, intelligent file removal

2. **Cleanup Operations Available**:
   - **Full Cleanup**: Remove cache, temporary files, old logs, and SLURM outputs
   - **Cache Only**: Remove only Python cache (`__pycache__`, `*.pyc`) and ML framework cache
   - **Temp Only**: Remove temporary files, SLURM outputs, and old logs
   - **Dry Run**: Analyze and report what would be cleaned without actually deleting

3. **Always Protected Files**:
   - Trained models in `models/*/latest/`
   - Final results in `figures/` and `results/`
   - Source code and configuration files
   - Current datasets and documentation
   - Git repository and version control files

4. **Smart Features**:
   - Size analysis and reporting (before/after disk usage)
   - Confirmation prompts for large deletions
   - Backup manifests for recovery if needed
   - Respect .gitignore patterns and git-tracked files

5. **Cleanup Targets**:
   - `__pycache__/` directories and Python bytecode files
   - `slurm_output_*/` temporary SLURM job directories
   - `wandb/`, `tensorboard_logs/`, `runs/` ML framework cache
   - `.nv/` GPU cache directories
   - `*.tmp`, `*.temp`, `*.log` files (especially old ones)
   - `.ipynb_checkpoints/` Jupyter cache
   - Old model checkpoints (keeping latest versions)

## Usage Examples

```bash
# Full cleanup with confirmation
/cleanup

# See what would be cleaned without deleting
/cleanup --dry-run

# Remove only Python and ML cache
/cleanup --cache-only

# Remove only temporary files and directories
/cleanup --temp-only

# Verbose cleanup with detailed reporting
/cleanup --verbose

# Clean old files (logs, checkpoints older than 7 days)
/cleanup --old-files
```

## Safety Guidelines

- **Always run dry-run first** for unfamiliar projects or large cleanups
- **Confirm model preservation**: Verify `models/*/latest/` directories are protected
- **Check for running processes**: Don't clean files that might be in use
- **Size reporting**: Show before/after disk usage for transparency
- **Recovery information**: Provide guidance on restoring accidentally deleted files

## Integration Notes

This cleanup system integrates with:
- Your existing `.gitignore` patterns
- SLURM job management (clean outputs after completion)  
- Model training pipelines (preserve important checkpoints)
- Development workflow (maintain clean working directory)

The cleanup-manager agent will handle all technical details while ensuring your valuable training results and models remain safe.