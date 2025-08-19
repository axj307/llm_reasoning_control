# Workspace Cleanup Plan

## SLURM Scripts Cleanup

### Keep (4 essential scripts)
```bash
# Essential working scripts
slurm/complete_di_pipeline.sbatch       # Main DI pipeline ⭐
slurm/standard_pipeline.sbatch          # DI + VDP pipeline ⭐  
slurm/quick_test_di.sbatch              # Quick testing ⭐
slurm/train_working_notebook_slurm.sbatch # Working baseline ⭐
```

### Delete (20+ redundant scripts)
```bash
# Remove these redundant/experimental scripts
rm slurm/compare_sft_grpo.sbatch
rm slurm/complete_di_sft_grpo.sbatch
rm slurm/debug_llm_outputs.sbatch
rm slurm/di_pipeline.sbatch
rm slurm/evaluate_existing_model.sbatch
rm slurm/evaluate_grpo_working_params.sbatch
rm slurm/extend_sft_exact_notebook.sbatch
rm slurm/extend_sft_training.sbatch
rm slurm/quick_test_pipeline.sbatch
rm slurm/test_pipeline_verification.sbatch
rm slurm/train_and_evaluate_complete.sbatch
rm slurm/train_complete_pipeline.sbatch
rm slurm/train_diverse_sft.sbatch
rm slurm/train_diverse_sft_no_vllm.sbatch
rm slurm/train_evaluate_grpo.sbatch
rm slurm/train_evaluate_sft.sbatch
rm slurm/train_evaluate_universal.sbatch
rm slurm/train_grpo.sbatch
rm slurm/train_sft_only.sbatch
rm slurm/train_working_notebook_fixed.sbatch
```

## Datasets Cleanup

### Keep (6 essential files)
```bash
# Essential DI datasets
datasets/di_train.pkl                   # Main training data ⭐
datasets/di_train.json                  # JSON format ⭐
datasets/di_eval.pkl                    # Main evaluation data ⭐
datasets/di_eval.json                   # JSON format ⭐
datasets/di_info.pkl                    # Dataset metadata ⭐
datasets/di_info.json                   # JSON format ⭐
```

### Delete (50+ test/duplicate files)
```bash
# Remove all test and duplicate datasets
rm datasets/di_complete_*
rm datasets/di_eval_complete_*
rm datasets/di_quick_test_*
rm datasets/quick_test_*
rm datasets/test_baseline_*
rm datasets/double_integrator_45train_*
rm datasets/di_eval_quick_test_*
```

### Create (missing VDP datasets)
```bash
# Generate VDP datasets for universal training later
python scripts/generate_data.py --systems van_der_pol --total-samples 200 --dataset-name vdp
```

## Safe Cleanup Commands

### Step 1: Backup Current State
```bash
# Create backup before cleanup
tar -czf workspace_backup_$(date +%Y%m%d_%H%M%S).tar.gz slurm/ datasets/
```

### Step 2: Clean SLURM Scripts
```bash
# Keep only essential scripts
cd slurm/
mkdir ../slurm_backup
mv *.sbatch ../slurm_backup/
mv ../slurm_backup/complete_di_pipeline.sbatch .
mv ../slurm_backup/standard_pipeline.sbatch .
mv ../slurm_backup/train_working_notebook_slurm.sbatch .
# Check if quick_test_di.sbatch exists
if [ -f ../slurm_backup/quick_test_di.sbatch ]; then
    mv ../slurm_backup/quick_test_di.sbatch .
fi
```

### Step 3: Clean Datasets
```bash
# Keep only essential datasets
cd datasets/
mkdir ../datasets_backup
mv * ../datasets_backup/
mv ../datasets_backup/di_train.* .
mv ../datasets_backup/di_eval.* .
mv ../datasets_backup/di_info.* .
```

### Step 4: Verify Cleanup
```bash
# Check what's left
echo "SLURM scripts remaining:"
ls -la slurm/
echo "Datasets remaining:"
ls -la datasets/
```

## Before/After Comparison

### Before Cleanup
- **SLURM scripts**: 23 files (mostly redundant)
- **Datasets**: 54 files (many duplicates/tests)
- **Disk usage**: ~500MB+ of redundant data

### After Cleanup  
- **SLURM scripts**: 3-4 essential files
- **Datasets**: 6 essential files
- **Disk usage**: ~50MB of essential data

## Benefits
1. **Clear workspace** - Easy to find what you need
2. **Reduced confusion** - No duplicate/experimental files
3. **Faster operations** - Less files to scan
4. **Better organization** - Only working, tested scripts
5. **Easier maintenance** - Fewer files to manage

## Recovery Plan
If you need any deleted files:
1. **Backup available** - All files backed up before deletion
2. **Git history** - Previous commits have the complex scripts
3. **Documentation** - IMPLEMENTATION_GUIDE.md has all features documented

## Next Steps After Cleanup
1. Test that essential pipeline still works
2. Generate VDP datasets for universal training
3. Add essential features (cache cleanup, docs) to clean workspace
4. Proceed with actual research/training work