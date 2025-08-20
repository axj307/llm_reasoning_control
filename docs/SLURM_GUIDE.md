# 🎯 SLURM Job Submission Guide

## What Each Script Does

Each SLURM script does **BOTH training AND evaluation** with automatic plot generation.

### 1. `train_evaluate_sft.sbatch` - SFT Only
- ✅ Trains SFT model on your chosen system
- ✅ Evaluates the trained model  
- ✅ Generates plots and saves them
- ✅ Creates summary report

### 2. `train_evaluate_grpo.sbatch` - SFT + GRPO
- ✅ Trains SFT model first
- ✅ Trains GRPO model using the SFT model
- ✅ Evaluates BOTH models separately
- ✅ Generates plots for both models
- ✅ Creates comprehensive summary report

### 3. `train_evaluate_universal.sbatch` - Universal Model
- ✅ Trains universal model on multiple systems
- ✅ Evaluates on all systems
- ✅ Generates plots for all systems

## 🚀 Simple Commands (What You Asked For)

### Case 1: Double Integrator with SFT Only
```bash
sbatch slurm/train_evaluate_sft.sbatch
```
**What happens:** Trains SFT model on double_integrator, evaluates it, saves everything to `figures/job_XXXXX/`

### Case 2: Double Integrator with SFT + GRPO
```bash
sbatch slurm/train_evaluate_grpo.sbatch
```
**What happens:** Trains both SFT and GRPO models, evaluates both, saves plots for both models

### Case 3: Van der Pol with SFT Only
```bash
sbatch --export=ENVIRONMENT=van_der_pol slurm/train_evaluate_sft.sbatch
```

### Case 4: Van der Pol with SFT + GRPO
```bash
sbatch --export=ENVIRONMENT=van_der_pol slurm/train_evaluate_grpo.sbatch
```

### Case 5: Universal Model (Both Systems)
```bash
sbatch slurm/train_evaluate_universal.sbatch
```

## 📊 What You Get After Each Job

Each job automatically creates:

### Files Created:
```
figures/job_XXXXX/         # All plots, summaries, and results
models/single_system/      # Trained models
logs/                      # SLURM output logs
```

### For SFT Jobs:
- `trajectory_plot.png` - Trajectory visualization
- `phase_portrait.png` - Phase space plot
- `control_input.png` - Control signals
- Training summary with metrics

### For GRPO Jobs:
- `sft_trajectory_plot.png` - SFT model plots
- `grpo_trajectory_plot.png` - GRPO model plots  
- Plus all the individual plots for each model
- Comparison summary

## 🎛️ Advanced Options (Optional)

If you want to customize:

```bash
# Custom LoRA rank
sbatch --export=ENVIRONMENT=double_integrator,LORA_RANK=16 slurm/train_evaluate_sft.sbatch

# More test cases for evaluation
sbatch --export=ENVIRONMENT=double_integrator,NUM_TEST_CASES=50 slurm/train_evaluate_sft.sbatch
```

## 📋 Quick Reference

| What You Want | Command |
|---------------|---------|
| DI + SFT | `sbatch slurm/train_evaluate_sft.sbatch` |
| DI + SFT+GRPO | `sbatch slurm/train_evaluate_grpo.sbatch` |
| VDP + SFT | `sbatch --export=ENVIRONMENT=van_der_pol slurm/train_evaluate_sft.sbatch` |
| VDP + SFT+GRPO | `sbatch --export=ENVIRONMENT=van_der_pol slurm/train_evaluate_grpo.sbatch` |
| Universal | `sbatch slurm/train_evaluate_universal.sbatch` |

## 🔍 Checking Job Status

```bash
squeue -u $USER              # Check your running jobs
cat logs/sft_eval_XXXXX.out  # Check job output  
cat logs/sft_eval_XXXXX.err  # Check job errors
```

## ❓ Do You Need the /slurm folder?

**YES, keep it!** The `/slurm` folder contains all the batch scripts that do the training and evaluation. Without it, you can't submit jobs to the cluster.

The folder structure is:
```
slurm/
├── train_evaluate_sft.sbatch      # SFT training + evaluation
├── train_evaluate_grpo.sbatch     # SFT+GRPO training + evaluation  
├── train_evaluate_universal.sbatch # Universal model
└── train_sft.sbatch               # SFT only (no evaluation)
```

## 💡 Pro Tips

1. **Always check logs first** if a job fails: `cat logs/sft_eval_XXXXX.err`
2. **Your plots are automatically saved** - no extra steps needed
3. **Jobs run completely automatically** - training → evaluation → plots → summary
4. **Models are saved permanently** in `models/` directory with timestamps