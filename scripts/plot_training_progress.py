#!/usr/bin/env python3
"""
Training progress visualization for SFT and GRPO training runs.
Monitors key parameters and metrics during training.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

def set_professional_style():
    """Set consistent professional styling for training plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 20,
        'axes.labelsize': 24,
        'axes.titlesize': 26,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'xtick.major.size': 8,
        'ytick.major.size': 8,
        'xtick.major.width': 2,
        'ytick.major.width': 2,
        'legend.fontsize': 18,
        'figure.titlesize': 28,
        'lines.linewidth': 3.0,
        'lines.markersize': 7,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.axisbelow': True
    })

def parse_wandb_logs(log_dir: str) -> Dict:
    """Parse wandb logs for training metrics."""
    wandb_dir = Path(log_dir) / "wandb"
    if not wandb_dir.exists():
        print(f"❌ No wandb directory found at {wandb_dir}")
        return {}
    
    # Look for run directories
    run_dirs = [d for d in wandb_dir.iterdir() if d.is_dir() and d.name.startswith("run-")]
    if not run_dirs:
        print(f"❌ No wandb run directories found")
        return {}
    
    # Use the most recent run
    latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
    print(f"📊 Using wandb run: {latest_run.name}")
    
    # Try to find the events file or history
    history_file = latest_run / "files" / "wandb-history.jsonl"
    if history_file.exists():
        return parse_wandb_history(history_file)
    
    print(f"❌ No wandb history file found in {latest_run}")
    return {}

def parse_wandb_history(history_file: Path) -> Dict:
    """Parse wandb history JSONL file."""
    metrics = {
        'step': [],
        'epoch': [],
        'train_loss': [],
        'eval_loss': [],
        'learning_rate': [],
        'grad_norm': [],
        'train_reward': [],
        'eval_reward': [],
        'kl_divergence': [],
        'policy_loss': [],
        'value_loss': []
    }
    
    try:
        with open(history_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    
                    # Extract common metrics
                    if '_step' in data:
                        metrics['step'].append(data['_step'])
                    
                    # Training metrics
                    for key in ['train/loss', 'train_loss', 'loss']:
                        if key in data and data[key] is not None:
                            metrics['train_loss'].append(data[key])
                            break
                    else:
                        if metrics['step'] and len(metrics['train_loss']) < len(metrics['step']):
                            metrics['train_loss'].append(None)
                    
                    # Evaluation metrics
                    for key in ['eval/loss', 'eval_loss', 'validation_loss']:
                        if key in data and data[key] is not None:
                            metrics['eval_loss'].append(data[key])
                            break
                    else:
                        if metrics['step'] and len(metrics['eval_loss']) < len(metrics['step']):
                            metrics['eval_loss'].append(None)
                    
                    # Learning rate
                    for key in ['train/learning_rate', 'learning_rate', 'lr']:
                        if key in data and data[key] is not None:
                            metrics['learning_rate'].append(data[key])
                            break
                    else:
                        if metrics['step'] and len(metrics['learning_rate']) < len(metrics['step']):
                            metrics['learning_rate'].append(None)
                    
                    # Gradient norm
                    for key in ['train/grad_norm', 'grad_norm', 'gradient_norm']:
                        if key in data and data[key] is not None:
                            metrics['grad_norm'].append(data[key])
                            break
                    else:
                        if metrics['step'] and len(metrics['grad_norm']) < len(metrics['step']):
                            metrics['grad_norm'].append(None)
                    
                    # GRPO specific metrics
                    for key in ['train/reward', 'reward', 'train_reward']:
                        if key in data and data[key] is not None:
                            metrics['train_reward'].append(data[key])
                            break
                    else:
                        if metrics['step'] and len(metrics['train_reward']) < len(metrics['step']):
                            metrics['train_reward'].append(None)
                    
                    for key in ['train/kl', 'kl_divergence', 'kl']:
                        if key in data and data[key] is not None:
                            metrics['kl_divergence'].append(data[key])
                            break
                    else:
                        if metrics['step'] and len(metrics['kl_divergence']) < len(metrics['step']):
                            metrics['kl_divergence'].append(None)
                    
                    for key in ['train/policy_loss', 'policy_loss']:
                        if key in data and data[key] is not None:
                            metrics['policy_loss'].append(data[key])
                            break
                    else:
                        if metrics['step'] and len(metrics['policy_loss']) < len(metrics['step']):
                            metrics['policy_loss'].append(None)
    
    except Exception as e:
        print(f"❌ Error parsing wandb history: {e}")
        return {}
    
    # Clean up metrics - remove None values and ensure equal lengths
    cleaned_metrics = {}
    max_len = len(metrics['step']) if metrics['step'] else 0
    
    for key, values in metrics.items():
        if values and any(v is not None for v in values):
            # Pad with None to match step length
            while len(values) < max_len:
                values.append(None)
            cleaned_metrics[key] = values[:max_len]
    
    print(f"✅ Parsed {len(cleaned_metrics['step']) if 'step' in cleaned_metrics else 0} training steps")
    return cleaned_metrics

def parse_console_logs(log_file: str) -> Dict:
    """Parse console output logs for training metrics."""
    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
        return {}
    
    metrics = {
        'step': [],
        'epoch': [],
        'train_loss': [],
        'eval_loss': [],
        'learning_rate': [],
        'grad_norm': []
    }
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                # Look for training step patterns
                step_match = re.search(r'Step (\d+)', line)
                loss_match = re.search(r'Loss: ([\d\.]+)', line)
                lr_match = re.search(r'LR: ([\d\.e-]+)', line)
                
                if step_match:
                    step = int(step_match.group(1))
                    metrics['step'].append(step)
                    
                    if loss_match:
                        metrics['train_loss'].append(float(loss_match.group(1)))
                    
                    if lr_match:
                        metrics['learning_rate'].append(float(lr_match.group(1)))
    
    except Exception as e:
        print(f"❌ Error parsing console logs: {e}")
        return {}
    
    print(f"✅ Parsed {len(metrics['step'])} steps from console logs")
    return metrics

def plot_training_progress(metrics: Dict, save_path: str, training_type: str = "Training"):
    """Create focused training progress visualization."""
    set_professional_style()
    
    if not metrics or 'step' not in metrics:
        print("❌ No valid metrics to plot")
        return
    
    steps = metrics['step']
    if not steps:
        print("❌ No training steps found")
        return
    
    # Determine if this is GRPO or SFT based on available metrics
    is_grpo = any(key in metrics for key in ['train_reward', 'kl_divergence', 'policy_loss'])
    
    if is_grpo:
        # GRPO: Focused 2x2 layout with essential metrics
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        plot_grpo_focused(metrics, axes, steps)
    else:
        # SFT: Standard 2x2 layout
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        plot_sft_focused(metrics, axes, steps)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✅ Training progress plot saved: {save_path}")
    
    plt.close()

def plot_grpo_focused(metrics: Dict, axes, steps: List):
    """Plot focused GRPO metrics in 2x2 layout."""
    # Color scheme for GRPO essentials
    colors = {
        'train_reward': '#43AA8B',    # Green - Primary objective
        'kl_divergence': '#F8961E',   # Orange - Stability measure
        'policy_loss': '#90E0EF',     # Light blue - Optimization progress
        'learning_rate': '#FCBF49',   # Yellow - Context
    }
    
    # Top-left: Training Reward (Most important)
    if 'train_reward' in metrics and any(x is not None for x in metrics['train_reward']):
        reward = [x for x in metrics['train_reward'] if x is not None]
        reward_steps = [steps[i] for i, x in enumerate(metrics['train_reward']) if x is not None]
        
        axes[0,0].plot(reward_steps, reward, color=colors['train_reward'], 
                      linewidth=3.0, marker='*', markersize=7, alpha=0.8)
        axes[0,0].set_xlabel('Training Step', fontweight='bold')
        axes[0,0].set_ylabel('Training Reward', fontweight='bold')
        axes[0,0].grid(True, alpha=0.3)
    
    # Top-right: KL Divergence (Stability)
    if 'kl_divergence' in metrics and any(x is not None for x in metrics['kl_divergence']):
        kl = [x for x in metrics['kl_divergence'] if x is not None]
        kl_steps = [steps[i] for i, x in enumerate(metrics['kl_divergence']) if x is not None]
        
        axes[0,1].plot(kl_steps, kl, color=colors['kl_divergence'], 
                      linewidth=3.0, marker='v', markersize=5, alpha=0.8)
        axes[0,1].set_xlabel('Training Step', fontweight='bold')
        axes[0,1].set_ylabel('KL Divergence', fontweight='bold')
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].set_yscale('log')
    
    # Bottom-left: Policy Loss (Optimization)
    if 'policy_loss' in metrics and any(x is not None for x in metrics['policy_loss']):
        policy_loss = [x for x in metrics['policy_loss'] if x is not None]
        policy_steps = [steps[i] for i, x in enumerate(metrics['policy_loss']) if x is not None]
        
        axes[1,0].plot(policy_steps, policy_loss, color=colors['policy_loss'], 
                      linewidth=3.0, marker='<', markersize=5, alpha=0.8)
        axes[1,0].set_xlabel('Training Step', fontweight='bold')
        axes[1,0].set_ylabel('Policy Loss', fontweight='bold')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].set_yscale('log')
    
    # Bottom-right: Learning Rate (Context)
    if 'learning_rate' in metrics and any(x is not None for x in metrics['learning_rate']):
        lr = [x for x in metrics['learning_rate'] if x is not None]
        lr_steps = [steps[i] for i, x in enumerate(metrics['learning_rate']) if x is not None]
        
        axes[1,1].plot(lr_steps, lr, color=colors['learning_rate'], 
                      linewidth=3.0, marker='^', markersize=5, alpha=0.8)
        axes[1,1].set_xlabel('Training Step', fontweight='bold')
        axes[1,1].set_ylabel('Learning Rate', fontweight='bold')
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].set_yscale('log')

def plot_sft_focused(metrics: Dict, axes, steps: List):
    """Plot focused SFT metrics in 2x2 layout."""
    # Color scheme for SFT essentials
    colors = {
        'train_loss': '#E63946',      # Red - Primary objective
        'eval_loss': '#F77F00',       # Orange - Validation
        'learning_rate': '#FCBF49',   # Yellow - Context
        'grad_norm': '#277DA1',       # Blue - Stability
    }
    
    # Top-left: Training Loss
    if 'train_loss' in metrics and any(x is not None for x in metrics['train_loss']):
        train_loss = [x for x in metrics['train_loss'] if x is not None]
        train_steps = [steps[i] for i, x in enumerate(metrics['train_loss']) if x is not None]
        
        axes[0,0].plot(train_steps, train_loss, color=colors['train_loss'], 
                      linewidth=3.0, marker='o', markersize=5, alpha=0.8)
        axes[0,0].set_xlabel('Training Step', fontweight='bold')
        axes[0,0].set_ylabel('Training Loss', fontweight='bold')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].set_yscale('log')
    
    # Top-right: Evaluation Loss
    if 'eval_loss' in metrics and any(x is not None for x in metrics['eval_loss']):
        eval_loss = [x for x in metrics['eval_loss'] if x is not None]
        eval_steps = [steps[i] for i, x in enumerate(metrics['eval_loss']) if x is not None]
        
        axes[0,1].plot(eval_steps, eval_loss, color=colors['eval_loss'], 
                      linewidth=3.0, marker='s', markersize=5, alpha=0.8)
        axes[0,1].set_xlabel('Training Step', fontweight='bold')
        axes[0,1].set_ylabel('Evaluation Loss', fontweight='bold')
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].set_yscale('log')
    
    # Bottom-left: Learning Rate
    if 'learning_rate' in metrics and any(x is not None for x in metrics['learning_rate']):
        lr = [x for x in metrics['learning_rate'] if x is not None]
        lr_steps = [steps[i] for i, x in enumerate(metrics['learning_rate']) if x is not None]
        
        axes[1,0].plot(lr_steps, lr, color=colors['learning_rate'], 
                      linewidth=3.0, marker='^', markersize=5, alpha=0.8)
        axes[1,0].set_xlabel('Training Step', fontweight='bold')
        axes[1,0].set_ylabel('Learning Rate', fontweight='bold')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].set_yscale('log')
    
    # Bottom-right: Gradient Norm
    if 'grad_norm' in metrics and any(x is not None for x in metrics['grad_norm']):
        grad_norm = [x for x in metrics['grad_norm'] if x is not None]
        grad_steps = [steps[i] for i, x in enumerate(metrics['grad_norm']) if x is not None]
        
        axes[1,1].plot(grad_steps, grad_norm, color=colors['grad_norm'], 
                      linewidth=3.0, marker='d', markersize=5, alpha=0.8)
        axes[1,1].set_xlabel('Training Step', fontweight='bold')
        axes[1,1].set_ylabel('Gradient Norm', fontweight='bold')
        axes[1,1].grid(True, alpha=0.3)

def plot_parameter_evolution(model_checkpoints: List[str], save_path: str):
    """Plot evolution of specific model parameters during training."""
    set_professional_style()
    
    # This would require loading model checkpoints and extracting specific parameters
    # For now, create a placeholder that shows the concept
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Simulate parameter evolution (replace with actual parameter extraction)
    steps = np.arange(0, 1000, 50)
    
    # Example: LoRA adapter weights
    lora_alpha = 64 + np.random.normal(0, 2, len(steps))
    lora_r = 32 + np.random.normal(0, 1, len(steps))
    
    # Example: Attention weights variance
    attn_weights_var = 0.1 * np.exp(-steps/500) + 0.01 + np.random.normal(0, 0.005, len(steps))
    
    # Example: Layer norm scaling
    layer_norm_scale = 1.0 + 0.1 * np.sin(steps/100) + np.random.normal(0, 0.02, len(steps))
    
    # Plot parameter evolution
    axes[0,0].plot(steps, lora_alpha, color='#E63946', linewidth=2.5, marker='o', markersize=4)
    axes[0,0].set_xlabel('Training Step', fontweight='bold')
    axes[0,0].set_ylabel('LoRA Alpha', fontweight='bold')
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].plot(steps, lora_r, color='#277DA1', linewidth=2.5, marker='s', markersize=4)
    axes[0,1].set_xlabel('Training Step', fontweight='bold')
    axes[0,1].set_ylabel('LoRA Rank', fontweight='bold')
    axes[0,1].grid(True, alpha=0.3)
    
    axes[1,0].plot(steps, attn_weights_var, color='#43AA8B', linewidth=2.5, marker='^', markersize=4)
    axes[1,0].set_xlabel('Training Step', fontweight='bold')
    axes[1,0].set_ylabel('Attention Weights Variance', fontweight='bold')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_yscale('log')
    
    axes[1,1].plot(steps, layer_norm_scale, color='#F77F00', linewidth=2.5, marker='d', markersize=4)
    axes[1,1].set_xlabel('Training Step', fontweight='bold')
    axes[1,1].set_ylabel('Layer Norm Scale', fontweight='bold')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✅ Parameter evolution plot saved: {save_path}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot training progress for SFT and GRPO runs")
    parser.add_argument("--log-dir", type=str, help="Directory containing training logs")
    parser.add_argument("--wandb-dir", type=str, help="Wandb directory path")
    parser.add_argument("--console-log", type=str, help="Console log file path")
    parser.add_argument("--save-dir", type=str, default="figures/training", help="Directory to save plots")
    parser.add_argument("--training-type", type=str, default="Training", help="Training type (SFT, GRPO, etc.)")
    
    args = parser.parse_args()
    
    print(f"🎯 Training Progress Visualization")
    print(f"📁 Save directory: {args.save_dir}")
    
    metrics = {}
    
    # Try to parse wandb logs first
    if args.wandb_dir or args.log_dir:
        log_dir = args.wandb_dir or args.log_dir
        wandb_metrics = parse_wandb_logs(log_dir)
        if wandb_metrics:
            metrics.update(wandb_metrics)
    
    # Try console logs as fallback
    if args.console_log and not metrics:
        console_metrics = parse_console_logs(args.console_log)
        if console_metrics:
            metrics.update(console_metrics)
    
    if not metrics:
        print("❌ No training metrics found. Please provide valid log directories or files.")
        print("💡 Usage examples:")
        print("  python plot_training_progress.py --log-dir ./wandb")
        print("  python plot_training_progress.py --console-log ./training.log")
        return
    
    # Create training progress plot
    save_path = os.path.join(args.save_dir, f"{args.training_type.lower()}_progress.png")
    plot_training_progress(metrics, save_path, args.training_type)
    
    # Create parameter evolution plot (placeholder)
    param_save_path = os.path.join(args.save_dir, f"{args.training_type.lower()}_parameters.png")
    plot_parameter_evolution([], param_save_path)
    
    print(f"🎉 Training visualization completed!")

if __name__ == "__main__":
    main()
