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
    # Handle both direct wandb directory and parent directory containing wandb
    if Path(log_dir).name == "wandb":
        wandb_dir = Path(log_dir)
    else:
        wandb_dir = Path(log_dir) / "wandb"
    
    if not wandb_dir.exists():
        print(f"❌ No wandb directory found at {wandb_dir}")
        return {}
    
    # Look for run directories
    run_dirs = [d for d in wandb_dir.iterdir() if d.is_dir() and d.name.startswith("run-")]
    if not run_dirs:
        print(f"❌ No wandb run directories found in {wandb_dir}")
        return {}
    
    # Use the most recent run
    latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
    print(f"📊 Using wandb run: {latest_run.name}")
    
    # Try multiple sources for training metrics
    # 1. Try wandb-history.jsonl (standard wandb format)
    history_file = latest_run / "files" / "wandb-history.jsonl"
    if history_file.exists():
        print(f"📊 Found wandb history file")
        return parse_wandb_history(history_file)
    
    # 2. Try wandb-summary.json (contains final metrics)
    summary_file = latest_run / "files" / "wandb-summary.json"
    if summary_file.exists():
        print(f"📊 Found wandb summary file, extracting available metrics")
        return parse_wandb_summary(summary_file)
    
    # 3. Try output.log (console output from wandb)
    output_log = latest_run / "files" / "output.log"
    if output_log.exists():
        print(f"📊 Found wandb output log, parsing training metrics")
        return parse_console_logs(str(output_log))
    
    print(f"❌ No parseable wandb files found in {latest_run}")
    print(f"💡 Available files: {[f.name for f in (latest_run / 'files').iterdir() if f.is_file()]}")
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

def parse_wandb_summary(summary_file: Path) -> Dict:
    """Parse wandb summary JSON file for final metrics."""
    try:
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        # Create basic metrics structure with available final values
        metrics = {
            'step': [1],  # Single step for final metrics
            'train_loss': [],
            'eval_loss': [],
            'learning_rate': [],
            'train_reward': [],
            'grad_norm': [],
            'global_step': [],
            'epoch': [],
            'runtime': [],
            'samples_per_second': [],
            'steps_per_second': [],
            'final_metrics': True  # Flag to indicate these are final metrics
        }
        
        # Extract available metrics from summary
        metric_mappings = {
            'train_loss': ['train/loss', 'train_loss', 'loss'],
            'eval_loss': ['eval/loss', 'eval_loss', 'validation_loss'],
            'learning_rate': ['train/learning_rate', 'learning_rate', 'lr'],
            'train_reward': ['train/reward', 'reward', 'train_reward'],
            'grad_norm': ['train/grad_norm', 'grad_norm', 'gradient_norm'],
            'global_step': ['train/global_step', 'global_step', 'step'],
            'epoch': ['train/epoch', 'epoch'],
            'runtime': ['train_runtime', 'runtime'],
            'samples_per_second': ['train_samples_per_second', 'samples_per_second'],
            'steps_per_second': ['train_steps_per_second', 'steps_per_second']
        }
        
        for metric_key, possible_keys in metric_mappings.items():
            for key in possible_keys:
                if key in summary and summary[key] is not None:
                    metrics[metric_key] = [summary[key]]
                    print(f"📊 Found {metric_key}: {summary[key]}")
                    break
            else:
                metrics[metric_key] = [None]
        
        print(f"✅ Parsed final metrics from wandb summary")
        return metrics
        
    except Exception as e:
        print(f"❌ Error parsing wandb summary: {e}")
        return {}

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
        'grad_norm': [],
        'train_reward': []
    }
    
    try:
        with open(log_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                step = None
                # Look for training metrics in dictionary format like {'loss': 0.0141, 'grad_norm': ...}
                if "{'loss':" in line or '{"loss":' in line:
                    # This is a training metrics line
                    step = len(metrics['step']) + 1  # Increment step counter
                
                if step is not None:
                    metrics['step'].append(step)
                    
                    # Extract metrics from dictionary format
                    # Look for 'loss': value
                    loss_match = re.search(r"'loss':\s*([\d\.e-]+)", line)
                    if loss_match:
                        metrics['train_loss'].append(float(loss_match.group(1)))
                    else:
                        metrics['train_loss'].append(None)
                    
                    # Look for 'learning_rate': value  
                    lr_match = re.search(r"'learning_rate':\s*([\d\.e-]+)", line)
                    if lr_match:
                        metrics['learning_rate'].append(float(lr_match.group(1)))
                    else:
                        metrics['learning_rate'].append(None)
                    
                    # Look for 'grad_norm': value
                    grad_match = re.search(r"'grad_norm':\s*([\d\.e-]+)", line)
                    if grad_match:
                        metrics['grad_norm'].append(float(grad_match.group(1)))
                    else:
                        metrics['grad_norm'].append(None)
                        
                    # Look for 'reward': value (GRPO)
                    reward_match = re.search(r"'reward':\s*([\d\.e-]+)", line)
                    if reward_match:
                        metrics['train_reward'].append(float(reward_match.group(1)))
                    else:
                        metrics['train_reward'].append(None)
                        
                    # Look for 'epoch': value
                    epoch_match = re.search(r"'epoch':\s*([\d\.e-]+)", line)
                    if epoch_match:
                        metrics['epoch'].append(float(epoch_match.group(1)))
                    else:
                        metrics['epoch'].append(None)
    
    except Exception as e:
        print(f"❌ Error parsing console logs: {e}")
        return {}
    
    # Clean up metrics - remove entries with no useful data
    if metrics['step']:
        print(f"✅ Parsed {len(metrics['step'])} steps from console logs")
        return metrics
    else:
        print(f"❌ No training steps found in console logs")
        return {}

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
    
    # Check if we have final metrics only (single data point)
    is_final_metrics = metrics.get('final_metrics', False) or len(steps) == 1
    
    if is_final_metrics:
        print("📊 Creating final metrics summary (single data point from wandb summary)")
        plot_final_metrics_summary(metrics, save_path, training_type)
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

def plot_final_metrics_summary(metrics: Dict, save_path: str, training_type: str):
    """Create a summary visualization for final training metrics."""
    set_professional_style()
    
    # Create a summary card-style visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.axis('off')  # Hide axes for text-based summary
    
    # Extract available metrics
    available_metrics = []
    
    metric_display = {
        'train_loss': ('Training Loss', '•'),
        'learning_rate': ('Learning Rate', '•'),
        'grad_norm': ('Gradient Norm', '•'),
        'global_step': ('Global Steps', '•'),
        'epoch': ('Epochs', '•'),
        'runtime': ('Training Time (s)', '•'),
        'samples_per_second': ('Samples/Second', '•'),
        'steps_per_second': ('Steps/Second', '•'),
        'train_reward': ('Training Reward', '•'),
        'eval_loss': ('Validation Loss', '•'),
    }
    
    y_pos = 0.9
    x_left = 0.1
    x_right = 0.6
    
    # Title
    ax.text(0.5, 0.95, f'{training_type} Training Summary', 
            fontsize=24, fontweight='bold', ha='center', transform=ax.transAxes)
    
    # Add metrics in two columns
    col1_metrics = []
    col2_metrics = []
    
    for metric_key, (display_name, emoji) in metric_display.items():
        if metric_key in metrics and metrics[metric_key] and metrics[metric_key][0] is not None:
            value = metrics[metric_key][0]
            
            # Format the value appropriately
            if metric_key in ['learning_rate']:
                formatted_value = f"{value:.2e}"
            elif metric_key in ['train_loss', 'grad_norm', 'train_reward', 'eval_loss']:
                formatted_value = f"{value:.4f}"
            elif metric_key in ['runtime']:
                formatted_value = f"{value:.1f}"
            elif metric_key in ['samples_per_second', 'steps_per_second']:
                formatted_value = f"{value:.2f}"
            else:
                formatted_value = f"{value}"
            
            metric_text = f"{emoji} {display_name}: {formatted_value}"
            
            if len(col1_metrics) <= len(col2_metrics):
                col1_metrics.append(metric_text)
            else:
                col2_metrics.append(metric_text)
    
    # Display first column
    for i, metric_text in enumerate(col1_metrics):
        ax.text(x_left, y_pos - i*0.08, metric_text, 
                fontsize=16, transform=ax.transAxes, fontweight='bold')
    
    # Display second column
    for i, metric_text in enumerate(col2_metrics):
        ax.text(x_right, y_pos - i*0.08, metric_text, 
                fontsize=16, transform=ax.transAxes, fontweight='bold')
    
    # Add footer info
    footer_y = 0.15
    ax.text(0.5, footer_y, 'Final Training Metrics from Wandb Summary', 
            fontsize=12, ha='center', transform=ax.transAxes, style='italic', alpha=0.7)
    ax.text(0.5, footer_y - 0.05, 'For detailed progress curves, enable wandb history logging', 
            fontsize=10, ha='center', transform=ax.transAxes, alpha=0.6)
    
    # Add a subtle border
    from matplotlib.patches import Rectangle
    border = Rectangle((0.05, 0.05), 0.9, 0.9, linewidth=2, 
                      edgecolor='#2E86AB', facecolor='none', transform=ax.transAxes)
    ax.add_patch(border)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✅ Training metrics summary saved: {save_path}")
    
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

def plot_parameter_evolution(model_checkpoints: List[str], save_path: str, metrics: Dict = None):
    """Plot evolution of specific model parameters during training."""
    # Skip generating fake parameter plots - only create if we have real model checkpoints
    if not model_checkpoints:
        print("ℹ️  Skipping parameter evolution plot - no model checkpoints provided")
        print("💡 To enable parameter evolution plots, provide model checkpoint paths")
        return
    
    set_professional_style()
    
    # TODO: Implement actual parameter extraction from model checkpoints
    # This would require loading model checkpoints and extracting specific parameters
    # For now, we skip this entirely to avoid fake data
    
    print(f"⚠️  Parameter evolution plotting not yet implemented for model checkpoints")
    print(f"📍 Model checkpoints provided: {len(model_checkpoints)}")
    return

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
        print("💡 Common solutions:")
        print("  1. Check if wandb logging was enabled during training")
        print("  2. Look for console/SLURM output logs with training metrics")
        print("  3. Verify the log directory path is correct")
        print("💡 Usage examples:")
        print("  python plot_training_progress.py --log-dir .")
        print("  python plot_training_progress.py --wandb-dir wandb")
        print("  python plot_training_progress.py --console-log slurm_logs/job_123.out")
        
        # Helpful diagnostic info
        print("\n🔍 Diagnostic information:")
        if args.log_dir:
            log_path = Path(args.log_dir)
            if log_path.exists():
                print(f"  Directory exists: {log_path}")
                if (log_path / "wandb").exists():
                    print(f"  Found wandb directory with {len(list((log_path / 'wandb').iterdir()))} items")
                else:
                    print(f"  No wandb directory found in {log_path}")
            else:
                print(f"  Directory not found: {log_path}")
        
        if args.wandb_dir:
            wandb_path = Path(args.wandb_dir)
            if wandb_path.exists():
                print(f"  Wandb directory exists: {wandb_path}")
                print(f"  Contents: {[item.name for item in wandb_path.iterdir()]}")
            else:
                print(f"  Wandb directory not found: {wandb_path}")
        
        return
    
    # Create training progress plot
    save_path = os.path.join(args.save_dir, f"{args.training_type.lower()}_progress.png")
    plot_training_progress(metrics, save_path, args.training_type)
    
    # Create parameter evolution plot only if we have model checkpoints
    # Skip placeholder/fake parameter plots
    print(f"ℹ️  Parameter evolution plots skipped (no model checkpoints provided)")
    print(f"💡 Only generating actual training progress plots from real data")
    
    print(f"🎉 Training visualization completed!")

if __name__ == "__main__":
    main()
