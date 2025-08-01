"""
Comprehensive plotting utilities for control system evaluation.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import seaborn as sns

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def create_comprehensive_plot(
    times: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    controls: np.ndarray,
    positions_opt: Optional[np.ndarray] = None,
    velocities_opt: Optional[np.ndarray] = None,
    controls_opt: Optional[np.ndarray] = None,
    title: str = "Control System Trajectory Analysis",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12)
) -> None:
    """
    Create a comprehensive multi-panel plot showing:
    - Phase space trajectory
    - Position over time
    - Velocity over time
    - Control inputs over time
    
    Args:
        times: Time points
        positions: Position trajectory
        velocities: Velocity trajectory
        controls: Control inputs
        positions_opt: Optional optimal position trajectory
        velocities_opt: Optional optimal velocity trajectory
        controls_opt: Optional optimal control inputs
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size
    """
    # Create figure with custom layout
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 2, height_ratios=[1.2, 1, 1], width_ratios=[1, 1])
    
    # Phase space plot (larger, spans two columns)
    ax_phase = fig.add_subplot(gs[0, :])
    
    # Position plot
    ax_pos = fig.add_subplot(gs[1, 0])
    
    # Velocity plot
    ax_vel = fig.add_subplot(gs[1, 1])
    
    # Control plot (spans two columns)
    ax_control = fig.add_subplot(gs[2, :])
    
    # 1. Phase Space Plot
    ax_phase.plot(positions, velocities, 'b-', linewidth=2.5, label='Predicted', marker='o', 
                  markersize=4, markevery=max(1, len(positions)//20))
    if positions_opt is not None and velocities_opt is not None:
        ax_phase.plot(positions_opt, velocities_opt, 'g--', linewidth=2, label='Optimal', 
                      alpha=0.7, marker='s', markersize=3, markevery=max(1, len(positions_opt)//20))
    
    # Mark start and end points
    ax_phase.plot(positions[0], velocities[0], 'ro', markersize=12, label='Start', zorder=5)
    ax_phase.plot(positions[-1], velocities[-1], 'r^', markersize=12, label='Target', zorder=5)
    ax_phase.plot(0, 0, 'k*', markersize=15, label='Origin', zorder=5)
    
    # Add trajectory direction arrows
    n_arrows = min(5, len(positions) - 1)
    arrow_indices = np.linspace(0, len(positions)-2, n_arrows, dtype=int)
    for i in arrow_indices:
        dx = positions[i+1] - positions[i]
        dy = velocities[i+1] - velocities[i]
        ax_phase.annotate('', xy=(positions[i+1], velocities[i+1]), 
                         xytext=(positions[i], velocities[i]),
                         arrowprops=dict(arrowstyle='->', color='blue', alpha=0.5))
    
    ax_phase.set_xlabel('Position', fontsize=12)
    ax_phase.set_ylabel('Velocity', fontsize=12)
    ax_phase.set_title('Phase Space Trajectory', fontsize=14, fontweight='bold')
    ax_phase.grid(True, alpha=0.3)
    ax_phase.legend(loc='best')
    ax_phase.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax_phase.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # 2. Position over Time
    ax_pos.plot(times, positions, 'b-', linewidth=2.5, label='Predicted', marker='o', 
                markersize=4, markevery=max(1, len(times)//20))
    if positions_opt is not None:
        ax_pos.plot(times, positions_opt, 'g--', linewidth=2, label='Optimal', alpha=0.7)
    
    ax_pos.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Target')
    ax_pos.fill_between(times, -1, 1, alpha=0.1, color='gray', label='Bounds')
    ax_pos.set_xlabel('Time (s)', fontsize=12)
    ax_pos.set_ylabel('Position', fontsize=12)
    ax_pos.set_title('Position Trajectory', fontsize=14, fontweight='bold')
    ax_pos.grid(True, alpha=0.3)
    ax_pos.legend(loc='best')
    ax_pos.set_ylim([-1.2, 1.2])
    
    # 3. Velocity over Time
    ax_vel.plot(times, velocities, 'b-', linewidth=2.5, label='Predicted', marker='o', 
                markersize=4, markevery=max(1, len(times)//20))
    if velocities_opt is not None:
        ax_vel.plot(times, velocities_opt, 'g--', linewidth=2, label='Optimal', alpha=0.7)
    
    ax_vel.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Target')
    ax_vel.fill_between(times, -1, 1, alpha=0.1, color='gray', label='Bounds')
    ax_vel.set_xlabel('Time (s)', fontsize=12)
    ax_vel.set_ylabel('Velocity', fontsize=12)
    ax_vel.set_title('Velocity Trajectory', fontsize=14, fontweight='bold')
    ax_vel.grid(True, alpha=0.3)
    ax_vel.legend(loc='best')
    ax_vel.set_ylim([-1.2, 1.2])
    
    # 4. Control Inputs over Time
    control_times = times[:-1]  # Control is applied between time steps
    ax_control.step(control_times, controls, 'b-', linewidth=2.5, where='post', 
                    label='Predicted', marker='o', markersize=4, markevery=max(1, len(control_times)//20))
    if controls_opt is not None:
        ax_control.step(control_times, controls_opt, 'g--', linewidth=2, where='post', 
                       label='Optimal', alpha=0.7)
    
    ax_control.fill_between(control_times, -3, 3, alpha=0.1, color='gray', label='Bounds', step='post')
    ax_control.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax_control.set_xlabel('Time (s)', fontsize=12)
    ax_control.set_ylabel('Control Input', fontsize=12)
    ax_control.set_title('Control Sequence', fontsize=14, fontweight='bold')
    ax_control.grid(True, alpha=0.3)
    ax_control.legend(loc='best')
    ax_control.set_ylim([-3.5, 3.5])
    
    # Overall title
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_batch_comparison_plot(
    results: List[Dict[str, Any]],
    title: str = "Batch Evaluation Results",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 10)
) -> None:
    """
    Create a comparison plot for multiple trajectories.
    
    Args:
        results: List of evaluation results
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size
    """
    n_results = len(results)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Define color palette
    colors = plt.cm.viridis(np.linspace(0, 1, n_results))
    
    # 1. Phase Space Comparison
    ax = axes[0]
    for i, result in enumerate(results):
        positions = result['positions']
        velocities = result['velocities']
        label = result.get('label', f'Test {i+1}')
        
        ax.plot(positions, velocities, color=colors[i], linewidth=2, 
                label=label, alpha=0.7, marker='o', markersize=3, 
                markevery=max(1, len(positions)//10))
    
    ax.plot(0, 0, 'k*', markersize=15, label='Origin', zorder=5)
    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('Velocity', fontsize=12)
    ax.set_title('Phase Space Trajectories', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # 2. Control Effort Comparison
    ax = axes[1]
    control_efforts = []
    labels = []
    for i, result in enumerate(results):
        controls = result['controls']
        effort = np.sum(np.square(controls))
        control_efforts.append(effort)
        labels.append(result.get('label', f'Test {i+1}'))
    
    bars = ax.bar(range(n_results), control_efforts, color=colors)
    ax.set_xlabel('Test Case', fontsize=12)
    ax.set_ylabel('Total Control Effort (∑u²)', fontsize=12)
    ax.set_title('Control Effort Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(range(n_results))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, effort in zip(bars, control_efforts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{effort:.2f}', ha='center', va='bottom')
    
    # 3. Final Error Comparison
    ax = axes[2]
    final_errors = []
    for i, result in enumerate(results):
        positions = result['positions']
        velocities = result['velocities']
        final_error = np.sqrt(positions[-1]**2 + velocities[-1]**2)
        final_errors.append(final_error)
    
    bars = ax.bar(range(n_results), final_errors, color=colors)
    ax.set_xlabel('Test Case', fontsize=12)
    ax.set_ylabel('Final Error (||[x,v]||₂)', fontsize=12)
    ax.set_title('Final State Error', fontsize=14, fontweight='bold')
    ax.set_xticks(range(n_results))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, error in zip(bars, final_errors):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{error:.4f}', ha='center', va='bottom')
    
    # 4. Convergence Time Comparison
    ax = axes[3]
    convergence_times = []
    threshold = 0.01  # Convergence threshold
    
    for i, result in enumerate(results):
        positions = result['positions']
        velocities = result['velocities']
        times = result['times']
        
        # Find convergence time
        converged = False
        conv_time = times[-1]
        for j, (p, v, t) in enumerate(zip(positions, velocities, times)):
            if np.sqrt(p**2 + v**2) < threshold:
                conv_time = t
                converged = True
                break
        
        convergence_times.append(conv_time)
    
    bars = ax.bar(range(n_results), convergence_times, color=colors)
    ax.set_xlabel('Test Case', fontsize=12)
    ax.set_ylabel('Convergence Time (s)', fontsize=12)
    ax.set_title(f'Time to Reach Target (ε={threshold})', fontsize=14, fontweight='bold')
    ax.set_xticks(range(n_results))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, conv_time in zip(bars, convergence_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{conv_time:.2f}', ha='center', va='bottom')
    
    # Overall title
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_control_heatmap(
    control_sequences: List[np.ndarray],
    labels: Optional[List[str]] = None,
    title: str = "Control Sequence Heatmap",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Create a heatmap visualization of multiple control sequences.
    
    Args:
        control_sequences: List of control sequences
        labels: Optional labels for each sequence
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size
    """
    # Convert to 2D array
    max_len = max(len(seq) for seq in control_sequences)
    heatmap_data = np.full((len(control_sequences), max_len), np.nan)
    
    for i, seq in enumerate(control_sequences):
        heatmap_data[i, :len(seq)] = seq
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(heatmap_data, cmap='RdBu_r', aspect='auto', vmin=-3, vmax=3)
    
    # Set labels
    if labels is None:
        labels = [f'Test {i+1}' for i in range(len(control_sequences))]
    
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Test Case', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Control Input', fontsize=12)
    
    # Add grid
    ax.set_xticks(np.arange(max_len + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(control_sequences) + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Save or show
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()