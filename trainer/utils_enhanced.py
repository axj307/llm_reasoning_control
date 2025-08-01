"""
Enhanced utility functions for comprehensive visualizations.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from typing import List, Tuple
from control import simulate_trajectory, solve_double_integrator


def visualize_all_trajectories(
    test_cases: List[Tuple[float, float, List[float]]],
    dt: float,
    save_path: str = "combined_evaluation.png",
    show_optimal: bool = True
):
    """
    Create a comprehensive single figure showing all trajectories.
    
    Args:
        test_cases: List of (x0, v0, controls) tuples
        dt: Time step
        save_path: Path to save the figure
        show_optimal: Whether to show optimal LQR solution
    """
    # Create figure with custom layout
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1], width_ratios=[1, 1])
    
    # Create subplots
    ax_phase = fig.add_subplot(gs[0, :])  # Phase space (spans both columns)
    ax_pos = fig.add_subplot(gs[1, 0])    # Position
    ax_vel = fig.add_subplot(gs[1, 1])    # Velocity
    
    # Color palette for different trajectories
    colors = plt.cm.viridis(np.linspace(0, 1, len(test_cases)))
    
    # Plot each trajectory
    for i, (x0, v0, controls) in enumerate(test_cases):
        # Simulate trajectory
        positions, velocities, times = simulate_trajectory(x0, v0, controls, dt)
        
        # Create label
        label = f"IC: [{x0:.2f}, {v0:.2f}]"
        
        # Phase space plot
        ax_phase.plot(positions, velocities, '-', color=colors[i], linewidth=2.5, 
                     label=label, marker='o', markersize=4, markevery=max(1, len(positions)//20))
        
        # Mark start point
        ax_phase.plot(positions[0], velocities[0], 'o', color=colors[i], 
                     markersize=12, markeredgecolor='black', markeredgewidth=2)
        
        # Mark end point
        ax_phase.plot(positions[-1], velocities[-1], '^', color=colors[i], 
                     markersize=12, markeredgecolor='black', markeredgewidth=2)
        
        # Position plot
        ax_pos.plot(times, positions, '-', color=colors[i], linewidth=2.5, 
                   label=label, marker='o', markersize=3, markevery=max(1, len(times)//20))
        
        # Velocity plot
        ax_vel.plot(times, velocities, '-', color=colors[i], linewidth=2.5, 
                   label=label, marker='o', markersize=3, markevery=max(1, len(times)//20))
        
        # Optionally show optimal trajectory
        if show_optimal:
            optimal_controls = solve_double_integrator(x0, v0, dt, len(controls))
            opt_positions, opt_velocities, _ = simulate_trajectory(x0, v0, optimal_controls, dt)
            
            # Plot optimal as dashed line
            ax_phase.plot(opt_positions, opt_velocities, '--', color=colors[i], 
                         linewidth=1.5, alpha=0.5)
            ax_pos.plot(times, opt_positions, '--', color=colors[i], 
                       linewidth=1.5, alpha=0.5)
            ax_vel.plot(times, opt_velocities, '--', color=colors[i], 
                       linewidth=1.5, alpha=0.5)
    
    # Format phase space plot
    ax_phase.plot(0, 0, 'k*', markersize=20, label='Target', zorder=5)
    ax_phase.set_xlabel('Position', fontsize=14)
    ax_phase.set_ylabel('Velocity', fontsize=14)
    ax_phase.set_title('Phase Space Trajectories (Solid: Model, Dashed: Optimal)', fontsize=16, fontweight='bold')
    ax_phase.grid(True, alpha=0.3)
    ax_phase.legend(loc='best', fontsize=10)
    ax_phase.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax_phase.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax_phase.set_xlim([-1.1, 1.1])
    ax_phase.set_ylim([-1.1, 1.1])
    
    # Format position plot
    ax_pos.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Target')
    ax_pos.fill_between(times, -1, 1, alpha=0.1, color='gray', label='Bounds')
    ax_pos.set_xlabel('Time (s)', fontsize=14)
    ax_pos.set_ylabel('Position', fontsize=14)
    ax_pos.set_title('Position Trajectories', fontsize=16, fontweight='bold')
    ax_pos.grid(True, alpha=0.3)
    ax_pos.legend(loc='best', fontsize=10)
    ax_pos.set_ylim([-1.2, 1.2])
    
    # Format velocity plot
    ax_vel.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Target')
    ax_vel.fill_between(times, -1, 1, alpha=0.1, color='gray', label='Bounds')
    ax_vel.set_xlabel('Time (s)', fontsize=14)
    ax_vel.set_ylabel('Velocity', fontsize=14)
    ax_vel.set_title('Velocity Trajectories', fontsize=16, fontweight='bold')
    ax_vel.grid(True, alpha=0.3)
    ax_vel.legend(loc='best', fontsize=10)
    ax_vel.set_ylim([-1.2, 1.2])
    
    # Overall title
    fig.suptitle('Double Integrator Control Evaluation - All Test Cases', fontsize=18, fontweight='bold')
    
    # Add control panel
    fig.text(0.5, 0.02, 'All trajectories shown with their predicted (solid) and optimal (dashed) solutions', 
             ha='center', fontsize=12, style='italic')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.05)
    
    # Save
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined visualization saved to {save_path}")


def visualize_all_trajectories_with_controls(
    test_cases: List[Tuple[float, float, List[float]]],
    dt: float,
    save_path: str = "combined_evaluation_full.png",
    show_optimal: bool = True
):
    """
    Create an even more comprehensive figure including control inputs.
    
    Args:
        test_cases: List of (x0, v0, controls) tuples
        dt: Time step
        save_path: Path to save the figure
        show_optimal: Whether to show optimal LQR solution
    """
    # Create figure with custom layout
    fig = plt.figure(figsize=(24, 16))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1.2, 1, 1], width_ratios=[1, 1])
    
    # Create subplots
    ax_phase = fig.add_subplot(gs[0, :])     # Phase space (spans both columns)
    ax_pos = fig.add_subplot(gs[1, 0])       # Position
    ax_vel = fig.add_subplot(gs[1, 1])       # Velocity
    ax_control = fig.add_subplot(gs[2, :])   # Control (spans both columns)
    
    # Color palette
    colors = plt.cm.viridis(np.linspace(0, 1, len(test_cases)))
    
    # Plot each trajectory
    for i, (x0, v0, controls) in enumerate(test_cases):
        # Simulate trajectory
        positions, velocities, times = simulate_trajectory(x0, v0, controls, dt)
        control_times = times[:-1]
        
        # Create label
        label = f"IC: [{x0:.2f}, {v0:.2f}]"
        
        # Phase space
        ax_phase.plot(positions, velocities, '-', color=colors[i], linewidth=2.5, 
                     label=label, marker='o', markersize=4, markevery=max(1, len(positions)//20))
        ax_phase.plot(positions[0], velocities[0], 'o', color=colors[i], 
                     markersize=12, markeredgecolor='black', markeredgewidth=2)
        ax_phase.plot(positions[-1], velocities[-1], '^', color=colors[i], 
                     markersize=12, markeredgecolor='black', markeredgewidth=2)
        
        # Position
        ax_pos.plot(times, positions, '-', color=colors[i], linewidth=2.5, 
                   label=label, marker='o', markersize=3, markevery=max(1, len(times)//20))
        
        # Velocity
        ax_vel.plot(times, velocities, '-', color=colors[i], linewidth=2.5, 
                   label=label, marker='o', markersize=3, markevery=max(1, len(times)//20))
        
        # Control
        ax_control.step(control_times, controls, '-', color=colors[i], linewidth=2.5, 
                       where='post', label=label, marker='o', markersize=3, 
                       markevery=max(1, len(control_times)//20))
        
        # Show optimal if requested
        if show_optimal:
            optimal_controls = solve_double_integrator(x0, v0, dt, len(controls))
            opt_positions, opt_velocities, _ = simulate_trajectory(x0, v0, optimal_controls, dt)
            
            ax_phase.plot(opt_positions, opt_velocities, '--', color=colors[i], 
                         linewidth=1.5, alpha=0.5)
            ax_pos.plot(times, opt_positions, '--', color=colors[i], 
                       linewidth=1.5, alpha=0.5)
            ax_vel.plot(times, opt_velocities, '--', color=colors[i], 
                       linewidth=1.5, alpha=0.5)
            ax_control.step(control_times, optimal_controls, '--', color=colors[i], 
                           linewidth=1.5, alpha=0.5, where='post')
    
    # Format phase space
    ax_phase.plot(0, 0, 'k*', markersize=20, label='Target', zorder=5)
    ax_phase.set_xlabel('Position', fontsize=14)
    ax_phase.set_ylabel('Velocity', fontsize=14)
    ax_phase.set_title('Phase Space Trajectories', fontsize=16, fontweight='bold')
    ax_phase.grid(True, alpha=0.3)
    ax_phase.legend(loc='best', fontsize=10)
    ax_phase.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax_phase.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax_phase.set_xlim([-1.1, 1.1])
    ax_phase.set_ylim([-1.1, 1.1])
    
    # Format position
    ax_pos.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax_pos.fill_between(times, -1, 1, alpha=0.1, color='gray')
    ax_pos.set_xlabel('Time (s)', fontsize=14)
    ax_pos.set_ylabel('Position', fontsize=14)
    ax_pos.set_title('Position Trajectories', fontsize=16, fontweight='bold')
    ax_pos.grid(True, alpha=0.3)
    ax_pos.set_ylim([-1.2, 1.2])
    
    # Format velocity
    ax_vel.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax_vel.fill_between(times, -1, 1, alpha=0.1, color='gray')
    ax_vel.set_xlabel('Time (s)', fontsize=14)
    ax_vel.set_ylabel('Velocity', fontsize=14)
    ax_vel.set_title('Velocity Trajectories', fontsize=16, fontweight='bold')
    ax_vel.grid(True, alpha=0.3)
    ax_vel.set_ylim([-1.2, 1.2])
    
    # Format control
    ax_control.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax_control.fill_between(control_times, -3, 3, alpha=0.1, color='gray', step='post')
    ax_control.set_xlabel('Time (s)', fontsize=14)
    ax_control.set_ylabel('Control Input', fontsize=14)
    ax_control.set_title('Control Sequences', fontsize=16, fontweight='bold')
    ax_control.grid(True, alpha=0.3)
    ax_control.legend(loc='best', fontsize=10)
    ax_control.set_ylim([-3.5, 3.5])
    
    # Overall title
    fig.suptitle('Comprehensive Control System Evaluation', fontsize=20, fontweight='bold')
    
    # Add metrics panel
    metrics_text = "Performance Metrics:\n"
    for i, (x0, v0, controls) in enumerate(test_cases):
        positions, velocities, _ = simulate_trajectory(x0, v0, controls, dt)
        final_error = np.sqrt(positions[-1]**2 + velocities[-1]**2)
        control_effort = np.sum(np.square(controls))
        metrics_text += f"IC [{x0:.2f}, {v0:.2f}]: Final Error = {final_error:.4f}, Control Effort = {control_effort:.2f}\n"
    
    fig.text(0.98, 0.5, metrics_text, transform=fig.transFigure, fontsize=10,
             verticalalignment='center', horizontalalignment='right',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.94, right=0.85)
    
    # Save
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Full combined visualization saved to {save_path}")