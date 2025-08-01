"""
Enhanced visualization with publication-quality styling matching benchmark style.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from typing import List, Tuple, Optional
from control import simulate_trajectory, solve_double_integrator
from plotting_styles import (
    setup_beautiful_plotting, style_axes, add_beautiful_legend,
    save_beautiful_figure, BEAUTIFUL_COLORS, TYPOGRAPHY
)


def create_benchmark_style_comparison(
    test_cases: List[Tuple[float, float, List[float]]],
    dt: float,
    save_path: str = "benchmark_comparison",
    show_optimal: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10)
):
    """
    Create publication-ready comparison plot matching benchmark style exactly.
    
    Args:
        test_cases: List of (x0, v0, controls) tuples
        dt: Time step
        save_path: Path to save the figure
        show_optimal: Whether to show optimal LQR solution
        title: Optional custom title
        figsize: Figure size
    """
    # Set up beautiful plotting style
    setup_beautiful_plotting()
    
    # Create figure with 2x2 layout
    fig = plt.figure(figsize=figsize)
    
    # Create subplots
    ax_phase = fig.add_subplot(2, 2, 1)
    ax_pos = fig.add_subplot(2, 2, 2)
    ax_vel = fig.add_subplot(2, 2, 3)
    ax_control = fig.add_subplot(2, 2, 4)
    
    # Define colors matching the benchmark
    optimal_color = 'blue'     # Blue for optimal/LQR
    model_color = 'red'        # Red for model
    
    # Get comparison color scheme
    colors = {
        'target': '#DC143C',       # Deep red for targets
        'bounds': 'red',           # Red for bounds
        'start': '#FF4500',        # Orange-red for start
        'end': '#4B0082',          # Indigo for end
    }
    
    # Target state (assuming origin as target)
    target_position, target_velocity = 0.0, 0.0
    target_radius = 0.1  # Matching benchmark
    
    # PHASE SPACE PLOT
    # Draw the target circle
    target_circle = plt.Circle((target_position, target_velocity), target_radius, 
                              color=colors['target'], alpha=0.3, linewidth=0)
    ax_phase.add_patch(target_circle)
    
    # Draw target marker
    ax_phase.plot(target_position, target_velocity, 'X', color=colors['target'], 
                 markersize=10, markeredgewidth=2.5, label='Target')
    
    # Plot trajectories
    for i, (x0, v0, controls) in enumerate(test_cases):
        # Simulate model trajectory
        positions, velocities, times = simulate_trajectory(x0, v0, controls, dt)
        
        # Plot model trajectory
        ax_phase.plot(positions, velocities, '-', color=model_color, 
                     alpha=0.8, linewidth=2.5, label='Model' if i == 0 else None)
        
        # Plot optimal trajectory if requested
        if show_optimal:
            optimal_controls = solve_double_integrator(x0, v0, dt, len(controls))
            opt_positions, opt_velocities, _ = simulate_trajectory(x0, v0, optimal_controls, dt)
            ax_phase.plot(opt_positions, opt_velocities, '-', color=optimal_color,
                         alpha=0.8, linewidth=2.5, label='Optimal' if i == 0 else None)
        
        # Add directional arrows (early in trajectory)
        if len(positions) > 10:
            early_idx = min(3, len(positions) // 15)  # Around 0.3 seconds
            
            # Model arrow
            if early_idx + 3 < len(positions):
                dx = positions[early_idx + 3] - positions[early_idx]
                dy = velocities[early_idx + 3] - velocities[early_idx]
                magnitude = np.sqrt(dx**2 + dy**2)
                if magnitude > 0:
                    dx_norm = 0.06 * dx / magnitude
                    dy_norm = 0.06 * dy / magnitude
                    ax_phase.annotate('', 
                                     xy=(positions[early_idx] + dx_norm, 
                                         velocities[early_idx] + dy_norm),
                                     xytext=(positions[early_idx], velocities[early_idx]),
                                     arrowprops=dict(arrowstyle='->', color=model_color, 
                                                   alpha=0.9, lw=2.2))
            
            # Optimal arrow
            if show_optimal and early_idx + 3 < len(opt_positions):
                dx = opt_positions[early_idx + 3] - opt_positions[early_idx]
                dy = opt_velocities[early_idx + 3] - opt_velocities[early_idx]
                magnitude = np.sqrt(dx**2 + dy**2)
                if magnitude > 0:
                    dx_norm = 0.06 * dx / magnitude
                    dy_norm = 0.06 * dy / magnitude
                    ax_phase.annotate('', 
                                     xy=(opt_positions[early_idx] + dx_norm, 
                                         opt_velocities[early_idx] + dy_norm),
                                     xytext=(opt_positions[early_idx], opt_velocities[early_idx]),
                                     arrowprops=dict(arrowstyle='->', color=optimal_color, 
                                                   alpha=0.9, lw=2.2))
    
    # Style phase space - no title
    style_axes(ax_phase, title=None, xlabel='Position', ylabel='Velocity')
    
    # Add legend with transparent box
    if show_optimal:
        legend_elements = [
            plt.Line2D([0], [0], color=optimal_color, linewidth=2.5, label='Optimal'),
            plt.Line2D([0], [0], color=model_color, linewidth=2.5, label='Model')
        ]
    else:
        legend_elements = [
            plt.Line2D([0], [0], color=model_color, linewidth=2.5, label='Model')
        ]
    
    legend = ax_phase.legend(handles=legend_elements, loc='upper right', ncol=1,
                            fontsize=TYPOGRAPHY['legend_size'], frameon=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('gray')
    legend.get_frame().set_alpha(0.9)
    legend.get_frame().set_linewidth(0.8)
    
    # POSITION VS TIME PLOT
    ax_pos.axhline(y=target_position, color=colors['target'], linestyle='--', 
                  alpha=0.9, linewidth=2.5, label='Target Position', zorder=1)
    
    for i, (x0, v0, controls) in enumerate(test_cases):
        positions, velocities, times = simulate_trajectory(x0, v0, controls, dt)
        
        # Model position
        ax_pos.plot(times, positions, '-', color=model_color, alpha=0.8, 
                   linewidth=2.5, label='Model' if i == 0 else None)
        
        # Optimal position
        if show_optimal:
            optimal_controls = solve_double_integrator(x0, v0, dt, len(controls))
            opt_positions, _, _ = simulate_trajectory(x0, v0, optimal_controls, dt)
            ax_pos.plot(times, opt_positions, '-', color=optimal_color, alpha=0.8,
                       linewidth=2.5, label='Optimal' if i == 0 else None)
    
    style_axes(ax_pos, title=None, xlabel='Time (s)', ylabel='Position')
    
    # Position legend
    legend_elements = [
        plt.Line2D([0], [0], color=colors['target'], linestyle='--', linewidth=2.5, label='Target Position'),
    ]
    legend = ax_pos.legend(handles=legend_elements, loc='upper right', ncol=1,
                          fontsize=TYPOGRAPHY['legend_size'], frameon=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('gray')
    legend.get_frame().set_alpha(0.9)
    legend.get_frame().set_linewidth(0.8)
    
    # VELOCITY VS TIME PLOT
    ax_vel.axhline(y=target_velocity, color=colors['target'], linestyle='--', 
                  alpha=0.9, linewidth=2.5, label='Target Velocity', zorder=1)
    
    for i, (x0, v0, controls) in enumerate(test_cases):
        positions, velocities, times = simulate_trajectory(x0, v0, controls, dt)
        
        # Model velocity
        ax_vel.plot(times, velocities, '-', color=model_color, alpha=0.8,
                   linewidth=2.5, label='Model' if i == 0 else None)
        
        # Optimal velocity
        if show_optimal:
            optimal_controls = solve_double_integrator(x0, v0, dt, len(controls))
            _, opt_velocities, _ = simulate_trajectory(x0, v0, optimal_controls, dt)
            ax_vel.plot(times, opt_velocities, '-', color=optimal_color, alpha=0.8,
                       linewidth=2.5, label='Optimal' if i == 0 else None)
    
    style_axes(ax_vel, title=None, xlabel='Time (s)', ylabel='Velocity')
    
    # Velocity legend
    legend_elements = [
        plt.Line2D([0], [0], color=colors['target'], linestyle='--', linewidth=2.5, label='Target Velocity'),
    ]
    legend = ax_vel.legend(handles=legend_elements, loc='upper right', ncol=1,
                          fontsize=TYPOGRAPHY['legend_size'], frameon=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('gray')
    legend.get_frame().set_alpha(0.9)
    legend.get_frame().set_linewidth(0.8)
    
    # CONTROL INPUT PLOT
    # Draw action bounds
    bound_low = -3.0
    bound_high = 3.0
    ax_control.axhline(y=bound_low, color=colors['bounds'], linestyle='--', 
                      alpha=0.8, linewidth=2.5, zorder=1)
    ax_control.axhline(y=bound_high, color=colors['bounds'], linestyle='--', 
                      alpha=0.8, linewidth=2.5, label='Action Bounds', zorder=1)
    
    for i, (x0, v0, controls) in enumerate(test_cases):
        control_times = np.arange(len(controls)) * dt
        
        # Model controls
        ax_control.plot(control_times, controls, '-', color=model_color, alpha=0.8,
                       linewidth=2.5, label='Model' if i == 0 else None)
        
        # Optimal controls
        if show_optimal:
            optimal_controls = solve_double_integrator(x0, v0, dt, len(controls))
            ax_control.plot(control_times, optimal_controls, '-', color=optimal_color,
                           alpha=0.8, linewidth=2.5, label='Optimal' if i == 0 else None)
    
    style_axes(ax_control, title=None, xlabel='Time (s)', ylabel='Control')
    
    # Control legend
    legend_elements = [
        plt.Line2D([0], [0], color=colors['bounds'], linestyle='--', linewidth=2.5, label='Action Bounds'),
    ]
    legend = ax_control.legend(handles=legend_elements, loc='upper right', ncol=1,
                              fontsize=TYPOGRAPHY['legend_size'], frameon=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('gray')
    legend.get_frame().set_alpha(0.9)
    legend.get_frame().set_linewidth(0.8)
    
    # Better spacing between subplots
    plt.tight_layout(pad=4, h_pad=3.0, w_pad=3.0)
    
    # Add captions BELOW subplots
    ax_phase.text(0.5, -0.25, '(a) Phase Space', transform=ax_phase.transAxes, 
                  ha='center', va='top', fontsize=16)
    ax_pos.text(0.5, -0.25, '(b) Position vs Time', transform=ax_pos.transAxes, 
                ha='center', va='top', fontsize=16)
    ax_vel.text(0.5, -0.25, '(c) Velocity vs Time', transform=ax_vel.transAxes, 
                ha='center', va='top', fontsize=16)
    ax_control.text(0.5, -0.25, '(d) Control vs Time', transform=ax_control.transAxes, 
                    ha='center', va='top', fontsize=16)
    
    # Force matplotlib to draw the grids immediately
    fig.canvas.draw()
    
    # Save figure
    save_beautiful_figure(fig, save_path, formats=['pdf', 'png'])
    
    print(f"Benchmark-style comparison plot saved to {save_path}")
    
    return fig


def create_clean_comparison_plot(
    test_cases: List[Tuple[float, float, List[float]]],
    dt: float,
    save_path: str = "clean_comparison",
    show_optimal: bool = True,
    figsize: Tuple[int, int] = (15, 12)
):
    """
    Create a clean comparison plot without individual IC labels.
    Professional style with consistent colors across all trajectories.
    """
    setup_beautiful_plotting()
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)
    
    ax_phase = fig.add_subplot(gs[0, 0])
    ax_pos = fig.add_subplot(gs[0, 1])
    ax_vel = fig.add_subplot(gs[1, 0])
    ax_control = fig.add_subplot(gs[1, 1])
    
    # Use consistent colors for all trajectories
    model_color = '#d62728'      # Red for model
    optimal_color = '#1f77b4'    # Blue for optimal
    target_color = '#2ca02c'     # Green for target
    bounds_color = '#ff7f0e'     # Orange for bounds
    
    # Target state
    target_pos, target_vel = 0.0, 0.0
    
    # PHASE SPACE
    ax_phase.scatter(target_pos, target_vel, s=200, c=target_color, marker='*',
                    edgecolor='white', linewidth=2, label='Target', zorder=10)
    
    # Plot all trajectories without individual labels
    for i, (x0, v0, controls) in enumerate(test_cases):
        positions, velocities, times = simulate_trajectory(x0, v0, controls, dt)
        
        # Model trajectory
        ax_phase.plot(positions, velocities, '-', color=model_color, 
                     alpha=0.7, linewidth=2.0)
        
        # Mark start point
        if i == 0:  # Only label first trajectory
            ax_phase.scatter(positions[0], velocities[0], s=80, c=model_color,
                           marker='o', edgecolor='white', linewidth=1.5, 
                           label='Start', zorder=5)
        else:
            ax_phase.scatter(positions[0], velocities[0], s=80, c=model_color,
                           marker='o', edgecolor='white', linewidth=1.5, zorder=5)
        
        # Optimal trajectory
        if show_optimal:
            optimal_controls = solve_double_integrator(x0, v0, dt, len(controls))
            opt_pos, opt_vel, _ = simulate_trajectory(x0, v0, optimal_controls, dt)
            ax_phase.plot(opt_pos, opt_vel, '--', color=optimal_color,
                         alpha=0.7, linewidth=2.0)
    
    ax_phase.set_xlim([-1.1, 1.1])
    ax_phase.set_ylim([-1.1, 1.1])
    style_axes(ax_phase, title=None, xlabel='Position', ylabel='Velocity', grid=True, box=True)
    
    # Simple legend
    handles = [plt.Line2D([0], [0], color=model_color, linewidth=2.5, label='Model'),
               plt.Line2D([0], [0], color=optimal_color, linewidth=2.5, linestyle='--', label='Optimal')]
    ax_phase.legend(handles=handles, loc='best', frameon=True, fancybox=True, 
                   shadow=False, framealpha=0.9)
    
    # POSITION VS TIME
    ax_pos.axhline(y=target_pos, color=target_color, linestyle=':', 
                  linewidth=2.0, alpha=0.8, label='Target')
    
    for x0, v0, controls in test_cases:
        positions, velocities, times = simulate_trajectory(x0, v0, controls, dt)
        ax_pos.plot(times, positions, '-', color=model_color, alpha=0.7, linewidth=2.0)
        
        if show_optimal:
            optimal_controls = solve_double_integrator(x0, v0, dt, len(controls))
            opt_pos, _, _ = simulate_trajectory(x0, v0, optimal_controls, dt)
            ax_pos.plot(times, opt_pos, '--', color=optimal_color, alpha=0.7, linewidth=2.0)
    
    ax_pos.set_ylim([-1.2, 1.2])
    ax_pos.axhspan(-1, 1, alpha=0.05, color='gray')
    style_axes(ax_pos, title=None, xlabel='Time (s)', ylabel='Position', grid=True, box=True)
    
    # VELOCITY VS TIME
    ax_vel.axhline(y=target_vel, color=target_color, linestyle=':', 
                  linewidth=2.0, alpha=0.8, label='Target')
    
    for x0, v0, controls in test_cases:
        positions, velocities, times = simulate_trajectory(x0, v0, controls, dt)
        ax_vel.plot(times, velocities, '-', color=model_color, alpha=0.7, linewidth=2.0)
        
        if show_optimal:
            optimal_controls = solve_double_integrator(x0, v0, dt, len(controls))
            _, opt_vel, _ = simulate_trajectory(x0, v0, optimal_controls, dt)
            ax_vel.plot(times, opt_vel, '--', color=optimal_color, alpha=0.7, linewidth=2.0)
    
    ax_vel.set_ylim([-1.2, 1.2])
    ax_vel.axhspan(-1, 1, alpha=0.05, color='gray')
    style_axes(ax_vel, title=None, xlabel='Time (s)', ylabel='Velocity', grid=True, box=True)
    
    # CONTROL VS TIME
    ax_control.axhline(y=-3, color=bounds_color, linestyle='--', 
                      linewidth=2.0, alpha=0.8)
    ax_control.axhline(y=3, color=bounds_color, linestyle='--', 
                      linewidth=2.0, alpha=0.8, label='Bounds')
    
    for x0, v0, controls in test_cases:
        control_times = np.arange(len(controls)) * dt
        ax_control.plot(control_times, controls, '-', color=model_color, alpha=0.7, linewidth=2.0)
        
        if show_optimal:
            optimal_controls = solve_double_integrator(x0, v0, dt, len(controls))
            ax_control.plot(control_times, optimal_controls, '--', color=optimal_color, 
                           alpha=0.7, linewidth=2.0)
    
    ax_control.set_ylim([-3.5, 3.5])
    ax_control.axhspan(-3, 3, alpha=0.05, color='gray')
    style_axes(ax_control, title=None, xlabel='Time (s)', ylabel='Control u(t)', grid=True, box=True)
    ax_control.legend(['Bounds'], loc='upper right', frameon=True, fancybox=True, 
                     shadow=False, framealpha=0.9)
    
    # Add subplot captions below
    fig.text(0.25, 0.48, '(a) Phase Space', ha='center', fontsize=16, fontweight='bold')
    fig.text(0.75, 0.48, '(b) Position vs Time', ha='center', fontsize=16, fontweight='bold')
    fig.text(0.25, 0.02, '(c) Velocity vs Time', ha='center', fontsize=16, fontweight='bold')
    fig.text(0.75, 0.02, '(d) Control vs Time', ha='center', fontsize=16, fontweight='bold')
    
    # No title as requested
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    # Save
    save_beautiful_figure(fig, save_path, formats=['png', 'pdf'])
    
    return fig