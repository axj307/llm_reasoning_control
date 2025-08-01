"""
Enhanced visualization utilities with publication-quality styling - Fixed version.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from typing import List, Tuple, Optional
from control import simulate_trajectory, solve_double_integrator
from plotting_styles import (
    setup_beautiful_plotting, style_axes, add_beautiful_legend,
    add_directional_arrows, add_performance_annotation,
    save_beautiful_figure, BEAUTIFUL_COLORS, TYPOGRAPHY, PHASE_COLORS,
    PUBLICATION_COLORS
)


def create_publication_comparison_plot(
    test_cases: List[Tuple[float, float, List[float]]],
    dt: float,
    save_path: str = "publication_control_comparison.png",
    show_optimal: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 12)
):
    """
    Create publication-ready 4-subplot comparison plot (phase space, position, velocity, control).
    
    Args:
        test_cases: List of (x0, v0, controls) tuples
        dt: Time step
        save_path: Path to save the figure
        show_optimal: Whether to show optimal LQR solution
        title: Optional custom title (ignored now)
        figsize: Figure size
    """
    # Set up beautiful plotting style
    setup_beautiful_plotting()
    
    # Create figure with 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Use single colors for all trajectories
    model_color = '#d62728'      # Red for model
    optimal_color = '#1f77b4'    # Blue for optimal
    
    # Process each test case
    for i, (x0, v0, controls) in enumerate(test_cases):
        # Simulate trajectory
        positions, velocities, times = simulate_trajectory(x0, v0, controls, dt)
        control_times = times[:-1]  # Controls are one step shorter
        
        # Get optimal controls if requested
        if show_optimal:
            optimal_controls = solve_double_integrator(x0, v0, dt, len(controls))
            opt_positions, opt_velocities, _ = simulate_trajectory(x0, v0, optimal_controls, dt)
        
        # Subplot 1: Phase Space (top-left)
        ax = axes[0, 0]
        ax.plot(positions, velocities, '-', color=model_color, linewidth=2.5, 
                alpha=0.8)
        
        # Add directional arrows
        add_directional_arrows(ax, positions, velocities, color=model_color, alpha=0.6)
        
        # Mark start and end points
        ax.scatter(positions[0], velocities[0], s=120, c=model_color, 
                  marker='o', edgecolor='white', linewidth=2, zorder=5)
        ax.scatter(positions[-1], velocities[-1], s=150, c=PUBLICATION_COLORS['target'], 
                  marker='X', edgecolor='white', linewidth=2, zorder=5)
        
        # Plot optimal trajectory if requested
        if show_optimal:
            ax.plot(opt_positions, opt_velocities, '--', color=optimal_color, 
                   linewidth=2.0, alpha=0.5)
        
        # Subplot 2: Position vs Time (top-right)
        ax = axes[0, 1]
        ax.plot(times, positions, '-', color=model_color, linewidth=2.5, 
                alpha=0.8)
        if show_optimal:
            ax.plot(times, opt_positions, '--', color=optimal_color, 
                   linewidth=1.5, alpha=0.5)
        
        # Subplot 3: Velocity vs Time (bottom-left)
        ax = axes[1, 0]
        ax.plot(times, velocities, '-', color=model_color, linewidth=2.5, 
                alpha=0.8)
        if show_optimal:
            ax.plot(times, opt_velocities, '--', color=optimal_color, 
                   linewidth=1.5, alpha=0.5)
        
        # Subplot 4: Control vs Time (bottom-right)
        ax = axes[1, 1]
        ax.step(control_times, controls, '-', color=model_color, linewidth=2.5, 
                where='post', alpha=0.8)
        if show_optimal:
            ax.step(control_times, optimal_controls, '--', color=optimal_color, 
                   linewidth=1.5, alpha=0.5, where='post')
    
    # Style phase space subplot
    ax = axes[0, 0]
    ax.scatter(0, 0, s=200, c=PUBLICATION_COLORS['target'], marker='*',
              edgecolor='white', linewidth=2, label='Target', zorder=6)
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    style_axes(ax, title=None, xlabel='Position', ylabel='Velocity', grid=True, box=True)
    
    # Add legend with custom positioning for phase space
    handles = []
    if show_optimal:
        handles.append(plt.Line2D([0], [0], color=model_color, linewidth=2.5, label='Model'))
        handles.append(plt.Line2D([0], [0], color=optimal_color, linewidth=2.5, linestyle='--', label='Optimal'))
    else:
        handles.append(plt.Line2D([0], [0], color=model_color, linewidth=2.5, label='Model'))
    
    add_beautiful_legend(ax, handles=handles, location='upper right', ncol=1)
    
    # Style position subplot
    ax = axes[0, 1]
    ax.axhline(y=0, color=PUBLICATION_COLORS['target'], linestyle='--', linewidth=2, alpha=0.5)
    ax.fill_between(times, -1, 1, alpha=0.1, color='gray')
    ax.set_ylim([-1.2, 1.2])
    style_axes(ax, title=None, xlabel='Time (s)', ylabel='Position', grid=True, box=True)
    
    # Style velocity subplot
    ax = axes[1, 0]
    ax.axhline(y=0, color=PUBLICATION_COLORS['target'], linestyle='--', linewidth=2, alpha=0.5)
    ax.fill_between(times, -1, 1, alpha=0.1, color='gray')
    ax.set_ylim([-1.2, 1.2])
    style_axes(ax, title=None, xlabel='Time (s)', ylabel='Velocity', grid=True, box=True)
    
    # Style control subplot
    ax = axes[1, 1]
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.axhline(-3, color=PUBLICATION_COLORS['action_bounds'], linestyle='--', linewidth=2.5, alpha=0.8, label='Bounds')
    ax.axhline(3, color=PUBLICATION_COLORS['action_bounds'], linestyle='--', linewidth=2.5, alpha=0.8)
    ax.fill_between(control_times, -3, 3, alpha=0.05, color='gray', step='post')
    ax.set_ylim([-3.5, 3.5])
    style_axes(ax, title=None, xlabel='Time (s)', ylabel='Control u(t)', grid=True, box=True)
    
    # Add subplot captions below
    ax = axes[0, 0]
    ax.text(0.5, -0.25, '(a) Phase Space', transform=ax.transAxes, 
            ha='center', va='top', fontsize=16, fontweight='bold')
    ax = axes[0, 1]
    ax.text(0.5, -0.25, '(b) Position vs Time', transform=ax.transAxes, 
            ha='center', va='top', fontsize=16, fontweight='bold')
    ax = axes[1, 0]
    ax.text(0.5, -0.25, '(c) Velocity vs Time', transform=ax.transAxes, 
            ha='center', va='top', fontsize=16, fontweight='bold')
    ax = axes[1, 1]
    ax.text(0.5, -0.25, '(d) Control vs Time', transform=ax.transAxes, 
            ha='center', va='top', fontsize=16, fontweight='bold')
    
    # No title as requested
    
    # Final layout adjustment
    plt.tight_layout(pad=3.0, h_pad=2.5, w_pad=2.5)
    plt.subplots_adjust(top=0.96, bottom=0.08)
    
    # Save figure
    save_beautiful_figure(fig, save_path, formats=['png', 'pdf'])
    print(f"Publication-quality figure saved to {save_path}")
    
    return fig


def create_model_trajectories_plot(
    test_cases: List[Tuple[float, float, List[float]]],
    dt: float,
    save_path: str = "model_trajectories.png",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 12),
    add_metrics: bool = True
):
    """
    Create publication-ready plots for model trajectories without optimal baseline.
    Useful for visualizing model-only results.
    
    Args:
        test_cases: List of (x0, v0, controls) tuples
        dt: Time step
        save_path: Path to save the figure
        title: Optional custom title (ignored now)
        figsize: Figure size
        add_metrics: Whether to add performance metrics
    """
    # Set up beautiful plotting style
    setup_beautiful_plotting()
    
    # Create figure with 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Use single color for all model trajectories
    model_color = '#d62728'  # Red for model
    
    # Process each test case
    for i, (x0, v0, controls) in enumerate(test_cases):
        # Simulate trajectory
        positions, velocities, times = simulate_trajectory(x0, v0, controls, dt)
        control_times = times[:-1]
        
        # Phase space plot
        ax = axes[0, 0]
        ax.plot(positions, velocities, '-', color=model_color, linewidth=2.5, 
                alpha=0.8)
        add_directional_arrows(ax, positions, velocities, color=model_color)
        
        # Mark start and end
        ax.scatter(positions[0], velocities[0], s=120, c=model_color, 
                  marker='o', edgecolor='white', linewidth=2, zorder=5)
        ax.scatter(positions[-1], velocities[-1], s=150, c='black', 
                  marker='s', edgecolor='white', linewidth=2, zorder=5)
        
        # Time series plots
        axes[0, 1].plot(times, positions, '-', color=model_color, linewidth=2.5, alpha=0.8)
        axes[1, 0].plot(times, velocities, '-', color=model_color, linewidth=2.5, alpha=0.8)
        axes[1, 1].step(control_times, controls, '-', color=model_color, linewidth=2.5, 
                       where='post', alpha=0.8)
    
    # Style all subplots - no titles
    subplot_config = [
        (0, 0, None, 'Position', 'Velocity'),
        (0, 1, None, 'Time (s)', 'Position'),
        (1, 0, None, 'Time (s)', 'Velocity'),
        (1, 1, None, 'Time (s)', 'Control u(t)')
    ]
    
    for (row, col, subplot_title, xlabel, ylabel) in subplot_config:
        ax = axes[row, col]
        style_axes(ax, title=subplot_title, xlabel=xlabel, ylabel=ylabel, grid=True, box=True)
        
        if row == 0 and col == 0:  # Phase space
            ax.scatter(0, 0, s=200, c=PUBLICATION_COLORS['target'], marker='*',
                      edgecolor='white', linewidth=2, label='Target', zorder=6)
            ax.set_xlim([-1.1, 1.1])
            ax.set_ylim([-1.1, 1.1])
            add_beautiful_legend(ax, location='best')
        
        elif col == 1:  # Position/Control plots
            if row == 0:  # Position
                ax.axhline(y=0, color=PUBLICATION_COLORS['target'], linestyle='--', 
                          linewidth=2, alpha=0.5)
                ax.fill_between(times, -1, 1, alpha=0.1, color='gray')
                ax.set_ylim([-1.2, 1.2])
            else:  # Control
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
                ax.axhline(-3, color=PUBLICATION_COLORS['action_bounds'], 
                          linestyle='--', linewidth=2.5, alpha=0.8)
                ax.axhline(3, color=PUBLICATION_COLORS['action_bounds'], 
                          linestyle='--', linewidth=2.5, alpha=0.8)
                ax.set_ylim([-3.5, 3.5])
        
        elif row == 1 and col == 0:  # Velocity
            ax.axhline(y=0, color=PUBLICATION_COLORS['target'], linestyle='--', 
                      linewidth=2, alpha=0.5)
            ax.fill_between(times, -1, 1, alpha=0.1, color='gray')
            ax.set_ylim([-1.2, 1.2])
    
    # Add subplot captions
    captions = ['(a) Phase Space', '(b) Position vs Time', '(c) Velocity vs Time', '(d) Control vs Time']
    for i, ax in enumerate(axes.flat):
        ax.text(0.5, -0.25, captions[i], transform=ax.transAxes, 
                ha='center', va='top', fontsize=16, fontweight='bold')
    
    # No title as requested
    
    # Add metrics if requested
    if add_metrics:
        metrics_text = "Final States:\n"
        for i, (x0, v0, controls) in enumerate(test_cases):
            positions, velocities, _ = simulate_trajectory(x0, v0, controls, dt)
            final_error = np.sqrt(positions[-1]**2 + velocities[-1]**2)
            metrics_text += f"Test {i+1}: Error = {final_error:.4f}\n"
        
        add_performance_annotation(fig, metrics_text, position='right')
    
    # Final layout
    plt.tight_layout(pad=3.0, h_pad=2.5, w_pad=2.5)
    if add_metrics:
        plt.subplots_adjust(top=0.96, right=0.85, bottom=0.08)
    else:
        plt.subplots_adjust(top=0.96, bottom=0.08)
    
    # Save figure
    save_beautiful_figure(fig, save_path, formats=['png', 'pdf'])
    print(f"Model trajectories figure saved to {save_path}")
    
    return fig


def create_control_dashboard(
    test_cases: List[Tuple[float, float, List[float]]],
    dt: float,
    save_path: str = "control_dashboard.png",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 12)
):
    """
    Create a comprehensive control system dashboard with additional visualizations.
    
    Args:
        test_cases: List of (x0, v0, controls) tuples
        dt: Time step
        save_path: Path to save the figure
        title: Optional custom title (ignored now)
        figsize: Figure size
    """
    setup_beautiful_plotting()
    
    # Create figure with custom grid
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 3, height_ratios=[1.5, 1, 1], width_ratios=[1, 1, 1],
                          hspace=0.3, wspace=0.3)
    
    # Main phase portrait (spans 2 columns)
    ax_phase = fig.add_subplot(gs[0, :2])
    
    # Metrics summary (top right)
    ax_metrics = fig.add_subplot(gs[0, 2])
    
    # Time series plots (middle row)
    ax_pos = fig.add_subplot(gs[1, 0])
    ax_vel = fig.add_subplot(gs[1, 1])
    ax_control = fig.add_subplot(gs[1, 2])
    
    # Error analysis (bottom row)
    ax_error = fig.add_subplot(gs[2, :2])
    ax_effort = fig.add_subplot(gs[2, 2])
    
    # Use single color for all trajectories
    model_color = '#d62728'  # Red for model
    
    # Initialize metrics storage
    all_errors = []
    all_efforts = []
    
    for i, (x0, v0, controls) in enumerate(test_cases):
        positions, velocities, times = simulate_trajectory(x0, v0, controls, dt)
        control_times = times[:-1]
        
        # Phase space
        ax_phase.plot(positions, velocities, '-', color=model_color, linewidth=2.5, 
                     alpha=0.8)
        add_directional_arrows(ax_phase, positions, velocities, color=model_color)
        ax_phase.scatter(positions[0], velocities[0], s=120, c=model_color, 
                        marker='o', edgecolor='white', linewidth=2, zorder=5)
        ax_phase.scatter(positions[-1], velocities[-1], s=150, c='black', 
                        marker='s', edgecolor='white', linewidth=2, zorder=5)
        
        # Time series
        ax_pos.plot(times, positions, '-', color=model_color, linewidth=2.5, alpha=0.8)
        ax_vel.plot(times, velocities, '-', color=model_color, linewidth=2.5, alpha=0.8)
        ax_control.step(control_times, controls, '-', color=model_color, linewidth=2.5, 
                       where='post', alpha=0.8)
        
        # Error over time
        pos_arr = np.array(positions)
        vel_arr = np.array(velocities)
        errors = np.sqrt(pos_arr**2 + vel_arr**2)
        ax_error.plot(times, errors, '-', color=model_color, linewidth=2.5, 
                     alpha=0.8)
        
        # Cumulative control effort
        cumulative_effort = np.cumsum(np.square(controls)) * dt
        ax_effort.plot(control_times, cumulative_effort, '-', color=model_color, 
                      linewidth=2.5, alpha=0.8)
        
        # Store metrics
        all_errors.append(errors[-1])
        all_efforts.append(cumulative_effort[-1])
    
    # Style phase space - no title
    ax_phase.scatter(0, 0, s=200, c=PUBLICATION_COLORS['target'], marker='*',
                    edgecolor='white', linewidth=2, label='Target', zorder=6)
    ax_phase.set_xlim([-1.1, 1.1])
    ax_phase.set_ylim([-1.1, 1.1])
    style_axes(ax_phase, title=None, xlabel='Position', 
              ylabel='Velocity', grid=True, box=True)
    add_beautiful_legend(ax_phase, location='best')
    
    # Style metrics summary
    ax_metrics.axis('off')
    metrics_text = "Summary Statistics\n" + "="*20 + "\n\n"
    metrics_text += f"Test Cases: {len(test_cases)}\n"
    metrics_text += f"Mean Final Error: {np.mean(all_errors):.4f}\n"
    metrics_text += f"Max Final Error: {np.max(all_errors):.4f}\n"
    metrics_text += f"Min Final Error: {np.min(all_errors):.4f}\n"
    metrics_text += f"Mean Control Effort: {np.mean(all_efforts):.2f}\n"
    
    ax_metrics.text(0.1, 0.9, metrics_text, transform=ax_metrics.transAxes,
                   fontsize=12, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    # Style time series plots - no titles
    for ax, ylabel in [(ax_pos, 'Position'),
                       (ax_vel, 'Velocity'),
                       (ax_control, 'Control u(t)')]:
        style_axes(ax, title=None, xlabel='Time (s)', ylabel=ylabel, grid=True, box=True)
        if ax in [ax_pos, ax_vel]:
            ax.axhline(y=0, color=PUBLICATION_COLORS['target'], linestyle='--', 
                      linewidth=2, alpha=0.5)
            ax.set_ylim([-1.2, 1.2])
        else:  # Control
            ax.set_ylim([-3.5, 3.5])
            ax.axhline(-3, color=PUBLICATION_COLORS['action_bounds'], 
                      linestyle='--', linewidth=2.5, alpha=0.8)
            ax.axhline(3, color=PUBLICATION_COLORS['action_bounds'], 
                      linestyle='--', linewidth=2.5, alpha=0.8)
    
    # Style error plot - no title
    ax_error.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    style_axes(ax_error, title=None, xlabel='Time (s)', 
              ylabel='Error', grid=True, box=True)
    ax_error.set_yscale('log')
    ax_error.set_ylim([1e-4, 2])
    
    # Style effort plot - no title
    style_axes(ax_effort, title=None, xlabel='Time (s)', 
              ylabel='∑u²dt', grid=True, box=True)
    
    # Add captions below subplots
    ax_phase.text(0.5, -0.2, '(a) Phase Space Trajectories', transform=ax_phase.transAxes,
                  ha='center', va='top', fontsize=16, fontweight='bold')
    ax_pos.text(0.5, -0.25, '(b) Position', transform=ax_pos.transAxes,
                ha='center', va='top', fontsize=16, fontweight='bold')
    ax_vel.text(0.5, -0.25, '(c) Velocity', transform=ax_vel.transAxes,
                ha='center', va='top', fontsize=16, fontweight='bold')
    ax_control.text(0.5, -0.25, '(d) Control', transform=ax_control.transAxes,
                    ha='center', va='top', fontsize=16, fontweight='bold')
    ax_error.text(0.5, -0.2, '(e) State Error ||[x, v]||', transform=ax_error.transAxes,
                  ha='center', va='top', fontsize=16, fontweight='bold')
    ax_effort.text(0.5, -0.25, '(f) Cumulative Effort', transform=ax_effort.transAxes,
                   ha='center', va='top', fontsize=16, fontweight='bold')
    
    # No overall title as requested
    
    # Save figure
    save_beautiful_figure(fig, save_path, formats=['png', 'pdf'])
    print(f"Control dashboard saved to {save_path}")
    
    return fig