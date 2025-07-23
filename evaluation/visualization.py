"""Visualization utilities for control trajectories."""

import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
from environments import get_system
from .plotting_utils import (
    setup_beautiful_plotting, create_beautiful_figure, style_axes,
    beautiful_line_plot, add_beautiful_legend, get_color_scheme,
    plot_phase_portrait, add_control_bounds, add_grid_and_box,
    save_beautiful_figure, FIGURE_SIZES
)


def plot_trajectories(trajectories: Dict[str, Dict[str, Any]], 
                     system_name: str,
                     initial_state: Optional[Tuple[float, float]] = None,
                     title: Optional[str] = None,
                     figsize: Optional[str] = None) -> plt.Figure:
    """
    Plot multiple trajectories for comparison with beautiful styling.
    
    Args:
        trajectories: Dictionary of {label: trajectory_dict}
        system_name: Name of the system for bounds and labels
        initial_state: Initial state for title
        title: Optional custom title
        figsize: Figure size key from FIGURE_SIZES or tuple
        
    Returns:
        matplotlib Figure object
    """
    # Setup beautiful plotting
    setup_beautiful_plotting()
    
    # Get system info for bounds and labels
    system = get_system(system_name)()
    state_bounds = system.get_state_bounds()
    control_bounds = system.get_control_bounds()
    state_names = getattr(system, 'get_state_names', lambda: ['Position', 'Velocity'])()
    
    # Create beautiful figure
    if figsize is None:
        figsize = 'time_series'
    
    if title is None:
        title = f'{system_name.replace("_", " ").title()} Control Trajectories'
    
    fig = create_beautiful_figure(figsize=figsize, title=title)
    axs = fig.subplots(3, 1)
    
    # Get color scheme for trajectory plots
    colors = get_color_scheme('trajectory')
    trajectory_colors = get_color_scheme('comparison')
    
    # Plot each trajectory
    for i, (label, traj) in enumerate(trajectories.items()):
        if traj is None:
            continue
            
        states = traj['states']
        controls = traj['controls']
        times = traj['times']
        
        # Get color for this trajectory
        if label.lower() in trajectory_colors:
            color = trajectory_colors[label.lower()]
        else:
            color = colors['gradient_professional'][i % len(colors['gradient_professional'])]
        
        # Position plot
        beautiful_line_plot(times, states[:, 0], ax=axs[0], 
                          label=f'{label}', color=color)
        
        # Velocity plot  
        beautiful_line_plot(times, states[:, 1], ax=axs[1], 
                          label=f'{label}', color=color)
        
        # Control plot
        control_times = times[:-1]  # Controls are one step shorter
        axs[2].step(control_times, controls, where='post', label=f'{label}', 
                   color=color, linewidth=2.5, alpha=0.8)
    
    # Style all axes with grid and box
    style_axes(axs[0], title=f'{state_names[0]} vs Time', 
              xlabel='Time (s)', ylabel=f'{state_names[0]}', 
              grid=True, box=True)
    
    style_axes(axs[1], title=f'{state_names[1]} vs Time',
              xlabel='Time (s)', ylabel=f'{state_names[1]}', 
              grid=True, box=True)
    
    style_axes(axs[2], title='Control Input vs Time',
              xlabel='Time (s)', ylabel='Control u(t)', 
              grid=True, box=True)
    
    # Add bounds visualization
    if state_bounds and len(state_bounds) >= 2:
        for i, bounds in enumerate(state_bounds[:2]):
            if bounds and len(bounds) == 2:
                axs[i].axhline(bounds[0], color=trajectory_colors['bounds'], 
                             linestyle='--', alpha=0.5, label='Bounds' if i == 0 else "")
                axs[i].axhline(bounds[1], color=trajectory_colors['bounds'], 
                             linestyle='--', alpha=0.5)
    
    if control_bounds and len(control_bounds) == 2:
        axs[2].axhline(control_bounds[0], color=trajectory_colors['bounds'], 
                      linestyle='--', alpha=0.5, label='Control Bounds')
        axs[2].axhline(control_bounds[1], color=trajectory_colors['bounds'], 
                      linestyle='--', alpha=0.5)
    
    # Add enhanced grid and box to all subplots
    for ax in axs:
        add_grid_and_box(ax, grid_style='both', box=True)
    
    # Add legends to all subplots
    for ax in axs:
        add_beautiful_legend(ax, location='best')
    
    plt.tight_layout()
    return fig


def plot_phase_portrait(trajectories: Dict[str, Dict[str, Any]], 
                       system_name: str,
                       initial_state: Optional[Tuple[float, float]] = None,
                       title: Optional[str] = None,
                       figsize: Optional[str] = None) -> plt.Figure:
    """
    Plot phase portrait (state1 vs state2) for trajectories using beautiful styling.
    
    Args:
        trajectories: Dictionary of {label: trajectory_dict}
        system_name: Name of the system for bounds and labels
        initial_state: Initial state to mark
        title: Optional custom title
        figsize: Figure size key from FIGURE_SIZES or tuple
        
    Returns:
        matplotlib Figure object
    """
    # Setup beautiful plotting
    setup_beautiful_plotting()
    
    # Get system info
    system = get_system(system_name)()
    state_bounds = system.get_state_bounds()
    state_names = getattr(system, 'get_state_names', lambda: ['State 1', 'State 2'])()
    
    # Create beautiful figure
    if figsize is None:
        figsize = 'phase_portrait'
    
    if title is None:
        title = f'{system_name.replace("_", " ").title()} Phase Portrait'
    
    fig = create_beautiful_figure(figsize=figsize, title=title)
    ax = fig.gca()
    
    # Get beautiful colors for trajectories
    trajectory_colors = get_color_scheme('comparison')
    colors_scheme = get_color_scheme('trajectory')
    
    for i, (label, traj) in enumerate(trajectories.items()):
        if traj is None:
            continue
            
        states = traj['states']
        
        # Get appropriate color
        if label.lower() in trajectory_colors:
            color = trajectory_colors[label.lower()]
        else:
            color = colors_scheme['gradient_professional'][i % len(colors_scheme['gradient_professional'])]
        
        # Plot trajectory with beautiful styling
        beautiful_line_plot(states[:, 0], states[:, 1], ax=ax, 
                          label=label, color=color, linewidth=2.5)
        
        # Mark start and end with beautiful styling
        ax.scatter(states[0, 0], states[0, 1], s=80, 
                  c=colors_scheme['start'], marker='o', 
                  edgecolor='white', linewidth=1.5, zorder=5)
        ax.scatter(states[-1, 0], states[-1, 1], s=80, 
                  c=colors_scheme['end'], marker='s', 
                  edgecolor='white', linewidth=1.5, zorder=5)
    
    # Mark target (origin) with beautiful styling
    ax.scatter(0, 0, s=150, c=trajectory_colors['target'], 
              marker='*', edgecolor='white', linewidth=2, 
              label='Target', zorder=5)
    
    # Mark initial state if provided
    if initial_state:
        ax.scatter(initial_state[0], initial_state[1], s=100, 
                  c=colors_scheme['initial'], marker='o', 
                  edgecolor='white', linewidth=2, 
                  label='Initial State', zorder=5)
    
    # State bounds with beautiful styling
    pos_bounds = state_bounds[0]
    vel_bounds = state_bounds[1]
    ax.axhline(y=vel_bounds[0], color=trajectory_colors['bounds'], 
              linestyle='--', alpha=0.6, linewidth=1.5)
    ax.axhline(y=vel_bounds[1], color=trajectory_colors['bounds'], 
              linestyle='--', alpha=0.6, linewidth=1.5)
    ax.axvline(x=pos_bounds[0], color=trajectory_colors['bounds'], 
              linestyle='--', alpha=0.6, linewidth=1.5)
    ax.axvline(x=pos_bounds[1], color=trajectory_colors['bounds'], 
              linestyle='--', alpha=0.6, linewidth=1.5)
    
    # Reference lines at origin
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    # Style axes with grid and box
    style_axes(ax, xlabel=state_names[0], ylabel=state_names[1], 
              grid=True, box=True)
    
    # Add enhanced grid and box
    add_grid_and_box(ax, grid_style='both', box=True)
    
    # Add beautiful legend
    add_beautiful_legend(ax, location='best')
    
    return fig


def plot_comparison(results: List[Dict[str, Any]], 
                   systems: Optional[List[str]] = None,
                   figsize: Optional[str] = None) -> plt.Figure:
    """
    Plot comparison of multiple results using beautiful styling.
    
    Args:
        results: List of inference results
        systems: Optional list of system names to filter by
        figsize: Figure size key from FIGURE_SIZES or tuple
        
    Returns:
        matplotlib Figure object
    """
    # Setup beautiful plotting
    setup_beautiful_plotting()
    
    if systems:
        results = [r for r in results if r['system_name'] in systems]
    
    valid_results = [r for r in results if r.get('valid_format', False)]
    
    if not valid_results:
        print("No valid results to plot")
        return None
    
    n_results = len(valid_results)
    cols = min(3, n_results)
    rows = (n_results + cols - 1) // cols
    
    # Create beautiful figure
    if figsize is None:
        figsize = 'comparison_2x2' if n_results > 2 else 'comparison_2x1'
    
    fig = create_beautiful_figure(figsize=figsize, 
                                 title='Model vs Optimal Control Comparison')
    
    for i, result in enumerate(valid_results):
        ax = fig.add_subplot(rows, cols, i + 1)
        
        trajectories = {}
        
        if result.get('model_trajectory'):
            trajectories['Model'] = result['model_trajectory']
        
        if result.get('optimal_trajectory'):
            trajectories['Optimal'] = result['optimal_trajectory']
        
        system_name = result['system_name']
        initial_state = result['initial_state']
        
        # Get beautiful colors for trajectories
        trajectory_colors = get_color_scheme('comparison')
        
        # Plot each trajectory with beautiful styling
        for label, traj in trajectories.items():
            if traj is None:
                continue
                
            states = traj['states']
            
            # Get appropriate color
            if label.lower() in trajectory_colors:
                color = trajectory_colors[label.lower()]
            else:
                color = trajectory_colors['model'] if 'model' in label.lower() else trajectory_colors['baseline']
            
            # Use beautiful line plotting
            beautiful_line_plot(states[:, 0], states[:, 1], ax=ax, 
                              label=label, color=color)
            
            # Mark start and end points
            colors_scheme = get_color_scheme('trajectory')
            ax.scatter(states[0, 0], states[0, 1], s=80, 
                      c=colors_scheme['start'], marker='o', 
                      edgecolor='white', linewidth=1.5, zorder=5)
            ax.scatter(states[-1, 0], states[-1, 1], s=80, 
                      c=colors_scheme['end'], marker='s', 
                      edgecolor='white', linewidth=1.5, zorder=5)
        
        # Mark target with beautiful styling
        ax.scatter(0, 0, s=150, c=trajectory_colors['target'], 
                  marker='*', edgecolor='white', linewidth=2, 
                  label='Target', zorder=5)
        
        # Style the axes with grid and box
        title = f'{system_name.replace("_", " ").title()}\nInit: ({initial_state[0]:.2f}, {initial_state[1]:.2f})'
        style_axes(ax, title=title, xlabel='Position', ylabel='Velocity', 
                  grid=True, box=True)
        
        # Add performance annotation
        if result.get('model_trajectory'):
            final_error = result['model_trajectory']['final_error']
            from .plotting_utils import add_performance_annotation
            add_performance_annotation(ax, final_error, prefix="Final Error", 
                                     x_pos=0.02, y_pos=0.98)
        
        # Add beautiful legend
        add_beautiful_legend(ax, location='upper right')
    
    plt.tight_layout()
    return fig


def plot_metrics_comparison(batch_results: List[Dict[str, Any]],
                           metric_names: Optional[List[str]] = None,
                           figsize: Optional[str] = None) -> plt.Figure:
    """
    Plot comparison of metrics across multiple results with beautiful styling.
    
    Args:
        batch_results: List of results with computed metrics
        metric_names: List of metrics to plot
        figsize: Figure size key from FIGURE_SIZES or tuple
        
    Returns:
        matplotlib Figure object
    """
    # Setup beautiful plotting
    setup_beautiful_plotting()
    
    if metric_names is None:
        metric_names = ['final_error', 'lqr_cost', 'total_control_effort', 
                       'mean_control_change']
    
    from .metrics import compute_control_metrics
    
    # Extract metrics
    all_metrics = []
    labels = []
    
    for i, result in enumerate(batch_results):
        if (result.get('valid_format', False) and 
            result.get('model_trajectory') is not None and
            result.get('optimal_trajectory') is not None):
            
            metrics = compute_control_metrics(
                result['model_trajectory'],
                result['optimal_trajectory']
            )
            all_metrics.append(metrics['model_metrics'])
            
            system_name = result.get('system_name', 'Unknown')
            initial_state = result.get('initial_state', (0, 0))
            labels.append(f"{system_name}\n{initial_state}")
    
    if not all_metrics:
        print("No valid metrics to plot")
        return None
    
    # Create beautiful figure
    if figsize is None:
        figsize = 'comparison_2x2'
    
    fig = create_beautiful_figure(figsize=figsize, title='Performance Metrics Comparison')
    
    # Create subplots
    n_metrics = len(metric_names)
    cols = min(2, n_metrics)
    rows = (n_metrics + cols - 1) // cols
    
    axes = fig.subplots(rows, cols)
    if n_metrics == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    # Get beautiful colors for metrics
    colors_scheme = get_color_scheme('metrics')
    
    for i, metric_name in enumerate(metric_names):
        row, col = i // cols, i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        # Extract metric values
        values = [metrics.get(metric_name, 0) for metrics in all_metrics]
        
        # Create beautiful bar plot with performance-based coloring
        bar_colors = []
        for val in values:
            if metric_name.endswith('_error') or metric_name in ['lqr_cost']:
                # Lower is better - use performance gradient
                if val < 0.01:
                    bar_colors.append(colors_scheme['excellent'])
                elif val < 0.1:
                    bar_colors.append(colors_scheme['good'])
                else:
                    bar_colors.append(colors_scheme['poor'])
            else:
                # Neutral coloring for other metrics
                bar_colors.append(colors_scheme['neutral'])
        
        bars = ax.bar(range(len(values)), values, color=bar_colors, 
                     alpha=0.8, edgecolor='white', linewidth=1.5)
        
        # Style the axes with grid and box
        style_axes(ax, ylabel=metric_name.replace('_', ' ').title(), 
                  grid=True, box=True)
        
        # Set x-axis labels
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
        
        # Add enhanced grid and box
        add_grid_and_box(ax, grid_style='major', box=True)
        
        # Add value annotations on bars
        from .plotting_utils import add_value_annotations
        add_value_annotations(ax, bars, values, format_str='.3f')
    
    # Hide unused subplots
    for i in range(n_metrics, rows * cols):
        row, col = i // cols, i % cols
        if rows > 1:
            axes[row, col].set_visible(False)
        else:
            axes[col].set_visible(False)
    
    plt.tight_layout()
    return fig


def save_plots(figures: List[plt.Figure], filenames: List[str], 
              directory: str = "plots", dpi: int = 300):
    """Save multiple figures to files using beautiful plotting standards."""
    import os
    os.makedirs(directory, exist_ok=True)
    
    for fig, filename in zip(figures, filenames):
        filepath = os.path.join(directory, filename)
        # Use the beautiful plotting save function for consistent formatting
        save_beautiful_figure(fig, filepath, dpi=dpi)
        print(f"Saved plot to {filepath}")