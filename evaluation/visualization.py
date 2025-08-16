"""Visualization utilities for control trajectories."""

import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
from environments import get_system
from .plotting_utils import (
    setup_beautiful_plotting, create_beautiful_figure, style_axes,
    beautiful_line_plot, add_beautiful_legend, get_color_scheme,
    plot_phase_portrait, add_control_bounds, add_grid_and_box,
    save_beautiful_figure, FIGURE_SIZES, create_publication_figure,
    add_subplot_labels, plot_phase_space_subplot, PUBLICATION_COLORS,
    PHASE_COLORS, add_directional_arrows, create_enhanced_comparison_figure,
    add_enhanced_subplot_labels, plot_enhanced_phase_space_subplot
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
    
    # Create beautiful figure with single subplot for all trajectories
    if figsize is None:
        figsize = 'comparison_single'
    
    fig = create_beautiful_figure(figsize=figsize, 
                                 title='Model vs Optimal Control Comparison (All Test Cases)')
    
    # Create single axis for all trajectories
    ax = fig.add_subplot(1, 1, 1)
    
    # Get beautiful colors for trajectories
    trajectory_colors = get_color_scheme('comparison')
    model_color = trajectory_colors.get('model', '#2E86AB')
    optimal_color = trajectory_colors.get('baseline', '#A23B72')
    
    # Track if we've added legend labels
    model_legend_added = False
    optimal_legend_added = False
    
    for i, result in enumerate(valid_results):
        trajectories = {}
        
        if result.get('model_trajectory'):
            trajectories['Model'] = result['model_trajectory']
        
        if result.get('optimal_trajectory'):
            trajectories['Optimal'] = result['optimal_trajectory']
        
        system_name = result['system_name']
        initial_state = result['initial_state']
        
        # Plot each trajectory with beautiful styling
        for label, traj in trajectories.items():
            if traj is None:
                continue
                
            states = traj['states']
            
            # Use consistent colors and only add legend once per type
            if label == 'Model':
                color = model_color
                legend_label = label if not model_legend_added else None
                model_legend_added = True
                alpha = 0.7
            else:  # Optimal
                color = optimal_color
                legend_label = label if not optimal_legend_added else None
                optimal_legend_added = True
                alpha = 0.5
            
            # Use beautiful line plotting
            beautiful_line_plot(states[:, 0], states[:, 1], ax=ax, 
                              label=legend_label, color=color, alpha=alpha, linewidth=1.5)
            
            # Mark start and end points with smaller markers for clarity
            colors_scheme = get_color_scheme('trajectory')
            ax.scatter(states[0, 0], states[0, 1], s=40, 
                      c=colors_scheme['start'], marker='o', 
                      edgecolor='white', linewidth=1, zorder=4, alpha=0.8)
            ax.scatter(states[-1, 0], states[-1, 1], s=40, 
                      c=colors_scheme['end'], marker='s', 
                      edgecolor='white', linewidth=1, zorder=4, alpha=0.8)
    
    # Mark target with beautiful styling (only once, outside the loop)
    ax.scatter(0, 0, s=200, c=trajectory_colors.get('target', '#F18F01'), 
              marker='*', edgecolor='white', linewidth=2, 
              label='Target', zorder=6)
    
    # Style the axes with grid and box
    system_name = valid_results[0]['system_name']  # All should be same system
    title = f'{system_name.replace("_", " ").title()} Control Trajectories\n({n_results} test cases)'
    style_axes(ax, title=title, xlabel='Position', ylabel='Velocity', 
              grid=True, box=True)
    
    # Add system bounds if available
    if valid_results:
        try:
            system = get_system(system_name)()
            state_bounds = system.get_state_bounds()
            ax.set_xlim(state_bounds[0])
            ax.set_ylim(state_bounds[1])
        except:
            pass
    
    # Add performance summary
    model_errors = [r['model_trajectory']['final_error'] for r in valid_results 
                   if r.get('model_trajectory') and 'final_error' in r['model_trajectory']]
    if model_errors:
        mean_error = sum(model_errors) / len(model_errors)
        from .plotting_utils import add_performance_annotation
        add_performance_annotation(ax, mean_error, prefix="Mean Final Error", 
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
        # NOTE: add_value_annotations not implemented yet
        # from .plotting_utils import add_value_annotations
        # add_value_annotations(ax, bars, values, format_str='.3f')
        
        # Simple value annotation
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}',
                    ha='center', va='bottom', fontsize=10)
    
    # Hide unused subplots
    for i in range(n_metrics, rows * cols):
        row, col = i // cols, i % cols
        if rows > 1:
            axes[row, col].set_visible(False)
        else:
            axes[col].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_publication_comparison(trajectories: Dict[str, Dict[str, Any]], 
                               system_name: str,
                               initial_state: Optional[Tuple[float, float]] = None,
                               title: Optional[str] = None,
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Create publication-ready 4-subplot comparison plot (phase space, position, velocity, control).
    
    Args:
        trajectories: Dictionary of {label: trajectory_dict} (e.g., {'Model': traj1, 'Optimal': traj2})
        system_name: Name of the control system
        initial_state: Initial state for title
        title: Optional custom title
        save_path: Optional path to save the figure (without extension)
        
    Returns:
        matplotlib Figure object
    """
    # Create publication figure with 2x2 layout
    fig, axes = create_publication_figure()
    
    # Get system information
    system = get_system(system_name)()
    state_bounds = system.get_state_bounds()
    control_bounds = system.get_control_bounds()
    state_names = getattr(system, 'get_state_names', lambda: ['Position', 'Velocity'])()
    
    # Define subplot configuration
    subplot_config = [
        (0, 0, 'Phase Space', 'Position', 'Velocity'),
        (0, 1, f'{state_names[0]} vs Time', 'Time (s)', state_names[0]),
        (1, 0, f'{state_names[1]} vs Time', 'Time (s)', state_names[1]),
        (1, 1, 'Control vs Time', 'Time (s)', 'Control u(t)')
    ]
    
    colors = get_color_scheme('comparison')
    
    for i, (row, col, subplot_title, xlabel, ylabel) in enumerate(subplot_config):
        ax = axes[row, col]
        
        if i == 0:  # Phase space plot
            plot_phase_space_subplot(ax, trajectories, title=subplot_title, show_arrows=True)
            
        else:  # Time series plots
            for j, (label, traj) in enumerate(trajectories.items()):
                if traj is None:
                    continue
                    
                states = traj['states']
                controls = traj['controls']
                times = traj['times']
                
                # Get appropriate color and style
                if label.lower() == 'model':
                    color = colors['model']
                    linewidth = 2.5
                    alpha = 0.8
                    linestyle = '-'
                elif label.lower() in ['optimal', 'lqr']:
                    color = colors['lqr']
                    linewidth = 2.5
                    alpha = 0.7
                    linestyle = '--'
                else:
                    color = PHASE_COLORS[j % len(PHASE_COLORS)]
                    linewidth = 2.5
                    alpha = 0.8
                    linestyle = '-'
                
                if i == 1:  # Position vs time
                    ax.plot(times, states[:, 0], color=color, linewidth=linewidth, 
                           alpha=alpha, linestyle=linestyle, label=label)
                    
                elif i == 2:  # Velocity vs time  
                    ax.plot(times, states[:, 1], color=color, linewidth=linewidth,
                           alpha=alpha, linestyle=linestyle, label=label)
                    
                elif i == 3:  # Control vs time
                    control_times = times[:-1]  # Controls are one step shorter
                    ax.plot(control_times, controls, color=color, linewidth=linewidth,
                           alpha=alpha, linestyle=linestyle, label=label)
            
            # Add phase transitions (dotted black lines at specific times if needed)
            if i > 0:  # For time series plots
                # Add phase transition markers if available in trajectory data
                for label, traj in trajectories.items():
                    if traj and 'phase_times' in traj:
                        for phase_time in traj['phase_times']:
                            ax.axvline(phase_time, color=PUBLICATION_COLORS['phase_transition'], 
                                     linestyle=':', linewidth=2, alpha=0.7)
            
            # Add bounds and styling
            if i == 1 and state_bounds and len(state_bounds) >= 1:  # Position bounds
                bounds = state_bounds[0]
                ax.axhline(bounds[0], color=PUBLICATION_COLORS['action_bounds'], 
                          linestyle='--', linewidth=2.5, alpha=0.8)
                ax.axhline(bounds[1], color=PUBLICATION_COLORS['action_bounds'], 
                          linestyle='--', linewidth=2.5, alpha=0.8)
                          
            elif i == 2 and state_bounds and len(state_bounds) >= 2:  # Velocity bounds
                bounds = state_bounds[1] 
                ax.axhline(bounds[0], color=PUBLICATION_COLORS['action_bounds'], 
                          linestyle='--', linewidth=2.5, alpha=0.8)
                ax.axhline(bounds[1], color=PUBLICATION_COLORS['action_bounds'], 
                          linestyle='--', linewidth=2.5, alpha=0.8)
                          
            elif i == 3 and control_bounds:  # Control bounds
                ax.axhline(control_bounds[0], color=PUBLICATION_COLORS['action_bounds'], 
                          linestyle='--', linewidth=2.5, alpha=0.8, label='Control Bounds')
                ax.axhline(control_bounds[1], color=PUBLICATION_COLORS['action_bounds'], 
                          linestyle='--', linewidth=2.5, alpha=0.8)
            
            # Style axes
            style_axes(ax, title=subplot_title, xlabel=xlabel, ylabel=ylabel, 
                      grid=True, box=True)
            add_grid_and_box(ax, grid_style='both', box=True)
            
            # Add legends strategically
            if i == 1:  # Position plot - show phases
                legend_items = [line for line in ax.get_lines() if line.get_label() and 'Model' in line.get_label()]
                if legend_items:
                    add_beautiful_legend(ax, location='best')
                    
            elif i == 2:  # Velocity plot - show transitions
                # Add legend for phase transitions if present
                phase_lines = [line for line in ax.get_lines() if line.get_linestyle() == ':']
                if phase_lines:
                    ax.legend(['Phase Transition'], loc='best', fontsize=16,
                             frameon=True, framealpha=0.9, facecolor='white',
                             edgecolor='gray')
                             
            elif i == 3:  # Control plot - show bounds
                bounds_lines = [line for line in ax.get_lines() if 'Bounds' in str(line.get_label())]
                if bounds_lines:
                    add_beautiful_legend(ax, location='best')
    
    # Add subplot labels
    add_subplot_labels(axes)
    
    # Set overall title
    if title is None:
        if initial_state:
            title = f'{system_name.replace("_", " ").title()} Control Comparison\nInitial State: {initial_state}'
        else:
            title = f'{system_name.replace("_", " ").title()} Control Comparison'
    
    fig.suptitle(title, fontsize=20, fontweight='bold', y=0.95)
    
    # Final layout adjustment
    fig.canvas.draw()
    plt.tight_layout(pad=4, h_pad=3.0, w_pad=3.0)
    
    # Save if path provided
    if save_path:
        save_beautiful_figure(fig, save_path, formats=['pdf', 'png'])
    
    return fig


def plot_model_only_trajectories(trajectories: Dict[str, Dict[str, Any]], 
                                 system_name: str,
                                 title: Optional[str] = None,
                                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Create publication-ready plots for model trajectories without optimal baseline.
    
    Args:
        trajectories: Dictionary of {label: trajectory_dict} 
        system_name: Name of the control system
        title: Optional custom title
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    # Create publication figure
    fig, axes = create_publication_figure()
    
    # Get system information
    system = get_system(system_name)()
    state_bounds = system.get_state_bounds()
    control_bounds = system.get_control_bounds()
    state_names = getattr(system, 'get_state_names', lambda: ['Position', 'Velocity'])()
    
    # Plot configuration same as comparison version
    subplot_config = [
        (0, 0, 'Phase Space', 'Position', 'Velocity'),
        (0, 1, f'{state_names[0]} vs Time', 'Time (s)', state_names[0]),
        (1, 0, f'{state_names[1]} vs Time', 'Time (s)', state_names[1]),
        (1, 1, 'Control vs Time', 'Time (s)', 'Control u(t)')
    ]
    
    # Use model-focused colors
    model_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (row, col, subplot_title, xlabel, ylabel) in enumerate(subplot_config):
        ax = axes[row, col]
        
        if i == 0:  # Phase space plot
            plot_phase_space_subplot(ax, trajectories, title=subplot_title, show_arrows=True)
            
        else:  # Time series plots
            for j, (label, traj) in enumerate(trajectories.items()):
                if traj is None:
                    continue
                    
                states = traj['states']
                controls = traj['controls']
                times = traj['times']
                
                color = model_colors[j % len(model_colors)]
                
                if i == 1:  # Position vs time
                    ax.plot(times, states[:, 0], color=color, linewidth=2.5, 
                           alpha=0.8, label=label)
                elif i == 2:  # Velocity vs time  
                    ax.plot(times, states[:, 1], color=color, linewidth=2.5,
                           alpha=0.8, label=label)
                elif i == 3:  # Control vs time
                    control_times = times[:-1]
                    ax.plot(control_times, controls, color=color, linewidth=2.5,
                           alpha=0.8, label=label)
            
            # Add bounds (same as comparison version)
            if i == 1 and state_bounds and len(state_bounds) >= 1:
                bounds = state_bounds[0]
                ax.axhline(bounds[0], color=PUBLICATION_COLORS['action_bounds'], 
                          linestyle='--', linewidth=2.5, alpha=0.8)
                ax.axhline(bounds[1], color=PUBLICATION_COLORS['action_bounds'], 
                          linestyle='--', linewidth=2.5, alpha=0.8)
            elif i == 2 and state_bounds and len(state_bounds) >= 2:
                bounds = state_bounds[1]
                ax.axhline(bounds[0], color=PUBLICATION_COLORS['action_bounds'], 
                          linestyle='--', linewidth=2.5, alpha=0.8)
                ax.axhline(bounds[1], color=PUBLICATION_COLORS['action_bounds'], 
                          linestyle='--', linewidth=2.5, alpha=0.8)
            elif i == 3 and control_bounds:
                ax.axhline(control_bounds[0], color=PUBLICATION_COLORS['action_bounds'], 
                          linestyle='--', linewidth=2.5, alpha=0.8, label='Bounds')
                ax.axhline(control_bounds[1], color=PUBLICATION_COLORS['action_bounds'], 
                          linestyle='--', linewidth=2.5, alpha=0.8)
            
            # Style axes and add legends
            style_axes(ax, title=subplot_title, xlabel=xlabel, ylabel=ylabel, 
                      grid=True, box=True)
            add_grid_and_box(ax, grid_style='both', box=True)
            add_beautiful_legend(ax, location='best')
    
    # Add subplot labels
    add_subplot_labels(axes)
    
    # Set title
    if title is None:
        title = f'{system_name.replace("_", " ").title()} Model Trajectories'
    fig.suptitle(title, fontsize=20, fontweight='bold', y=0.95)
    
    # Final layout
    fig.canvas.draw()
    plt.tight_layout(pad=4, h_pad=3.0, w_pad=3.0)
    
    if save_path:
        save_beautiful_figure(fig, save_path, formats=['pdf', 'png'])
    
    return fig


def create_individual_phase_plot(trajectories: Dict[str, Dict[str, Any]], 
                                system_name: str,
                                figsize: Tuple[int, int] = (8, 8),
                                save_path: Optional[str] = None) -> plt.Figure:
    """Create standalone phase space plot."""
    setup_beautiful_plotting()
    fig, ax = plt.subplots(figsize=figsize)
    
    plot_phase_space_subplot(ax, trajectories)
    
    # Style and finalize
    plt.tight_layout()
    fig.canvas.draw()
    
    if save_path:
        save_beautiful_figure(fig, save_path, formats=['pdf', 'png'])
    
    return fig


def create_individual_state_plot(trajectories: Dict[str, Dict[str, Any]], 
                                 system_name: str,
                                 state_index: int = 0,
                                 figsize: Tuple[int, int] = (10, 6),
                                 save_path: Optional[str] = None) -> plt.Figure:
    """Create standalone state vs time plot."""
    setup_beautiful_plotting()
    fig, ax = plt.subplots(figsize=figsize)
    
    system = get_system(system_name)()
    state_names = getattr(system, 'get_state_names', lambda: ['Position', 'Velocity'])()
    state_bounds = system.get_state_bounds()
    
    colors = get_color_scheme('comparison')
    
    for j, (label, traj) in enumerate(trajectories.items()):
        if traj is None:
            continue
            
        states = traj['states']
        times = traj['times']
        
        # Get appropriate color and style
        if label.lower() == 'model':
            color = colors['model']
            linestyle = '-'
        elif label.lower() in ['optimal', 'lqr']:
            color = colors['lqr']
            linestyle = '--'
        else:
            color = PHASE_COLORS[j % len(PHASE_COLORS)]
            linestyle = '-'
        
        ax.plot(times, states[:, state_index], color=color, linewidth=2.5,
               alpha=0.8, linestyle=linestyle, label=label)
    
    # Add bounds
    if state_bounds and len(state_bounds) > state_index:
        bounds = state_bounds[state_index]
        ax.axhline(bounds[0], color=PUBLICATION_COLORS['action_bounds'], 
                  linestyle='--', linewidth=2.5, alpha=0.8)
        ax.axhline(bounds[1], color=PUBLICATION_COLORS['action_bounds'], 
                  linestyle='--', linewidth=2.5, alpha=0.8)
    
    # Style and finalize
    state_name = state_names[state_index] if state_index < len(state_names) else f'State {state_index}'
    style_axes(ax, title=f'{state_name} vs Time', 
              xlabel='Time (s)', ylabel=state_name, grid=True, box=True)
    add_grid_and_box(ax, grid_style='both', box=True)
    add_beautiful_legend(ax, location='best')
    
    plt.tight_layout()
    fig.canvas.draw()
    
    if save_path:
        save_beautiful_figure(fig, save_path, formats=['pdf', 'png'])
    
    return fig


def create_individual_control_plot(trajectories: Dict[str, Dict[str, Any]], 
                                  system_name: str,
                                  figsize: Tuple[int, int] = (10, 6),
                                  save_path: Optional[str] = None) -> plt.Figure:
    """Create standalone control vs time plot."""
    setup_beautiful_plotting()
    fig, ax = plt.subplots(figsize=figsize)
    
    system = get_system(system_name)()
    control_bounds = system.get_control_bounds()
    
    colors = get_color_scheme('comparison')
    
    for j, (label, traj) in enumerate(trajectories.items()):
        if traj is None:
            continue
            
        controls = traj['controls']
        times = traj['times']
        control_times = times[:-1]  # Controls are one step shorter
        
        # Get appropriate color and style
        if label.lower() == 'model':
            color = colors['model']
            linestyle = '-'
        elif label.lower() in ['optimal', 'lqr']:
            color = colors['lqr']
            linestyle = '--'
        else:
            color = PHASE_COLORS[j % len(PHASE_COLORS)]
            linestyle = '-'
        
        ax.plot(control_times, controls, color=color, linewidth=2.5,
               alpha=0.8, linestyle=linestyle, label=label)
    
    # Add control bounds
    if control_bounds:
        ax.axhline(control_bounds[0], color=PUBLICATION_COLORS['action_bounds'], 
                  linestyle='--', linewidth=2.5, alpha=0.8, label='Control Bounds')
        ax.axhline(control_bounds[1], color=PUBLICATION_COLORS['action_bounds'], 
                  linestyle='--', linewidth=2.5, alpha=0.8)
    
    # Style and finalize
    style_axes(ax, title='Control vs Time', 
              xlabel='Time (s)', ylabel='Control u(t)', grid=True, box=True)
    add_grid_and_box(ax, grid_style='both', box=True)
    add_beautiful_legend(ax, location='best')
    
    plt.tight_layout()
    fig.canvas.draw()
    
    if save_path:
        save_beautiful_figure(fig, save_path, formats=['pdf', 'png'])
    
    return fig


def generate_publication_plots(trajectories: Dict[str, Dict[str, Any]], 
                              system_name: str,
                              initial_state: Optional[Tuple[float, float]] = None,
                              base_filename: str = "control_analysis",
                              plot_dir: str = "plots") -> List[str]:
    """
    Generate complete set of publication-ready plots.
    
    Args:
        trajectories: Dictionary of trajectory data
        system_name: Name of control system
        initial_state: Initial state for titles
        base_filename: Base name for saved files
        plot_dir: Directory to save plots
        
    Returns:
        List of saved file paths
    """
    import os
    os.makedirs(plot_dir, exist_ok=True)
    
    saved_files = []
    
    # Main 4-subplot comparison
    fig_main = plot_publication_comparison(trajectories, system_name, initial_state)
    main_path = os.path.join(plot_dir, f"{base_filename}_comparison")
    save_beautiful_figure(fig_main, main_path, formats=['pdf', 'png'])
    saved_files.extend([f"{main_path}.pdf", f"{main_path}.png"])
    plt.close(fig_main)
    
    # Individual plots
    # Phase space
    fig_phase = create_individual_phase_plot(trajectories, system_name)
    phase_path = os.path.join(plot_dir, f"{base_filename}_phase_space")
    save_beautiful_figure(fig_phase, phase_path, formats=['pdf', 'png'])
    saved_files.extend([f"{phase_path}.pdf", f"{phase_path}.png"])
    plt.close(fig_phase)
    
    # Position plot
    fig_pos = create_individual_state_plot(trajectories, system_name, state_index=0)
    pos_path = os.path.join(plot_dir, f"{base_filename}_position")
    save_beautiful_figure(fig_pos, pos_path, formats=['pdf', 'png'])
    saved_files.extend([f"{pos_path}.pdf", f"{pos_path}.png"])
    plt.close(fig_pos)
    
    # Velocity plot
    fig_vel = create_individual_state_plot(trajectories, system_name, state_index=1)
    vel_path = os.path.join(plot_dir, f"{base_filename}_velocity")
    save_beautiful_figure(fig_vel, vel_path, formats=['pdf', 'png'])
    saved_files.extend([f"{vel_path}.pdf", f"{vel_path}.png"])
    plt.close(fig_vel)
    
    # Control plot
    fig_ctrl = create_individual_control_plot(trajectories, system_name)
    ctrl_path = os.path.join(plot_dir, f"{base_filename}_control")
    save_beautiful_figure(fig_ctrl, ctrl_path, formats=['pdf', 'png'])
    saved_files.extend([f"{ctrl_path}.pdf", f"{ctrl_path}.png"])
    plt.close(fig_ctrl)
    
    return saved_files


def plot_control_comparison(trajectories: Dict[str, Dict[str, Any]], 
                           system_name: str,
                           initial_state: Optional[Tuple[float, float]] = None,
                           title: Optional[str] = None,
                           save_path: Optional[str] = None,
                           show_comparison: bool = True) -> plt.Figure:
    """
    Create enhanced 4-subplot control comparison plot with publication-ready styling.
    
    This is the main enhanced plotting function that replaces individual plotting
    methods with a unified, professional visualization approach.
    
    Args:
        trajectories: Dictionary of {label: trajectory_dict} (e.g., {'Model': traj1, 'Optimal': traj2})
        system_name: Name of the control system
        initial_state: Initial state for title
        title: Optional custom title
        save_path: Optional path to save the figure (without extension)
        show_comparison: If True, show model vs optimal comparison; if False, show model-only plots
        
    Returns:
        matplotlib Figure object
    """
    # Create enhanced comparison figure with 2x2 layout
    if title is None:
        if show_comparison and len(trajectories) > 1:
            title = f'{system_name.replace("_", " ").title()} Control Comparison'
        else:
            title = f'{system_name.replace("_", " ").title()} Control Analysis'
        
        if initial_state:
            title += f'\nInitial State: {initial_state}'
    
    fig, axes = create_enhanced_comparison_figure(title=title)
    
    # Get system information
    system = get_system(system_name)()
    state_bounds = system.get_state_bounds()
    control_bounds = system.get_control_bounds()
    state_names = getattr(system, 'get_state_names', lambda: ['Position', 'Velocity'])()
    
    # Define subplot configuration
    subplot_config = [
        (0, 0, 'Phase Space', 'Position', 'Velocity'),
        (0, 1, f'{state_names[0]} vs Time', 'Time (s)', state_names[0]),
        (1, 0, f'{state_names[1]} vs Time', 'Time (s)', state_names[1]),
        (1, 1, 'Control vs Time', 'Time (s)', 'Control u(t)')
    ]
    
    colors = get_color_scheme('comparison')
    
    for i, (row, col, subplot_title, xlabel, ylabel) in enumerate(subplot_config):
        ax = axes[row, col]
        
        if i == 0:  # Phase space plot - use enhanced version
            plot_enhanced_phase_space_subplot(ax, trajectories, title=subplot_title, 
                                            show_arrows=True, show_target=True)
            
        else:  # Time series plots
            for j, (label, traj) in enumerate(trajectories.items()):
                if traj is None:
                    continue
                    
                states = traj['states']
                controls = traj['controls']
                times = traj['times']
                
                # Get appropriate color and style based on label
                if label.lower() == 'model':
                    color = colors['model']
                    linewidth = 2.5
                    alpha = 0.8
                    linestyle = '-'
                elif label.lower() in ['optimal', 'lqr']:
                    color = colors['lqr']
                    linewidth = 2.5
                    alpha = 0.7
                    linestyle = '--'
                else:
                    color = PHASE_COLORS[j % len(PHASE_COLORS)]
                    linewidth = 2.5
                    alpha = 0.8
                    linestyle = '-'
                
                if i == 1:  # Position vs time
                    ax.plot(times, states[:, 0], color=color, linewidth=linewidth, 
                           alpha=alpha, linestyle=linestyle, label=label)
                    
                elif i == 2:  # Velocity vs time  
                    ax.plot(times, states[:, 1], color=color, linewidth=linewidth,
                           alpha=alpha, linestyle=linestyle, label=label)
                    
                elif i == 3:  # Control vs time
                    control_times = times[:-1]  # Controls are one step shorter
                    ax.plot(control_times, controls, color=color, linewidth=linewidth,
                           alpha=alpha, linestyle=linestyle, label=label)
            
            # Add phase transitions if available in trajectory data
            if i > 0:  # For time series plots
                for label, traj in trajectories.items():
                    if traj and 'phase_times' in traj:
                        for phase_time in traj['phase_times']:
                            ax.axvline(phase_time, color=PUBLICATION_COLORS['phase_transition'], 
                                     linestyle=':', linewidth=2, alpha=0.7)
            
            # Add system bounds with enhanced styling
            if i == 1 and state_bounds and len(state_bounds) >= 1:  # Position bounds
                bounds = state_bounds[0]
                ax.axhline(bounds[0], color=PUBLICATION_COLORS['action_bounds'], 
                          linestyle='--', linewidth=2.5, alpha=0.8)
                ax.axhline(bounds[1], color=PUBLICATION_COLORS['action_bounds'], 
                          linestyle='--', linewidth=2.5, alpha=0.8)
                          
            elif i == 2 and state_bounds and len(state_bounds) >= 2:  # Velocity bounds
                bounds = state_bounds[1] 
                ax.axhline(bounds[0], color=PUBLICATION_COLORS['action_bounds'], 
                          linestyle='--', linewidth=2.5, alpha=0.8)
                ax.axhline(bounds[1], color=PUBLICATION_COLORS['action_bounds'], 
                          linestyle='--', linewidth=2.5, alpha=0.8)
                          
            elif i == 3 and control_bounds:  # Control bounds
                ax.axhline(control_bounds[0], color=PUBLICATION_COLORS['action_bounds'], 
                          linestyle='--', linewidth=2.5, alpha=0.8, label='Control Bounds')
                ax.axhline(control_bounds[1], color=PUBLICATION_COLORS['action_bounds'], 
                          linestyle='--', linewidth=2.5, alpha=0.8)
            
            # Enhanced axis styling
            style_axes(ax, title=subplot_title, xlabel=xlabel, ylabel=ylabel, 
                      grid=True, box=True)
            add_grid_and_box(ax, grid_style='both', box=True)
            
            # Strategic legend placement
            if i == 1:  # Position plot - show model vs optimal
                model_lines = [line for line in ax.get_lines() 
                              if 'model' in str(line.get_label()).lower()]
                if model_lines and show_comparison:
                    add_beautiful_legend(ax, location='best')
                    
            elif i == 2:  # Velocity plot - show phase transitions if present
                phase_lines = [line for line in ax.get_lines() if line.get_linestyle() == ':']
                if phase_lines:
                    # Custom legend for phase transitions
                    ax.legend(['Phase Transition'], loc='best', fontsize=16,
                             frameon=True, framealpha=0.9, facecolor='white',
                             edgecolor='gray')
                             
            elif i == 3:  # Control plot - show bounds
                bounds_lines = [line for line in ax.get_lines() 
                               if 'bounds' in str(line.get_label()).lower()]
                if bounds_lines:
                    add_beautiful_legend(ax, location='best')
    
    # Add enhanced subplot labels
    add_enhanced_subplot_labels(axes)
    
    # Final layout adjustment
    fig.canvas.draw()
    plt.tight_layout(pad=4, h_pad=3.0, w_pad=3.0)
    
    # Save if path provided
    if save_path:
        save_beautiful_figure(fig, save_path, formats=['pdf', 'png'])
    
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


def plot_all_trajectories_comparison(results, system_name, save_path=None):
    """
    Create a single comparison figure with all trajectories.
    
    This function was moved from plot_all_trajectories.py to consolidate 
    plotting functionality in the evaluation module.
    
    Args:
        results: List of evaluation results
        system_name: Name of the control system
        save_path: Path to save the figure
    """
    import numpy as np
    setup_beautiful_plotting()
    
    # Filter valid results
    valid_results = [r for r in results if r.get('valid_format', False)]
    if not valid_results:
        print("No valid results to plot")
        return None
    
    n_cases = len(valid_results)
    
    # Create figure with 2x2 subplots (position, velocity, phase space, control)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Colors for different trajectories
    colors = plt.cm.tab10(np.linspace(0, 1, n_cases))
    
    for i, result in enumerate(valid_results):
        # Handle trajectory data structure - simulate_trajectory returns a dict
        model_traj_data = result['model_trajectory']
        optimal_traj_data = result['optimal_trajectory']
        
        # Extract states array from trajectory dict
        if isinstance(model_traj_data, dict) and 'states' in model_traj_data:
            model_traj = model_traj_data['states']
        else:
            model_traj = model_traj_data
            
        if isinstance(optimal_traj_data, dict) and 'states' in optimal_traj_data:
            optimal_traj = optimal_traj_data['states']
        else:
            optimal_traj = optimal_traj_data
        
        # Generate time steps based on trajectory length
        time_steps = list(range(len(optimal_traj)))
        initial_state = result['initial_state']
        
        # Extract data - handle both numpy arrays and lists
        if hasattr(optimal_traj, 'shape'):  # numpy array
            x1_opt = optimal_traj[:, 0].tolist()
            x2_opt = optimal_traj[:, 1].tolist()
        else:  # list of states
            x1_opt = [s[0] for s in optimal_traj]
            x2_opt = [s[1] for s in optimal_traj]
            
        if model_traj is not None:
            if hasattr(model_traj, 'shape'):  # numpy array
                x1_model = model_traj[:, 0].tolist()
                x2_model = model_traj[:, 1].tolist()
            else:  # list of states
                x1_model = [s[0] for s in model_traj]
                x2_model = [s[1] for s in model_traj]
        else:
            # If model trajectory is None, use same length as optimal but with zeros
            x1_model = [0] * len(x1_opt)
            x2_model = [0] * len(x2_opt)
        
        # Handle controls if available
        u_opt = result.get('optimal_controls', [])
        u_model = result.get('model_controls', [])
        
        if u_opt and len(u_opt) > 0:
            # Create time array that matches control length
            control_len = len(u_opt)
            time_control = list(range(control_len))
            
            # Ensure model controls match optimal controls length for plotting
            if u_model and len(u_model) != control_len:
                if len(u_model) > control_len:
                    u_model = u_model[:control_len]
                else:
                    u_model = u_model + [0] * (control_len - len(u_model))
            elif not u_model:
                # If no model controls, create zeros for comparison
                u_model = [0] * control_len
        else:
            u_model = None
            u_opt = None
            time_control = None
        
        # Position subplot
        ax = axes[0]
        ax.plot(time_steps, x1_model, '-', color=colors[i], linewidth=2, 
                label=f'Model {i+1}' if i == 0 else None)
        ax.plot(time_steps, x1_opt, '--', color=colors[i], linewidth=2, 
                label=f'Optimal {i+1}' if i == 0 else None, alpha=0.7)
        
        # Velocity subplot
        ax = axes[1]
        ax.plot(time_steps, x2_model, '-', color=colors[i], linewidth=2)
        ax.plot(time_steps, x2_opt, '--', color=colors[i], linewidth=2, alpha=0.7)
        
        # Phase space subplot
        ax = axes[2]
        ax.plot(x1_model, x2_model, '-', color=colors[i], linewidth=2)
        ax.plot(x1_opt, x2_opt, '--', color=colors[i], linewidth=2, alpha=0.7)
        ax.plot(initial_state[0], initial_state[1], 'o', color=colors[i], 
                markersize=8, markeredgecolor='black', markeredgewidth=1)
        ax.plot(0, 0, 'x', color='red', markersize=10, markeredgewidth=3)
        
        # Control subplot
        if u_model is not None and u_opt is not None:
            ax = axes[3]
            ax.plot(time_control, u_model, '-', color=colors[i], linewidth=2)
            ax.plot(time_control, u_opt, '--', color=colors[i], linewidth=2, alpha=0.7)
    
    # Configure subplots
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Position')
    
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Velocity')
    
    axes[2].set_xlabel('Position')
    axes[2].set_ylabel('Velocity')
    
    axes[3].set_xlabel('Time [s]')
    axes[3].set_ylabel('Control')
    
    # Add legend only to first subplot
    axes[0].legend(loc='best', frameon=True, fancybox=True, shadow=True)
    
    # Add grid to all subplots
    for ax in axes:
        add_grid_and_box(ax, grid_style='both', box=True)
    
    # Add centralized subplot labels
    for i, label in enumerate(['(a)', '(b)', '(c)', '(d)']):
        axes[i].text(0.05, 0.95, label, transform=axes[i].transAxes,
                    fontsize=12, fontweight='bold', va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             edgecolor='gray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison figure to {save_path}")
    
    return fig