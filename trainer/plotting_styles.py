"""
Professional plotting styles and utilities for control system visualizations.
Extracted and adapted from user-provided styling examples.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.patches import Circle
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec

# Color palettes for professional appearance
BEAUTIFUL_COLORS = {
    'primary': '#2E86AB',      # Deep blue
    'secondary': '#A23B72',    # Deep pink/purple
    'accent': '#F18F01',       # Warm orange
    'success': '#C73E1D',      # Deep red for emphasis
    
    # Control-specific color scheme
    'control_scheme': {
        'state': '#2E86AB',         # Blue for state
        'control': '#F18F01',       # Orange for control
        'reference': '#6A994E',     # Green for reference/target
        'model': '#2E86AB',         # Blue for model predictions
        'optimal': '#A23B72',       # Pink for optimal/LQR
        'bounds': '#C73E1D',        # Red for constraints
        'feasible': '#E8F5E9',      # Light green for feasible region
    },
    
    # Professional gradient colors for trajectories
    'gradient_professional': [
        '#1f77b4',  # Professional blue
        '#ff7f0e',  # Professional orange  
        '#2ca02c',  # Professional green
        '#d62728',  # Professional red
        '#9467bd',  # Professional purple
        '#8c564b',  # Professional brown
        '#e377c2',  # Professional pink
        '#7f7f7f',  # Professional gray
    ],
    
    # Neutral colors
    'light_gray': '#F5F5F5',
    'medium_gray': '#9E9E9E',
    'dark_gray': '#424242',
    'black': '#212121',
    'white': '#FFFFFF',
    'grid': '#BDBDBD',
}

# Professional typography settings
TYPOGRAPHY = {
    'title_size': 24,
    'subtitle_size': 20,
    'axis_label_size': 16,
    'tick_label_size': 14,
    'legend_size': 14,
    'annotation_size': 12,
    'small_text_size': 10,
    'annotation_weight': 'bold',
}

# Phase and waypoint colors
PHASE_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
PUBLICATION_COLORS = {
    'target': '#DC143C',           # Deep red for final targets
    'phase_transition': 'black',   # Black for phase transitions
    'action_bounds': 'red',        # Red for action bounds
    'grid': 'gray',                # Gray for grid lines
    'model': '#1f77b4',           # Professional blue for model
    'optimal': '#ff7f0e',         # Professional orange for optimal
}


def setup_beautiful_plotting():
    """Set up matplotlib for beautiful, publication-quality plots."""
    mpl.rcParams.update({
        # Figure settings
        'figure.facecolor': 'white',
        'figure.edgecolor': 'none',
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'none',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        
        # Font settings
        'font.size': TYPOGRAPHY['tick_label_size'],
        'axes.titlesize': TYPOGRAPHY['subtitle_size'],
        'axes.labelsize': TYPOGRAPHY['axis_label_size'],
        'xtick.labelsize': TYPOGRAPHY['tick_label_size'],
        'ytick.labelsize': TYPOGRAPHY['tick_label_size'],
        'legend.fontsize': TYPOGRAPHY['legend_size'],
        'figure.titlesize': TYPOGRAPHY['title_size'],
        
        # Axes settings
        'axes.facecolor': 'white',
        'axes.edgecolor': BEAUTIFUL_COLORS['dark_gray'],
        'axes.linewidth': 1.5,  # Thicker box lines
        'axes.grid': True,
        'axes.axisbelow': True,
        'axes.spines.top': True,     # Show top spine for box
        'axes.spines.right': True,   # Show right spine for box
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        
        # Grid settings - more visible
        'grid.color': BEAUTIFUL_COLORS['medium_gray'],
        'grid.linestyle': '-',       # Solid lines for better visibility
        'grid.linewidth': 0.5,       # Thinner lines
        'grid.alpha': 0.3,           # Lower alpha for subtlety
        
        # Line settings
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
        
        # Tick settings
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
        'xtick.color': BEAUTIFUL_COLORS['dark_gray'],
        'ytick.color': BEAUTIFUL_COLORS['dark_gray'],
        
        # Legend settings
        'legend.frameon': True,
        'legend.fancybox': True,
        'legend.shadow': False,
        'legend.framealpha': 0.9,
        'legend.facecolor': 'white',
        'legend.edgecolor': BEAUTIFUL_COLORS['medium_gray'],
        'legend.borderpad': 0.5,
        'legend.columnspacing': 1.0,
        'legend.handlelength': 2.0,
        'legend.handletextpad': 0.5,
    })


def style_axes(ax, title=None, xlabel=None, ylabel=None, grid=True, box=True):
    """Apply beautiful styling to a matplotlib axes object."""
    # Set labels and title
    if title:
        ax.set_title(title, fontsize=TYPOGRAPHY['subtitle_size'], 
                    fontweight='bold', pad=20,
                    color=BEAUTIFUL_COLORS['black'])
    
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=TYPOGRAPHY['axis_label_size'],
                     fontweight='normal', color=BEAUTIFUL_COLORS['dark_gray'])
    
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=TYPOGRAPHY['axis_label_size'],
                     fontweight='normal', color=BEAUTIFUL_COLORS['dark_gray'])
    
    # Configure spines - if box=True, show all four
    if box:
        for spine_name in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine_name].set_visible(True)
            ax.spines[spine_name].set_color(BEAUTIFUL_COLORS['dark_gray'])
            ax.spines[spine_name].set_linewidth(1.5)
    
    # Configure grid - enhanced visibility
    if grid:
        ax.grid(True, linestyle='-', alpha=0.3, linewidth=0.5,
               color=BEAUTIFUL_COLORS['medium_gray'])
        ax.set_axisbelow(True)
        
        # Add minor grid for precision
        ax.grid(True, which='minor', linestyle=':', alpha=0.2, linewidth=0.3,
               color=BEAUTIFUL_COLORS['medium_gray'])
    
    # Style ticks
    ax.tick_params(axis='both', which='major', 
                  labelsize=TYPOGRAPHY['tick_label_size'],
                  colors=BEAUTIFUL_COLORS['dark_gray'],
                  direction='out', length=6, width=1)
    ax.tick_params(axis='both', which='minor',
                  colors=BEAUTIFUL_COLORS['dark_gray'],
                  direction='out', length=3, width=0.8)
    
    # Enable minor ticks for better grid
    ax.minorticks_on()


def add_beautiful_legend(ax, title=None, location='best', ncol=1, **kwargs):
    """Add a beautiful, styled legend to the plot."""
    legend_params = {
        'title': title,
        'loc': location,
        'frameon': True, 
        'fancybox': True, 
        'shadow': False,
        'fontsize': TYPOGRAPHY['legend_size'],
        'title_fontsize': TYPOGRAPHY['legend_size'],
        'framealpha': 0.95, 
        'facecolor': 'white',
        'edgecolor': BEAUTIFUL_COLORS['medium_gray'],
        'borderpad': 0.5, 
        'columnspacing': 1.0,
        'handlelength': 2.0, 
        'handletextpad': 0.5,
        'ncol': ncol
    }
    
    # Allow user to override defaults
    legend_params.update(kwargs)
    
    legend = ax.legend(**legend_params)
    
    # Style legend title
    if title and legend.get_title():
        legend.get_title().set_fontweight('bold')
        legend.get_title().set_color(BEAUTIFUL_COLORS['black'])
    
    return legend


def add_subplot_labels(axes):
    """Add (a), (b), (c), (d) labels to subplots."""
    labels = ['(a) Phase Space', '(b) Position vs Time', '(c) Velocity vs Time', '(d) Control vs Time']
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    for i, (label, (row, col)) in enumerate(zip(labels, positions)):
        if isinstance(axes, np.ndarray):
            ax = axes[row, col] if len(axes.shape) == 2 else axes[i]
        else:
            ax = axes
        ax.text(0.02, 0.98, label, transform=ax.transAxes, fontsize=16,
                fontweight='bold', va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='lightgray', alpha=0.9))


def add_directional_arrows(ax, x, y, color='black', alpha=0.6, density=5):
    """Add directional arrows to phase space trajectory."""
    if len(x) < density * 2:
        return
    
    # Add arrows at regular intervals
    indices = np.linspace(density, len(x) - density, min(5, len(x)//density), dtype=int)
    for idx in indices:
        if idx < len(x) - 1:
            dx = x[idx+1] - x[idx]
            dy = y[idx+1] - y[idx]
            # Scale arrow for visibility
            scale = 0.3
            ax.arrow(x[idx], y[idx], dx*scale, dy*scale,
                    head_width=0.03, head_length=0.03,
                    fc=color, ec=color, alpha=alpha, zorder=3)


def add_performance_annotation(ax, metrics_text, position='right'):
    """Add a styled performance annotation box."""
    if position == 'right':
        x_pos, y_pos = 0.98, 0.5
        ha, va = 'right', 'center'
    else:  # top-right
        x_pos, y_pos = 0.95, 0.95
        ha, va = 'right', 'top'
    
    ax.text(x_pos, y_pos, metrics_text,
            transform=ax.figure.transFigure,
            ha=ha, va=va,
            fontsize=TYPOGRAPHY['annotation_size'],
            fontweight=TYPOGRAPHY['annotation_weight'],
            color=BEAUTIFUL_COLORS['black'],
            bbox=dict(boxstyle='round,pad=0.5',
                      facecolor=BEAUTIFUL_COLORS['light_gray'],
                      edgecolor=BEAUTIFUL_COLORS['medium_gray'],
                      alpha=0.9))


def save_beautiful_figure(fig, filename, formats=['pdf', 'png'], dpi=300):
    """Save figure in multiple formats with beautiful settings."""
    import os
    base_filename = os.path.splitext(filename)[0]
    
    saved_files = []
    for fmt in formats:
        full_filename = f"{base_filename}.{fmt}"
        
        fig.savefig(full_filename, format=fmt, dpi=dpi, 
                   bbox_inches='tight', facecolor='white', 
                   edgecolor='none', pad_inches=0.1)
        
        saved_files.append(full_filename)
    
    return saved_files