"""
Beautiful plotting styles and utilities for AI/ML research figures.

JUNIOR NOTE: This module centralizes all styling to ensure consistent, 
publication-quality figures across all plotting scripts.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator, SymmetricalLogLocator, MaxNLocator
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

# JUNIOR NOTE: Color palettes designed for accessibility and visual appeal
BEAUTIFUL_COLORS = {
    # Primary colors - vibrant but professional
    'primary': '#2E86AB',      # Deep blue
    'secondary': '#A23B72',    # Deep pink/purple
    'accent': '#F18F01',       # Warm orange
    'success': '#C73E1D',      # Deep red for emphasis
    
    # Extended palette for multi-line plots
    'palette_qualitative': [
        '#2E86AB',  # Deep blue
        '#A23B72',  # Deep pink
        '#F18F01',  # Orange
        '#C73E1D',  # Red
        '#6A994E',  # Green
        '#7209B7',  # Purple
        '#F72585',  # Bright pink
        '#4361EE',  # Bright blue
    ],
    
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
    
    # Control-specific color scheme
    'control_scheme': {
        'state': '#2E86AB',         # Blue for state
        'control': '#F18F01',       # Orange for control
        'reference': '#6A994E',     # Green for reference/target
        'model': '#2E86AB',         # Blue for model predictions
        'optimal': '#A23B72',       # Pink for optimal/LQR
        'bounds': '#C73E1D',        # Red for constraints
        'feasible': '#E8F5E9',      # Light green for feasible region
        'infeasible': '#FFEBEE',    # Light red for infeasible region
    },
    
    # Gradient palettes for heatmaps/continuous data
    'gradient_blue': ['#E3F2FD', '#1976D2', '#0D47A1'],
    'gradient_warm': ['#FFF3E0', '#FF9800', '#E65100'],
    'gradient_diverging': ['#0D47A1', '#FFFFFF', '#B71C1C'],  # Blue-White-Red
    
    # Neutral colors
    'light_gray': '#F5F5F5',
    'medium_gray': '#9E9E9E',
    'dark_gray': '#424242',
    'black': '#212121',
    'white': '#FFFFFF',
    
    # Semantic colors
    'model': '#2E86AB',        # Blue for model results
    'baseline': '#A23B72',     # Pink for baseline/LQR
    'target': '#6A994E',       # Green for targets
    'bounds': '#C73E1D',       # Red for bounds/limits
    
    # JUNIOR NOTE: These colors are for the performance annotation box.
    # Think of it like setting the font, background, and border color
    # for a sticky note on our plot.
    'text': '#212121',         # Black for text
    'background': '#FFFFFF',   # White for background
    'grid': '#BDBDBD',         # Lighter gray for annotation grid
}

# JUNIOR NOTE: Professional typography settings
TYPOGRAPHY = {
    'title_size': 24,
    'subtitle_size': 20,
    'axis_label_size': 16,
    'tick_label_size': 14,
    'legend_size': 16,          # Increased for better readability
    'annotation_size': 12,
    'small_text_size': 10,
    'annotation_weight': 'bold', # JUNIOR NOTE: Make annotation text bold
    
    # Font families
    'title_font': 'DejaVu Sans',
    'body_font': 'DejaVu Sans',
    'math_font': 'DejaVu Sans',
}

# JUNIOR NOTE: Standard figure sizes for different purposes
FIGURE_SIZES = {
    'single_plot': (10, 6),
    'wide_plot': (12, 6),
    'square_plot': (8, 8),
    'comparison_2x1': (16, 6),
    'comparison_2x2': (14, 10),
    'comparison_3x2': (18, 12),
    'combined_dashboard': (20, 12),
    'control_dashboard': (16, 10),  # For control system dashboards
    'phase_portrait': (8, 8),       # Square for phase portraits
    'time_series': (12, 8),         # Wide for time series
}

def setup_beautiful_plotting():
    """
    Set up matplotlib and seaborn for beautiful, publication-quality plots.
    
    JUNIOR NOTE: Call this function at the start of any plotting script
    to ensure consistent styling across all figures.
    """
    # Set seaborn style with custom modifications
    sns.set_theme(style="whitegrid", palette=BEAUTIFUL_COLORS['palette_qualitative'])
    sns.set_context("paper", font_scale=1.0)
    
    # Configure matplotlib rcParams for publication quality
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
        'font.family': TYPOGRAPHY['body_font'],
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
        'lines.markeredgewidth': 0,
        
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
        
        # Color cycle
        'axes.prop_cycle': plt.cycler('color', BEAUTIFUL_COLORS['palette_qualitative']),
    })
    
    print("✅ Beautiful plotting style configured!")

def create_beautiful_figure(figsize=None, title=None, suptitle_y=0.95):
    """
    Create a beautifully styled figure with consistent formatting.
    
    Args:
        figsize: Tuple of (width, height) or key from FIGURE_SIZES
        title: Main title for the figure
        suptitle_y: Y position for the main title
        
    Returns:
        fig: Matplotlib figure object
    """
    # Handle figsize input
    if figsize is None:
        figsize = FIGURE_SIZES['single_plot']
    elif isinstance(figsize, str) and figsize in FIGURE_SIZES:
        figsize = FIGURE_SIZES[figsize]
    
    fig = plt.figure(figsize=figsize, facecolor='white')
    
    if title:
        fig.suptitle(title, fontsize=TYPOGRAPHY['title_size'], 
                    fontweight='bold', y=suptitle_y,
                    color=BEAUTIFUL_COLORS['black'])
    
    return fig

def style_axes(ax, title=None, xlabel=None, ylabel=None, 
               grid=True, spines=['left', 'bottom', 'top', 'right'], box=True):
    """
    Apply beautiful styling to a matplotlib axes object.
    
    Args:
        ax: Matplotlib axes object
        title: Title for the subplot
        xlabel: X-axis label
        ylabel: Y-axis label
        grid: Whether to show grid
        spines: Which spines to show ['top', 'right', 'left', 'bottom']
        box: If True, show all four spines to create a box around the plot
    """
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
    
    # Configure spines - if box=True, override spines to show all four
    if box:
        spines_to_show = ['top', 'right', 'left', 'bottom']
    else:
        spines_to_show = spines
    
    for spine_name in ['top', 'right', 'left', 'bottom']:
        if spine_name in spines_to_show:
            ax.spines[spine_name].set_visible(True)
            ax.spines[spine_name].set_color(BEAUTIFUL_COLORS['dark_gray'])
            ax.spines[spine_name].set_linewidth(1.5)  # Thicker for better box appearance
        else:
            ax.spines[spine_name].set_visible(False)
    
    # Configure grid - enhanced visibility
    if grid:
        ax.grid(True, linestyle='-', alpha=0.3, linewidth=0.5,
               color=BEAUTIFUL_COLORS['medium_gray'])
        ax.set_axisbelow(True)
        
        # Add minor grid for even more precision (optional)
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

def add_grid_and_box(ax, grid_style='both', box=True):
    """
    Add enhanced grid and box styling to any existing plot.
    
    Args:
        ax: Matplotlib axes object
        grid_style: 'major', 'minor', or 'both'
        box: Whether to show all four spines
    """
    if box:
        # Show all spines for box
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(BEAUTIFUL_COLORS['dark_gray'])
            spine.set_linewidth(1.5)
    
    if grid_style in ['major', 'both']:
        ax.grid(True, which='major', linestyle='-', alpha=0.3, linewidth=0.5,
               color=BEAUTIFUL_COLORS['medium_gray'])
    
    if grid_style in ['minor', 'both']:
        ax.grid(True, which='minor', linestyle=':', alpha=0.2, linewidth=0.3,
               color=BEAUTIFUL_COLORS['medium_gray'])
        ax.minorticks_on()
    
    ax.set_axisbelow(True)

def beautiful_line_plot(x, y, ax=None, label=None, color=None, alpha=1.0,
                       linewidth=None, linestyle='-', marker=None,
                       smooth=False, smooth_window=10, show_variance=False,
                       robust_smoothing=True, outlier_threshold=0.02):
    """
    Create a beautiful line plot with optional smoothing and variance shading.
    
    JUNIOR NOTE: We've enhanced this function with robust smoothing! When 
    `robust_smoothing=True`, the thick trend line will ignore extreme spikes
    and follow the main pattern. Think of it like noise-canceling headphones
    for your data - you see the music (trend) without the background noise (spikes).
    
    Args:
        x, y: Data arrays
        ax: Matplotlib axes (creates new if None)
        label: Line label for legend
        color: Line color (uses palette if None)
        alpha: Line transparency
        linewidth: Line width
        linestyle: Line style
        marker: Marker style
        smooth: Whether to apply smoothing
        smooth_window: Window size for smoothing
        show_variance: If True and smooth is True, show a shaded region for variance.
        robust_smoothing: If True, use robust smoothing that ignores outliers
        outlier_threshold: Fraction of data to consider as outliers (0.02 = top/bottom 1% each)
        
    Returns:
        ax: Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGURE_SIZES['single_plot'])
    
    # Convert to numpy arrays
    x, y = np.array(x), np.array(y)
    
    # Set default styling
    if color is None:
        color = BEAUTIFUL_COLORS['primary']
    if linewidth is None:
        linewidth = 2.5
    
    # Plot raw data if smoothing is enabled (with transparency)
    if smooth and len(y) >= smooth_window:
        if robust_smoothing:
            # Use our robust smoothing that ignores extreme spikes
            y_smooth = robust_moving_average(y, window=smooth_window, 
                                           outlier_threshold=outlier_threshold)
            x_smooth = x  # Same length as original data
            
            # JUNIOR NOTE: For robust smoothing, we disable variance shading by default
            # because the variance calculation would still be affected by outliers.
            # Instead, we show the raw data as a transparent background.
            if show_variance:
                # Calculate robust standard deviation (using percentiles instead of std)
                data = pd.Series(y)
                # Use interquartile range as a robust measure of spread
                q75 = data.rolling(window=smooth_window, min_periods=1).quantile(0.75)
                q25 = data.rolling(window=smooth_window, min_periods=1).quantile(0.25)
                iqr = (q75 - q25) * 0.5  # Scale IQR to approximate std
                
                ax.fill_between(x_smooth, y_smooth - iqr, y_smooth + iqr, 
                                color=color, alpha=0.15, edgecolor='none')
            else:
                # Show raw data with transparency
                ax.plot(x, y, color=color, alpha=0.25, linewidth=linewidth*0.5,
                       linestyle='-', marker=None)
        else:
            # Use original pandas rolling mean for backward compatibility
            data = pd.Series(y)
            y_smooth = data.rolling(window=smooth_window, min_periods=1).mean()
            x_smooth = x
            
            if show_variance:
                # Calculate standard deviation for the shaded region
                y_std = data.rolling(window=smooth_window, min_periods=1).std().fillna(0)
                ax.fill_between(x_smooth, y_smooth - y_std, y_smooth + y_std, 
                                color=color, alpha=0.2, edgecolor='none')
            else:
                # Show raw data with transparency
                ax.plot(x, y, color=color, alpha=0.3, linewidth=linewidth*0.6,
                       linestyle='-', marker=None)
        
        # Plot the smoothed line on top
        ax.plot(x_smooth, y_smooth, color=color, alpha=alpha, 
               linewidth=linewidth, linestyle=linestyle, marker=marker,
               label=label)
    else:
        # Plot without smoothing
        ax.plot(x, y, color=color, alpha=alpha, linewidth=linewidth,
               linestyle=linestyle, marker=marker, label=label)
    
    return ax

def beautiful_scatter_plot(x, y, ax=None, label=None, color=None, 
                          size=None, alpha=0.7, marker='o', edgecolor='white'):
    """
    Create a beautiful scatter plot.
    
    Args:
        x, y: Data arrays
        ax: Matplotlib axes (creates new if None)
        label: Scatter label for legend
        color: Point color
        size: Point size
        alpha: Point transparency
        marker: Marker style
        edgecolor: Edge color for markers
        
    Returns:
        ax: Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGURE_SIZES['single_plot'])
    
    if color is None:
        color = BEAUTIFUL_COLORS['primary']
    if size is None:
        size = 100
    
    ax.scatter(x, y, c=color, s=size, alpha=alpha, marker=marker,
              edgecolors=edgecolor, linewidth=1.5, label=label)
    
    return ax

def plot_phase_portrait(x, y, ax=None, color=None, alpha=0.8, 
                       show_direction=True, show_start_end=True,
                       gradient_color=False):
    """
    Create a beautiful phase portrait for control systems.
    
    Args:
        x, y: State variables (e.g., position and velocity)
        ax: Matplotlib axes (creates new if None)
        color: Line color
        alpha: Line transparency
        show_direction: Add arrows showing trajectory direction
        show_start_end: Mark start and end points
        gradient_color: Use gradient coloring along trajectory
        
    Returns:
        ax: Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGURE_SIZES['phase_portrait'])
    
    colors = get_color_scheme('trajectory')
    
    if gradient_color:
        # Create gradient colored line
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create color map
        lc = LineCollection(segments, cmap='viridis', alpha=alpha)
        lc.set_array(np.linspace(0, 1, len(x)))
        lc.set_linewidth(2.5)
        line = ax.add_collection(lc)
        ax.autoscale()
    else:
        # Standard colored line
        if color is None:
            color = colors['phase_space']
        ax.plot(x, y, color=color, alpha=alpha, linewidth=2.5)
    
    # Add direction arrows
    if show_direction and len(x) > 10:
        # Add arrows at 25%, 50%, 75% of trajectory
        for frac in [0.25, 0.5, 0.75]:
            idx = int(len(x) * frac)
            if idx < len(x) - 1:
                dx = x[idx+1] - x[idx]
                dy = y[idx+1] - y[idx]
                ax.arrow(x[idx], y[idx], dx*0.3, dy*0.3,
                        head_width=0.05, head_length=0.05,
                        fc=color if not gradient_color else 'black',
                        ec=color if not gradient_color else 'black',
                        alpha=0.6)
    
    # Mark start and end
    if show_start_end:
        ax.scatter(x[0], y[0], s=150, c=colors['start'], 
                  marker='o', edgecolor='white', linewidth=2,
                  label='Start', zorder=5)
        ax.scatter(x[-1], y[-1], s=150, c=colors['end'], 
                  marker='s', edgecolor='white', linewidth=2,
                  label='End', zorder=5)
    
    return ax

def add_control_bounds(ax, bounds, axis='both', alpha=0.2):
    """
    Add shaded regions for control/state bounds.
    
    Args:
        ax: Matplotlib axes
        bounds: Dict with 'lower' and 'upper' keys, or tuple (lower, upper)
        axis: 'x', 'y', or 'both'
        alpha: Transparency of shaded region
    """
    colors = get_color_scheme('control_scheme')
    
    if isinstance(bounds, tuple):
        bounds = {'lower': bounds[0], 'upper': bounds[1]}
    
    if axis in ['x', 'both']:
        ax.axvspan(bounds['lower'], bounds['upper'], 
                  alpha=alpha, color=colors['feasible'], 
                  label='Feasible Region')
    
    if axis in ['y', 'both']:
        ax.axhspan(bounds['lower'], bounds['upper'], 
                  alpha=alpha, color=colors['feasible'])
    
    # Add bound lines
    if axis in ['x', 'both']:
        ax.axvline(bounds['lower'], color=colors['bounds'], 
                  linestyle='--', alpha=0.5)
        ax.axvline(bounds['upper'], color=colors['bounds'], 
                  linestyle='--', alpha=0.5)
    
    if axis in ['y', 'both']:
        ax.axhline(bounds['lower'], color=colors['bounds'], 
                  linestyle='--', alpha=0.5)
        ax.axhline(bounds['upper'], color=colors['bounds'], 
                  linestyle='--', alpha=0.5)

def setup_log_scale(ax, axis='y', symmetric=False, threshold=1e-1):
    """
    Configure an axis for logarithmic or symmetrical logarithmic scaling.
    
    JUNIOR NOTE: We're adjusting the 'base' of the LogLocator. The default
    is base=10, which gives ticks at 10^0, 10^1, 10^2, etc. By changing it
    to base=100, we get ticks at every other power of ten (10^0, 10^2, 10^4),
    which can make a dense plot much cleaner.
    """
    locator_base = 100.0  # Use base 100 for ticks at 10^0, 10^2, 10^4, etc.
    
    if axis == 'y':
        if symmetric:
            ax.set_yscale('symlog', linthresh=threshold)
            locator = SymmetricalLogLocator(base=locator_base, linthresh=threshold)
        else:
            ax.set_yscale('log')
            locator = LogLocator(base=locator_base)
        ax.yaxis.set_major_locator(locator)
    elif axis == 'x':
        if symmetric:
            ax.set_xscale('symlog', linthresh=threshold)
            locator = SymmetricalLogLocator(base=locator_base, linthresh=threshold)
        else:
            ax.set_xscale('log')
            locator = LogLocator(base=locator_base)
        ax.xaxis.set_major_locator(locator)

def add_beautiful_legend(ax, title=None, location='best', **kwargs):
    """
    Add a beautiful, styled legend to the plot.
    
    JUNIOR NOTE: We've added `**kwargs` to this function. This is a powerful
    Python feature that lets us pass any extra keyword arguments (like `handles`
    or `ncol`) directly to the `ax.legend()` call. This makes our helper
    function much more flexible without having to define every possible
    legend option ourselves.
    
    Args:
        ax: Matplotlib axes object
        title: Legend title
        location: Legend location
        **kwargs: Additional keyword arguments passed to ax.legend()
        
    Returns:
        legend: Matplotlib legend object
    """
    # Default legend settings that define our "beautiful" style
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
        'handletextpad': 0.5
    }
    
    # Allow user to override any of our defaults or add new parameters
    legend_params.update(kwargs)
    
    legend = ax.legend(**legend_params)
    
    # Style legend title
    if title and legend.get_title():
        legend.get_title().set_fontweight('bold')
        legend.get_title().set_color(BEAUTIFUL_COLORS['black'])
    
    return legend

def add_compact_legend(ax, loc='best', ncol=2, title=None, 
                      bbox_to_anchor=None, handles=None):
    """
    Add an extra-compact legend for space-constrained plots.
    
    JUNIOR NOTE: This is specifically designed for subplots where space is tight.
    Uses multiple columns and minimal padding to maximize plot area.
    
    Args:
        ax: Matplotlib axes object
        loc: Legend location
        ncol: Number of columns (default 2 for compactness)
        title: Legend title
        bbox_to_anchor: Precise positioning tuple (x, y)
        handles: Custom legend handles
        
    Returns:
        legend: Matplotlib legend object
    """
    legend = ax.legend(handles=handles, loc=loc, title=title, ncol=ncol,
                      bbox_to_anchor=bbox_to_anchor,
                      fontsize=TYPOGRAPHY['legend_size'],  # Use standard legend size for better readability
                      title_fontsize=TYPOGRAPHY['legend_size'],
                      frameon=True, framealpha=0.9, facecolor='white',
                      edgecolor=BEAUTIFUL_COLORS['light_gray'],
                      borderpad=0.4,      # Slightly more padding for readability
                      columnspacing=0.8,  # Better column spacing
                      handlelength=1.8,   # Longer legend handles for clarity
                      handletextpad=0.5,  # Better handle-text spacing
                      labelspacing=0.4)   # Better row spacing
    
    # Style legend title
    if title and legend.get_title():
        legend.get_title().set_fontweight('bold')
        legend.get_title().set_color(BEAUTIFUL_COLORS['black'])
    
    return legend

def save_beautiful_figure(fig, filename, formats=['pdf', 'png'], dpi=300):
    """
    Save figure in multiple formats with beautiful settings.
    
    Args:
        fig: Matplotlib figure object
        filename: Base filename (without extension)
        formats: List of formats to save ['pdf', 'png', 'svg']
        dpi: Resolution for raster formats
    """
    # Remove any existing extension from filename
    import os
    base_filename = os.path.splitext(filename)[0]
    
    saved_files = []
    for fmt in formats:
        full_filename = f"{base_filename}.{fmt}"
        
        if fmt.lower() == 'pdf':
            fig.savefig(full_filename, format='pdf', dpi=dpi, 
                       bbox_inches='tight', facecolor='white', 
                       edgecolor='none', pad_inches=0.1)
        elif fmt.lower() == 'png':
            fig.savefig(full_filename, format='png', dpi=dpi,
                       bbox_inches='tight', facecolor='white',
                       edgecolor='none', pad_inches=0.1)
        elif fmt.lower() == 'svg':
            fig.savefig(full_filename, format='svg', 
                       bbox_inches='tight', facecolor='white',
                       edgecolor='none', pad_inches=0.1)
        
        saved_files.append(full_filename)
    
    return saved_files

def create_color_annotation(text, color, size='medium'):
    """
    Create a colored text annotation for legends or labels.
    
    Args:
        text: Text to display
        color: Color for the text
        size: Text size ('small', 'medium', 'large')
        
    Returns:
        Text object ready for plotting
    """
    size_map = {
        'small': TYPOGRAPHY['small_text_size'],
        'medium': TYPOGRAPHY['annotation_size'],
        'large': TYPOGRAPHY['axis_label_size']
    }
    
    return {'text': text, 'color': color, 'fontsize': size_map.get(size, size)}

# JUNIOR NOTE: Helper function to get colors for different plot types
def get_color_scheme(plot_type='default'):
    """
    Get appropriate color schemes for different types of plots.
    
    Args:
        plot_type: Type of plot ('training', 'comparison', 'trajectory', 'heatmap', 'control_scheme')
        
    Returns:
        Dictionary with color assignments
    """
    schemes = {
        'training': {
            'reward': BEAUTIFUL_COLORS['primary'],
            'loss': BEAUTIFUL_COLORS['secondary'],
            'policy_loss': BEAUTIFUL_COLORS['accent'],
            'entropy': BEAUTIFUL_COLORS['success'],
            'success_rate': BEAUTIFUL_COLORS['target']
        },
        'comparison': {
            'model': '#1f77b4',        # Professional blue for model
            'baseline': '#ff7f0e',     # Professional orange for baseline/LQR
            'lqr': '#ff7f0e',         # Professional orange for LQR
            'target': '#228B22',       # Forest green for targets (more visible)
            'bounds': '#d62728',       # Professional red for bounds/limits
            'phase_transition': '#7f7f7f'  # Professional gray for phase markers
        },
        'trajectory': {
            'phase_space': BEAUTIFUL_COLORS['primary'],
            'position': '#1f77b4',     # Professional blue
            'velocity': '#ff7f0e',     # Professional orange
            'control': '#2ca02c',      # Professional green
            'target': '#228B22',       # Forest green for targets (more visible)
            'start': '#FF4500',        # Orange-red for start (distinct from target)
            'end': '#4B0082',          # Indigo for end (distinct from target)
            'bounds': '#d62728'        # Professional red for bounds
        },
        'control_scheme': BEAUTIFUL_COLORS['control_scheme'],
        'default': BEAUTIFUL_COLORS['palette_qualitative']
    }
    
    return schemes.get(plot_type, schemes['default'])

def add_performance_annotation(ax, value, prefix="Final", auto_position=False, x_pos=None, y_pos=None):
    """
    Add a styled performance annotation to the plot.
    
    JUNIOR NOTE: We're making this function more powerful. Previously, it could
    only auto-place the annotation in a corner. Now, by adding `x_pos` and `y_pos`,
    we can give it exact coordinates (from 0.0 to 1.0) to place the annotation
    box anywhere on the plot. This is essential for complex plots where the
    corners might already be crowded.
    """
    # Format the value string
    if isinstance(value, (int, float)):
        if abs(value) > 1e-2 and abs(value) < 1e3:
            text = f"{prefix}: {value:.3f}" if prefix else f"{value:.3f}"
        else:
            text = f"{prefix}: {value:.2e}" if prefix else f"{value:.2e}"
    else:
        # If a prefix is provided, add it. Otherwise, just use the value.
        text = f"{prefix}\n{value}" if prefix else str(value)
        
    # Determine position for the annotation
    if x_pos is not None and y_pos is not None:
        # User has provided exact coordinates.
        # 'left', 'top' alignment means (x_pos, y_pos) is the top-left corner of the text box.
        ha, va = 'left', 'top'
        x_text, y_text = x_pos, y_pos
    elif auto_position:
        # Get the y-data of the last line plotted on the axes
        if ax.lines:
            last_line = ax.lines[-1]
            y_data = last_line.get_ydata()
            
            if len(y_data) > 0:
                y_min, y_max = ax.get_ylim()
                final_y_normalized = (y_data[-1] - y_min) / (y_max - y_min)
                
                if final_y_normalized > 0.5:
                    vertical_alignment = 'bottom'
                    y_pos_auto = 0.05
                else:
                    vertical_alignment = 'top'
                    y_pos_auto = 0.95
            else:
                vertical_alignment = 'top'
                y_pos_auto = 0.95
        else:
            vertical_alignment = 'top'
            y_pos_auto = 0.95
                
        ha, va = 'right', vertical_alignment
        x_text, y_text = 0.95, y_pos_auto
    else:
        # Default legacy position (top-right)
        ha, va = 'right', 'top'
        x_text, y_text = 0.95, 0.95

    ax.text(x_text, y_text, text,
            transform=ax.transAxes,
            ha=ha,
            va=va,
            fontsize=TYPOGRAPHY['annotation_size'],
            fontweight=TYPOGRAPHY['annotation_weight'],
            color=BEAUTIFUL_COLORS['text'],
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor=BEAUTIFUL_COLORS['background'],
                      edgecolor=BEAUTIFUL_COLORS['grid'],
                      alpha=0.8))

def robust_moving_average(values, window=20, outlier_threshold=0.02):
    """
    Create a robust moving average that ignores extreme outliers.
    
    Args:
        values: Array of values to smooth
        window: Window size for moving average
        outlier_threshold: Fraction of data to consider as outliers (0.02 = top/bottom 1% each)
        
    Returns:
        smoothed_values: Robust moving average that ignores spikes
        
    JUNIOR NOTE: This function filters out extreme spikes before smoothing,
    so the thick trend line follows the main pattern instead of being 
    pulled around by occasional huge spikes. Think of it like noise-canceling
    headphones - we hear the music (trend) without the background noise (spikes).
    """
    values = np.array(values)
    
    if len(values) < window:
        return values
    
    # Calculate rolling percentiles to identify outliers dynamically
    percentile_low = outlier_threshold * 100  # e.g., 2% 
    percentile_high = (1 - outlier_threshold) * 100  # e.g., 98%
    
    # Create a robust smoothed version
    smoothed = np.zeros_like(values, dtype=float)
    
    for i in range(len(values)):
        # Define window bounds
        start_idx = max(0, i - window // 2)
        end_idx = min(len(values), i + window // 2 + 1)
        window_values = values[start_idx:end_idx]
        
        # Filter out outliers in this window
        if len(window_values) > 3:  # Need at least a few points
            low_thresh = np.percentile(window_values, percentile_low)
            high_thresh = np.percentile(window_values, percentile_high)
            
            # Keep values within the reasonable range
            filtered_values = window_values[
                (window_values >= low_thresh) & (window_values <= high_thresh)
            ]
            
            # If we filtered out too much, fall back to less aggressive filtering
            if len(filtered_values) < len(window_values) * 0.3:  # Keep at least 30%
                low_thresh = np.percentile(window_values, 5)
                high_thresh = np.percentile(window_values, 95)
                filtered_values = window_values[
                    (window_values >= low_thresh) & (window_values <= high_thresh)
                ]
        else:
            filtered_values = window_values
        
        # Take the mean of the filtered values
        smoothed[i] = np.mean(filtered_values) if len(filtered_values) > 0 else values[i]
    
    return smoothed

# Control-specific plotting functions
def plot_control_dashboard(results, system_name, save_path=None):
    """
    Create a comprehensive control system dashboard.
    
    Args:
        results: Dictionary with control results
        system_name: Name of the control system
        save_path: Path to save the figure
        
    Returns:
        fig: Matplotlib figure
    """
    setup_beautiful_plotting()
    fig = create_beautiful_figure(figsize='control_dashboard', 
                                 title=f'{system_name.replace("_", " ").title()} Control Dashboard')
    
    # Create grid of subplots
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Phase portrait
    ax_phase = fig.add_subplot(gs[0, 0])
    if 'states' in results:
        states = np.array(results['states'])
        plot_phase_portrait(states[:, 0], states[:, 1], ax=ax_phase,
                          gradient_color=True, show_direction=True)
    style_axes(ax_phase, title='Phase Portrait', 
              xlabel='Position', ylabel='Velocity')
    
    # State trajectories
    ax_states = fig.add_subplot(gs[0, 1:])
    if 'states' in results and 'time' in results:
        time = results['time']
        states = np.array(results['states'])
        colors = get_color_scheme('trajectory')
        
        beautiful_line_plot(time, states[:, 0], ax=ax_states, 
                          label='Position', color=colors['position'])
        beautiful_line_plot(time, states[:, 1], ax=ax_states, 
                          label='Velocity', color=colors['velocity'])
        
        # Add reference if available
        if 'reference' in results:
            ax_states.axhline(results['reference'][0], color=colors['target'],
                            linestyle='--', alpha=0.7, label='Target')
    
    style_axes(ax_states, title='State Trajectories', 
              xlabel='Time (s)', ylabel='State Value')
    add_beautiful_legend(ax_states)
    
    # Control inputs
    ax_control = fig.add_subplot(gs[1, :2])
    if 'controls' in results and 'time' in results:
        time = results['time'][:-1]  # Control has one less point
        controls = np.array(results['controls'])
        colors = get_color_scheme('control_scheme')
        
        beautiful_line_plot(time, controls, ax=ax_control,
                          color=colors['control'], label='Control Input')
        
        # Add bounds if available
        if 'control_bounds' in results:
            bounds = results['control_bounds']
            add_control_bounds(ax_control, bounds, axis='y')
    
    style_axes(ax_control, title='Control Inputs', 
              xlabel='Time (s)', ylabel='Control u(t)')
    
    # Performance metrics
    ax_metrics = fig.add_subplot(gs[1, 2])
    if 'metrics' in results:
        metrics = results['metrics']
        
        # Create bar chart of metrics
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        bars = ax_metrics.bar(range(len(metric_names)), metric_values,
                            color=BEAUTIFUL_COLORS['palette_qualitative'][:len(metric_names)])
        
        ax_metrics.set_xticks(range(len(metric_names)))
        ax_metrics.set_xticklabels(metric_names, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax_metrics.text(bar.get_x() + bar.get_width()/2., height,
                          f'{value:.3f}', ha='center', va='bottom',
                          fontsize=TYPOGRAPHY['small_text_size'])
    
    style_axes(ax_metrics, title='Performance Metrics', ylabel='Value')
    
    plt.tight_layout()
    
    if save_path:
        save_beautiful_figure(fig, save_path, formats=['pdf', 'png'])
    
    return fig

# Test function to verify styling
def test_beautiful_styling():
    """Test function to create sample plots with beautiful styling."""
    setup_beautiful_plotting()
    
    # Generate sample data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x) * np.exp(-x/10)
    y2 = np.cos(x) * np.exp(-x/8)
    
    fig = create_beautiful_figure(figsize='comparison_2x1', 
                                 title='Beautiful Styling Test')
    
    # Create subplots
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    
    # Plot with beautiful styling
    beautiful_line_plot(x, y1, ax=ax1, label='Model', 
                       color=BEAUTIFUL_COLORS['primary'], smooth=True)
    beautiful_line_plot(x, y2, ax=ax1, label='Baseline',
                       color=BEAUTIFUL_COLORS['secondary'], smooth=True)
    
    style_axes(ax1, title='Training Progress', 
              xlabel='Epoch', ylabel='Performance', box=True, grid=True)
    add_beautiful_legend(ax1, title='Methods')
    
    # Scatter plot
    beautiful_scatter_plot(x[::5], y1[::5], ax=ax2, 
                          color=BEAUTIFUL_COLORS['accent'], label='Data Points')
    
    style_axes(ax2, title='Sample Data', 
              xlabel='Time', ylabel='Value', box=True, grid=True)
    add_beautiful_legend(ax2)
    
    # Add extra grid styling to demonstrate
    add_grid_and_box(ax1, grid_style='both', box=True)
    add_grid_and_box(ax2, grid_style='both', box=True)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Test the styling
    test_fig = test_beautiful_styling()
    print("✅ Styling test completed! Check the generated figure.")
    plt.show()