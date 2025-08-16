# Publication-Ready Plotting Guide

This guide explains how to use the enhanced publication-ready plotting capabilities in the Universal Control LLM Framework.

## Overview

The enhanced plotting system provides publication-quality visualizations that follow scientific standards for control systems and trajectory analysis. All plots maintain consistent styling, professional color schemes, and clear layouts suitable for papers and presentations.

## Key Features

✅ **Professional Standards**: Adheres to scientific publication standards with clean layouts and consistent typography  
✅ **4-Subplot Layout**: Phase space, position vs time, velocity vs time, and control vs time  
✅ **System-Agnostic**: Works with any control system (double_integrator, van_der_pol, etc.)  
✅ **Multiple Formats**: Generates both PDF (vector) and PNG (raster) outputs  
✅ **Individual Plots**: Create standalone plots for each subplot type  
✅ **Directional Arrows**: Phase space plots include trajectory direction indicators  
✅ **Bounds Visualization**: Shows control and state constraints  
✅ **Beautiful Styling**: Professional color schemes and consistent formatting  

## Quick Start

### 1. Basic Usage with Evaluation Script

Generate publication plots during model evaluation:

```bash
# Generate 4-subplot comparison plots
python scripts/evaluate_model.py \
    --model-path models/single_system/double_integrator/grpo/latest \
    --model-type single_system \
    --publication-plots \
    --save-plots

# Generate individual standalone plots  
python scripts/evaluate_model.py \
    --model-path models/single_system/double_integrator/grpo/latest \
    --model-type single_system \
    --individual-plots \
    --save-plots
```

### 2. Benchmark Generation

Create comprehensive publication plots for multiple systems:

```bash
# Generate benchmark plots for multiple systems
python scripts/create_publication_benchmark.py \
    --systems double_integrator,van_der_pol \
    --num-cases 3 \
    --plot-types comparison,individual \
    --plot-dir benchmark_plots
```

### 3. Programmatic Usage

```python
from evaluation.visualization import (
    plot_publication_comparison, 
    plot_model_only_trajectories,
    generate_publication_plots
)

# Create 4-subplot comparison plot
trajectories = {'Model': model_traj, 'Optimal': optimal_traj}
fig = plot_publication_comparison(trajectories, "double_integrator", 
                                 initial_state=(0.5, -0.3),
                                 save_path="output/comparison")

# Generate complete set of plots
saved_files = generate_publication_plots(trajectories, "double_integrator",
                                        base_filename="analysis",
                                        plot_dir="publication_plots")
```

## Plot Types Available

### 1. **Publication Comparison** (`plot_publication_comparison`)
- 4-subplot layout with phase space, position, velocity, and control
- Model vs Optimal trajectory comparison
- Professional styling with consistent colors
- Directional arrows in phase space
- Control and state bounds visualization

### 2. **Model-Only Trajectories** (`plot_model_only_trajectories`)
- Same 4-subplot layout but for model trajectories without optimal baseline
- Useful when optimal solutions aren't available
- Maintains consistent styling

### 3. **Individual Plots**
- `create_individual_phase_plot`: Standalone phase space plot
- `create_individual_state_plot`: Position or velocity vs time
- `create_individual_control_plot`: Control vs time
- Perfect for presentations or focused analysis

### 4. **Complete Plot Generation** (`generate_publication_plots`)
- Generates all plot types in one call
- Creates both comparison and individual plots
- Saves in multiple formats (PDF + PNG)

## Color Standards

The plotting system uses scientifically-appropriate color schemes:

- **Model Trajectories**: Professional blue (`#1f77b4`)
- **Optimal/LQR**: Professional orange (`#ff7f0e`) with dashed lines
- **Targets**: Deep red (`#DC143C`) with X markers
- **Control Bounds**: Red dashed lines (`#d62728`)
- **Phase Transitions**: Black dotted lines
- **Grid**: Subtle gray for background

## Layout Standards

- **Figure Size**: 15×12 inches for 4-subplot layouts
- **Subplot Labels**: (a), (b), (c), (d) positioned consistently
- **Typography**: 16pt for labels, 20pt for titles
- **Spacing**: Optimized padding for readability
- **Legends**: Positioned strategically to avoid overlapping data

## File Organization

Generated files follow consistent naming conventions:

```
plots/
├── system_case_1_comparison.pdf           # 4-subplot comparison
├── system_case_1_comparison.png
├── system_case_1_phase_space.pdf          # Individual phase plot
├── system_case_1_phase_space.png
├── system_case_1_position.pdf             # Individual position plot
├── system_case_1_position.png
├── system_case_1_velocity.pdf             # Individual velocity plot
├── system_case_1_velocity.png
├── system_case_1_control.pdf              # Individual control plot
└── system_case_1_control.png
```

## Integration with Existing Code

The enhanced plotting system is fully integrated with the existing evaluation pipeline:

### Enhanced `evaluate_model.py`
- Added `--publication-plots` flag for 4-subplot comparisons
- Added `--individual-plots` flag for standalone plots
- Maintains backward compatibility with existing functionality

### System-Agnostic Design
Works seamlessly with all control systems:
- `double_integrator`: Position/velocity dynamics
- `van_der_pol`: Nonlinear oscillator  
- Any future systems following the `BaseEnvironment` interface

### Beautiful Plotting Utilities
Enhanced `plotting_utils.py` with:
- Publication-ready constants and color schemes
- Subplot label functions
- Directional arrow utilities
- Professional styling functions

## Quality Verification

Run the test suite to verify everything works:

```bash
python test_publication_plots.py
```

This will:
- Test all plotting functions
- Generate sample plots
- Verify system-agnostic design
- Create quality verification plots in `test_plots/`

## Example Output

The system generates publication-ready plots with:

1. **Phase Space (a)**: Trajectory in state space with directional arrows
2. **Position vs Time (b)**: State evolution with bounds and phase information  
3. **Velocity vs Time (c)**: Second state variable with transitions
4. **Control vs Time (d)**: Control inputs with constraint bounds

All plots maintain consistent professional styling suitable for:
- Academic papers and journals
- Conference presentations
- Technical reports
- Thesis documents

## Advanced Usage

### Custom Styling
Modify colors and styling in `plotting_utils.py`:

```python
PUBLICATION_COLORS = {
    'target': '#DC143C',           # Deep red for targets
    'model': '#1f77b4',           # Professional blue
    'optimal': '#ff7f0e',         # Professional orange
    # ... customize as needed
}
```

### Multiple Systems
The benchmark script can handle any number of systems:

```bash
python scripts/create_publication_benchmark.py \
    --systems double_integrator,van_der_pol,custom_system \
    --num-cases 5
```

### Presentation Mode
For presentations, use individual plots with larger fonts:

```python
fig = create_individual_phase_plot(trajectories, system_name, figsize=(10, 10))
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: Ensure matplotlib, numpy, and seaborn are installed
2. **Font Issues**: The system uses DejaVu Sans - install if missing
3. **Memory**: Large numbers of plots may require more RAM
4. **File Permissions**: Ensure write access to output directories

### Testing
Always test with the provided test script before important use:

```bash
python test_publication_plots.py
```

## Summary

The enhanced plotting system provides everything needed for publication-ready control system visualization:

- **Professional Quality**: Meets scientific publication standards
- **Easy Integration**: Works with existing evaluation pipeline  
- **System Agnostic**: Handles any control system
- **Multiple Formats**: PDF and PNG outputs
- **Comprehensive**: Both comparison and individual plots
- **Well Tested**: Includes verification test suite

Use these tools to create beautiful, consistent, and professional visualizations for your control systems research!