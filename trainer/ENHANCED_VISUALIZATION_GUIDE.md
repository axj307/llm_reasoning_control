# Enhanced Visualization Guide

## Overview
The enhanced visualization system provides publication-quality plots for control system analysis with professional styling, comprehensive metrics, and multiple visualization types.

## Key Features

### 1. **Publication-Quality Styling**
- Professional color schemes with accessibility in mind
- Consistent typography and spacing
- Enhanced grid and box styling for clarity
- High-resolution output in multiple formats (PNG, PDF)

### 2. **Three Main Visualization Types**

#### a) Publication Comparison Plot
```python
create_publication_comparison_plot(
    test_cases=trajectories,
    dt=0.1,
    save_path="comparison",
    show_optimal=True
)
```
- **4-panel layout**: Phase space, Position, Velocity, Control
- Shows both model predictions (solid) and optimal solutions (dashed)
- Performance metrics annotation
- Subplot labels: (a), (b), (c), (d)

#### b) Model Trajectories Plot
```python
create_model_trajectories_plot(
    test_cases=trajectories,
    dt=0.1,
    save_path="model_only",
    add_metrics=True
)
```
- Focus on model-only results without baseline
- Optional metrics summary
- Useful for presenting final results

#### c) Control Dashboard
```python
create_control_dashboard(
    test_cases=trajectories,
    dt=0.1,
    save_path="dashboard"
)
```
- Comprehensive 6-panel dashboard
- Includes error analysis and cumulative effort
- Summary statistics panel
- Best for detailed performance analysis

### 3. **Visual Elements**

#### Colors
- **Trajectory colors**: Professional gradient palette
- **Target**: Deep red X marker
- **Bounds**: Red dashed lines
- **Start points**: Colored circles with white edges
- **End points**: Black squares (or red X if near target)

#### Styling
- **Grid**: Both major and minor grids for precision
- **Box**: Complete frame around each subplot
- **Arrows**: Directional indicators in phase space
- **Typography**: Clear hierarchy with different font sizes

## Usage Examples

### Basic Usage
```python
from utils_enhanced_v2 import create_publication_comparison_plot

# Prepare test cases
test_cases = [
    (0.5, -0.3, controls1),
    (0.7, 0.2, controls2),
    (-0.6, -0.4, controls3)
]

# Create visualization
create_publication_comparison_plot(
    test_cases=test_cases,
    dt=0.1,
    save_path="my_results",
    show_optimal=True,
    title="Custom Title Here"
)
```

### Using with Evaluation
```python
# In main_enhanced.py
python trainer/main_enhanced.py --mode eval --viz-type all
```

This creates all three visualization types:
- `publication_comparison.png/pdf`
- `model_trajectories.png/pdf`
- `control_dashboard.png/pdf`

### Customization Options

#### Visualization Type Selection
```bash
# Only comparison plot
python trainer/main_enhanced.py --mode eval --viz-type comparison

# Only dashboard
python trainer/main_enhanced.py --mode eval --viz-type dashboard

# All types
python trainer/main_enhanced.py --mode eval --viz-type all
```

#### Hide Optimal Solutions
```bash
python trainer/main_enhanced.py --mode eval --viz-type comparison --no-optimal
```

## Testing

Run the comprehensive test script:
```bash
python trainer/test_enhanced_viz.py
```

This creates sample visualizations in `enhanced_visualizations/`:
- Test with varying noise levels
- Different numbers of trajectories
- With and without metrics
- Multiple initial conditions

## Best Practices

1. **For Papers**: Use `create_publication_comparison_plot` with 3-5 test cases
2. **For Presentations**: Use `create_model_trajectories_plot` without metrics
3. **For Analysis**: Use `create_control_dashboard` with all test cases
4. **Export**: Always save in both PNG (for viewing) and PDF (for papers)

## Customization

### Adding New Color Schemes
Edit `plotting_styles.py`:
```python
BEAUTIFUL_COLORS['my_scheme'] = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    # ... add more colors
}
```

### Modifying Typography
Edit `TYPOGRAPHY` dictionary in `plotting_styles.py`:
```python
TYPOGRAPHY = {
    'title_size': 24,      # Main title
    'subtitle_size': 20,   # Subplot titles
    'axis_label_size': 16, # Axis labels
    # ... modify as needed
}
```

### Custom Metrics
Add to the metrics text in any visualization:
```python
metrics_text = "Custom Metrics:\n"
metrics_text += f"My Metric: {value:.4f}\n"
add_performance_annotation(fig, metrics_text)
```

## Comparison with Original

| Feature | Original | Enhanced |
|---------|----------|----------|
| Styling | Basic matplotlib | Publication-ready |
| Colors | Default palette | Professional gradient |
| Grid | Simple | Major + minor grids |
| Borders | Partial spines | Complete box frame |
| Labels | None | (a), (b), (c), (d) |
| Metrics | In title | Dedicated annotation box |
| Formats | PNG only | PNG + PDF |
| Typography | Default sizes | Hierarchical sizing |

## Troubleshooting

1. **Plots look crowded**: Reduce number of test cases to 3-5
2. **Text too small**: Adjust `TYPOGRAPHY` values in `plotting_styles.py`
3. **Colors not distinct**: Use fewer test cases or modify color palette
4. **PDF not saving**: Ensure matplotlib backend supports PDF export

## Future Enhancements

- Animation support for trajectory evolution
- Interactive plots with plotly
- 3D phase space visualization
- Multi-system comparison plots
- Statistical significance indicators