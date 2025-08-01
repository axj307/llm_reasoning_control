# Combined Visualization Guide

## Overview
The new visualization system creates a single comprehensive figure showing all test trajectories together, instead of individual plots for each test case.

## Key Features

### 1. **All-in-One Visualization**
- **Phase Space**: Shows all trajectories in position-velocity space
- **Position over Time**: All trajectories' positions on one plot
- **Velocity over Time**: All trajectories' velocities on one plot
- **Control Inputs** (optional): All control sequences on one plot

### 2. **Visual Elements**
- **Solid Lines**: Model predictions
- **Dashed Lines**: Optimal (LQR) solutions
- **Different Colors**: Each test case gets a unique color
- **Start Points**: Marked with circles
- **End Points**: Marked with triangles
- **Target (Origin)**: Marked with a star

## Usage

### Using Updated main.py

```bash
# Run evaluation with combined visualization
python trainer/main_v2.py --mode eval

# This will create:
# - outputs/combined_evaluation_full.png (with control subplot)
# - outputs/combined_evaluation.png (without control subplot)
```

### Using the Functions Directly

```python
from utils_enhanced import visualize_all_trajectories_with_controls

# Prepare test cases: list of (x0, v0, controls) tuples
test_cases = [
    (0.5, -0.3, controls1),
    (0.7, 0.2, controls2),
    (-0.6, -0.4, controls3)
]

# Create visualization
visualize_all_trajectories_with_controls(
    test_cases=test_cases,
    dt=0.1,
    save_path="my_evaluation.png",
    show_optimal=True  # Shows optimal trajectories as dashed lines
)
```

### Using the Evaluation Module

```python
from trainer_module_v3 import ControlTrainerV3

# Load model and evaluate
trainer = ControlTrainerV3(model_name="path/to/model")
results = trainer.evaluate(num_test_cases=5, visualize=True)

# This automatically creates combined visualizations
```

## Visualization Types

### 1. **Full Combined (4 subplots)**
```python
visualize_all_trajectories_with_controls(...)
```
- Top: Phase space (spans 2 columns)
- Bottom Left: Position vs Time
- Bottom Right: Velocity vs Time
- Bottom: Control vs Time (spans 2 columns)

### 2. **Simple Combined (3 subplots)**
```python
visualize_all_trajectories(...)
```
- Top: Phase space (spans 2 columns)
- Bottom Left: Position vs Time
- Bottom Right: Velocity vs Time

## Benefits

1. **Easy Comparison**: See all trajectories at once
2. **Space Efficient**: One figure instead of many
3. **Performance Overview**: Quickly identify which initial conditions are challenging
4. **Model vs Optimal**: Direct comparison with dashed optimal trajectories

## Example Output

The combined visualization will show:
- Multiple colored trajectories (one per test case)
- Each trajectory shown in all subplots with consistent color
- Clear legend indicating initial conditions
- Bounds shown as shaded regions
- Target (origin) clearly marked

## Testing

Run the test script to see examples:
```bash
python trainer/test_combined_viz.py
```

This creates sample visualizations with different noise levels to demonstrate the capabilities.