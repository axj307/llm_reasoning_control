# Evaluation Pipeline Module

The evaluation module provides comprehensive tools for evaluating control system models with rich visualizations and detailed metrics.

## Features

### 1. **Comprehensive Multi-Panel Visualizations**
The module creates publication-quality plots showing:
- **Phase Space Trajectory**: 2D plot of position vs velocity with trajectory direction
- **Position over Time**: Shows how position evolves with bounds
- **Velocity over Time**: Shows velocity evolution with bounds  
- **Control Inputs**: Step plot of control sequence with bounds

All in a single figure for easy analysis!

### 2. **Detailed Trajectory Analysis**
- Final state error metrics
- Constraint violation tracking
- Control effort and smoothness metrics
- Convergence analysis
- Phase space properties (path length, area)
- Energy metrics

### 3. **Batch Evaluation Capabilities**
- Evaluate multiple test cases simultaneously
- Aggregate statistics across test cases
- Compare predicted vs optimal trajectories
- Success rate and convergence rate metrics

### 4. **Model Comparison Tools**
- Compare multiple models on same test set
- Automatic ranking by performance metrics
- Detailed comparison reports

## Usage

### Basic Evaluation

```python
from trainer_module_v3 import ControlTrainerV3

# Load a trained model
trainer = ControlTrainerV3(model_name="path/to/your/model")

# Evaluate on standard benchmark
results = trainer.evaluate_on_benchmark(
    benchmark_name="standard",  # or "easy", "hard"
    visualize=True
)

# Or evaluate with custom test cases
results = trainer.evaluate(
    num_test_cases=20,
    visualize=True
)
```

### Direct Usage of Evaluation Components

```python
from evaluation import EvaluationManager, DoubleIntegratorEvaluator

# Create evaluation manager
eval_manager = EvaluationManager()

# Get system-specific evaluator
evaluator = eval_manager.get_evaluator("double_integrator", dt=0.1, steps=50)

# Evaluate single trajectory
result = evaluator.evaluate_single(
    initial_state=np.array([0.5, -0.3]),
    predicted_controls=predicted_controls,
    optimal_controls=optimal_controls  # Optional
)

# Create visualization
evaluator.visualize_single(
    initial_state=np.array([0.5, -0.3]),
    predicted_controls=predicted_controls,
    optimal_controls=optimal_controls,
    save_path="trajectory_analysis.png"
)
```

### Model Comparison

```python
# Evaluate multiple models
model_paths = [
    "models/sft_model",
    "models/grpo_model",
    "models/universal_model"
]

for model_path in model_paths:
    trainer = ControlTrainerV3(model_name=model_path)
    trainer.evaluate_on_benchmark("standard")

# Compare results
comparison = eval_manager.compare_models(
    ["sft_model", "grpo_model", "universal_model"],
    system="double_integrator"
)
```

## Visualization Examples

### Single Trajectory Visualization
The comprehensive plot shows:
- Top: Phase space with start/end points and trajectory direction
- Middle Left: Position trajectory with bounds and target
- Middle Right: Velocity trajectory with bounds and target
- Bottom: Control sequence with bounds

### Batch Comparison Visualization
Multiple plots showing:
- Phase space trajectories overlay
- Control effort comparison bar chart
- Final error comparison
- Convergence time analysis

## Metrics Explained

### Trajectory Metrics
- `final_state_error`: Euclidean distance from target state
- `control_effort`: Sum of squared control inputs
- `convergence_time`: Time to reach within threshold of target
- `phase_space_path_length`: Total distance traveled in phase space

### Constraint Metrics
- `position_violations`: Number of time steps violating position bounds
- `velocity_violations`: Number of time steps violating velocity bounds
- `control_violations`: Number of control inputs outside bounds

### Comparison Metrics (when optimal trajectory available)
- `position_rmse`: Root mean square error in position tracking
- `velocity_rmse`: Root mean square error in velocity tracking
- `control_rmse`: Root mean square error in control tracking
- `control_effort_ratio`: Ratio of predicted to optimal control effort

## Output Files

Evaluation results are saved in `evaluation_results/` directory:
- `{model}_{system}_eval_{timestamp}.pkl`: Full results with trajectories
- `{model}_{system}_eval_{timestamp}_summary.json`: Summary metrics
- `{model}_{system}_visualizations/`: Directory with all plots
  - `batch_evaluation.png`: Batch comparison plots
  - `batch_evaluation_heatmap.png`: Control sequence heatmap
  - `best_case_1.png`, `best_case_2.png`, ...: Best performing trajectories
  - `worst_case_1.png`, `worst_case_2.png`, ...: Worst performing trajectories

## Extending the Module

To add a new control system:

1. Create a new evaluator class inheriting from `BaseEvaluator`
2. Implement required methods:
   - `evaluate_single()`
   - `evaluate_batch()`
   - `visualize_single()`
   - `visualize_batch()`
3. Register in `EvaluationManager._register_evaluators()`

Example:
```python
from evaluation import BaseEvaluator

class VanDerPolEvaluator(BaseEvaluator):
    def __init__(self, dt: float = 0.1, steps: int = 100):
        super().__init__("van_der_pol", dt, steps)
        # Initialize system-specific components
        
    # Implement required methods...
```