# Model Evaluation and Visualization Guide

This guide explains how to evaluate your trained SFT and GRPO models with comprehensive visualizations and analysis.

## Overview

The evaluation system provides three main capabilities:

1. **Comprehensive Model Evaluation** - Test individual models on diverse initial conditions
2. **Model Comparison** - Compare SFT vs GRPO vs Optimal control performance 
3. **Automatic Integration** - Built into the complete training pipeline

## Quick Start

### Run Complete Pipeline with Evaluation

The easiest way is to use the complete pipeline which includes evaluation:

```bash
# Submit complete pipeline (includes training + evaluation)
sbatch slurm/train_complete_pipeline.sbatch
```

This will:
- Train SFT model
- Train GRPO model  
- Evaluate both models on diverse test cases
- Generate comparison plots
- Create comprehensive visualizations

### Manual Evaluation

#### 1. Comprehensive Single Model Evaluation

```bash
# Evaluate SFT model
python scripts/comprehensive_model_evaluation.py \
    --model-path models/single_system/double_integrator/sft/latest \
    --model-type single_system \
    --system double_integrator \
    --test-type both \
    --grid-size 5 \
    --num-random 50 \
    --output-dir evaluation_results/sft \
    --save-results

# Evaluate GRPO model
python scripts/comprehensive_model_evaluation.py \
    --model-path models/single_system/double_integrator/grpo/latest \
    --model-type single_system \
    --system double_integrator \
    --test-type both \
    --grid-size 5 \
    --num-random 50 \
    --output-dir evaluation_results/grpo \
    --save-results
```

#### 2. Model Comparison

```bash
# Compare SFT vs GRPO vs Optimal
python scripts/model_comparison_evaluation.py \
    --system double_integrator \
    --model-types sft,grpo \
    --num-test-cases 50 \
    --comparison-cases 5 \
    --output-dir model_comparison_results \
    --save-results
```

## Evaluation Features

### Test Case Generation

**Grid Testing** (`--test-type grid`):
- Systematic grid of initial conditions
- Covers state space uniformly
- Good for finding failure regions

**Random Testing** (`--test-type random`):
- Random initial conditions within bounds
- Tests model robustness
- More realistic distribution

**Both** (`--test-type both`):
- Combines grid + random testing
- Most comprehensive evaluation

### Generated Visualizations

#### 1. Success Rate Heatmap
- Shows success/failure across initial conditions
- Identifies problematic regions
- Color-coded success rates

#### 2. Trajectory Comparison Grid  
- Model vs optimal trajectories
- Multiple test cases side-by-side
- Phase space plots with success indicators

#### 3. Performance Summary
- Final error distributions
- Success rate vs initial distance
- Control effort analysis
- Statistical summaries

#### 4. Model Comparison Plots
- SFT vs GRPO vs Optimal performance
- Success rate comparisons
- Control effort comparisons
- Detailed trajectory comparisons

## Output Structure

After running evaluation, you'll get:

```
evaluation_results/
├── sft/
│   ├── double_integrator_success_heatmap.png
│   ├── double_integrator_trajectory_comparison.png
│   ├── double_integrator_performance_summary.png
│   └── double_integrator_evaluation_results.json
├── grpo/
│   ├── double_integrator_success_heatmap.png
│   ├── double_integrator_trajectory_comparison.png
│   ├── double_integrator_performance_summary.png
│   └── double_integrator_evaluation_results.json
└── comparison/
    ├── double_integrator_performance_comparison.png
    ├── double_integrator_comparison_case_1.png
    ├── double_integrator_comparison_case_2.png
    ├── ...
    └── double_integrator_comparison_results.json
```

## Key Metrics

### Success Metrics
- **Success Rate**: Percentage reaching target region (||state|| < 0.1)
- **Mean Final Error**: Average distance from target at end
- **Success vs Initial Distance**: How success rate varies with starting distance

### Performance Metrics  
- **Control Effort**: Total absolute control input
- **Trajectory Smoothness**: Control input variations
- **Convergence Speed**: Time to reach target region

### Comparison Metrics
- **Relative Performance**: SFT vs GRPO improvement
- **Robustness**: Consistency across test cases
- **Optimality Gap**: Distance from optimal control performance

## Understanding Results

### Good Model Indicators
- ✅ High success rate (>80%)
- ✅ Low mean final error (<0.05)
- ✅ Consistent performance across initial conditions
- ✅ Smooth control trajectories

### Potential Issues
- ❌ Low success rate (<50%)
- ❌ High variance in performance
- ❌ Failure in specific state regions
- ❌ Erratic control behavior

### SFT vs GRPO Expected Patterns
- **SFT**: More consistent but potentially suboptimal
- **GRPO**: Better performance but potentially more variable
- **Optimal**: Best performance baseline for comparison

## Configuration Options

### Test Configuration
```bash
--test-type both          # Grid + random testing
--grid-size 7            # 7x7 grid = 49 systematic test points
--num-random 100         # 100 additional random test cases
--temperature 0.1        # Low temperature for consistent output
```

### Visualization Options
```bash
--save-results           # Save JSON results for analysis
--show-plots            # Display plots interactively
--output-dir my_results # Custom output directory
```

### Hardware Options
```bash
--gpu-id 0              # Use specific GPU
```

## Integration with Training Pipeline

The complete pipeline (`slurm/train_complete_pipeline.sbatch`) automatically:

1. **Trains Models**: SFT → GRPO sequence
2. **Finds Models**: Automatically locates latest trained models
3. **Evaluates Both**: Runs comprehensive evaluation on each
4. **Compares Models**: SFT vs GRPO comparison if both exist
5. **Organizes Output**: Creates structured results in `figures/job_${SLURM_JOB_ID}/`

## Troubleshooting

### Model Not Found
```
❌ SFT model not found at: models/single_system/double_integrator/sft/latest
```
**Solution**: Ensure training completed successfully and model was saved.

### Evaluation Errors
```
❌ Failed evaluation: CUDA out of memory
```
**Solution**: Reduce `--num-test-cases` or use smaller `--grid-size`.

### No Valid Predictions
```
Success rate: 0.0% (0/50 cases)
```
**Solution**: Check model format, ensure chat template is correct, verify model training.

## Advanced Usage

### Custom Test Cases
Modify the evaluation scripts to test specific challenging scenarios:

```python
# Add to comprehensive_model_evaluation.py
custom_test_cases = [
    (0.9, 0.9),   # Near boundary
    (-0.9, -0.9), # Opposite corner
    (0.1, -0.9),  # High velocity
    # Add your specific test cases
]
```

### Additional Metrics
Extend the evaluation with domain-specific metrics:

```python
# Add to metrics computation
settling_time = compute_settling_time(trajectory)
overshoot = compute_overshoot(trajectory)
steady_state_error = compute_steady_state_error(trajectory)
```

This evaluation system provides comprehensive analysis of your trained models, helping you understand their strengths, weaknesses, and readiness for deployment.