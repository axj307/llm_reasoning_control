# Results Directory

This directory stores important plots and figures that should be version controlled.

## Directory Structure:

- `final_plots/` - Final evaluation plots and comparisons
- `paper_figures/` - Publication-ready figures
- `benchmarks/` - Performance benchmark results

## Usage:

When you generate important plots that you want to keep in Git:

```python
# In your notebooks or scripts
plt.savefig('results/final_plots/di_grpo_performance.png', dpi=300)
plt.savefig('results/paper_figures/figure_1_control_comparison.pdf')
```

These plots will be tracked by Git while temporary plots in `plots/` will be ignored.