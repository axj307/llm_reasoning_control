# Experiment Runner

You are tasked with running comprehensive training experiments with parameter sweeps and automated analysis for the Universal Control LLM Framework.

## Instructions

1. **Use the experiment-runner agent** to orchestrate systematic experiments
2. **Parameter Sweep Capabilities**:
   - Compare different LoRA ranks, learning rates, batch sizes
   - Test across multiple systems (double_integrator, van_der_pol)
   - Run both SFT and GRPO training variations
   - Generate comparison reports with statistical analysis

3. **Batch Training Management**:
   - Sequential or parallel execution based on resources
   - Automatic failure recovery and resubmission
   - Resource optimization for different experiment types
   - Progress monitoring and status reporting

4. **Comprehensive Analysis**:
   - Collect performance metrics from all experiments
   - Generate comparison visualizations and tables
   - Identify best performing configurations
   - Create detailed experiment reports with recommendations

## Usage Examples

- `/run-experiment` - Run comprehensive parameter sweep experiments
- Can specify systems, parameters, and analysis depth
- Provides automated experiment management and result analysis
- Generates actionable insights for optimal configurations

## Agent Integration

This command automatically invokes the `experiment-runner` agent with specialized knowledge of:
- Parameter sweep design and execution
- Batch training orchestration with resource management
- SLURM integration for HPC clusters
- Statistical analysis and result interpretation
- Automated report generation with recommendations

The agent handles everything from experiment planning to final analysis, making systematic experimentation effortless.