# Configuration Optimizer

You are tasked with optimizing training configurations and hyperparameters for the Universal Control LLM Framework.

## Instructions

1. **Use the config-optimizer agent** for intelligent configuration optimization
2. **Hardware-Aware Optimization**:
   - Optimize for specific GPU memory constraints
   - Adjust batch sizes and gradient accumulation
   - Select optimal LoRA ranks and sequence lengths
   - Balance performance vs resource usage

3. **System-Specific Tuning**:
   - Tailor configurations for linear vs nonlinear systems
   - Optimize SFT and GRPO hyperparameters separately
   - Adjust learning rates and training schedules
   - Configure universal vs specialist model parameters

4. **Automated Hyperparameter Search**:
   - Bayesian optimization for complex parameter spaces
   - Grid search for systematic exploration
   - Multi-objective optimization (performance vs efficiency)
   - Statistical validation of optimal configurations

5. **Configuration Templates**:
   - Generate pre-optimized templates for common scenarios
   - Quick development vs production configurations
   - Resource-constrained optimization strategies
   - Custom configuration generation

## Usage Examples

- `/optimize-config` - Optimize training configuration for specific requirements
- Can target performance, speed, or memory efficiency
- Provides automated hyperparameter search capabilities
- Generates validated configurations with performance estimates

## Agent Integration

This command automatically invokes the `config-optimizer` agent with specialized knowledge of:
- Transformer architecture and LoRA fine-tuning optimization
- Hardware resource management and memory profiling
- Hyperparameter search algorithms and statistical validation
- System-specific optimization strategies
- Configuration template generation and management

The agent provides intelligent configuration optimization that maximizes performance while respecting constraints, eliminating manual hyperparameter tuning.