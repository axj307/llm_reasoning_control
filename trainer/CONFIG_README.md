# YAML Configuration System

## Overview

The YAML configuration system provides a flexible, hierarchical configuration management for the LLM control training pipeline. It supports:

- **Configuration inheritance** - Build on base configurations
- **Environment variables** - Use `${VAR_NAME:default}` syntax
- **Command-line overrides** - Override any parameter from CLI
- **Type validation** - Pydantic schemas ensure correct types
- **Multiple config files** - Merge configurations in order

## Directory Structure

```
configs/
├── base/
│   ├── base.yaml              # Base configuration with all defaults
│   ├── model.yaml             # Model presets (qwen3_4b, llama3_8b, etc.)
│   └── training.yaml          # Training strategy presets (sft, grpo, dpo)
├── environments/
│   ├── double_integrator.yaml # Double integrator specific settings
│   ├── van_der_pol.yaml       # Van der Pol oscillator settings
│   └── orbit_raising.yaml     # Orbit raising settings
├── experiments/
│   ├── quick_test.yaml        # Quick testing configuration
│   ├── full_training.yaml     # Full training configuration
│   └── ablation_study.yaml    # Ablation study settings
└── presets/
    ├── small_gpu.yaml         # Settings for 8GB GPUs
    ├── standard.yaml          # Standard settings
    └── high_performance.yaml  # High-end GPU settings
```

## Usage

### Basic Usage

```bash
# Use default configuration
python main_yaml.py

# Use specific configuration
python main_yaml.py --config configs/experiments/quick_test.yaml

# Use environment-specific config
python main_yaml.py --config configs/environments/double_integrator.yaml
```

### Command-Line Overrides

Override any configuration parameter:

```bash
# Override single parameter
python main_yaml.py --overrides training.learning_rate=1e-4

# Override multiple parameters
python main_yaml.py --overrides training.batch_size=8 lora.rank=64 model.name=unsloth/llama-3-8b

# Override nested parameters
python main_yaml.py --overrides control.dt=0.05 control.steps=100
```

### Multiple Configurations

Load multiple configs (later ones override earlier):

```bash
# Base + environment + experiment
python main_yaml.py --configs base/base.yaml,environments/double_integrator.yaml,experiments/quick_test.yaml

# Preset + custom
python main_yaml.py --configs presets/small_gpu.yaml,experiments/my_experiment.yaml
```

### Environment Variables

Use environment variables in configs:

```yaml
model:
  name: ${MODEL_NAME:unsloth/Qwen3-4B-Base}
  
gpu:
  memory_utilization: ${GPU_MEMORY:0.7}
```

Then:
```bash
MODEL_NAME=unsloth/llama-3-8b GPU_MEMORY=0.9 python main_yaml.py
```

## Configuration Examples

### Quick Test

```yaml
# configs/experiments/quick_test.yaml
defaults:
  - /base/base
  - /base/training@training: quick_test

dataset:
  num_samples: 50

training:
  max_steps: 10
  per_device_train_batch_size: 2
```

### Double Integrator

```yaml
# configs/environments/double_integrator.yaml
defaults:
  - /base/base

system:
  name: "double_integrator"
  description: "Double integrator control system"

control:
  dt: 0.1
  steps: 50
  control_bounds: [-3.0, 3.0]
```

### Small GPU Setup

```yaml
# configs/presets/small_gpu.yaml
defaults:
  - /base/base

model:
  max_seq_length: 2048
  
lora:
  rank: 16

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
```

## Python API

```python
from config import ConfigManager

# Create manager
config_manager = ConfigManager()

# Load configuration
config = config_manager.load("experiments/quick_test.yaml")

# Access values
print(config.model.name)
print(config.training.learning_rate)

# Get training config for specific strategy
sft_config = config_manager.get_training_config("sft")
grpo_config = config_manager.get_training_config("grpo")

# Save configuration
config_manager.save("outputs/my_config.yaml")
```

## Creating New Configurations

1. **For new experiments**: Create in `configs/experiments/`
2. **For new systems**: Create in `configs/environments/`
3. **For hardware presets**: Create in `configs/presets/`

### Example: New Experiment

```yaml
# configs/experiments/my_experiment.yaml
defaults:
  - /base/base
  - /environments/double_integrator

# Override specific settings
dataset:
  num_samples: 1000

training:
  num_train_epochs: 5
  learning_rate: 3e-4

lora:
  rank: 64
  
evaluation:
  num_test_cases: 20
```

## Validation

All configurations are validated using Pydantic schemas:

- **Type checking**: Ensures correct data types
- **Range validation**: e.g., learning_rate > 0
- **Required fields**: Missing required fields cause errors
- **Custom validation**: e.g., max_seq_length >= 2048

## Best Practices

1. **Use inheritance**: Start with `defaults:` to inherit settings
2. **Override sparingly**: Only override what you need to change
3. **Document configs**: Add comments explaining non-obvious settings
4. **Version control**: Keep configs in git for reproducibility
5. **Naming convention**: Use descriptive names for experiments

## Troubleshooting

### Common Issues

1. **File not found**: Check config path relative to `configs/` directory
2. **Validation errors**: Check Pydantic error messages for invalid values
3. **Override syntax**: Use dots for nested params: `training.learning_rate=0.001`
4. **YAML syntax**: Ensure proper indentation and valid YAML

### Debug Configuration

```bash
# See final merged configuration
python -c "from config import ConfigManager; cm = ConfigManager(); cm.load('your_config.yaml'); cm.save('debug.yaml')"
```

## Advanced Features

### Conditional Configuration

Use different settings based on environment:

```yaml
training:
  learning_rate: ${LR:2e-4}
  batch_size: ${BATCH_SIZE:4}
```

### Config Groups

Organize related configs:

```yaml
defaults:
  - base: base
  - model: qwen3_4b
  - training: sft
  - optimizer: adamw_8bit
```

This allows modular configuration composition.