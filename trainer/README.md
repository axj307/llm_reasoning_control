# Control Trainer

A modular implementation for training language models on control tasks using Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO).

## Overview

This package provides a clean, easy-to-understand implementation for training models to solve double integrator control problems. The code is organized into simple modules focusing on clarity over complexity.

## Installation

Make sure you have the conda environment activated:
```bash
conda activate unsloth_env
```

## Usage

### Quick Start

Train a model with default settings:
```bash
python trainer/main.py --mode train
```

### Command Line Options

```bash
python trainer/main.py [options]

Options:
  --mode {train,eval,both}  Training mode (default: both)
  --num-samples N          Number of training samples (default: 500)
  --sft-epochs N           Number of SFT epochs (default: 2)
  --grpo-steps N           Number of GRPO steps (default: 50)
  --gpu N                  GPU to use (default: random)
  --skip-sft               Skip SFT training
  --skip-grpo              Skip GRPO training
```

### Examples

1. **Train with more samples:**
   ```bash
   python trainer/main.py --mode train --num-samples 1000
   ```

2. **Evaluate only:**
   ```bash
   python trainer/main.py --mode eval
   ```

3. **Train SFT only:**
   ```bash
   python trainer/main.py --mode train --skip-grpo
   ```

4. **Use specific GPU:**
   ```bash
   python trainer/main.py --gpu 0
   ```

### Python API

```python
from trainer import ControlTrainer, create_dataset

# Create trainer
trainer = ControlTrainer()

# Generate dataset
dataset = create_dataset(num_samples=500)

# Setup and train
trainer.setup_model()
trainer.train(dataset)

# Generate control sequence
prompt = "Control a double integrator system with initial state [position=0.5, velocity=-0.3]..."
output = trainer.generate(prompt)
print(output)
```

## Module Structure

- `config.py` - All hyperparameters in one place
- `control.py` - LQR solver and system dynamics
- `data.py` - Dataset generation
- `rewards.py` - GRPO reward functions
- `trainer.py` - Simple combined trainer class
- `utils.py` - Visualization and helpers
- `logger.py` - Basic logging setup
- `main.py` - Command-line interface

## Key Features

- **Simple Design**: Easy to understand and modify
- **Modular Structure**: Clear separation of concerns
- **Configurable**: All settings in one config file
- **Logging**: Built-in logging support
- **Visualization**: Plot control trajectories

## Customization

To modify hyperparameters, edit `config.py`:
```python
# Example: Change LoRA rank
LORA_RANK = 64  # Default: 32

# Example: Change learning rates
SFT_LEARNING_RATE = 1e-4  # Default: 2e-4
GRPO_LEARNING_RATE = 1e-5  # Default: 5e-6
```

## Output

Models are saved to `outputs/` directory:
- SFT checkpoints: `outputs/sft/`
- GRPO checkpoints: `outputs/grpo/`
- Final LoRA model: `di_control_lora/`

## Troubleshooting

1. **Out of memory**: Reduce batch size in `config.py`
2. **Training too slow**: Decrease number of GRPO generations
3. **Poor performance**: Increase training samples or epochs

## Next Steps

This simple implementation can be extended by:
- Adding more control systems beyond double integrator
- Implementing different reward functions
- Supporting multi-GPU training
- Adding more sophisticated logging and monitoring