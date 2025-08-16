# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Universal Control LLM Framework - A modular framework for training Large Language Models to solve control problems using Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO). Supports both specialist models for individual control systems and universal models that can control multiple systems.

## Key Commands

### Environment Setup
```bash
conda activate unsloth_env
```

### Data Generation (Run Once)
```bash
# Generate datasets for control systems
python scripts/generate_data.py --systems double_integrator --total-samples 2000 --split-ratio 0.9 --dataset-name di
python scripts/generate_data.py --systems van_der_pol --total-samples 2000 --split-ratio 0.9 --dataset-name vdp

# List available datasets
python scripts/train_single_system.py --list-datasets
```

### Training
```bash
# Train single system (SFT + GRPO)
python scripts/train_single_system.py --system double_integrator --dataset-name di --training-type both --lora-rank 8

# Train universal model
python scripts/train_universal.py --systems double_integrator,van_der_pol --datasets di,vdp --training-type both --lora-rank 8
```

### Evaluation
```bash
# Evaluate models
python scripts/evaluate_model.py --model-path models/single_system/double_integrator/grpo/latest --model-type single_system --eval-dataset di --save-plots

# List available models
python scripts/list_models.py --detailed
```

### Linting & Testing
```bash
# Code formatting
black .

# Linting
flake8 .

# Run tests
pytest
```

## Architecture Overview

The framework follows a modular design with clear separation of concerns:

### Core Components

1. **Control Environments** (`environments/`)
   - Base class: `BaseEnvironment` defines interface for all control systems
   - Each system (e.g., `double_integrator.py`, `van_der_pol.py`) implements dynamics, bounds, and simulation
   - New systems can be added by extending `BaseEnvironment`

2. **Optimal Control Solvers** (`core/solvers/`)
   - `lqr_solver.py`: Linear Quadratic Regulator for linear systems
   - `vdp_solver.py`: Numerical optimization for nonlinear systems
   - Solvers generate optimal trajectories used for training

3. **Training Pipeline** (`training/`)
   - `sft_training.py`: Supervised fine-tuning on optimal control solutions
   - `grpo_training.py`: Reinforcement learning with multiple reward components
   - Two-stage training: SFT provides baseline, GRPO improves performance

4. **Data Pipeline** (`core/data_pipeline.py`)
   - Generates training examples with optimal control solutions
   - Creates reasoning explanations for interpretability
   - Supports mixed datasets for universal models

5. **Model Management** (`core/model_manager.py`)
   - Versioning with timestamps and metadata
   - Automatic organization by model type and system
   - Easy loading with `latest` symlinks

### Model Types

- **Specialist Models**: Trained on single control system, stored in `models/single_system/{system}/`
- **Universal Models**: Trained on multiple systems, stored in `models/universal/`

### Key Design Patterns

1. **Configuration-Driven**: YAML configs in `configs/` directory control all aspects
2. **LoRA Adapters**: Efficient fine-tuning of large models (Qwen3-4B-Base)
3. **Clean Data Pipeline**: Separate data generation from training for reproducibility
4. **Automatic GPU Management**: `gpu_utils.py` handles GPU selection

### Adding New Control Systems

1. Create new environment class extending `BaseEnvironment`
2. Implement required methods: `simulate_step()`, `get_bounds()`, `compute_reward()`
3. Add corresponding optimal control solver if needed
4. Update `__init__.py` files to register the new system
5. Generate data and train using existing scripts

### Important Files

- `config.py`: Central configuration loader and defaults
- `data_utils.py`: Data processing and formatting utilities
- `evaluation/metrics.py`: Performance metrics computation
- `evaluation/visualization.py`: Plotting trajectories and phase portraits

The framework is designed for research in LLM-based control, with emphasis on interpretability, extensibility, and fair comparison between specialist and universal models.