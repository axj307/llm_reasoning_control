# Universal Control LLM Framework

A modular framework for training Large Language Models to solve control problems using Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO). This framework supports multiple control systems and can train both specialist models for individual systems and universal models that can control any trained system.

## 🌟 Key Features

- **Universal Control**: One model that can control multiple different systems
- **Extensible Architecture**: Easy to add new control systems
- **Multiple Training Modes**: SFT → GRPO pipeline with specialist and universal training
- **Advanced Evaluation**: Comprehensive metrics, visualization, and MPC-style inference
- **Production Ready**: Model versioning, checkpointing, and Slurm-compatible scripts

## 🏗️ Architecture

```
llm_reasoning_control/
├── config.py                     # YAML-based configuration loader
├── configs/                      # Environment-specific configurations
│   ├── base_config.yaml         # Common settings (model, training)
│   ├── double_integrator.yaml   # DI-specific parameters
│   └── van_der_pol.yaml         # VDP-specific parameters
├── core/                         # Core framework modules
│   ├── data_pipeline.py         # Multi-system data generation
│   ├── model_manager.py         # Model saving/loading with versioning
│   └── solvers/                 # Optimal control solvers
│       ├── lqr_solver.py        # LQR for linear systems
│       └── vdp_solver.py        # Numerical optimization for VDP
├── environments/                 # Control environments
│   ├── base_environment.py      # Base class for all environments
│   ├── double_integrator.py     # Double integrator (ẍ = u)
│   └── van_der_pol.py          # Van der Pol oscillator
├── training/                     # Training implementations
│   ├── sft_training.py          # Supervised fine-tuning
│   └── grpo_training.py         # Group relative policy optimization
├── evaluation/                   # Evaluation pipeline
│   ├── inference.py             # Standard and MPC inference
│   ├── metrics.py               # Performance metrics
│   └── visualization.py         # Plotting and visualization
├── scripts/                      # Training and evaluation scripts
│   ├── generate_data.py         # Standalone data generation
│   ├── train_single_system.py   # Single-environment training
│   ├── train_universal.py       # Multi-environment training
│   └── evaluate_model.py        # Model evaluation
├── datasets/                     # Pre-generated training/eval data
├── models/                       # Saved model organization
│   ├── single_system/           # Environment-specific specialists
│   │   ├── double_integrator/   # DI models (sft/, grpo/)
│   │   └── van_der_pol/         # VDP models (sft/, grpo/)
│   └── universal/               # Multi-environment models
└── gpu_utils.py                 # Automatic GPU selection
```

## 🚀 Quick Start

### 1. Activate Environment
```bash
conda activate unsloth_env
```

### 2. Generate Datasets (Once per System)

Generate clean, separate datasets for each system:

```bash
# Generate Double Integrator dataset
python scripts/generate_data.py \
    --systems double_integrator \
    --total-samples 2000 \
    --split-ratio 0.9 \
    --dataset-name di

# Generate Van der Pol dataset  
python scripts/generate_data.py \
    --systems van_der_pol \
    --total-samples 2000 \
    --split-ratio 0.9 \
    --dataset-name vdp

# Check available datasets
python scripts/train_single_system.py --list-datasets
```

This creates simple, clean datasets:
- `di_train.pkl` + `di_eval.pkl` + `di_info.json`
- `vdp_train.pkl` + `vdp_eval.pkl` + `vdp_info.json`

### 3. Mix and Match Training

Use your clean datasets for any training combination:

```bash
# Train DI specialist model
python scripts/train_single_system.py \
    --system double_integrator \
    --dataset-name di \
    --training-type both \
    --lora-rank 8

# Train VDP specialist model  
python scripts/train_single_system.py \
    --system van_der_pol \
    --dataset-name vdp \
    --training-type both \
    --lora-rank 8

# Train universal model (mix datasets during training)
python scripts/train_universal.py \
    --systems double_integrator,van_der_pol \
    --datasets di,vdp \
    --training-type both \
    --lora-rank 8
```

**Outputs**: 
- DI specialist: `models/single_system/double_integrator/sft|grpo/latest/`
- VDP specialist: `models/single_system/van_der_pol/sft|grpo/latest/`  
- Universal model: `models/universal/sft|grpo/latest/`

### 4. Evaluate Any Model on Any Dataset

Use your clean datasets to evaluate any trained model:

```bash
# Evaluate DI specialist on DI dataset
python scripts/evaluate_model.py \
    --model-path models/single_system/double_integrator/grpo/latest \
    --model-type single_system \
    --eval-dataset di \
    --save-plots

# Evaluate universal model on DI dataset
python scripts/evaluate_model.py \
    --model-path models/universal/grpo/latest \
    --model-type universal \
    --systems double_integrator \
    --eval-dataset di \
    --save-plots

# Evaluate universal model on VDP dataset  
python scripts/evaluate_model.py \
    --model-path models/universal/grpo/latest \
    --model-type universal \
    --systems van_der_pol \
    --eval-dataset vdp \
    --save-plots
```

## 🗂️ Clean Dataset Pipeline

### Simple Dataset Structure
```bash
# Check what datasets you have
python scripts/train_single_system.py --list-datasets

# Example output:
# 📂 Available datasets:
#    • di
#    • vdp
```

### Dataset Naming Convention
- `{system}` (e.g., `di`, `vdp`)
- Super simple, clean, one dataset per system

### Mix and Match During Training
```bash
# Generate clean datasets once
python scripts/generate_data.py --systems double_integrator --total-samples 2000 --dataset-name di
python scripts/generate_data.py --systems van_der_pol --total-samples 2000 --dataset-name vdp

# Mix and match for any training experiment:
# Specialist training
python scripts/train_single_system.py --system double_integrator --dataset-name di --training-type sft
python scripts/train_single_system.py --system double_integrator --dataset-name di --training-type both --lora-rank 16

# Universal training (mix datasets)
python scripts/train_universal.py --systems double_integrator,van_der_pol --datasets di,vdp --training-type both

# Cross-evaluation (any model on any dataset)
python scripts/evaluate_model.py --model-path models/universal/grpo/latest --eval-dataset di
python scripts/evaluate_model.py --model-path models/universal/grpo/latest --eval-dataset vdp
```

## 🔄 Progressive Training Workflow

Simple workflow using clean datasets:

### 1. Generate Clean Datasets (Once)

```bash
# Generate separate clean datasets
python scripts/generate_data.py --systems double_integrator --total-samples 2000 --dataset-name di
python scripts/generate_data.py --systems van_der_pol --total-samples 2000 --dataset-name vdp
```

### 2. Train Models (Mix and Match)

```bash
# Train DI specialist
python scripts/train_single_system.py --system double_integrator --dataset-name di --training-type both

# Train VDP specialist  
python scripts/train_single_system.py --system van_der_pol --dataset-name vdp --training-type both

# Train universal model (mix clean datasets)
python scripts/train_universal.py --systems double_integrator,van_der_pol --datasets di,vdp --training-type both
```

**Clean Model Organization:**
- DI specialist: `models/single_system/double_integrator/grpo/latest/`
- VDP specialist: `models/single_system/van_der_pol/grpo/latest/`
- Universal model: `models/universal/grpo/latest/`

### 3. Transfer Learning (Optional)

Use your trained DI model to bootstrap training on new environments:

```bash
# Start VDP training from trained DI model
python scripts/train_universal.py \
    --systems van_der_pol \
    --datasets vdp \
    --training-type both \
    --base-model-path models/single_system/double_integrator/grpo/latest

# Create universal model by extending DI model
python scripts/train_universal.py \
    --systems double_integrator,van_der_pol \
    --datasets di,vdp \
    --training-type both \
    --base-model-path models/single_system/double_integrator/grpo/latest
```

**Benefits:**
- Faster convergence on new environments
- Preserves existing knowledge
- Reduced training time

### 5. Compare Model Performance

```bash
# Evaluate universal model
python scripts/evaluate_model.py \
    --model-path models/universal/grpo/latest \
    --model-type universal \
    --num-test-cases 10

# List all available models
python scripts/list_models.py --detailed
```

## 🎓 Detailed Training Workflows

### SFT Training Only (Separate)

Train just the Supervised Fine-Tuning model without GRPO:

```bash
# Generate data first
python scripts/generate_data.py \
    --systems double_integrator \
    --train-samples 500 \
    --eval-samples 100 \
    --dataset-name di_sft_only_500_100

# Train SFT model only
python scripts/train_single_system.py \
    --system double_integrator \
    --dataset-name di_sft_only_500_100 \
    --training-type sft \
    --lora-rank 8 \
    --max-seq-length 1024
```

**Result**: SFT model saved to `models/single_system/double_integrator/sft/latest/`

### SFT + GRPO Training (Combined)

Train the complete pipeline in one run:

```bash
# Use same dataset as above or generate new one
python scripts/train_single_system.py \
    --system double_integrator \
    --dataset-name di_sft_only_500_100 \
    --training-type both \
    --lora-rank 8 \
    --max-seq-length 1024
```

**Results**: 
- SFT model: `models/single_system/double_integrator/sft/latest/`
- GRPO model: `models/single_system/double_integrator/grpo/latest/`

### GRPO Training Only (Using Existing SFT)

Train GRPO starting from a pre-trained SFT model:

```bash
python scripts/train_single_system.py \
    --system double_integrator \
    --dataset-name di_sft_only_500_100 \
    --training-type grpo \
    --load-sft-model models/single_system/double_integrator/sft/latest \
    --lora-rank 8
```

**Result**: GRPO model saved to `models/single_system/double_integrator/grpo/latest/`

### Model Loading in Jupyter Notebooks

Load trained models for evaluation and testing:

```python
import sys
sys.path.append('.')

from core.model_manager import UniversalModelManager

# Load SFT model
manager = UniversalModelManager()
sft_model, sft_tokenizer, sft_lora, sft_metadata = manager.load_single_system_model("double_integrator", model_type="sft")

# Load GRPO model  
grpo_model, grpo_tokenizer, grpo_lora, grpo_metadata = manager.load_single_system_model("double_integrator", model_type="grpo")

# Load universal model
universal_model, universal_tokenizer, universal_lora, universal_metadata = manager.load_universal_model()

print("Models loaded successfully!")
```

### 4. Interactive Testing (Jupyter)

```bash
jupyter notebook notebooks/test_universal_model.ipynb
```

## 🎯 Supported Systems

### Current Systems

- **Double Integrator**: `ẍ = u`
  - State bounds: position, velocity ∈ [-1, 1]
  - Control bounds: u ∈ [-3, 3]
  - Solver: LQR (Linear Quadratic Regulator)

- **Van der Pol Oscillator**: `ẍ - μ(1-x²)ẋ + x = u`
  - State bounds: position, velocity ∈ [-2, 2]
  - Control bounds: u ∈ [-5, 5]
  - Solver: Numerical optimization

### Adding New Systems

1. Create new system class in `systems/` inheriting from `BaseSystem`
2. Implement required methods: `simulate_step`, `get_bounds`, etc.
3. Add corresponding solver in `solvers/`
4. Update `systems/__init__.py` and `solvers/__init__.py`
5. Train model with new system:

```bash
python scripts/add_new_system.py --new-system your_system --num-samples 200
```

## 📊 Training Pipeline

### 1. Data Generation
- Generates training examples with optimal control solutions
- Creates reasoning explanations for each solution
- Supports both single-system and mixed-system datasets

### 2. Supervised Fine-Tuning (SFT)
- Teaches model the format and basic control knowledge
- Uses teacher-forcing with optimal solutions
- Configurable epochs, batch size, learning rate

### 3. Group Relative Policy Optimization (GRPO)
- Reinforcement learning with multiple reward functions:
  - Format matching (exact and approximate)
  - Control performance (final error, constraint satisfaction)
  - Trajectory quality (smoothness, optimality)

## 🎮 Model Types

### Universal Models
- Trained on multiple systems simultaneously
- Can switch between different control problems
- Stored in `models/universal/`

### Specialist Models
- Trained on single systems for maximum performance
- Optimized for specific control problems
- Stored in `models/single_system/{system}/`

### Model Versioning
- Automatic timestamping and metadata tracking
- Easy loading with `latest` symlinks
- Supports incremental learning and system extension

## 📈 Evaluation Features

### Standard Evaluation
- Batch inference on multiple test cases
- Performance metrics vs optimal solutions
- Constraint satisfaction analysis
- Visualization of trajectories and phase portraits

### MPC-Style Evaluation
- Model Predictive Control simulation
- Receding horizon inference
- Robustness to prediction errors

### Metrics
- Final state error
- Control effort and smoothness
- LQR cost comparison
- Constraint violation rates
- Overall performance scores

## 🛠️ Configuration

All settings centralized in `config.py`:

```python
# Model settings
MODEL_CONFIG = {
    "base_model_name": "unsloth/Qwen3-4B-Base",
    "max_seq_length": 2048,
    "lora_rank": 16,
}

# Training settings
SFT_CONFIG = {
    "learning_rate": 2e-4,
    "num_train_epochs": 4,
    # ... other SFT parameters
}

GRPO_CONFIG = {
    "learning_rate": 5e-6,
    "max_steps": 100,
    # ... other GRPO parameters
}
```

## 📝 Usage Examples

### Training Examples

```bash
# Universal model with custom configuration
python scripts/train_universal.py \
    --systems double_integrator,van_der_pol \
    --samples-per-system 300 \
    --lora-rank 32 \
    --output-base ./my_training

# Single system specialist
python scripts/train_single_system.py \
    --system van_der_pol \
    --num-samples 800 \
    --training-type grpo \
    --load-sft-model models/single_system/van_der_pol/sft/latest
```

### Evaluation Examples

```bash
# Comprehensive evaluation
python scripts/evaluate_model.py \
    --model-path models/universal/grpo/latest \
    --model-type universal \
    --systems double_integrator,van_der_pol \
    --test-type both \
    --num-test-cases 20 \
    --save-plots

# MPC evaluation only
python scripts/evaluate_model.py \
    --model-path models/universal/grpo/latest \
    --model-type universal \
    --test-type mpc \
    --mpc-horizon 5
```

### Model Management

```bash
# List all models
python scripts/list_models.py

# Filter by type and system
python scripts/list_models.py \
    --model-type single_system \
    --system double_integrator \
    --detailed

# Add new system to existing universal model
python scripts/add_new_system.py \
    --new-system pendulum \
    --base-model latest \
    --incremental
```

## 🔬 Research Applications

This framework supports research in:

- **LLM Control**: Training language models for control tasks
- **Multi-System Learning**: Universal models vs specialists
- **Reasoning in Control**: Interpretable control decisions
- **Few-Shot Control**: Quick adaptation to new systems
- **Safe Control**: Constraint-aware control generation

## 📋 Requirements

```
torch>=2.0.0
transformers>=4.30.0
unsloth
trl>=0.7.0
vllm
datasets>=2.0.0
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.5.0
pyyaml>=6.0
wandb>=0.15.0  # Optional, for experiment tracking
```

## 🤝 Contributing

1. **Add New Systems**: Follow the `BaseSystem` interface
2. **Improve Solvers**: Add better optimal control algorithms
3. **Enhance Evaluation**: Add new metrics and visualization
4. **Training Improvements**: Better reward functions, training strategies

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on [Unsloth](https://github.com/unslothai/unsloth) for efficient LLM fine-tuning
- Uses [TRL](https://github.com/huggingface/trl) for GRPO implementation
- Inspired by recent advances in LLM reasoning and control theory

---

## 📞 Getting Help

- **Issues**: Report bugs or request features via GitHub Issues
- **Discussions**: Ask questions in GitHub Discussions
- **Documentation**: Check the `notebooks/` directory for examples
- **Configuration**: Run `python config.py` to validate your setup

## 🌟 Complete Clean Workflow

Since you already have DI dataset, here's your simplified workflow:

### Step 1: Prepare Clean Datasets
```bash
# List your available datasets
python scripts/train_single_system.py --list-datasets

# If you don't have clean DI dataset, generate it:
python scripts/generate_data.py --systems double_integrator --total-samples 2000 --dataset-name di

# Generate VDP dataset
python scripts/generate_data.py --systems van_der_pol --total-samples 2000 --dataset-name vdp
```

### Step 2: Train Your Models
```bash
# Train DI specialist  
python scripts/train_single_system.py --system double_integrator --dataset-name di --training-type both

# Train universal model (mix clean datasets)
python scripts/train_universal.py --systems double_integrator,van_der_pol --datasets di,vdp --training-type both
```

### Step 3: Cross-Evaluate Everything
```bash
# Test DI specialist on DI data
python scripts/evaluate_model.py --model-path models/single_system/double_integrator/grpo/latest --eval-dataset di --save-plots

# Test universal model on DI data  
python scripts/evaluate_model.py --model-path models/universal/grpo/latest --eval-dataset di --save-plots

# Test universal model on VDP data
python scripts/evaluate_model.py --model-path models/universal/grpo/latest --eval-dataset vdp --save-plots
```

### Step 4: Load in Jupyter
```python
from core.model_manager import UniversalModelManager

manager = UniversalModelManager()
model, tokenizer, lora_request, metadata = manager.load_universal_model()
print(f"Model trained on: {metadata.get('trained_systems', [])}")
```

**Clean Workflow Benefits:**
- ✅ One clean dataset per system (di, vdp)
- ✅ Mix and match datasets during training
- ✅ Cross-evaluate any model on any dataset
- ✅ Super simple naming, no confusion
- ✅ Focus on LLM training, not data generation

Happy controlling! 🎛️🤖