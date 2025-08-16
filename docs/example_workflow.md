# 📋 **Separate Data Pipeline Workflow**

This guide shows how to use the separated data generation and training pipeline for efficient experimentation.

## 🗂️ **Workflow Overview**

1. **Generate Data Once** → Use for multiple training runs
2. **Train Models** → Load pre-generated data instantly  
3. **Evaluate Models** → Use pre-generated evaluation data

---

## 📊 **Step 1: Generate Training & Evaluation Data**

### Generate data for a single system:
```bash
python scripts/generate_data.py \
    --systems double_integrator \
    --train-samples 400 \
    --eval-samples 100 \
    --dataset-name di_400train_100eval
```

### Generate data for multiple systems:
```bash
python scripts/generate_data.py \
    --systems double_integrator van_der_pol \
    --train-samples 300 \
    --eval-samples 100 \
    --dataset-name multi_system_300_100
```

### Auto-split from total samples:
```bash
python scripts/generate_data.py \
    --systems double_integrator \
    --total-samples 500 \
    --split-ratio 0.8 \
    --dataset-name di_500_autosplit
```

### Preview data before generating:
```bash
python scripts/generate_data.py \
    --systems double_integrator \
    --preview
```

**Output**: Creates files in `datasets/` directory:
- `{dataset_name}_train.pkl` - Training data
- `{dataset_name}_eval.pkl` - Evaluation data  
- `{dataset_name}_info.json` - Dataset metadata

---

## 🏋️ **Step 2: Train Models Using Pre-Generated Data**

### List available datasets:
```bash
python scripts/train_single_system.py --list-datasets
```

### Train using pre-generated dataset:
```bash
python scripts/train_single_system.py \
    --system double_integrator \
    --dataset-name di_400train_100eval \
    --training-type sft
```

### Train with GRPO:
```bash
python scripts/train_single_system.py \
    --system double_integrator \
    --dataset-name di_400train_100eval \
    --training-type both
```

### Force generate new data (ignore pre-generated):
```bash
python scripts/train_single_system.py \
    --system double_integrator \
    --generate-data \
    --num-samples 200 \
    --training-type sft
```

---

## 📈 **Step 3: Evaluate Using Pre-Generated Test Data**

### Evaluate with pre-generated eval dataset:
```bash
python scripts/evaluate_model.py \
    --model-path models/single_system/double_integrator/sft/latest \
    --model-type single_system \
    --eval-dataset di_400train_100eval
```

### Evaluate with direct file path:
```bash
python scripts/evaluate_model.py \
    --model-path models/single_system/double_integrator/sft/latest \
    --model-type single_system \
    --eval-data-file datasets/di_400train_100eval_eval.pkl
```

### Generate new test cases (traditional method):
```bash
python scripts/evaluate_model.py \
    --model-path models/single_system/double_integrator/sft/latest \
    --model-type single_system \
    --num-test-cases 50
```

---

## 🚀 **Benefits of Separate Data Pipeline**

### ✅ **Efficiency**
- Generate optimal trajectories once, use many times
- No repeated expensive numerical simulations
- Faster training iterations

### ✅ **Reproducibility** 
- Same data across all experiments
- Consistent evaluation benchmarks
- Better comparison between models

### ✅ **Flexibility**
- Mix different data sizes easily
- Compare models on identical test sets
- Archive datasets for future use

### ✅ **Resource Management**
- Generate data on high-compute nodes
- Train models on different hardware
- Separate data generation from GPU usage

---

## 📁 **Directory Structure**

```
datasets/
├── di_400train_100eval_train.pkl          # Training data
├── di_400train_100eval_eval.pkl           # Evaluation data
├── di_400train_100eval_info.json          # Dataset metadata
├── multi_system_300_100_train.pkl         # Multi-system training
├── multi_system_300_100_eval.pkl          # Multi-system evaluation
└── multi_system_300_100_info.json         # Multi-system metadata

models/
├── single_system/
│   ├── double_integrator/
│   │   ├── sft/latest/                     # Trained on di_400train_100eval
│   │   └── grpo/latest/                    # Trained on di_400train_100eval
└── universal/
    ├── sft/latest/                         # Trained on multi_system_300_100
    └── grpo/latest/                        # Trained on multi_system_300_100
```

---

## 🎯 **Example Complete Workflow**

```bash
# 1. Generate data (once)
python scripts/generate_data.py \
    --systems double_integrator \
    --train-samples 500 \
    --eval-samples 100 \
    --dataset-name di_baseline_500_100

# 2. Train SFT model
python scripts/train_single_system.py \
    --system double_integrator \
    --dataset-name di_baseline_500_100 \
    --training-type sft

# 3. Train GRPO model  
python scripts/train_single_system.py \
    --system double_integrator \
    --dataset-name di_baseline_500_100 \
    --training-type grpo

# 4. Evaluate both models on same test data
python scripts/evaluate_model.py \
    --model-path models/single_system/double_integrator/sft/latest \
    --model-type single_system \
    --eval-dataset di_baseline_500_100 \
    --save-plots

python scripts/evaluate_model.py \
    --model-path models/single_system/double_integrator/grpo/latest \
    --model-type single_system \
    --eval-dataset di_baseline_500_100 \
    --save-plots
```

Now you can experiment with different:
- LoRA ranks
- Learning rates  
- Training epochs
- Model architectures

All while using the **exact same training and evaluation data** for fair comparisons! 🎉