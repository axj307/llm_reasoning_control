"""
Simple configuration file for all hyperparameters.
"""

# Model settings
MODEL_NAME = "unsloth/Qwen3-4B-Base"
MAX_SEQ_LENGTH = 2048
LORA_RANK = 32
GPU_MEMORY_UTILIZATION = 0.7

# Control system settings
DT = 0.1  # Time step duration
STEPS = 50  # Number of control steps

# Dataset settings
NUM_SAMPLES = 500  # Number of training samples

# SFT training settings
SFT_BATCH_SIZE = 4
SFT_EPOCHS = 2
SFT_LEARNING_RATE = 2e-4
SFT_WARMUP_STEPS = 5

# GRPO training settings
GRPO_BATCH_SIZE = 1
GRPO_NUM_GENERATIONS = 4
GRPO_MAX_STEPS = 50
GRPO_LEARNING_RATE = 5e-6
GRPO_TEMPERATURE = 1.0

# Common training settings
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "linear"
OPTIMIZER = "adamw_8bit"
SEED = 3407

# Output settings
OUTPUT_DIR = "outputs"
MODEL_SAVE_NAME = "di_control_lora"

# Formatting tokens
REASONING_START = "<REASONING>"
REASONING_END = "</REASONING>"
SOLUTION_START = "<CONTROLS>"
SOLUTION_END = "</CONTROLS>"