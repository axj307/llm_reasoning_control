"""Configuration file for the universal control LLM system."""

from typing import Dict, List, Any

# ========================
# Model Configuration
# ========================
MODEL_CONFIG = {
    "base_model_name": "unsloth/Qwen3-4B-Base",
    "max_seq_length": 2048,
    "lora_rank": 16,
    "load_in_4bit": True,
    "gpu_memory_utilization": 0.7,
}

# ========================
# System Configuration
# ========================
SYSTEM_CONFIG = {
    "dt": 0.1,  # Time step
    "steps": 50,  # Number of control steps
    "reasoning_start": "<REASONING>",
    "reasoning_end": "</REASONING>",
    "solution_start": "<CONTROLS>",
    "solution_end": "</CONTROLS>",
}

# ========================
# Training Configuration
# ========================
SFT_CONFIG = {
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 1,
    "warmup_steps": 10,
    "num_train_epochs": 4,
    "learning_rate": 2e-4,
    "logging_steps": 5,
    "eval_steps": 100,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "seed": 3407,
    "report_to": "wandb",
    "save_steps": 500,
    "save_total_limit": 3,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
}

GRPO_CONFIG = {
    "learning_rate": 5e-6,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "linear",
    "optim": "adamw_8bit",
    "logging_steps": 1,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 1,
    "num_generations": 8,
    "max_completion_length": 2048,
    "max_steps": 100,
    "save_steps": 500,
    "report_to": "wandb",
    "temperature": 1.0,
    "min_p": 0.1,
    "top_p": 1.0,
    "top_k": -1,
    "seed": 3407,
}

# ========================
# Data Generation Configuration
# ========================
DATA_CONFIG = {
    "default_samples_per_system": 200,
    "train_eval_split": 0.9,
    "random_seed": 42,
    "save_datasets": True,
    "dataset_directory": "data/saved_datasets",
}

# ========================
# Evaluation Configuration
# ========================
EVAL_CONFIG = {
    "sampling_params": {
        "temperature": 0.7,
        "top_k": 50,
        "max_tokens": 1024,
    },
    "num_test_cases": 10,
    "mpc_horizon": 10,
    "metrics_to_compute": [
        "final_error",
        "lqr_cost", 
        "total_control_effort",
        "mean_control_change",
        "constraints_satisfied"
    ],
    "plot_config": {
        "figsize": (12, 10),
        "dpi": 300,
        "save_plots": True,
        "plot_directory": "plots",
    }
}

# ========================
# Path Configuration
# ========================
PATHS = {
    "models_dir": "models",
    "data_dir": "data",
    "plots_dir": "plots",
    "logs_dir": "logs",
    "checkpoints_dir": "checkpoints",
    "notebooks_dir": "notebooks",
}

# ========================
# System-Specific Configuration
# ========================
SYSTEM_SPECIFIC_CONFIG = {
    "double_integrator": {
        "state_bounds": [(-1.0, 1.0), (-1.0, 1.0)],
        "control_bounds": (-3.0, 3.0),
        "initial_bounds": [(-0.8, 0.8), (-0.8, 0.8)],
        "lqr_weights": {
            "Q": [10.0, 10.0],  # Position, velocity
            "R": 0.1,
        },
    },
    "van_der_pol": {
        "state_bounds": [(-2.0, 2.0), (-2.0, 2.0)],
        "control_bounds": (-5.0, 5.0),
        "initial_bounds": [(-1.5, 1.5), (-1.5, 1.5)],
        "mu": 1.0,  # Van der Pol parameter
        "solver_config": {
            "Q": [10.0, 5.0],  # Position, velocity  
            "R": 0.1,
            "maxiter": 1000,
        },
    },
}

# ========================
# Available Systems
# ========================
AVAILABLE_SYSTEMS = ["double_integrator", "van_der_pol"]

# Future systems can be added here:
# FUTURE_SYSTEMS = ["pendulum", "cartpole", "quadrotor"]

# ========================
# Utility Functions
# ========================
def get_system_config(system_name: str) -> Dict[str, Any]:
    """Get configuration for a specific system."""
    if system_name not in SYSTEM_SPECIFIC_CONFIG:
        raise ValueError(f"Unknown system: {system_name}")
    return SYSTEM_SPECIFIC_CONFIG[system_name]

def get_universal_prompt_template() -> str:
    """Get the universal system prompt template."""
    dt = SYSTEM_CONFIG["dt"]
    steps = SYSTEM_CONFIG["steps"]
    total_time = dt * steps
    reasoning_start = SYSTEM_CONFIG["reasoning_start"]
    reasoning_end = SYSTEM_CONFIG["reasoning_end"]
    solution_start = SYSTEM_CONFIG["solution_start"]
    solution_end = SYSTEM_CONFIG["solution_end"]
    
    return f"""You are a universal control systems expert.
Given any control system with its description, initial state, and constraints,
generate a sequence of {steps} control inputs to reach the target state in {total_time:.2f} seconds.
Analyze the system dynamics, identify the appropriate control approach, and ensure all constraints are satisfied.
Explain your approach between {reasoning_start} and {reasoning_end}.
Then provide exactly {steps} control values as a comma-separated list between {solution_start} and {solution_end}."""

def get_model_save_path(model_type: str, system: str, training_type: str, 
                       run_name: str = "latest") -> str:
    """Get the save path for a model."""
    models_dir = PATHS["models_dir"]
    
    if model_type == "universal":
        return f"{models_dir}/universal/{training_type}/{run_name}"
    elif model_type == "single_system":
        return f"{models_dir}/single_system/{system}/{training_type}/{run_name}"
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def validate_config():
    """Validate the configuration settings."""
    # Check that all required keys are present
    required_configs = [
        MODEL_CONFIG, SYSTEM_CONFIG, SFT_CONFIG, GRPO_CONFIG, 
        DATA_CONFIG, EVAL_CONFIG, PATHS
    ]
    
    errors = []
    
    # Validate model config
    if MODEL_CONFIG["lora_rank"] <= 0:
        errors.append("LoRA rank must be positive")
    
    if MODEL_CONFIG["max_seq_length"] <= 0:
        errors.append("Max sequence length must be positive")
    
    # Validate system config
    if SYSTEM_CONFIG["dt"] <= 0:
        errors.append("Time step (dt) must be positive")
    
    if SYSTEM_CONFIG["steps"] <= 0:
        errors.append("Number of steps must be positive")
    
    # Validate training config
    if SFT_CONFIG["learning_rate"] <= 0:
        errors.append("SFT learning rate must be positive")
    
    if GRPO_CONFIG["learning_rate"] <= 0:
        errors.append("GRPO learning rate must be positive")
    
    # Validate data config
    if not (0 < DATA_CONFIG["train_eval_split"] < 1):
        errors.append("Train/eval split must be between 0 and 1")
    
    if errors:
        raise ValueError("Configuration errors found:\n" + "\n".join(errors))
    
    print("Configuration validation passed!")

# Validate config on import
validate_config()

# ========================
# Export All Config
# ========================
ALL_CONFIG = {
    "model": MODEL_CONFIG,
    "system": SYSTEM_CONFIG,
    "sft": SFT_CONFIG,
    "grpo": GRPO_CONFIG,
    "data": DATA_CONFIG,
    "eval": EVAL_CONFIG,
    "paths": PATHS,
    "systems": SYSTEM_SPECIFIC_CONFIG,
    "available_systems": AVAILABLE_SYSTEMS,
}

# Print configuration summary on import
if __name__ == "__main__":
    print("=== Universal Control LLM Configuration ===")
    print(f"Base model: {MODEL_CONFIG['base_model_name']}")
    print(f"Available systems: {', '.join(AVAILABLE_SYSTEMS)}")
    print(f"Default dt: {SYSTEM_CONFIG['dt']}, steps: {SYSTEM_CONFIG['steps']}")
    print(f"SFT learning rate: {SFT_CONFIG['learning_rate']}")
    print(f"GRPO learning rate: {GRPO_CONFIG['learning_rate']}")
    print("Configuration loaded successfully!")
else:
    print("Universal Control LLM configuration loaded")