"""Configuration loader for the universal control LLM system."""

import yaml
import os
from typing import Dict, List, Any, Optional
from pathlib import Path

class ConfigLoader:
    """Configuration loader that merges YAML files."""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self._base_config = None
        self._training_config = None
        self._environment_configs = {}
        
        # Load configurations on initialization
        self._load_configurations()
    
    def _load_configurations(self):
        """Load all configuration files."""
        # Load base config
        base_config_path = self.config_dir / "base_config.yaml"
        if base_config_path.exists():
            with open(base_config_path, 'r') as f:
                self._base_config = yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"Base config not found: {base_config_path}")
        
        # Load training config
        training_config_path = self.config_dir / "training_config.yaml"
        if training_config_path.exists():
            with open(training_config_path, 'r') as f:
                self._training_config = yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"Training config not found: {training_config_path}")
        
        # Load environment-specific configs
        for config_file in self.config_dir.glob("*.yaml"):
            if config_file.name not in ["base_config.yaml", "training_config.yaml"]:
                env_name = config_file.stem
                with open(config_file, 'r') as f:
                    self._environment_configs[env_name] = yaml.safe_load(f)
    
    def get_config(self, environment: Optional[str] = None) -> Dict[str, Any]:
        """
        Get merged configuration for a specific environment or universal config.
        
        Args:
            environment: Environment name (e.g., 'double_integrator') or None for universal
        
        Returns:
            Merged configuration dictionary
        """
        # Start with base config
        config = self._base_config.copy()
        
        # Merge training config
        config.update(self._training_config)
        
        # If environment is specified, merge environment-specific config
        if environment and environment in self._environment_configs:
            env_config = self._environment_configs[environment]
            config = self._deep_merge(config, env_config)
        
        return config
    
    def get_environment_config(self, environment: str) -> Dict[str, Any]:
        """Get environment-specific configuration."""
        if environment not in self._environment_configs:
            raise ValueError(f"Environment config not found: {environment}")
        
        return self._environment_configs[environment]["environment"]
    
    def get_available_environments(self) -> List[str]:
        """Get list of available environments."""
        return list(self._environment_configs.keys())
    
    def _deep_merge(self, base_dict: Dict, override_dict: Dict) -> Dict:
        """Deep merge two dictionaries, with override_dict taking precedence."""
        result = base_dict.copy()
        
        for key, value in override_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration settings."""
        errors = []
        
        # Validate model config
        model_config = config.get("model", {})
        if model_config.get("lora_rank", 0) <= 0:
            errors.append("LoRA rank must be positive")
        
        if model_config.get("max_seq_length", 0) <= 0:
            errors.append("Max sequence length must be positive")
        
        # Validate system config
        system_config = config.get("system", {})
        if system_config.get("dt", 0) <= 0:
            errors.append("Time step (dt) must be positive")
        
        if system_config.get("steps", 0) <= 0:
            errors.append("Number of steps must be positive")
        
        # Validate training config
        sft_config = config.get("sft", {})
        if sft_config.get("learning_rate", 0) <= 0:
            errors.append("SFT learning rate must be positive")
        
        grpo_config = config.get("grpo", {})
        if grpo_config.get("learning_rate", 0) <= 0:
            errors.append("GRPO learning rate must be positive")
        
        # Validate data config
        data_config = config.get("data", {})
        train_eval_split = data_config.get("train_eval_split", 0.5)
        if not (0 < train_eval_split < 1):
            errors.append("Train/eval split must be between 0 and 1")
        
        if errors:
            raise ValueError("Configuration errors found:\\n" + "\\n".join(errors))
        
        return True

# Global config loader instance
_config_loader = ConfigLoader()

# ========================
# Convenience Functions
# ========================

def get_config(environment: Optional[str] = None) -> Dict[str, Any]:
    """Get configuration for environment or universal config."""
    return _config_loader.get_config(environment)

def get_environment_config(environment: str) -> Dict[str, Any]:
    """Get environment-specific configuration."""
    return _config_loader.get_environment_config(environment)

def get_available_environments() -> List[str]:
    """Get list of available environments."""
    return _config_loader.get_available_environments()

def get_system_config(system_name: str) -> Dict[str, Any]:
    """Get configuration for a specific system (backward compatibility)."""
    return get_environment_config(system_name)

def get_universal_prompt_template() -> str:
    """Get the universal system prompt template."""
    config = get_config()
    system_config = config["system"]
    
    dt = system_config["dt"]
    steps = system_config["steps"]
    total_time = dt * steps
    reasoning_start = system_config["reasoning_start"]
    reasoning_end = system_config["reasoning_end"]
    solution_start = system_config["solution_start"]
    solution_end = system_config["solution_end"]
    
    return f"""You are a universal control systems expert.
Given any control system with its description, initial state, and constraints,
generate a sequence of {steps} control inputs to reach the target state in {total_time:.2f} seconds.
Analyze the system dynamics, identify the appropriate control approach, and ensure all constraints are satisfied.
Explain your approach between {reasoning_start} and {reasoning_end}.
Then provide exactly {steps} control values as a comma-separated list between {solution_start} and {solution_end}."""

def get_model_save_path(model_type: str, system: str, training_type: str, 
                       run_name: str = "latest") -> str:
    """Get the save path for a model."""
    config = get_config()
    models_dir = config["paths"]["models_dir"]
    
    if model_type == "universal":
        return f"{models_dir}/universal/{training_type}/{run_name}"
    elif model_type == "single_system":
        return f"{models_dir}/single_system/{system}/{training_type}/{run_name}"
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def validate_config():
    """Validate the configuration settings."""
    config = get_config()
    _config_loader.validate_config(config)
    print("Configuration validation passed!")

# ========================
# Backward Compatibility
# ========================

# Create backward compatible variables
def _create_backward_compatible_config():
    """Create backward compatible config variables."""
    base_config = get_config()
    
    global MODEL_CONFIG, SYSTEM_CONFIG, SFT_CONFIG, GRPO_CONFIG
    global DATA_CONFIG, EVAL_CONFIG, PATHS, AVAILABLE_SYSTEMS
    global SYSTEM_SPECIFIC_CONFIG, ALL_CONFIG
    
    MODEL_CONFIG = base_config["model"]
    SYSTEM_CONFIG = base_config["system"] 
    SFT_CONFIG = base_config["sft"]
    GRPO_CONFIG = base_config["grpo"]
    DATA_CONFIG = base_config["data"]
    EVAL_CONFIG = base_config["eval"]
    PATHS = base_config["paths"]
    AVAILABLE_SYSTEMS = get_available_environments()
    
    # Create system specific config from environment configs
    SYSTEM_SPECIFIC_CONFIG = {}
    for env_name in AVAILABLE_SYSTEMS:
        env_config = get_environment_config(env_name)
        SYSTEM_SPECIFIC_CONFIG[env_name] = env_config
    
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

# Initialize backward compatible config
_create_backward_compatible_config()

# Validate config on import
validate_config()

# Print configuration summary
if __name__ == "__main__":
    print("=== Universal Control LLM Configuration ===")
    print(f"Base model: {MODEL_CONFIG['base_model_name']}")
    print(f"Available environments: {', '.join(AVAILABLE_SYSTEMS)}")
    print(f"Default dt: {SYSTEM_CONFIG['dt']}, steps: {SYSTEM_CONFIG['steps']}")
    print(f"SFT learning rate: {SFT_CONFIG['learning_rate']}")
    print(f"GRPO learning rate: {GRPO_CONFIG['learning_rate']}")
    print("Configuration loaded successfully!")
else:
    print("Universal Control LLM configuration loaded from YAML files")