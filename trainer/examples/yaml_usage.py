#!/usr/bin/env python3
"""
Example usage of the YAML configuration system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ConfigManager, Config


def example_basic_loading():
    """Example: Basic configuration loading."""
    print("=== Basic Configuration Loading ===")
    
    # Create config manager
    config_manager = ConfigManager()
    
    # Load base configuration
    config = config_manager.load("base/base.yaml")
    
    print(f"Model: {config.model.name}")
    print(f"LoRA rank: {config.lora.rank}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Control steps: {config.control.steps}")
    print()


def example_environment_specific():
    """Example: Load environment-specific configuration."""
    print("=== Environment-Specific Configuration ===")
    
    config_manager = ConfigManager()
    
    # Load double integrator config (inherits from base)
    config = config_manager.load("environments/double_integrator.yaml")
    
    print(f"System: {config.system.name}")
    print(f"Description: {config.system.description}")
    print(f"LQR Q matrix: {config.lqr.Q}")
    print(f"Initial state ranges: {config.dataset.initial_state_ranges}")
    print()


def example_experiment_config():
    """Example: Load experiment configuration."""
    print("=== Experiment Configuration ===")
    
    config_manager = ConfigManager()
    
    # Load quick test configuration
    config = config_manager.load("experiments/quick_test.yaml")
    
    print(f"Dataset size: {config.dataset.num_samples}")
    print(f"Training steps: {config.training.max_steps}")
    print(f"Batch size: {config.training.per_device_train_batch_size}")
    print(f"Output dir: {config.training.output_dir}")
    print()


def example_with_overrides():
    """Example: Configuration with overrides."""
    print("=== Configuration with Overrides ===")
    
    config_manager = ConfigManager()
    
    # Load with overrides
    overrides = {
        "model": {"name": "unsloth/llama-3-8b-bnb-4bit"},
        "lora": {"rank": 64},
        "training": {"learning_rate": 1e-4}
    }
    
    config = config_manager.load("base/base.yaml", overrides=overrides)
    
    print(f"Model (overridden): {config.model.name}")
    print(f"LoRA rank (overridden): {config.lora.rank}")
    print(f"Learning rate (overridden): {config.training.learning_rate}")
    print()


def example_multiple_configs():
    """Example: Merge multiple configurations."""
    print("=== Multiple Configuration Files ===")
    
    config_manager = ConfigManager()
    
    # Load multiple configs (later ones override earlier ones)
    config = config_manager.load([
        "base/base.yaml",
        "environments/double_integrator.yaml",
        "experiments/quick_test.yaml"
    ])
    
    print(f"System: {config.system.name}")  # From double_integrator.yaml
    print(f"Dataset size: {config.dataset.num_samples}")  # From quick_test.yaml
    print(f"Model: {config.model.name}")  # From base.yaml
    print()


def example_save_config():
    """Example: Save configuration to file."""
    print("=== Save Configuration ===")
    
    config_manager = ConfigManager()
    config = config_manager.load("experiments/quick_test.yaml")
    
    # Save as YAML
    config_manager.save("outputs/my_config.yaml", format="yaml")
    print("Saved to outputs/my_config.yaml")
    
    # Save as JSON
    config_manager.save("outputs/my_config.json", format="json")
    print("Saved to outputs/my_config.json")
    print()


def example_training_config():
    """Example: Get strategy-specific training config."""
    print("=== Strategy-Specific Training Config ===")
    
    config_manager = ConfigManager()
    config = config_manager.load("experiments/full_training.yaml")
    
    # Get SFT-specific config
    sft_config = config_manager.get_training_config("sft")
    print(f"SFT epochs: {sft_config.num_train_epochs}")
    print(f"SFT learning rate: {sft_config.learning_rate}")
    
    # Get GRPO-specific config
    grpo_config = config_manager.get_training_config("grpo")
    print(f"GRPO max steps: {grpo_config.max_steps}")
    print(f"GRPO num generations: {grpo_config.num_generations}")
    print()


def example_validation():
    """Example: Configuration validation."""
    print("=== Configuration Validation ===")
    
    config_manager = ConfigManager()
    
    # This will validate the configuration
    try:
        # Invalid config (sequence length too short)
        overrides = {"model": {"max_seq_length": 100}}  # Less than 2048
        config = config_manager.load("base/base.yaml", overrides=overrides)
    except ValueError as e:
        print(f"Validation error: {e}")
    
    # Valid config
    overrides = {"model": {"max_seq_length": 4096}}
    config = config_manager.load("base/base.yaml", overrides=overrides)
    print(f"Valid config loaded with seq_length: {config.model.max_seq_length}")
    print()


if __name__ == "__main__":
    # Run all examples
    example_basic_loading()
    example_environment_specific()
    example_experiment_config()
    example_with_overrides()
    example_multiple_configs()
    example_save_config()
    example_training_config()
    example_validation()