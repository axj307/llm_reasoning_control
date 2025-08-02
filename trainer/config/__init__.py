"""
Configuration management system for LLM Control Training.

This module provides:
- YAML configuration loading with inheritance
- Pydantic validation schemas
- Environment variable substitution
- Command-line override support
"""

from .config_loader import ConfigLoader, load_config, merge_configs
from .config_schema import (
    ModelConfig,
    LoRAConfig,
    TrainingConfig,
    ControlConfig,
    DatasetConfig,
    FormattingConfig,
    GPUConfig,
    LoggingConfig,
    EvaluationConfig,
    Config
)
from .config_manager import ConfigManager, create_config_parser

__all__ = [
    # Loader
    "ConfigLoader",
    "load_config",
    "merge_configs",
    
    # Schemas
    "ModelConfig",
    "LoRAConfig", 
    "TrainingConfig",
    "ControlConfig",
    "DatasetConfig",
    "FormattingConfig",
    "GPUConfig",
    "LoggingConfig",
    "EvaluationConfig",
    "Config",
    
    # Manager
    "ConfigManager",
    "create_config_parser"
]