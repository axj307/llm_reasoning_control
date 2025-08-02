"""
Configuration manager that combines loading and validation.
Provides a high-level interface for working with configurations.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from .config_loader import ConfigLoader, load_config, merge_configs
from .config_schema import Config, TrainingConfig, GRPOConfig


class ConfigManager:
    """
    High-level configuration management.
    Handles loading, validation, and access to configuration.
    """
    
    def __init__(self, config_dir: Union[str, Path] = "configs"):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Root directory for configuration files
        """
        self.config_dir = Path(config_dir)
        self.loader = ConfigLoader(config_dir)
        self._config_dict: Optional[Dict[str, Any]] = None
        self._config: Optional[Config] = None
        
    def load(
        self, 
        config_path: Union[str, Path, List[Union[str, Path]]] = "base/base.yaml",
        overrides: Optional[Dict[str, Any]] = None,
        validate: bool = True
    ) -> Config:
        """
        Load and validate configuration.
        
        Args:
            config_path: Path(s) to configuration file(s)
            overrides: Dictionary of overrides
            validate: Whether to validate configuration
            
        Returns:
            Validated configuration object
        """
        # Load raw configuration
        self._config_dict = load_config(config_path, self.config_dir, overrides)
        
        # Validate if requested
        if validate:
            self._config = Config(**self._config_dict)
        else:
            self._config = None
            
        return self._config
        
    def get_dict(self) -> Dict[str, Any]:
        """Get raw configuration dictionary."""
        if self._config_dict is None:
            raise ValueError("No configuration loaded")
        return self._config_dict
        
    def get(self) -> Config:
        """Get validated configuration object."""
        if self._config is None:
            if self._config_dict is None:
                raise ValueError("No configuration loaded")
            self._config = Config(**self._config_dict)
        return self._config
        
    def save(self, path: Union[str, Path], format: str = "yaml") -> None:
        """
        Save current configuration to file.
        
        Args:
            path: Output file path
            format: Output format ('yaml' or 'json')
        """
        if self._config_dict is None:
            raise ValueError("No configuration loaded")
            
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "yaml":
            import yaml
            with open(path, 'w') as f:
                yaml.dump(self._config_dict, f, default_flow_style=False, sort_keys=False)
        elif format == "json":
            with open(path, 'w') as f:
                json.dump(self._config_dict, f, indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")
            
    def get_training_config(self, strategy: str = "sft") -> TrainingConfig:
        """
        Get training configuration for specific strategy.
        
        Args:
            strategy: Training strategy ('sft' or 'grpo')
            
        Returns:
            Training configuration
        """
        config = self.get()
        
        # Check for strategy-specific config first
        strategy_config_name = f"{strategy}_training"
        if hasattr(config, strategy_config_name) and getattr(config, strategy_config_name) is not None:
            return getattr(config, strategy_config_name)
            
        # Fall back to general training config
        return config.training
        
    def create_output_dir(self, suffix: Optional[str] = None) -> Path:
        """
        Create output directory based on configuration.
        
        Args:
            suffix: Optional suffix for directory name
            
        Returns:
            Path to created directory
        """
        config = self.get()
        
        # Base output directory
        output_dir = Path(config.training.output_dir)
        
        # Add timestamp and suffix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{timestamp}"
        if suffix:
            dir_name += f"_{suffix}"
            
        output_path = output_dir / dir_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration to output directory
        self.save(output_path / "config.yaml")
        
        return output_path
        
    @classmethod
    def from_args(cls, args: Optional[argparse.Namespace] = None) -> 'ConfigManager':
        """
        Create ConfigManager from command-line arguments.
        
        Args:
            args: Parsed arguments (if None, will parse from sys.argv)
            
        Returns:
            Configured ConfigManager instance
        """
        if args is None:
            parser = create_config_parser()
            args = parser.parse_args()
            
        # Create manager
        manager = cls(args.config_dir)
        
        # Parse config paths
        config_paths = []
        if args.config:
            config_paths.append(args.config)
        if args.configs:
            config_paths.extend(args.configs.split(','))
            
        # Default to base config if none specified
        if not config_paths:
            config_paths = ["base/base.yaml"]
            
        # Parse overrides
        overrides = parse_overrides(args.overrides) if args.overrides else {}
        
        # Load configuration
        manager.load(config_paths, overrides)
        
        return manager


def create_config_parser() -> argparse.ArgumentParser:
    """Create argument parser for configuration options."""
    parser = argparse.ArgumentParser(add_help=False)
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--configs",
        type=str,
        help="Comma-separated list of configuration files"
    )
    
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs",
        help="Configuration directory (default: configs)"
    )
    
    parser.add_argument(
        "--overrides",
        nargs="+",
        help="Configuration overrides (e.g., training.learning_rate=0.001)"
    )
    
    return parser


def parse_overrides(overrides: List[str]) -> Dict[str, Any]:
    """
    Parse command-line overrides into nested dictionary.
    
    Args:
        overrides: List of override strings (e.g., ["training.lr=0.001"])
        
    Returns:
        Nested dictionary of overrides
    """
    result = {}
    
    for override in overrides:
        if '=' not in override:
            raise ValueError(f"Invalid override format: {override}")
            
        key, value = override.split('=', 1)
        keys = key.split('.')
        
        # Parse value
        try:
            # Try to evaluate as Python literal
            parsed_value = eval(value)
        except:
            # Keep as string if evaluation fails
            parsed_value = value
            
        # Build nested dictionary
        current = result
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = parsed_value
        
    return result