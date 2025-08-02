"""
YAML configuration loader with support for:
- Inheritance via defaults
- Environment variable substitution
- Deep merging of configurations
"""

import os
import re
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from copy import deepcopy


class ConfigLoader:
    """Loads and processes YAML configuration files."""
    
    def __init__(self, config_dir: Union[str, Path] = "configs"):
        """
        Initialize configuration loader.
        
        Args:
            config_dir: Root directory for configuration files
        """
        self.config_dir = Path(config_dir)
        self._loaded_configs = {}  # Cache for loaded configs
        
    def load(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a configuration file with full processing.
        
        Args:
            config_path: Path to configuration file (absolute or relative to config_dir)
            
        Returns:
            Processed configuration dictionary
        """
        config_path = Path(config_path)
        
        # Make path absolute
        if not config_path.is_absolute():
            config_path = self.config_dir / config_path
            
        # Load base config
        config = self._load_yaml(config_path)
        
        # Process defaults (inheritance)
        if "defaults" in config:
            config = self._process_defaults(config, config_path.parent)
            
        # Process environment variables
        config = self._substitute_env_vars(config)
        
        return config
        
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load a YAML file."""
        # Check cache first
        path_str = str(path)
        if path_str in self._loaded_configs:
            return deepcopy(self._loaded_configs[path_str])
            
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
            
        with open(path, 'r') as f:
            config = yaml.safe_load(f) or {}
            
        # Cache the loaded config
        self._loaded_configs[path_str] = config
        
        return deepcopy(config)
        
    def _process_defaults(self, config: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
        """Process defaults section for inheritance."""
        defaults = config.pop("defaults", [])
        
        # Ensure defaults is a list
        if not isinstance(defaults, list):
            defaults = [defaults]
            
        # Start with empty base config
        merged_config = {}
        
        # Process each default
        for default in defaults:
            if isinstance(default, str):
                # Simple string default
                default_config = self._load_default(default, base_dir)
                merged_config = merge_configs(merged_config, default_config)
            elif isinstance(default, dict):
                # Advanced default with specific merge target
                for key, path in default.items():
                    default_config = self._load_default(path, base_dir)
                    if "@" in key:
                        # Handle Hydra-style group override
                        target_key = key.split("@")[1]
                        merged_config[target_key] = default_config
                    else:
                        merged_config[key] = default_config
                        
        # Merge current config on top
        merged_config = merge_configs(merged_config, config)
        
        return merged_config
        
    def _load_default(self, default_path: str, base_dir: Path) -> Dict[str, Any]:
        """Load a default configuration."""
        # Handle absolute paths (starting with /)
        if default_path.startswith("/"):
            # For absolute paths like /base/base, treat relative to config_dir
            default_path = default_path[1:]  # Remove leading slash
            # Add .yaml extension if not present
            if not default_path.endswith('.yaml'):
                default_path += '.yaml'
            return self.load(default_path)
        else:
            path = base_dir / default_path
            # Add .yaml extension if not present
            if not path.suffix:
                path = path.with_suffix(".yaml")
            return self.load(path)
        
    def _substitute_env_vars(self, config: Any) -> Any:
        """Recursively substitute environment variables in config."""
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str):
            # Look for ${VAR_NAME:default_value} pattern
            pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
            
            def replacer(match):
                var_name = match.group(1)
                default_value = match.group(2) or ""
                return os.environ.get(var_name, default_value)
                
            return re.sub(pattern, replacer, config)
        else:
            return config


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two configuration dictionaries.
    
    Args:
        base: Base configuration
        override: Configuration to merge on top
        
    Returns:
        Merged configuration
    """
    merged = deepcopy(base)
    
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursive merge for nested dicts
            merged[key] = merge_configs(merged[key], value)
        else:
            # Direct override
            merged[key] = deepcopy(value)
            
    return merged


def load_config(
    config_path: Union[str, Path, List[Union[str, Path]]],
    config_dir: Union[str, Path] = "configs",
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to load configuration(s).
    
    Args:
        config_path: Path or list of paths to configuration files
        config_dir: Root directory for configurations
        overrides: Dictionary of overrides to apply
        
    Returns:
        Loaded and merged configuration
    """
    loader = ConfigLoader(config_dir)
    
    # Handle single or multiple configs
    if isinstance(config_path, (str, Path)):
        config_paths = [config_path]
    else:
        config_paths = config_path
        
    # Load and merge all configs
    merged_config = {}
    for path in config_paths:
        config = loader.load(path)
        merged_config = merge_configs(merged_config, config)
        
    # Apply overrides
    if overrides:
        merged_config = merge_configs(merged_config, overrides)
        
    return merged_config