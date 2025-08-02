#!/usr/bin/env python3
"""
Test the YAML configuration system.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import ConfigManager, Config, load_config, merge_configs


def test_basic_loading():
    """Test basic configuration loading."""
    print("Testing basic configuration loading...")
    
    config_manager = ConfigManager()
    config = config_manager.load("base/base.yaml")
    
    assert isinstance(config, Config)
    assert config.model.name == "unsloth/Qwen3-4B-Base"
    assert config.model.max_seq_length == 2048
    assert config.lora.rank == 32
    assert config.training.learning_rate == 2e-4
    
    print("✓ Basic loading test passed")


def test_inheritance():
    """Test configuration inheritance."""
    print("\nTesting configuration inheritance...")
    
    config_manager = ConfigManager()
    config = config_manager.load("environments/double_integrator.yaml")
    
    # Should inherit from base
    assert config.model.name == "unsloth/Qwen3-4B-Base"
    # Should have system-specific settings
    assert config.system.name == "double_integrator"
    assert config.system.state_dim == 2
    
    print("✓ Inheritance test passed")


def test_overrides():
    """Test configuration overrides."""
    print("\nTesting configuration overrides...")
    
    config_manager = ConfigManager()
    overrides = {
        "model": {"name": "test-model"},
        "lora": {"rank": 128},
        "training": {"learning_rate": 1e-3}
    }
    
    config = config_manager.load("base/base.yaml", overrides=overrides)
    
    assert config.model.name == "test-model"
    assert config.lora.rank == 128
    assert config.training.learning_rate == 1e-3
    
    print("✓ Override test passed")


def test_validation():
    """Test configuration validation."""
    print("\nTesting configuration validation...")
    
    config_manager = ConfigManager()
    
    # Test invalid sequence length
    try:
        overrides = {"model": {"max_seq_length": 100}}
        config = config_manager.load("base/base.yaml", overrides=overrides)
        assert False, "Should have raised validation error"
    except Exception as e:
        # Check that it's a validation error for max_seq_length
        assert "max_seq_length" in str(e) or "2048" in str(e)
    
    # Test invalid learning rate
    try:
        overrides = {"training": {"learning_rate": 2.0}}
        config = config_manager.load("base/base.yaml", overrides=overrides)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "learning_rate" in str(e) or "less than or equal to 1" in str(e)
    
    print("✓ Validation test passed")


def test_merge_configs():
    """Test configuration merging."""
    print("\nTesting configuration merging...")
    
    base = {"model": {"name": "base"}, "training": {"lr": 1e-4}}
    override = {"model": {"name": "override"}, "training": {"batch_size": 8}}
    
    merged = merge_configs(base, override)
    
    assert merged["model"]["name"] == "override"
    assert merged["training"]["lr"] == 1e-4
    assert merged["training"]["batch_size"] == 8
    
    print("✓ Merge test passed")


def test_training_config():
    """Test strategy-specific training config."""
    print("\nTesting strategy-specific training config...")
    
    config_manager = ConfigManager()
    config = config_manager.load("experiments/full_training.yaml")
    
    # Get SFT config
    sft_config = config_manager.get_training_config("sft")
    assert sft_config.num_train_epochs == 3
    
    # Get GRPO config
    grpo_config = config_manager.get_training_config("grpo")
    assert hasattr(grpo_config, 'num_generations')
    
    print("✓ Training config test passed")


def test_save_load():
    """Test saving and loading configuration."""
    print("\nTesting save/load functionality...")
    
    config_manager = ConfigManager()
    config = config_manager.load("base/base.yaml")
    
    # Save to temp file in current directory
    temp_path = "./test_config_temp.yaml"
    config_manager.save(temp_path)
    
    # Check that file exists
    assert os.path.exists(temp_path)
    
    # Load saved config directly (not through config_dir)
    import yaml
    with open(temp_path, 'r') as f:
        loaded_dict = yaml.safe_load(f)
    
    assert loaded_dict['model']['name'] == config.model.name
    assert loaded_dict['lora']['rank'] == config.lora.rank
    
    # Clean up
    os.remove(temp_path)
    
    print("✓ Save/load test passed")


def main():
    """Run all tests."""
    print("=== Testing YAML Configuration System ===\n")
    
    try:
        test_basic_loading()
        test_inheritance()
        test_overrides()
        test_validation()
        test_merge_configs()
        test_training_config()
        test_save_load()
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())