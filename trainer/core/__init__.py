"""
Core abstract base classes for the modular trainer.
"""

from .base_trainer import BaseTrainer
from .base_environment import BaseEnvironment
from .base_reward import BaseReward, RewardRegistry, reward_registry

__all__ = [
    "BaseTrainer",
    "BaseEnvironment", 
    "BaseReward",
    "RewardRegistry",
    "reward_registry"
]