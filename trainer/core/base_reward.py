"""
Abstract base class for GRPO reward functions.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseReward(ABC):
    """Abstract base class for reward functions."""
    
    def __init__(self, name: str, weight: float = 1.0):
        """
        Initialize reward function.
        
        Args:
            name: Name of the reward function
            weight: Weight of this reward in combined scoring
        """
        self.name = name
        self.weight = weight
        
    @abstractmethod
    def __call__(self, prompts: Optional[List] = None, 
                 completions: Optional[List] = None, 
                 **kwargs) -> List[float]:
        """
        Calculate rewards for completions.
        
        Args:
            prompts: List of prompts
            completions: List of model completions
            **kwargs: Additional arguments
            
        Returns:
            List of reward scores
        """
        pass
        
    def get_config(self) -> Dict[str, Any]:
        """Get reward function configuration."""
        return {
            "name": self.name,
            "weight": self.weight,
            "type": self.__class__.__name__
        }


class RewardRegistry:
    """Registry for managing reward functions."""
    
    def __init__(self):
        self._rewards = {}
        
    def register(self, name: str, reward_class: type):
        """Register a reward function class."""
        if not issubclass(reward_class, BaseReward):
            raise ValueError(f"{reward_class} must inherit from BaseReward")
        self._rewards[name] = reward_class
        
    def create(self, name: str, **kwargs) -> BaseReward:
        """Create a reward function instance."""
        if name not in self._rewards:
            raise ValueError(f"Unknown reward function: {name}")
        # Pass name as first argument if not in kwargs
        if 'name' not in kwargs:
            kwargs['name'] = name
        return self._rewards[name](**kwargs)
        
    def list_available(self) -> List[str]:
        """List available reward functions."""
        return list(self._rewards.keys())


# Global registry instance
reward_registry = RewardRegistry()