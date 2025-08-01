"""
Factory for creating trainer instances.
"""

from typing import Any, Dict, Optional, Union
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import BaseTrainer
from .sft_trainer import SFTTrainerModule
from .grpo_trainer import GRPOTrainerModule
from logger import logger


class TrainerFactory:
    """Factory for creating trainer instances based on strategy."""
    
    # Registry of available trainers
    _trainers = {
        'sft': SFTTrainerModule,
        'grpo': GRPOTrainerModule,
    }
    
    @classmethod
    def create(cls, 
               trainer_type: str,
               model: Any,
               tokenizer: Any,
               config: Optional[Dict[str, Any]] = None) -> BaseTrainer:
        """
        Create a trainer instance.
        
        Args:
            trainer_type: Type of trainer ('sft' or 'grpo')
            model: The model to train
            tokenizer: The tokenizer
            config: Configuration dictionary
            
        Returns:
            Trainer instance
            
        Raises:
            ValueError: If trainer type is not recognized
        """
        if trainer_type not in cls._trainers:
            available = ', '.join(cls._trainers.keys())
            raise ValueError(f"Unknown trainer type: {trainer_type}. Available: {available}")
            
        config = config or {}
        trainer_class = cls._trainers[trainer_type]
        
        logger.info(f"Creating {trainer_type.upper()} trainer with config: {config}")
        
        return trainer_class(model, tokenizer, config)
        
    @classmethod
    def register(cls, name: str, trainer_class: type) -> None:
        """
        Register a new trainer type.
        
        Args:
            name: Name for the trainer type
            trainer_class: Trainer class (must inherit from BaseTrainer)
        """
        if not issubclass(trainer_class, BaseTrainer):
            raise ValueError(f"{trainer_class} must inherit from BaseTrainer")
            
        cls._trainers[name] = trainer_class
        logger.info(f"Registered new trainer type: {name}")
        
    @classmethod
    def list_available(cls) -> list:
        """List available trainer types."""
        return list(cls._trainers.keys())


def create_trainer(trainer_type: str, **kwargs) -> BaseTrainer:
    """
    Convenience function to create a trainer.
    
    Args:
        trainer_type: Type of trainer to create
        **kwargs: Arguments passed to TrainerFactory.create
        
    Returns:
        Trainer instance
    """
    return TrainerFactory.create(trainer_type, **kwargs)