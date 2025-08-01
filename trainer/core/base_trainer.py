"""
Abstract base class for all training strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from datasets import Dataset


class BaseTrainer(ABC):
    """Abstract base class for training strategies."""
    
    def __init__(self, model: Any, tokenizer: Any, config: Dict[str, Any]):
        """
        Initialize base trainer.
        
        Args:
            model: The model to train
            tokenizer: The tokenizer
            config: Configuration dictionary
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self._is_initialized = False
        
    @abstractmethod
    def setup(self) -> None:
        """Setup trainer-specific configurations."""
        pass
        
    @abstractmethod
    def train(self, dataset: 'Dataset', **kwargs) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            dataset: Training dataset
            **kwargs: Additional training arguments
            
        Returns:
            Dictionary containing training metrics
        """
        pass
        
    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint."""
        pass
        
    def validate(self, dataset: 'Dataset') -> Dict[str, Any]:
        """
        Validate the model. Default implementation can be overridden.
        
        Args:
            dataset: Validation dataset
            
        Returns:
            Dictionary containing validation metrics
        """
        return {}
        
    def cleanup(self) -> None:
        """Cleanup resources after training."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()