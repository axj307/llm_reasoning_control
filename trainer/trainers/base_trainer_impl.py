"""
Base trainer implementation with common functionality.
"""

import torch
import gc
from typing import Any, Dict, Optional
import datasets as hf_datasets
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import BaseTrainer
from logger import logger


class BaseTrainerImpl(BaseTrainer):
    """Base implementation with common functionality for all trainers."""
    
    def __init__(self, model: Any, tokenizer: Any, config: Dict[str, Any]):
        """Initialize base trainer implementation."""
        super().__init__(model, tokenizer, config)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
    def setup(self) -> None:
        """Common setup operations."""
        if self._is_initialized:
            return
            
        logger.info(f"Setting up {self.__class__.__name__}...")
        
        # Move model to device if needed
        if hasattr(self.model, 'to') and torch.cuda.is_available():
            self.model = self.model.to(self.device)
            
        self._is_initialized = True
        
    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint."""
        logger.info(f"Saving checkpoint to {path}")
        
        # Save model state
        if hasattr(self.model, 'save_lora'):
            # For LoRA models
            self.model.save_lora(path)
        elif hasattr(self.model, 'save_pretrained'):
            # For transformers models
            self.model.save_pretrained(path)
        else:
            # Generic PyTorch save
            torch.save(self.model.state_dict(), os.path.join(path, 'model.pt'))
            
        # Save tokenizer if available
        if self.tokenizer and hasattr(self.tokenizer, 'save_pretrained'):
            self.tokenizer.save_pretrained(path)
            
    def cleanup(self) -> None:
        """Enhanced cleanup with memory management."""
        super().cleanup()
        
        # Force garbage collection
        gc.collect()
        
        # Log memory usage if available
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            
    def _get_optimizer(self) -> torch.optim.Optimizer:
        """Get optimizer based on config."""
        optimizer_name = self.config.get('optimizer', 'adamw').lower()
        learning_rate = self.config.get('learning_rate', 2e-4)
        weight_decay = self.config.get('weight_decay', 0.01)
        
        if optimizer_name == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            return torch.optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
            
    def _prepare_dataset(self, dataset: hf_datasets.Dataset) -> hf_datasets.Dataset:
        """Prepare dataset for training."""
        # Override in subclasses for specific preparation
        return dataset