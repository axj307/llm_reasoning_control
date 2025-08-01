"""
Protocol interfaces for type checking and contracts.
"""

from typing import Protocol, Any, Dict, List, Optional, Tuple
from datasets import Dataset
import numpy as np


class TrainingStrategy(Protocol):
    """Protocol for training strategies."""
    
    def setup(self) -> None:
        """Setup the training strategy."""
        ...
        
    def train(self, dataset: Dataset, **kwargs) -> Dict[str, Any]:
        """Train the model."""
        ...
        
    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint."""
        ...


class DataLoader(Protocol):
    """Protocol for data loading and formatting."""
    
    def load_dataset(self, path: str) -> Dataset:
        """Load dataset from path."""
        ...
        
    def format_for_training(self, dataset: Dataset) -> Dataset:
        """Format dataset for training."""
        ...
        
    def create_dataset(self, num_samples: int) -> Dataset:
        """Create a new dataset."""
        ...


class ModelManager(Protocol):
    """Protocol for model management operations."""
    
    def load_model(self, model_name: str) -> Tuple[Any, Any]:
        """Load model and tokenizer."""
        ...
        
    def setup_lora(self, model: Any, config: Dict[str, Any]) -> Any:
        """Setup LoRA configuration."""
        ...
        
    def save_model(self, model: Any, path: str) -> None:
        """Save model to path."""
        ...
        
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from prompt."""
        ...


class Evaluator(Protocol):
    """Protocol for model evaluation."""
    
    def evaluate(self, model: Any, dataset: Dataset) -> Dict[str, Any]:
        """Evaluate model on dataset."""
        ...
        
    def compute_metrics(self, predictions: List, references: List) -> Dict[str, float]:
        """Compute evaluation metrics."""
        ...


class Visualizer(Protocol):
    """Protocol for visualization utilities."""
    
    def plot_trajectory(self, states: List[np.ndarray], actions: List[float]) -> None:
        """Plot system trajectory."""
        ...
        
    def plot_training_metrics(self, metrics: Dict[str, List[float]]) -> None:
        """Plot training metrics."""
        ...