"""
Dataset abstraction layer for control systems.
"""

from .base_dataset import BaseControlDataset
from .dataset_manager import DatasetManager

# Create global dataset manager instance
dataset_manager = DatasetManager()

__all__ = [
    "BaseControlDataset",
    "DatasetManager",
    "dataset_manager",
]