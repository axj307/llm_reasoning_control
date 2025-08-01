"""
Protocol interfaces for type checking.
"""

from .protocols import (
    TrainingStrategy,
    DataLoader,
    ModelManager,
    Evaluator,
    Visualizer
)

__all__ = [
    "TrainingStrategy",
    "DataLoader",
    "ModelManager",
    "Evaluator",
    "Visualizer"
]