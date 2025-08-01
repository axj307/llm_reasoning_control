"""
Trainer package for control system fine-tuning.
"""

from .trainer_module import ControlTrainer
from .data import create_dataset, generate_control_dataset
from .control import solve_double_integrator
from .utils import visualize_solution, parse_control_output

__all__ = [
    'ControlTrainer',
    'create_dataset',
    'generate_control_dataset',
    'solve_double_integrator',
    'visualize_solution',
    'parse_control_output',
]