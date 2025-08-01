"""
Abstract base class for control environments.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any
import numpy as np


class BaseEnvironment(ABC):
    """Abstract base class for control system environments."""
    
    def __init__(self, dt: float = 0.1, steps: int = 50):
        """
        Initialize environment.
        
        Args:
            dt: Time step duration
            steps: Number of control steps
        """
        self.dt = dt
        self.steps = steps
        self.state_dim = None  # To be set by subclass
        self.action_dim = None  # To be set by subclass
        self.state_bounds = None  # To be set by subclass
        self.action_bounds = None  # To be set by subclass
        
    @abstractmethod
    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.
        
        Returns:
            Initial state vector
        """
        pass
        
    @abstractmethod
    def step(self, state: np.ndarray, action: float) -> np.ndarray:
        """
        Take one step in the environment.
        
        Args:
            state: Current state
            action: Control action
            
        Returns:
            Next state
        """
        pass
        
    @abstractmethod
    def solve_optimal_control(self, initial_state: np.ndarray) -> np.ndarray:
        """
        Solve optimal control problem for given initial state.
        
        Args:
            initial_state: Initial state vector
            
        Returns:
            Optimal control sequence
        """
        pass
        
    @abstractmethod
    def get_problem_description(self, initial_state: np.ndarray) -> str:
        """
        Get natural language description of control problem.
        
        Args:
            initial_state: Initial state vector
            
        Returns:
            Problem description string
        """
        pass
        
    def simulate_trajectory(self, initial_state: np.ndarray, 
                          control_sequence: List[float]) -> Tuple[List[np.ndarray], List[float]]:
        """
        Simulate system trajectory given control sequence.
        
        Args:
            initial_state: Initial state
            control_sequence: List of control inputs
            
        Returns:
            Tuple of (states, actions) trajectories
        """
        states = [initial_state]
        state = initial_state.copy()
        
        for action in control_sequence:
            state = self.step(state, action)
            states.append(state.copy())
            
        return states[:-1], control_sequence
        
    def validate_control_sequence(self, control_sequence: List[float]) -> Dict[str, Any]:
        """
        Validate a control sequence.
        
        Args:
            control_sequence: Control sequence to validate
            
        Returns:
            Validation results dictionary
        """
        valid = True
        issues = []
        
        # Check length
        if len(control_sequence) != self.steps:
            valid = False
            issues.append(f"Expected {self.steps} steps, got {len(control_sequence)}")
            
        # Check bounds
        if self.action_bounds is not None:
            for i, action in enumerate(control_sequence):
                if action < self.action_bounds[0] or action > self.action_bounds[1]:
                    valid = False
                    issues.append(f"Action {i} out of bounds: {action}")
                    
        return {
            "valid": valid,
            "issues": issues,
            "length": len(control_sequence)
        }