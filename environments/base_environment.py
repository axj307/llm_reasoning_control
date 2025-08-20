"""Base class for all control environments."""

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any
import numpy as np


class BaseEnvironment(ABC):
    """Abstract base class for control environments."""
    
    def __init__(self, name: str, state_dim: int, control_dim: int, 
                 dt: float = 0.1, steps: int = 50):
        self.name = name
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.dt = dt
        self.steps = steps
        self.total_time = dt * steps
    
    @abstractmethod
    def get_state_bounds(self) -> List[Tuple[float, float]]:
        """Return bounds for each state variable as [(min, max), ...]."""
        pass
    
    @abstractmethod
    def get_control_bounds(self) -> Tuple[float, float]:
        """Return bounds for control input as (min, max)."""
        pass
    
    @abstractmethod
    def get_initial_state_bounds(self) -> List[Tuple[float, float]]:
        """Return bounds for sampling initial states."""
        pass
    
    @abstractmethod
    def simulate_step(self, state: np.ndarray, control: float) -> np.ndarray:
        """
        Simulate one time step of the system dynamics.
        
        Args:
            state: Current state vector
            control: Control input
            
        Returns:
            Next state vector
        """
        pass
    
    @abstractmethod
    def get_dynamics_description(self) -> str:
        """Return a string describing the system dynamics."""
        pass
    
    @abstractmethod
    def get_problem_description(self, initial_state: np.ndarray) -> str:
        """Generate problem description for a specific initial state."""
        pass
    
    def get_system_prompt(self, reasoning_start: str, reasoning_end: str,
                         solution_start: str, solution_end: str) -> str:
        """Generate system-specific prompt for the LLM."""
        return f"""You are a control systems expert.
Given a {self.name.replace('_', ' ')} system with the following dynamics:
{self.get_dynamics_description()}

Generate a sequence of {self.steps} control inputs to reach the target state in exactly {self.total_time:.2f} seconds.
State bounds: {self.get_state_bounds()}
Control bounds: {self.get_control_bounds()}

Explain your approach between {reasoning_start} and {reasoning_end}.
Then provide exactly {self.steps} control values as a comma-separated list between {solution_start} and {solution_end}."""
    
    def generate_random_initial_state(self) -> np.ndarray:
        """Generate a random initial state within bounds."""
        bounds = self.get_initial_state_bounds()
        state = []
        for (low, high) in bounds:
            state.append(np.random.uniform(low, high))
        return np.array(state)
    
    def simulate_trajectory(self, initial_state: np.ndarray, 
                          controls: List[float]) -> Dict[str, Any]:
        """
        Simulate full trajectory given initial state and control sequence.
        
        Returns:
            Dictionary with trajectory information
        """
        if len(controls) != self.steps:
            raise ValueError(f"Expected {self.steps} controls, got {len(controls)}")
        
        state = np.array(initial_state)
        states = [state.copy()]
        times = [0.0]
        valid_trajectory = True
        
        state_bounds = self.get_state_bounds()
        control_bounds = self.get_control_bounds()
        
        for i, u in enumerate(controls):
            # Check control bounds
            u_clamped = max(control_bounds[0], min(control_bounds[1], u))
            
            # Simulate one step
            state = self.simulate_step(state, u_clamped)
            
            # Check state bounds
            for j, (low, high) in enumerate(state_bounds):
                if not (low <= state[j] <= high):
                    valid_trajectory = False
            
            states.append(state.copy())
            times.append((i + 1) * self.dt)
        
        # Calculate final error (distance to origin/target)
        final_error = np.linalg.norm(state)
        
        return {
            'states': np.array(states),
            'controls': controls,
            'times': times,
            'valid_trajectory': valid_trajectory,
            'final_error': final_error,
            'initial_state': initial_state,
            'final_state': state
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Return system information as a dictionary."""
        return {
            'name': self.name,
            'state_dim': self.state_dim,
            'control_dim': self.control_dim,
            'dynamics': self.get_dynamics_description(),
            'state_bounds': self.get_state_bounds(),
            'control_bounds': self.get_control_bounds(),
            'default_dt': self.dt,
            'default_steps': self.steps
        }
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', state_dim={self.state_dim}, control_dim={self.control_dim})"


# Backward compatibility
BaseSystem = BaseEnvironment