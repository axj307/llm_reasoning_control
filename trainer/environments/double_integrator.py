"""
Double Integrator environment implementation.
"""

import numpy as np
import scipy.linalg as la
from typing import Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.base_environment import BaseEnvironment


class DoubleIntegratorEnvironment(BaseEnvironment):
    """Double integrator control environment: ẍ = u"""
    
    def __init__(self, dt: float = 0.1, steps: int = 50):
        """
        Initialize double integrator environment.
        
        Args:
            dt: Time step duration
            steps: Number of control steps
        """
        super().__init__(dt, steps)
        self.state_dim = 2  # [position, velocity]
        self.action_dim = 1  # acceleration control
        self.state_bounds = (-1.0, 1.0)  # State constraints
        self.action_bounds = (-3.0, 3.0)  # Control constraints
        
        # LQR matrices
        self.Q = np.eye(2)  # State cost
        self.R = np.array([[0.1]])  # Control cost
        
    def reset(self) -> np.ndarray:
        """Reset to random initial state."""
        x0 = np.random.uniform(-0.9, 0.9)
        v0 = np.random.uniform(-0.9, 0.9)
        return np.array([x0, v0])
        
    def step(self, state: np.ndarray, action: float) -> np.ndarray:
        """
        Take one step in the environment.
        
        Args:
            state: Current state [position, velocity]
            action: Control input (acceleration)
            
        Returns:
            Next state [position, velocity]
        """
        x, v = state
        
        # Update dynamics: ẍ = u
        v_new = v + action * self.dt
        x_new = x + v * self.dt
        
        # Clip to bounds
        x_new = np.clip(x_new, self.state_bounds[0], self.state_bounds[1])
        v_new = np.clip(v_new, self.state_bounds[0], self.state_bounds[1])
        
        return np.array([x_new, v_new])
        
    def solve_optimal_control(self, initial_state: np.ndarray) -> np.ndarray:
        """
        Solve LQR optimal control problem.
        
        Args:
            initial_state: Initial state [x0, v0]
            
        Returns:
            Optimal control sequence
        """
        x0, v0 = initial_state
        
        # System matrices for discrete-time LQR
        A = np.array([[1, self.dt], [0, 1]])
        B = np.array([[0], [self.dt]])
        
        # Solve discrete-time algebraic Riccati equation
        P = la.solve_discrete_are(A, B, self.Q, self.R)
        
        # Compute LQR gain
        K = la.inv(self.R + B.T @ P @ B) @ (B.T @ P @ A)
        
        # Generate control sequence
        controls = []
        x = initial_state.copy()
        
        for _ in range(self.steps):
            u = -K @ x
            u = float(u.flatten()[0])  # Ensure we can convert to scalar
            u = np.clip(u, self.action_bounds[0], self.action_bounds[1])
            controls.append(u)
            x = A @ x + B.flatten() * u  # B.flatten() to ensure correct shape
            
        return np.array(controls)
        
    def get_problem_description(self, initial_state: np.ndarray) -> str:
        """Get natural language description."""
        x0, v0 = initial_state
        total_time = self.dt * self.steps
        
        return (f"Control a double integrator system with initial state "
                f"[position={x0:.2f}, velocity={v0:.2f}] to reach the origin (0,0) "
                f"in {total_time:.2f} seconds using {self.steps} steps.")
                
    def get_state_names(self) -> Tuple[str, str]:
        """Get names of state variables."""
        return ("position", "velocity")