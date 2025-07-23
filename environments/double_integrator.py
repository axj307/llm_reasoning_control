"""Double integrator system implementation."""

import numpy as np
from typing import Tuple, List
from .base_environment import BaseEnvironment


class DoubleIntegrator(BaseEnvironment):
    """Double integrator system: ẍ = u"""
    
    def __init__(self, dt: float = 0.1, steps: int = 50):
        super().__init__(
            name="double_integrator",
            state_dim=2,  # position and velocity
            control_dim=1,  # acceleration
            dt=dt,
            steps=steps
        )
    
    def get_state_bounds(self) -> List[Tuple[float, float]]:
        """Position and velocity bounds: [-1, 1] for both."""
        return [(-1.0, 1.0), (-1.0, 1.0)]
    
    def get_control_bounds(self) -> Tuple[float, float]:
        """Control bounds: [-3, 3]."""
        return (-3.0, 3.0)
    
    def get_initial_state_bounds(self) -> List[Tuple[float, float]]:
        """Initial condition sampling bounds: [-0.8, 0.8] for both."""
        return [(-0.8, 0.8), (-0.8, 0.8)]
    
    def simulate_step(self, state: np.ndarray, control: float) -> np.ndarray:
        """
        Simulate one time step of double integrator.
        
        Args:
            state: [position, velocity]
            control: control input u (acceleration)
            
        Returns:
            new_state: [new_position, new_velocity]
        """
        position, velocity = state
        
        # Double integrator dynamics: ẍ = u
        # Using Euler integration:
        # v_{k+1} = v_k + u * dt
        # x_{k+1} = x_k + v_{k+1} * dt
        
        new_velocity = velocity + control * self.dt
        new_position = position + new_velocity * self.dt
        
        return np.array([new_position, new_velocity])
    
    def get_dynamics_description(self) -> str:
        """Return dynamics description."""
        return "Double integrator: ẍ = u (acceleration equals control input)"
    
    def get_problem_description(self, initial_state: np.ndarray) -> str:
        """Generate problem description for a specific initial state."""
        x0, v0 = initial_state
        return (f"Control a double integrator system with initial state "
                f"[position={x0:.2f}, velocity={v0:.2f}] to reach the origin (0,0) "
                f"in {self.total_time:.2f} seconds using {self.steps} steps. "
                f"Ensure all states remain within [-1,1] and controls within [-3,3].")
    
    def get_system_prompt(self, reasoning_start: str, reasoning_end: str,
                         solution_start: str, solution_end: str) -> str:
        """Get system-specific prompt for the LLM."""
        return f"""You are a control systems expert.
Given a double integrator system (ẍ = u) with initial position and velocity,
generate a sequence of {self.steps} control inputs to reach the origin (0,0) in exactly {self.total_time:.2f} seconds.
Position and velocity must stay within [-1, 1], and control inputs must be within [-3, 3].
Explain your approach between {reasoning_start} and {reasoning_end}.
Then provide exactly {self.steps} control values as a comma-separated list between {solution_start} and {solution_end}."""
    
    def get_state_names(self) -> List[str]:
        """Return names of state variables."""
        return ["position", "velocity"]
    
    def get_control_names(self) -> List[str]:
        """Return names of control variables."""
        return ["acceleration"]