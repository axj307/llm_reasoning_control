"""Van der Pol oscillator system implementation."""

import numpy as np
from typing import Tuple, List
from scipy.integrate import solve_ivp
from .base_environment import BaseEnvironment


class VanDerPol(BaseEnvironment):
    """Van der Pol oscillator system: ẍ - μ(1-x²)ẋ + x = u"""
    
    def __init__(self, mu: float = 1.0, dt: float = 0.1, steps: int = 50):
        super().__init__(
            name="van_der_pol",
            state_dim=2,  # position and velocity
            control_dim=1,  # control force
            dt=dt,
            steps=steps
        )
        self.mu = mu  # Damping parameter
    
    def get_state_bounds(self) -> List[Tuple[float, float]]:
        """Position and velocity bounds: [-2, 2] for both."""
        return [(-2.0, 2.0), (-2.0, 2.0)]
    
    def get_control_bounds(self) -> Tuple[float, float]:
        """Control bounds: [-5, 5]."""
        return (-5.0, 5.0)
    
    def get_initial_state_bounds(self) -> List[Tuple[float, float]]:
        """Initial condition sampling bounds: [-1.5, 1.5] for both."""
        return [(-1.5, 1.5), (-1.5, 1.5)]
    
    def simulate_step(self, state: np.ndarray, control: float) -> np.ndarray:
        """
        Simulate one time step of Van der Pol oscillator.
        
        Args:
            state: [position, velocity]
            control: control input u
            
        Returns:
            new_state: [new_position, new_velocity]
        """
        def vdp_dynamics(t, y, u):
            """Van der Pol dynamics with control."""
            x, v = y
            dxdt = v
            dvdt = self.mu * (1 - x**2) * v - x + u
            return [dxdt, dvdt]
        
        # Use RK45 integration for better accuracy with nonlinear dynamics
        sol = solve_ivp(
            lambda t, y: vdp_dynamics(t, y, control),
            [0, self.dt], 
            state, 
            method='RK45',
            t_eval=[self.dt]
        )
        
        return sol.y[:, -1]
    
    def get_dynamics_description(self) -> str:
        """Return dynamics description."""
        return f"Van der Pol oscillator: ẍ - μ(1-x²)ẋ + x = u, with μ={self.mu}"
    
    def get_problem_description(self, initial_state: np.ndarray) -> str:
        """Generate problem description for a specific initial state."""
        x0, v0 = initial_state
        return (f"Control a Van der Pol oscillator system (μ={self.mu}) with initial state "
                f"[position={x0:.2f}, velocity={v0:.2f}] to reach the origin (0,0) "
                f"in {self.total_time:.2f} seconds using {self.steps} steps. "
                f"Ensure all states remain within [-2,2] and controls within [-5,5].")
    
    def get_system_prompt(self, reasoning_start: str, reasoning_end: str,
                         solution_start: str, solution_end: str) -> str:
        """Get system-specific prompt for the LLM."""
        return f"""You are a control systems expert.
Given a Van der Pol oscillator system (ẍ - μ(1-x²)ẋ + x = u) with μ={self.mu}, initial position and velocity,
generate a sequence of {self.steps} control inputs to reach the origin (0,0) in exactly {self.total_time:.2f} seconds.
Position and velocity must stay within [-2, 2], and control inputs must be within [-5, 5].
Explain your approach between {reasoning_start} and {reasoning_end}.
Then provide exactly {self.steps} control values as a comma-separated list between {solution_start} and {solution_end}."""
    
    def get_state_names(self) -> List[str]:
        """Return names of state variables."""
        return ["position", "velocity"]
    
    def get_control_names(self) -> List[str]:
        """Return names of control variables."""
        return ["force"]
    
    def get_info(self):
        """Return system information including mu parameter."""
        info = super().get_info()
        info['mu'] = self.mu
        return info