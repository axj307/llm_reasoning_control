"""
Double integrator dataset implementation.
"""

import numpy as np
import sys
import os
from typing import Dict, Any, List

# Get the trainer directory path
trainer_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if trainer_path not in sys.path:
    sys.path.insert(0, trainer_path)

# Use absolute imports to avoid conflicts
from datasets.base_dataset import BaseControlDataset

# Import other modules directly
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parent_dir)

# Now import from root-level modules
from environments import get_environment
from config import REASONING_START, REASONING_END, SOLUTION_START, SOLUTION_END


class DoubleIntegratorDataset(BaseControlDataset):
    """Dataset for double integrator control problems."""
    
    def __init__(self, num_samples: int, dt: float, steps: int, seed: int = 42, **kwargs):
        """
        Initialize double integrator dataset.
        
        Args:
            num_samples: Number of samples to generate
            dt: Time step
            steps: Number of control steps
            seed: Random seed
            **kwargs: Additional arguments
        """
        super().__init__("double_integrator", num_samples, dt, steps, seed)
        
        # Create environment
        self.env = get_environment("double_integrator", dt=dt, steps=steps)
        
        # State bounds for initial conditions
        self.position_range = kwargs.get('position_range', (-0.8, 0.8))
        self.velocity_range = kwargs.get('velocity_range', (-0.8, 0.8))
        
    def _get_system_prompt(self) -> str:
        """Generate system prompt."""
        total_time = self.dt * self.steps
        return f"""You are a control systems expert.
Given a double integrator system (ẍ = u) with initial position and velocity,
generate a sequence of {self.steps} control inputs to reach the origin (0,0) in exactly {total_time:.2f} seconds.
Position and velocity must stay within [-1, 1], and control inputs must be within [-3, 3].
Explain your approach between {REASONING_START} and {REASONING_END}.
Then provide exactly {self.steps} control values as a comma-separated list between {SOLUTION_START} and {SOLUTION_END}."""
        
    def _format_problem(self, x0: float, v0: float) -> str:
        """Format problem statement."""
        total_time = self.dt * self.steps
        return (f"Control a double integrator system with initial state "
                f"[position={x0:.2f}, velocity={v0:.2f}] to reach the origin (0,0) "
                f"in {total_time:.2f} seconds using {self.steps} steps. "
                f"Ensure all states remain within [-1,1] and controls within [-3,3].")
                
    def _format_reasoning(self, x0: float, v0: float) -> str:
        """Generate reasoning text."""
        total_time = self.dt * self.steps
        return f"""For the double integrator system starting at position {x0:.2f} and velocity {v0:.2f}, I'll apply Linear Quadratic Regulator (LQR) control to reach the origin optimally in {total_time:.2f} seconds using {self.steps} steps.

        The LQR approach provides an optimal feedback control law by minimizing a quadratic cost function that balances:
        1. The error in state (position and velocity)
        2. The control effort used

        For a double integrator with dynamics:
        - ẋ = v
        - v̇ = u

        The discrete-time state-space representation is:
        - x(k+1) = Ax(k) + Bu(k)

        Where:
        - A = [[1, Δt], [0, 1]]
        - B = [[0.5(Δt)², Δt]]
        - Δt = {self.dt:.2f} seconds

        Computing the optimal gain matrix K through the Riccati equation gives a feedback law u = -Kx.
        This produces a smooth control sequence that brings the system to the origin while respecting constraints.

        The resulting {self.steps} control inputs applied over {total_time:.2f} seconds will optimally control the system to the target state."""
        
    def _format_controls(self, controls: np.ndarray) -> str:
        """Format control sequence as comma-separated string."""
        return ", ".join([f"{u:.3f}" for u in controls])
        
    def generate(self) -> None:
        """Generate dataset samples."""
        system_prompt = self._get_system_prompt()
        
        for i in range(self.num_samples):
            # Random initial conditions
            x0 = np.random.uniform(*self.position_range)
            v0 = np.random.uniform(*self.velocity_range)
            initial_state = np.array([x0, v0])
            
            # Solve for optimal control
            control_sequence = self.env.solve_optimal_control(initial_state)
            
            # Format reasoning and solution
            reasoning = self._format_reasoning(x0, v0)
            controls_str = self._format_controls(control_sequence)
            
            # Create complete response
            assistant_response = f"{reasoning}{REASONING_END}{SOLUTION_START}{controls_str}{SOLUTION_END}"
            
            # Create sample
            sample = {
                'prompt': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': self._format_problem(x0, v0)}
                ],
                'answer': controls_str,
                'Messages': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': self._format_problem(x0, v0)},
                    {'role': 'assistant', 'content': assistant_response}
                ],
                'system_type': 'double_integrator',
                'metadata': {
                    'initial_state': initial_state.tolist(),
                    'control_sequence': control_sequence.tolist(),
                    'dt': self.dt,
                    'steps': self.steps,
                    'total_time': self.dt * self.steps
                }
            }
            
            self.data.append(sample)