"""Universal data pipeline for multi-system training."""

import numpy as np
import random
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
try:
    from datasets import Dataset
except ImportError:
    try:
        import datasets
        Dataset = datasets.Dataset
    except:
        # Fallback - create a simple Dataset class if needed
        class Dataset:
            @staticmethod
            def from_dict(data):
                return data
            
            @staticmethod
            def from_list(data):
                return data

from environments import get_system, list_systems
from .solvers import get_solver
from .solvers.lqr_solver import compute_lqr_cost


class UniversalDataGenerator:
    """Generate training data for multiple control systems."""
    
    def __init__(self, systems: Optional[List[str]] = None,
                 dt: float = 0.1, steps: int = 50,
                 reasoning_start: str = "<REASONING>",
                 reasoning_end: str = "</REASONING>",
                 solution_start: str = "<CONTROLS>",
                 solution_end: str = "</CONTROLS>"):
        """
        Initialize data generator.
        
        Args:
            systems: List of system names to generate data for. If None, uses all.
            dt: Time step
            steps: Number of control steps
            reasoning_start/end: Tags for reasoning section
            solution_start/end: Tags for control solution
        """
        self.systems = systems or list_systems()
        self.dt = dt
        self.steps = steps
        self.reasoning_start = reasoning_start
        self.reasoning_end = reasoning_end
        self.solution_start = solution_start
        self.solution_end = solution_end
    
    def generate_system_data(self, system_name: str, num_samples: int) -> List[Dict[str, Any]]:
        """Generate data for a specific system."""
        # Create system instance
        system = get_system(system_name)(dt=self.dt, steps=self.steps)
        solver = get_solver(system_name)
        
        data = []
        
        for i in range(num_samples):
            # Generate random initial state
            initial_state = system.generate_random_initial_state()
            
            # Get problem description
            problem = system.get_problem_description(initial_state)
            
            # Get system prompt
            system_prompt = system.get_system_prompt(
                self.reasoning_start, self.reasoning_end,
                self.solution_start, self.solution_end
            )
            
            # Solve for optimal control
            controls = solver(initial_state, self.dt, self.steps)
            
            # Simulate trajectory from optimal controls
            trajectory, lqr_cost, final_error = self._simulate_trajectory_with_metrics(
                system, initial_state, controls
            )
            
            # Generate reasoning text
            reasoning = self._generate_reasoning(system_name, initial_state, controls)
            
            # Format control values
            control_str = ", ".join([f"{u:.3f}" for u in controls])
            
            # Create complete output
            complete_output = (f"{self.reasoning_start}{reasoning}{self.reasoning_end}"
                             f"{self.solution_start}{control_str}{self.solution_end}")
            
            # Create data entry - match working notebook format exactly
            data_entry = {
                "system_type": system_name,
                "initial_state": initial_state.tolist(),
                "controls": controls,
                "trajectory": trajectory,  # Full state trajectory from optimal controls
                "lqr_cost": lqr_cost,     # LQR cost for this trajectory
                "final_error": final_error, # Final distance from origin
                "system_prompt": system_prompt,
                "problem": problem,
                "reasoning": reasoning,
                "complete_output": complete_output,
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": problem}
                ],
                "answer": control_str,  # GRPO expects this field
                "Messages": [  # Working notebook has capital M
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": complete_output}
                ]
            }
            
            data.append(data_entry)
        
        return data
    
    def _simulate_trajectory_with_metrics(self, system, initial_state: np.ndarray, 
                                        controls: List[float]) -> Tuple[List[List[float]], float, float]:
        """
        Simulate trajectory and compute LQR cost and metrics.
        
        Args:
            system: Environment instance with simulate_step method
            initial_state: Initial state [x0, v0]
            controls: List of control inputs
            
        Returns:
            trajectory: List of states [[x0, v0], [x1, v1], ...]
            lqr_cost: Total LQR cost for this trajectory
            final_error: Final distance from origin
        """
        # Simulate trajectory
        trajectory = [initial_state.tolist()]
        state = initial_state.copy()
        
        for control in controls:
            # Apply control bounds (should already be satisfied by solver, but safety)
            u_min, u_max = system.get_control_bounds()
            control = max(u_min, min(u_max, control))
            
            # Simulate one step
            state = system.simulate_step(state, control)
            
            # Apply state bounds (though LQR should keep within bounds)
            state_bounds = system.get_state_bounds()
            for i, (s_min, s_max) in enumerate(state_bounds):
                state[i] = max(s_min, min(s_max, state[i]))
            
            trajectory.append(state.tolist())
        
        # Calculate LQR cost (only for double integrator for now)
        if system.name == "double_integrator":
            # LQR parameters (matching lqr_solver.py)
            Q = np.array([[10.0, 0.0], [0.0, 10.0]])
            R = 0.1
            
            # Convert trajectory to state arrays for cost calculation
            states = [np.array(s).reshape(-1, 1) for s in trajectory]
            lqr_cost = compute_lqr_cost(states, controls, Q, R)
        else:
            # For other systems, use a simple quadratic cost
            lqr_cost = 0.0
            for state, control in zip(trajectory[:-1], controls):
                lqr_cost += np.sum(np.array(state)**2) + 0.1 * control**2
            # Add final state cost
            final_state = trajectory[-1]
            lqr_cost += np.sum(np.array(final_state)**2)
        
        # Calculate final error (distance from origin)
        final_state = trajectory[-1]
        final_error = np.sqrt(sum(x**2 for x in final_state))
        
        return trajectory, float(lqr_cost), float(final_error)
    
    def generate_universal_dataset(self, samples_per_system: int = 200) -> List[Dict[str, Any]]:
        """Generate data for all configured systems."""
        all_data = []
        
        print(f"Generating data for systems: {', '.join(self.systems)}")
        
        for system_name in self.systems:
            print(f"  Generating {samples_per_system} samples for {system_name}...")
            system_data = self.generate_system_data(system_name, samples_per_system)
            all_data.extend(system_data)
        
        # Shuffle to mix different systems
        random.shuffle(all_data)
        
        print(f"Total samples generated: {len(all_data)}")
        return all_data
    
    def generate_single_system_dataset(self, system_name: str, 
                                     num_samples: int = 500) -> List[Dict[str, Any]]:
        """Generate data for a single system."""
        print(f"Generating {num_samples} samples for {system_name}...")
        data = self.generate_system_data(system_name, num_samples)
        
        # Shuffle the data
        random.shuffle(data)
        
        return data
    
    def add_new_system_data(self, existing_data: List[Dict[str, Any]], 
                          new_system_name: str, 
                          num_samples: int = 200) -> List[Dict[str, Any]]:
        """Add data for a new system to existing dataset."""
        print(f"Adding {num_samples} samples for new system: {new_system_name}")
        
        new_data = self.generate_system_data(new_system_name, num_samples)
        
        # Combine with existing data
        combined_data = existing_data + new_data
        
        # Shuffle to mix old and new
        random.shuffle(combined_data)
        
        return combined_data
    
    def _generate_reasoning(self, system_name: str, initial_state: np.ndarray, 
                          controls: List[float]) -> str:
        """Generate reasoning text for a solution."""
        if system_name == "double_integrator":
            return self._generate_di_reasoning(initial_state, controls)
        elif system_name == "van_der_pol":
            return self._generate_vdp_reasoning(initial_state, controls)
        else:
            # Generic reasoning for future systems
            return self._generate_generic_reasoning(system_name, initial_state, controls)
    
    def _generate_di_reasoning(self, initial_state: np.ndarray, controls: List[float]) -> str:
        """Generate reasoning for double integrator."""
        x0, v0 = initial_state
        total_time = self.dt * self.steps
        
        reasoning = f"""For the double integrator system starting at position {x0:.2f} and velocity {v0:.2f}, I'll apply Linear Quadratic Regulator (LQR) control to reach the origin optimally in {total_time:.2f} seconds using {self.steps} steps.

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
        
        return reasoning.strip()
    
    def _generate_vdp_reasoning(self, initial_state: np.ndarray, controls: List[float]) -> str:
        """Generate reasoning for Van der Pol oscillator."""
        x0, v0 = initial_state
        total_time = self.dt * self.steps
        
        reasoning = f"""For the Van der Pol oscillator starting at position {x0:.2f} and velocity {v0:.2f}, I'll design a control sequence to reach the origin optimally in {total_time:.2f} seconds using {self.steps} steps.

The Van der Pol oscillator follows the dynamics:
- ẋ = v
- v̇ = μ(1-x²)v - x + u
where μ=1 determines the strength of the nonlinearity.

Unlike the double integrator, this system exhibits self-sustained oscillations and has nonlinear dynamics.
To control it effectively, I need to:
1. Counteract the nonlinear damping term μ(1-x²)v
2. Add appropriate control to stabilize the system toward the origin
3. Ensure the state stays within constraints [-2,2] and controls within [-5,5]

I'll use a model-predictive approach with a quadratic cost function that penalizes:
- Deviation from the origin (state cost)
- Excessive control effort (control cost)
- With increasing weight on errors as we approach the final time

This results in a smooth control sequence that navigates the nonlinear dynamics
to bring the system to rest at the origin."""
        
        return reasoning.strip()
    
    def _generate_generic_reasoning(self, system_name: str, initial_state: np.ndarray, 
                                  controls: List[float]) -> str:
        """Generate generic reasoning for future systems."""
        total_time = self.dt * self.steps
        system = get_system(system_name)(dt=self.dt, steps=self.steps)
        
        reasoning = f"""For the {system_name.replace('_', ' ')} system with initial state {initial_state.tolist()}, I'll design an optimal control sequence to reach the target state in {total_time:.2f} seconds using {self.steps} steps.

System dynamics: {system.get_dynamics_description()}

Control approach:
1. Analyze the system dynamics and constraints
2. Apply appropriate control theory (LQR, MPC, or other methods)
3. Ensure all state and control constraints are satisfied
4. Minimize a cost function balancing state error and control effort

The computed control sequence will guide the system to the desired target while respecting all constraints."""
        
        return reasoning.strip()
    
    def save_dataset(self, data: List[Dict[str, Any]], filename: str):
        """Save dataset to file."""
        save_path = Path("data/saved_datasets")
        save_path.mkdir(parents=True, exist_ok=True)
        
        full_path = save_path / filename
        
        with open(full_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Dataset saved to {full_path}")
        
        # Also save metadata
        metadata = {
            "num_samples": len(data),
            "systems": list(set(d["system_type"] for d in data)),
            "dt": self.dt,
            "steps": self.steps,
            "filename": filename
        }
        
        metadata_path = full_path.with_suffix('.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_dataset(self, filename: str) -> List[Dict[str, Any]]:
        """Load dataset from file."""
        load_path = Path("data/saved_datasets") / filename
        
        if not load_path.exists():
            raise FileNotFoundError(f"Dataset not found: {load_path}")
        
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Loaded dataset with {len(data)} samples")
        return data
    
    def to_huggingface_dataset(self, data: List[Dict[str, Any]], 
                              for_training: bool = True) -> Dataset:
        """Convert to HuggingFace Dataset format."""
        if for_training:
            # For training, we need the full messages
            dataset_dict = {
                "Messages": [d["Messages"] for d in data],
                "prompt": [d["prompt"] for d in data],
                "answer": [d["answer"] for d in data],
                "system_type": [d["system_type"] for d in data],
                "initial_state": [d["initial_state"] for d in data],
                "controls": [d["controls"] for d in data]
            }
        else:
            # For other uses, include all fields
            dataset_dict = data
        
        return Dataset.from_dict(dataset_dict)
    
    def get_dataset(self, system_name: str, num_samples: int, for_training: bool = True):
        """Generate and return a HuggingFace dataset."""
        data = self.generate_single_system_dataset(system_name, num_samples)
        return self.to_huggingface_dataset(data, for_training=for_training)

    def split_dataset(self, data: List[Dict[str, Any]], 
                     train_ratio: float = 0.9) -> Tuple[List[Dict], List[Dict]]:
        """Split dataset into train and eval sets."""
        # Shuffle first
        shuffled_data = data.copy()
        random.shuffle(shuffled_data)
        
        # Split
        split_idx = int(len(shuffled_data) * train_ratio)
        train_data = shuffled_data[:split_idx]
        eval_data = shuffled_data[split_idx:]
        
        print(f"Split dataset: {len(train_data)} train, {len(eval_data)} eval")
        return train_data, eval_data