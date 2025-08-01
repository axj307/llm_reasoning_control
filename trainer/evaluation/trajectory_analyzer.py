"""
Trajectory analyzer for control system evaluation.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional


class TrajectoryAnalyzer:
    """Analyze control system trajectories and compute metrics."""
    
    def __init__(self, dt: float, position_bounds: Tuple[float, float] = (-1.0, 1.0),
                 velocity_bounds: Tuple[float, float] = (-1.0, 1.0),
                 control_bounds: Tuple[float, float] = (-3.0, 3.0)):
        """
        Initialize trajectory analyzer.
        
        Args:
            dt: Time step
            position_bounds: Position constraints
            velocity_bounds: Velocity constraints
            control_bounds: Control input constraints
        """
        self.dt = dt
        self.position_bounds = position_bounds
        self.velocity_bounds = velocity_bounds
        self.control_bounds = control_bounds
        
    def analyze_trajectory(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        controls: np.ndarray,
        target_state: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Analyze a single trajectory.
        
        Args:
            positions: Position trajectory
            velocities: Velocity trajectory
            controls: Control sequence
            target_state: Target state [position, velocity]
            
        Returns:
            Dictionary containing trajectory metrics
        """
        if target_state is None:
            target_state = np.array([0.0, 0.0])
            
        metrics = {}
        
        # Basic trajectory properties
        metrics['trajectory_length'] = len(positions)
        metrics['total_time'] = (len(positions) - 1) * self.dt
        
        # Final state metrics
        final_position = positions[-1]
        final_velocity = velocities[-1]
        metrics['final_position'] = float(final_position)
        metrics['final_velocity'] = float(final_velocity)
        metrics['final_position_error'] = float(abs(final_position - target_state[0]))
        metrics['final_velocity_error'] = float(abs(final_velocity - target_state[1]))
        metrics['final_state_error'] = float(np.linalg.norm([final_position - target_state[0], 
                                                             final_velocity - target_state[1]]))
        
        # Constraint violations
        metrics['position_violations'] = int(np.sum((positions < self.position_bounds[0]) | 
                                                    (positions > self.position_bounds[1])))
        metrics['velocity_violations'] = int(np.sum((velocities < self.velocity_bounds[0]) | 
                                                    (velocities > self.velocity_bounds[1])))
        metrics['control_violations'] = int(np.sum((controls < self.control_bounds[0]) | 
                                                   (controls > self.control_bounds[1])))
        metrics['total_violations'] = (metrics['position_violations'] + 
                                      metrics['velocity_violations'] + 
                                      metrics['control_violations'])
        
        # Control effort metrics
        metrics['control_effort'] = float(np.sum(np.square(controls)))
        metrics['control_variation'] = float(np.sum(np.square(np.diff(controls))))
        metrics['max_control'] = float(np.max(np.abs(controls)))
        metrics['mean_control'] = float(np.mean(np.abs(controls)))
        
        # Trajectory smoothness
        metrics['position_smoothness'] = float(np.sum(np.square(np.diff(positions, n=2))))
        metrics['velocity_smoothness'] = float(np.sum(np.square(np.diff(velocities))))
        
        # Convergence analysis
        convergence_info = self._analyze_convergence(positions, velocities, target_state)
        metrics.update(convergence_info)
        
        # Phase space metrics
        phase_space_info = self._analyze_phase_space(positions, velocities)
        metrics.update(phase_space_info)
        
        # Energy metrics
        energy_info = self._compute_energy_metrics(positions, velocities, controls)
        metrics.update(energy_info)
        
        return metrics
        
    def _analyze_convergence(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        target_state: np.ndarray,
        threshold: float = 0.01
    ) -> Dict[str, Any]:
        """Analyze convergence properties."""
        metrics = {}
        
        # Compute distance to target over time
        distances = np.sqrt((positions - target_state[0])**2 + 
                           (velocities - target_state[1])**2)
        
        # Find convergence time
        converged_indices = np.where(distances < threshold)[0]
        if len(converged_indices) > 0:
            metrics['converged'] = True
            metrics['convergence_time'] = float(converged_indices[0] * self.dt)
            metrics['convergence_index'] = int(converged_indices[0])
            
            # Check if it stays converged
            if np.all(distances[converged_indices[0]:] < threshold):
                metrics['stable_convergence'] = True
            else:
                metrics['stable_convergence'] = False
        else:
            metrics['converged'] = False
            metrics['convergence_time'] = float('inf')
            metrics['convergence_index'] = -1
            metrics['stable_convergence'] = False
            
        # Minimum distance achieved
        metrics['min_distance_to_target'] = float(np.min(distances))
        metrics['min_distance_time'] = float(np.argmin(distances) * self.dt)
        
        return metrics
        
    def _analyze_phase_space(
        self,
        positions: np.ndarray,
        velocities: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze phase space properties."""
        metrics = {}
        
        # Phase space path length
        path_length = 0
        for i in range(1, len(positions)):
            path_length += np.sqrt((positions[i] - positions[i-1])**2 + 
                                  (velocities[i] - velocities[i-1])**2)
        metrics['phase_space_path_length'] = float(path_length)
        
        # Phase space area (using shoelace formula)
        area = 0
        for i in range(len(positions) - 1):
            area += positions[i] * velocities[i+1] - positions[i+1] * velocities[i]
        metrics['phase_space_area'] = float(abs(area) / 2)
        
        # Maximum distance from origin
        distances_from_origin = np.sqrt(positions**2 + velocities**2)
        metrics['max_distance_from_origin'] = float(np.max(distances_from_origin))
        metrics['mean_distance_from_origin'] = float(np.mean(distances_from_origin))
        
        return metrics
        
    def _compute_energy_metrics(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        controls: np.ndarray
    ) -> Dict[str, Any]:
        """Compute energy-related metrics."""
        metrics = {}
        
        # Kinetic energy over time
        kinetic_energy = 0.5 * velocities**2
        metrics['initial_kinetic_energy'] = float(kinetic_energy[0])
        metrics['final_kinetic_energy'] = float(kinetic_energy[-1])
        metrics['max_kinetic_energy'] = float(np.max(kinetic_energy))
        metrics['total_kinetic_energy'] = float(np.sum(kinetic_energy) * self.dt)
        
        # Work done by control
        work = 0
        for i in range(len(controls)):
            work += controls[i] * velocities[i] * self.dt
        metrics['total_work_done'] = float(work)
        
        # Control power
        power = controls * velocities[:-1]
        metrics['max_power'] = float(np.max(np.abs(power)))
        metrics['mean_power'] = float(np.mean(np.abs(power)))
        
        return metrics
        
    def compare_trajectories(
        self,
        predicted_positions: np.ndarray,
        predicted_velocities: np.ndarray,
        predicted_controls: np.ndarray,
        optimal_positions: np.ndarray,
        optimal_velocities: np.ndarray,
        optimal_controls: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compare predicted trajectory with optimal trajectory.
        
        Returns:
            Dictionary containing comparison metrics
        """
        metrics = {}
        
        # Trajectory differences
        position_diff = predicted_positions - optimal_positions
        velocity_diff = predicted_velocities - optimal_velocities
        control_diff = predicted_controls - optimal_controls
        
        # RMS errors
        metrics['position_rmse'] = float(np.sqrt(np.mean(position_diff**2)))
        metrics['velocity_rmse'] = float(np.sqrt(np.mean(velocity_diff**2)))
        metrics['control_rmse'] = float(np.sqrt(np.mean(control_diff**2)))
        
        # Maximum errors
        metrics['max_position_error'] = float(np.max(np.abs(position_diff)))
        metrics['max_velocity_error'] = float(np.max(np.abs(velocity_diff)))
        metrics['max_control_error'] = float(np.max(np.abs(control_diff)))
        
        # State trajectory error
        state_errors = np.sqrt(position_diff**2 + velocity_diff**2)
        metrics['state_rmse'] = float(np.sqrt(np.mean(state_errors**2)))
        metrics['max_state_error'] = float(np.max(state_errors))
        
        # Control effort comparison
        pred_effort = float(np.sum(predicted_controls**2))
        opt_effort = float(np.sum(optimal_controls**2))
        metrics['control_effort_ratio'] = float(pred_effort / opt_effort if opt_effort > 0 else float('inf'))
        metrics['control_effort_difference'] = float(pred_effort - opt_effort)
        
        # Phase space similarity
        pred_path_length = self._compute_phase_path_length(predicted_positions, predicted_velocities)
        opt_path_length = self._compute_phase_path_length(optimal_positions, optimal_velocities)
        metrics['phase_path_length_ratio'] = float(pred_path_length / opt_path_length if opt_path_length > 0 else float('inf'))
        
        # Correlation coefficients
        if len(predicted_controls) == len(optimal_controls):
            metrics['control_correlation'] = float(np.corrcoef(predicted_controls, optimal_controls)[0, 1])
        
        return metrics
        
    def _compute_phase_path_length(self, positions: np.ndarray, velocities: np.ndarray) -> float:
        """Compute phase space path length."""
        path_length = 0
        for i in range(1, len(positions)):
            path_length += np.sqrt((positions[i] - positions[i-1])**2 + 
                                  (velocities[i] - velocities[i-1])**2)
        return path_length