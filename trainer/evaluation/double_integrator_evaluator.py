"""
Double integrator system evaluator implementation.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import os
import sys

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from .base_evaluator import BaseEvaluator
from .trajectory_analyzer import TrajectoryAnalyzer
from .plotting_utils import (
    create_comprehensive_plot,
    create_batch_comparison_plot,
    create_control_heatmap
)
from control import simulate_trajectory
from environments import get_environment
from logger import logger


class DoubleIntegratorEvaluator(BaseEvaluator):
    """Evaluator for double integrator control system."""
    
    def __init__(self, dt: float = 0.1, steps: int = 50):
        """
        Initialize double integrator evaluator.
        
        Args:
            dt: Time step
            steps: Number of control steps
        """
        super().__init__("double_integrator", dt, steps)
        
        # Create environment and analyzer
        self.env = get_environment("double_integrator", dt=dt, steps=steps)
        self.analyzer = TrajectoryAnalyzer(
            dt=dt,
            position_bounds=(-1.0, 1.0),
            velocity_bounds=(-1.0, 1.0),
            control_bounds=(-3.0, 3.0)
        )
        
    def evaluate_single(
        self,
        initial_state: np.ndarray,
        predicted_controls: np.ndarray,
        optimal_controls: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate a single trajectory.
        
        Args:
            initial_state: Initial state [position, velocity]
            predicted_controls: Predicted control sequence
            optimal_controls: Optional optimal control sequence
            **kwargs: Additional parameters
            
        Returns:
            Evaluation metrics dictionary
        """
        # Extract initial conditions
        x0, v0 = initial_state[0], initial_state[1]
        
        # Simulate predicted trajectory
        pred_positions, pred_velocities, times = simulate_trajectory(
            x0, v0, predicted_controls, self.dt
        )
        
        # Convert to numpy arrays
        pred_positions = np.array(pred_positions)
        pred_velocities = np.array(pred_velocities)
        pred_controls = np.array(predicted_controls)
        
        # Analyze predicted trajectory
        metrics = self.analyzer.analyze_trajectory(
            pred_positions, pred_velocities, pred_controls
        )
        
        # Add initial state info
        metrics['initial_position'] = float(x0)
        metrics['initial_velocity'] = float(v0)
        
        # If optimal controls provided, compare trajectories
        if optimal_controls is not None:
            opt_positions, opt_velocities, _ = simulate_trajectory(
                x0, v0, optimal_controls, self.dt
            )
            opt_positions = np.array(opt_positions)
            opt_velocities = np.array(opt_velocities)
            opt_controls = np.array(optimal_controls)
            
            # Compare trajectories
            comparison_metrics = self.analyzer.compare_trajectories(
                pred_positions, pred_velocities, pred_controls,
                opt_positions, opt_velocities, opt_controls
            )
            
            # Add comparison metrics with prefix
            for key, value in comparison_metrics.items():
                metrics[f'comparison_{key}'] = value
                
            # Store optimal trajectory for visualization
            metrics['optimal_positions'] = opt_positions
            metrics['optimal_velocities'] = opt_velocities
            metrics['optimal_controls'] = opt_controls
            
        # Store trajectory data for visualization
        metrics['positions'] = pred_positions
        metrics['velocities'] = pred_velocities
        metrics['controls'] = pred_controls
        metrics['times'] = np.array(times)
        
        return metrics
        
    def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]],
        predictions: List[np.ndarray],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of test cases.
        
        Args:
            test_cases: List of test cases with initial states
            predictions: List of predicted control sequences
            **kwargs: Additional parameters
            
        Returns:
            Batch evaluation metrics
        """
        if len(test_cases) != len(predictions):
            raise ValueError("Number of test cases must match number of predictions")
            
        batch_metrics = {
            'num_test_cases': len(test_cases),
            'individual_results': [],
            'aggregate_metrics': {}
        }
        
        # Evaluate each test case
        for i, (test_case, pred_controls) in enumerate(zip(test_cases, predictions)):
            # Extract initial state
            if 'initial_state' in test_case:
                initial_state = np.array(test_case['initial_state'])
            else:
                # Try to extract from metadata or other fields
                metadata = test_case.get('metadata', {})
                initial_state = np.array(metadata.get('initial_state', [0.0, 0.0]))
                
            # Get optimal controls if available
            optimal_controls = None
            if 'optimal_controls' in test_case:
                optimal_controls = np.array(test_case['optimal_controls'])
            elif 'metadata' in test_case and 'control_sequence' in test_case['metadata']:
                optimal_controls = np.array(test_case['metadata']['control_sequence'])
                
            # Evaluate single trajectory
            result = self.evaluate_single(
                initial_state, pred_controls, optimal_controls
            )
            
            # Add test case info
            result['test_case_id'] = i
            result['label'] = f"IC: [{initial_state[0]:.2f}, {initial_state[1]:.2f}]"
            
            batch_metrics['individual_results'].append(result)
            
        # Compute aggregate metrics
        self._compute_aggregate_metrics(batch_metrics)
        
        # Log summary
        logger.info(f"Batch evaluation completed: {batch_metrics['num_test_cases']} test cases")
        logger.info(f"Average final error: {batch_metrics['aggregate_metrics']['final_state_error_mean']:.4f}")
        logger.info(f"Success rate: {batch_metrics['aggregate_metrics']['success_rate']:.2%}")
        
        return batch_metrics
        
    def _compute_aggregate_metrics(self, batch_metrics: Dict[str, Any]) -> None:
        """Compute aggregate metrics from individual results."""
        results = batch_metrics['individual_results']
        
        if not results:
            return
            
        # Define metrics to aggregate
        metric_keys = [
            'final_state_error', 'control_effort', 'convergence_time',
            'total_violations', 'phase_space_path_length',
            'position_rmse', 'velocity_rmse', 'control_rmse'
        ]
        
        aggregate = batch_metrics['aggregate_metrics']
        
        # Compute statistics for each metric
        for key in metric_keys:
            values = []
            for result in results:
                if key in result:
                    value = result[key]
                    if value != float('inf'):  # Exclude infinite values
                        values.append(value)
                        
            if values:
                aggregate[f'{key}_mean'] = float(np.mean(values))
                aggregate[f'{key}_std'] = float(np.std(values))
                aggregate[f'{key}_min'] = float(np.min(values))
                aggregate[f'{key}_max'] = float(np.max(values))
                aggregate[f'{key}_median'] = float(np.median(values))
                
        # Compute success metrics
        success_threshold = 0.01
        successes = sum(1 for r in results if r.get('final_state_error', float('inf')) < success_threshold)
        aggregate['success_rate'] = successes / len(results)
        aggregate['num_successes'] = successes
        
        # Constraint violation rate
        violations = sum(1 for r in results if r.get('total_violations', 0) > 0)
        aggregate['violation_rate'] = violations / len(results)
        
        # Convergence rate
        converged = sum(1 for r in results if r.get('converged', False))
        aggregate['convergence_rate'] = converged / len(results)
        
    def visualize_single(
        self,
        initial_state: np.ndarray,
        predicted_controls: np.ndarray,
        optimal_controls: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Create comprehensive visualization for a single trajectory.
        
        Args:
            initial_state: Initial state
            predicted_controls: Predicted control sequence
            optimal_controls: Optional optimal controls
            save_path: Path to save visualization
            **kwargs: Additional parameters
        """
        # Evaluate to get trajectories
        result = self.evaluate_single(
            initial_state, predicted_controls, optimal_controls
        )
        
        # Extract trajectories
        times = result['times']
        positions = result['positions']
        velocities = result['velocities']
        controls = result['controls']
        
        # Extract optimal trajectories if available
        opt_positions = result.get('optimal_positions', None)
        opt_velocities = result.get('optimal_velocities', None)
        opt_controls = result.get('optimal_controls', None)
        
        # Create title with metrics
        title = f"Double Integrator Control Analysis\n"
        title += f"Initial State: [{initial_state[0]:.2f}, {initial_state[1]:.2f}] | "
        title += f"Final Error: {result['final_state_error']:.4f} | "
        title += f"Control Effort: {result['control_effort']:.2f}"
        
        # Create comprehensive plot
        create_comprehensive_plot(
            times=times,
            positions=positions,
            velocities=velocities,
            controls=controls,
            positions_opt=opt_positions,
            velocities_opt=opt_velocities,
            controls_opt=opt_controls,
            title=title,
            save_path=save_path
        )
        
    def visualize_batch(
        self,
        results: List[Dict[str, Any]],
        save_path: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Create batch visualization.
        
        Args:
            results: List of evaluation results
            save_path: Base path for saving visualizations
            **kwargs: Additional parameters
        """
        if not results:
            logger.warning("No results to visualize")
            return
            
        # Prepare data for batch comparison
        comparison_data = []
        control_sequences = []
        labels = []
        
        for result in results:
            comparison_data.append({
                'positions': result['positions'],
                'velocities': result['velocities'],
                'controls': result['controls'],
                'times': result['times'],
                'label': result.get('label', f"Test {result.get('test_case_id', 0) + 1}")
            })
            control_sequences.append(result['controls'])
            labels.append(result.get('label', f"Test {result.get('test_case_id', 0) + 1}"))
            
        # Create comparison plot
        if save_path:
            comparison_path = save_path.replace('.png', '_comparison.png')
        else:
            comparison_path = None
            
        create_batch_comparison_plot(
            results=comparison_data,
            title="Double Integrator Batch Evaluation",
            save_path=comparison_path
        )
        
        # Create control heatmap
        if save_path:
            heatmap_path = save_path.replace('.png', '_heatmap.png')
        else:
            heatmap_path = None
            
        create_control_heatmap(
            control_sequences=control_sequences,
            labels=labels,
            title="Control Sequence Heatmap",
            save_path=heatmap_path
        )
        
        logger.info(f"Batch visualizations created: {len(results)} trajectories")