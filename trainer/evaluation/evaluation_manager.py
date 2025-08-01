"""
Evaluation manager for coordinating system evaluations.
"""

import os
import json
import pickle
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import numpy as np

from .double_integrator_evaluator import DoubleIntegratorEvaluator
from .base_evaluator import BaseEvaluator
from logger import logger


class EvaluationManager:
    """Manage evaluations across different control systems."""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        """
        Initialize evaluation manager.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = output_dir
        self.evaluators = {}
        self.results_cache = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Register available evaluators
        self._register_evaluators()
        
    def _register_evaluators(self):
        """Register available system evaluators."""
        # Add more evaluators as systems are implemented
        self.evaluator_classes = {
            'double_integrator': DoubleIntegratorEvaluator,
            'di': DoubleIntegratorEvaluator,  # Alias
        }
        
    def get_evaluator(
        self,
        system: str,
        dt: float = 0.1,
        steps: int = 50,
        **kwargs
    ) -> BaseEvaluator:
        """
        Get or create evaluator for a system.
        
        Args:
            system: System name
            dt: Time step
            steps: Number of control steps
            **kwargs: Additional evaluator parameters
            
        Returns:
            System evaluator instance
        """
        key = f"{system}_{dt}_{steps}"
        
        if key not in self.evaluators:
            if system not in self.evaluator_classes:
                raise ValueError(f"Unknown system: {system}. Available: {list(self.evaluator_classes.keys())}")
                
            evaluator_class = self.evaluator_classes[system]
            self.evaluators[key] = evaluator_class(dt=dt, steps=steps, **kwargs)
            
        return self.evaluators[key]
        
    def evaluate_model(
        self,
        model_name: str,
        system: str,
        test_cases: List[Dict[str, Any]],
        predictions: List[Union[List[float], np.ndarray]],
        dt: float = 0.1,
        steps: int = 50,
        save_results: bool = True,
        visualize: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate a model on test cases.
        
        Args:
            model_name: Name of the model being evaluated
            system: Control system name
            test_cases: List of test cases
            predictions: List of predicted control sequences
            dt: Time step
            steps: Number of control steps
            save_results: Whether to save results to disk
            visualize: Whether to create visualizations
            **kwargs: Additional parameters
            
        Returns:
            Evaluation results dictionary
        """
        logger.info(f"Evaluating model '{model_name}' on {system} with {len(test_cases)} test cases")
        
        # Get evaluator
        evaluator = self.get_evaluator(system, dt, steps)
        
        # Convert predictions to numpy arrays
        predictions = [np.array(pred) for pred in predictions]
        
        # Run batch evaluation
        results = evaluator.evaluate_batch(test_cases, predictions, **kwargs)
        
        # Add metadata
        results['model_name'] = model_name
        results['system'] = system
        results['timestamp'] = datetime.now().isoformat()
        results['dt'] = dt
        results['steps'] = steps
        
        # Create visualizations
        if visualize:
            self._create_visualizations(model_name, system, evaluator, results)
            
        # Save results
        if save_results:
            self._save_results(model_name, system, results)
            
        # Cache results
        cache_key = f"{model_name}_{system}"
        self.results_cache[cache_key] = results
        
        return results
        
    def _create_visualizations(
        self,
        model_name: str,
        system: str,
        evaluator: BaseEvaluator,
        results: Dict[str, Any]
    ) -> None:
        """Create and save visualizations."""
        # Create subdirectory for this evaluation
        viz_dir = os.path.join(self.output_dir, f"{model_name}_{system}_visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Batch visualization
        batch_path = os.path.join(viz_dir, "batch_evaluation.png")
        evaluator.visualize_batch(
            results['individual_results'],
            save_path=batch_path
        )
        
        # Individual trajectory visualizations for top/bottom performers
        individual_results = results['individual_results']
        
        # Sort by final error
        sorted_results = sorted(individual_results, key=lambda x: x.get('final_state_error', float('inf')))
        
        # Visualize best and worst cases
        n_examples = min(3, len(sorted_results))
        
        # Best cases
        for i in range(n_examples):
            result = sorted_results[i]
            save_path = os.path.join(viz_dir, f"best_case_{i+1}.png")
            
            initial_state = np.array([result['initial_position'], result['initial_velocity']])
            evaluator.visualize_single(
                initial_state=initial_state,
                predicted_controls=result['controls'],
                optimal_controls=result.get('optimal_controls', None),
                save_path=save_path
            )
            
        # Worst cases
        for i in range(n_examples):
            result = sorted_results[-(i+1)]
            save_path = os.path.join(viz_dir, f"worst_case_{i+1}.png")
            
            initial_state = np.array([result['initial_position'], result['initial_velocity']])
            evaluator.visualize_single(
                initial_state=initial_state,
                predicted_controls=result['controls'],
                optimal_controls=result.get('optimal_controls', None),
                save_path=save_path
            )
            
        logger.info(f"Visualizations saved to: {viz_dir}")
        
    def _save_results(
        self,
        model_name: str,
        system: str,
        results: Dict[str, Any]
    ) -> None:
        """Save evaluation results to disk."""
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{model_name}_{system}_eval_{timestamp}"
        
        # Save full results as pickle
        pickle_path = os.path.join(self.output_dir, f"{base_name}.pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
            
        # Save summary as JSON (excluding large arrays)
        summary = {
            'model_name': results['model_name'],
            'system': results['system'],
            'timestamp': results['timestamp'],
            'num_test_cases': results['num_test_cases'],
            'aggregate_metrics': results['aggregate_metrics']
        }
        
        json_path = os.path.join(self.output_dir, f"{base_name}_summary.json")
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Results saved to: {pickle_path} and {json_path}")
        
    def compare_models(
        self,
        model_names: List[str],
        system: str,
        metric: str = 'final_state_error_mean'
    ) -> Dict[str, Any]:
        """
        Compare multiple models on the same system.
        
        Args:
            model_names: List of model names to compare
            system: System name
            metric: Metric to use for comparison
            
        Returns:
            Comparison results
        """
        comparison = {
            'system': system,
            'metric': metric,
            'models': {}
        }
        
        for model_name in model_names:
            cache_key = f"{model_name}_{system}"
            
            if cache_key in self.results_cache:
                results = self.results_cache[cache_key]
                aggregate = results.get('aggregate_metrics', {})
                
                comparison['models'][model_name] = {
                    'value': aggregate.get(metric, None),
                    'success_rate': aggregate.get('success_rate', 0),
                    'convergence_rate': aggregate.get('convergence_rate', 0),
                    'violation_rate': aggregate.get('violation_rate', 0),
                    'num_test_cases': results.get('num_test_cases', 0)
                }
            else:
                logger.warning(f"No cached results for {model_name} on {system}")
                
        # Rank models
        ranked = sorted(
            comparison['models'].items(),
            key=lambda x: x[1]['value'] if x[1]['value'] is not None else float('inf')
        )
        comparison['ranking'] = [name for name, _ in ranked]
        
        return comparison
        
    def generate_report(
        self,
        model_name: str,
        system: str,
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate a text report for model evaluation.
        
        Args:
            model_name: Model name
            system: System name
            save_path: Optional path to save report
            
        Returns:
            Report text
        """
        cache_key = f"{model_name}_{system}"
        
        if cache_key not in self.results_cache:
            return f"No results found for {model_name} on {system}"
            
        results = self.results_cache[cache_key]
        aggregate = results.get('aggregate_metrics', {})
        
        # Generate report
        report = f"{'='*60}\n"
        report += f"EVALUATION REPORT\n"
        report += f"{'='*60}\n\n"
        
        report += f"Model: {model_name}\n"
        report += f"System: {system}\n"
        report += f"Timestamp: {results.get('timestamp', 'N/A')}\n"
        report += f"Test Cases: {results.get('num_test_cases', 0)}\n\n"
        
        report += f"{'='*60}\n"
        report += f"SUMMARY METRICS\n"
        report += f"{'='*60}\n\n"
        
        # Success metrics
        report += f"Success Rate: {aggregate.get('success_rate', 0):.2%}\n"
        report += f"Convergence Rate: {aggregate.get('convergence_rate', 0):.2%}\n"
        report += f"Violation Rate: {aggregate.get('violation_rate', 0):.2%}\n\n"
        
        # Performance metrics
        report += f"Final State Error:\n"
        report += f"  Mean: {aggregate.get('final_state_error_mean', 0):.4f}\n"
        report += f"  Std:  {aggregate.get('final_state_error_std', 0):.4f}\n"
        report += f"  Min:  {aggregate.get('final_state_error_min', 0):.4f}\n"
        report += f"  Max:  {aggregate.get('final_state_error_max', 0):.4f}\n\n"
        
        report += f"Control Effort:\n"
        report += f"  Mean: {aggregate.get('control_effort_mean', 0):.2f}\n"
        report += f"  Std:  {aggregate.get('control_effort_std', 0):.2f}\n\n"
        
        # Tracking performance (if available)
        if 'position_rmse_mean' in aggregate:
            report += f"Position Tracking RMSE:\n"
            report += f"  Mean: {aggregate.get('position_rmse_mean', 0):.4f}\n"
            report += f"  Std:  {aggregate.get('position_rmse_std', 0):.4f}\n\n"
            
            report += f"Velocity Tracking RMSE:\n"
            report += f"  Mean: {aggregate.get('velocity_rmse_mean', 0):.4f}\n"
            report += f"  Std:  {aggregate.get('velocity_rmse_std', 0):.4f}\n\n"
            
            report += f"Control Tracking RMSE:\n"
            report += f"  Mean: {aggregate.get('control_rmse_mean', 0):.4f}\n"
            report += f"  Std:  {aggregate.get('control_rmse_std', 0):.4f}\n\n"
        
        report += f"{'='*60}\n"
        
        # Save report if requested
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to: {save_path}")
            
        return report