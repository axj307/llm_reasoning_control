"""
Base evaluator class for control system evaluation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
import numpy as np


class BaseEvaluator(ABC):
    """Abstract base class for evaluators."""
    
    def __init__(self, system_name: str, dt: float, steps: int):
        """
        Initialize evaluator.
        
        Args:
            system_name: Name of the control system
            dt: Time step
            steps: Number of control steps
        """
        self.system_name = system_name
        self.dt = dt
        self.steps = steps
        self.results = []
        
    @abstractmethod
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
            initial_state: Initial state of the system
            predicted_controls: Predicted control sequence
            optimal_controls: Optional optimal control sequence for comparison
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary containing evaluation metrics
        """
        pass
        
    @abstractmethod
    def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]],
        predictions: List[np.ndarray],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of test cases.
        
        Args:
            test_cases: List of test case dictionaries
            predictions: List of predicted control sequences
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary containing aggregate metrics
        """
        pass
        
    @abstractmethod
    def visualize_single(
        self,
        initial_state: np.ndarray,
        predicted_controls: np.ndarray,
        optimal_controls: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Visualize a single trajectory with comprehensive plots.
        
        Args:
            initial_state: Initial state
            predicted_controls: Predicted control sequence
            optimal_controls: Optional optimal control sequence
            save_path: Path to save visualization
            **kwargs: Additional visualization parameters
        """
        pass
        
    @abstractmethod
    def visualize_batch(
        self,
        results: List[Dict[str, Any]],
        save_path: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Visualize batch evaluation results.
        
        Args:
            results: List of evaluation results
            save_path: Path to save visualization
            **kwargs: Additional visualization parameters
        """
        pass
        
    def add_result(self, result: Dict[str, Any]) -> None:
        """Add a result to the evaluator's result list."""
        self.results.append(result)
        
    def clear_results(self) -> None:
        """Clear all stored results."""
        self.results = []
        
    def get_summary_metrics(self) -> Dict[str, Any]:
        """
        Get summary metrics from all stored results.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.results:
            return {}
            
        # Extract common metrics
        metrics = {}
        metric_keys = self.results[0].keys()
        
        for key in metric_keys:
            if isinstance(self.results[0][key], (int, float)):
                values = [r[key] for r in self.results if key in r]
                if values:
                    metrics[f"{key}_mean"] = np.mean(values)
                    metrics[f"{key}_std"] = np.std(values)
                    metrics[f"{key}_min"] = np.min(values)
                    metrics[f"{key}_max"] = np.max(values)
                    
        metrics['num_evaluations'] = len(self.results)
        return metrics