"""Evaluation metrics for control trajectories."""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional


def evaluate_trajectory(trajectory: Dict[str, Any], 
                       target_state: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Evaluate a control trajectory with various metrics.
    
    Args:
        trajectory: Trajectory dictionary from system simulation
        target_state: Target state (default: origin)
        
    Returns:
        Dictionary of evaluation metrics
    """
    if target_state is None:
        target_state = np.zeros(trajectory['initial_state'].shape)
    
    states = trajectory['states']
    controls = trajectory['controls']
    
    # Final state metrics
    final_state = trajectory['final_state']
    final_error = np.linalg.norm(final_state - target_state)
    final_position_error = abs(final_state[0] - target_state[0])
    final_velocity_error = abs(final_state[1] - target_state[1])
    
    # Trajectory quality metrics
    state_errors = [np.linalg.norm(state - target_state) for state in states]
    mean_state_error = np.mean(state_errors)
    max_state_error = np.max(state_errors)
    
    # Control effort metrics
    total_control_effort = sum(abs(u) for u in controls)
    mean_control_effort = np.mean([abs(u) for u in controls])
    max_control_effort = max(abs(u) for u in controls)
    control_variance = np.var(controls)
    
    # Control smoothness (rate of change)
    control_changes = [abs(controls[i] - controls[i-1]) for i in range(1, len(controls))]
    mean_control_change = np.mean(control_changes) if control_changes else 0.0
    max_control_change = max(control_changes) if control_changes else 0.0
    
    # Constraint satisfaction
    constraints_satisfied = trajectory.get('valid_trajectory', False)
    
    # Settling metrics (when does the system get close to target?)
    settling_tolerance = 0.1
    settling_time = None
    for i, error in enumerate(state_errors):
        if error < settling_tolerance:
            settling_time = i * (trajectory['times'][1] - trajectory['times'][0])
            break
    
    # Performance index (quadratic cost)
    Q = np.diag([10.0, 1.0])  # Position more important than velocity
    R = 0.1  # Control cost
    
    lqr_cost = 0.0
    for state, u in zip(states[:-1], controls):  # Exclude final state from control cost
        state_cost = float((state - target_state).T @ Q @ (state - target_state))
        control_cost = R * u**2
        lqr_cost += state_cost + control_cost
    
    # Final state cost (terminal cost)
    final_state_cost = float((final_state - target_state).T @ Q @ (final_state - target_state))
    lqr_cost += final_state_cost
    
    return {
        'final_error': final_error,
        'final_position_error': final_position_error,
        'final_velocity_error': final_velocity_error,
        'mean_state_error': mean_state_error,
        'max_state_error': max_state_error,
        'total_control_effort': total_control_effort,
        'mean_control_effort': mean_control_effort,
        'max_control_effort': max_control_effort,
        'control_variance': control_variance,
        'mean_control_change': mean_control_change,
        'max_control_change': max_control_change,
        'constraints_satisfied': constraints_satisfied,
        'settling_time': settling_time,
        'lqr_cost': lqr_cost,
        'trajectory_length': len(states) - 1,  # Number of steps
    }


def compute_control_metrics(model_trajectory: Dict[str, Any],
                          optimal_trajectory: Dict[str, Any],
                          target_state: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compare model trajectory with optimal trajectory.
    
    Args:
        model_trajectory: Model-generated trajectory
        optimal_trajectory: Optimal trajectory for comparison
        target_state: Target state (default: origin)
        
    Returns:
        Comparison metrics
    """
    model_metrics = evaluate_trajectory(model_trajectory, target_state)
    optimal_metrics = evaluate_trajectory(optimal_trajectory, target_state)
    
    # Compute relative performance
    comparison_metrics = {}
    
    for key in model_metrics:
        if key in optimal_metrics and optimal_metrics[key] != 0:
            if key.endswith('_error') or key in ['lqr_cost', 'total_control_effort']:
                # For error metrics, lower is better
                ratio = model_metrics[key] / optimal_metrics[key]
                comparison_metrics[f'{key}_ratio'] = ratio
                comparison_metrics[f'{key}_improvement'] = 1.0 - ratio
            elif key == 'constraints_satisfied':
                # Boolean metric
                comparison_metrics[f'{key}_match'] = (
                    model_metrics[key] == optimal_metrics[key]
                )
    
    # Overall performance score (weighted combination)
    weights = {
        'final_error': 0.4,
        'lqr_cost': 0.3,
        'total_control_effort': 0.2,
        'constraints_satisfied': 0.1
    }
    
    performance_score = 0.0
    for metric, weight in weights.items():
        if metric == 'constraints_satisfied':
            # Binary score
            score = 1.0 if model_metrics[metric] else 0.0
        else:
            # Normalized score (1.0 = same as optimal, <1.0 = worse, >1.0 = better)
            if optimal_metrics[metric] > 0:
                score = optimal_metrics[metric] / max(model_metrics[metric], 1e-8)
            else:
                score = 1.0
        
        performance_score += weight * score
    
    comparison_metrics['overall_performance_score'] = performance_score
    
    return {
        'model_metrics': model_metrics,
        'optimal_metrics': optimal_metrics,
        'comparison': comparison_metrics
    }


def compute_batch_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute aggregate metrics across a batch of results.
    
    Args:
        results: List of inference results from run_batch_inference
        
    Returns:
        Aggregate statistics
    """
    # Filter valid results
    valid_results = [r for r in results if r.get('valid_format', False) and 
                    r.get('model_trajectory') is not None]
    
    if not valid_results:
        return {'error': 'No valid results to analyze'}
    
    # Compute metrics for each result
    all_metrics = []
    comparison_scores = []
    
    for result in valid_results:
        if result['optimal_trajectory'] is not None:
            metrics = compute_control_metrics(
                result['model_trajectory'],
                result['optimal_trajectory']
            )
            all_metrics.append(metrics)
            comparison_scores.append(metrics['comparison']['overall_performance_score'])
    
    # Aggregate statistics
    success_rate = len(valid_results) / len(results)
    
    if comparison_scores:
        mean_performance = np.mean(comparison_scores)
        std_performance = np.std(comparison_scores)
        min_performance = np.min(comparison_scores)
        max_performance = np.max(comparison_scores)
    else:
        mean_performance = std_performance = min_performance = max_performance = 0.0
    
    # Aggregate specific metrics
    final_errors = []
    lqr_costs = []
    constraint_violations = 0
    
    for result in valid_results:
        if result['model_trajectory']:
            metrics = evaluate_trajectory(result['model_trajectory'])
            final_errors.append(metrics['final_error'])
            lqr_costs.append(metrics['lqr_cost'])
            if not metrics['constraints_satisfied']:
                constraint_violations += 1
    
    aggregate_stats = {
        'total_tests': len(results),
        'successful_extractions': len(valid_results),
        'success_rate': success_rate,
        'mean_performance_score': mean_performance,
        'std_performance_score': std_performance,
        'min_performance_score': min_performance,
        'max_performance_score': max_performance,
        'constraint_violation_rate': constraint_violations / len(valid_results) if valid_results else 0,
    }
    
    if final_errors:
        aggregate_stats.update({
            'mean_final_error': np.mean(final_errors),
            'std_final_error': np.std(final_errors),
            'median_final_error': np.median(final_errors),
        })
    
    if lqr_costs:
        aggregate_stats.update({
            'mean_lqr_cost': np.mean(lqr_costs),
            'std_lqr_cost': np.std(lqr_costs),
        })
    
    return aggregate_stats


def rank_performance(results: List[Dict[str, Any]], 
                    metric: str = 'overall_performance_score') -> List[Tuple[int, float]]:
    """
    Rank results by a specific performance metric.
    
    Args:
        results: List of results with computed metrics
        metric: Metric to rank by
        
    Returns:
        List of (index, score) tuples, sorted by performance
    """
    rankings = []
    
    for i, result in enumerate(results):
        if (result.get('valid_format', False) and 
            result.get('model_trajectory') is not None and
            result.get('optimal_trajectory') is not None):
            
            metrics = compute_control_metrics(
                result['model_trajectory'],
                result['optimal_trajectory']
            )
            
            if metric in metrics['comparison']:
                score = metrics['comparison'][metric]
                rankings.append((i, score))
    
    # Sort by score (higher is better for most metrics)
    rankings.sort(key=lambda x: x[1], reverse=True)
    
    return rankings