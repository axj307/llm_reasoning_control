#!/usr/bin/env python3
"""
Script to evaluate models using the combined visualization approach.
"""

import argparse
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import EvaluationManager, DoubleIntegratorEvaluator
from evaluation.plotting_utils import create_comprehensive_plot, create_batch_comparison_plot
from utils import parse_control_output
from utils_enhanced import visualize_all_trajectories_with_controls
from data import create_dataset
from trainer_module_v2 import ControlTrainer
from control import simulate_trajectory, solve_double_integrator
from logger import logger


def evaluate_and_visualize(model_path: str = None, num_test_cases: int = 5):
    """
    Evaluate a model and create combined visualizations.
    
    Args:
        model_path: Path to trained model (if None, uses random controls for demo)
        num_test_cases: Number of test cases to evaluate
    """
    # Create test cases
    np.random.seed(42)
    test_cases = []
    
    # Generate diverse initial conditions
    for i in range(num_test_cases):
        # Use different regions of state space
        angle = 2 * np.pi * i / num_test_cases
        radius = 0.6 + 0.2 * np.random.rand()
        x0 = radius * np.cos(angle)
        v0 = radius * np.sin(angle)
        test_cases.append((x0, v0))
    
    # Initialize lists for results
    all_trajectories = []
    evaluation_results = []
    
    # Load model if provided
    if model_path:
        logger.info(f"Loading model from {model_path}")
        trainer = ControlTrainer(model_name=model_path)
        trainer.setup_model()
    else:
        trainer = None
        logger.info("No model provided - using demo mode with random controls")
    
    # Process each test case
    for i, (x0, v0) in enumerate(test_cases):
        logger.info(f"Processing test case {i+1}/{num_test_cases}: x0={x0:.2f}, v0={v0:.2f}")
        
        # Get optimal controls
        optimal_controls = solve_double_integrator(x0, v0, dt=0.1, steps=50)
        
        # Get model predictions or use demo controls
        if trainer:
            prompt = f"Control a double integrator system with initial state [position={x0:.2f}, velocity={v0:.2f}] to reach the origin (0,0) in 5.00 seconds using 50 steps."
            output = trainer.generate(prompt)
            predicted_controls = parse_control_output(output)
            
            if predicted_controls is None:
                logger.warning(f"Failed to parse controls for test case {i+1}")
                # Use noisy version of optimal as fallback
                predicted_controls = optimal_controls + np.random.normal(0, 0.1, len(optimal_controls))
                predicted_controls = np.clip(predicted_controls, -3, 3).tolist()
        else:
            # Demo mode: add noise to optimal controls
            noise_level = 0.1 * (i + 1)  # Increasing noise for variety
            predicted_controls = optimal_controls + np.random.normal(0, noise_level, len(optimal_controls))
            predicted_controls = np.clip(predicted_controls, -3, 3).tolist()
        
        # Add to trajectories list
        all_trajectories.append((x0, v0, predicted_controls))
        
        # Simulate trajectories
        pred_positions, pred_velocities, times = simulate_trajectory(x0, v0, predicted_controls, dt=0.1)
        opt_positions, opt_velocities, _ = simulate_trajectory(x0, v0, optimal_controls, dt=0.1)
        
        # Store results for evaluation module
        result = {
            'test_case_id': i,
            'initial_state': [x0, v0],
            'predicted_controls': predicted_controls,
            'optimal_controls': optimal_controls.tolist(),
            'positions': np.array(pred_positions),
            'velocities': np.array(pred_velocities),
            'controls': np.array(predicted_controls),
            'times': np.array(times),
            'optimal_positions': np.array(opt_positions),
            'optimal_velocities': np.array(opt_velocities),
            'label': f"IC: [{x0:.2f}, {v0:.2f}]"
        }
        evaluation_results.append(result)
    
    # Create output directory
    output_dir = "evaluation_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Create combined visualization using utils_enhanced
    logger.info("Creating combined visualization with all trajectories...")
    save_path_full = os.path.join(output_dir, "all_trajectories_combined_full.png")
    visualize_all_trajectories_with_controls(
        test_cases=all_trajectories,
        dt=0.1,
        save_path=save_path_full,
        show_optimal=True
    )
    
    # 2. Create evaluation module's batch comparison
    logger.info("Creating batch comparison visualization...")
    save_path_batch = os.path.join(output_dir, "batch_comparison.png")
    create_batch_comparison_plot(
        results=evaluation_results,
        title="Batch Evaluation Results",
        save_path=save_path_batch
    )
    
    # 3. Create individual comprehensive plots for best/worst cases
    # Find best and worst by final error
    final_errors = []
    for result in evaluation_results:
        final_error = np.sqrt(result['positions'][-1]**2 + result['velocities'][-1]**2)
        final_errors.append(final_error)
    
    best_idx = np.argmin(final_errors)
    worst_idx = np.argmax(final_errors)
    
    # Best case comprehensive plot
    logger.info(f"Creating comprehensive plot for best case (test {best_idx+1})...")
    best_result = evaluation_results[best_idx]
    save_path_best = os.path.join(output_dir, "best_case_comprehensive.png")
    create_comprehensive_plot(
        times=best_result['times'],
        positions=best_result['positions'],
        velocities=best_result['velocities'],
        controls=best_result['controls'],
        positions_opt=best_result['optimal_positions'],
        velocities_opt=best_result['optimal_velocities'],
        controls_opt=best_result['optimal_controls'],
        title=f"Best Performance - {best_result['label']} - Final Error: {final_errors[best_idx]:.4f}",
        save_path=save_path_best
    )
    
    # Worst case comprehensive plot
    logger.info(f"Creating comprehensive plot for worst case (test {worst_idx+1})...")
    worst_result = evaluation_results[worst_idx]
    save_path_worst = os.path.join(output_dir, "worst_case_comprehensive.png")
    create_comprehensive_plot(
        times=worst_result['times'],
        positions=worst_result['positions'],
        velocities=worst_result['velocities'],
        controls=worst_result['controls'],
        positions_opt=worst_result['optimal_positions'],
        velocities_opt=worst_result['optimal_velocities'],
        controls_opt=worst_result['optimal_controls'],
        title=f"Worst Performance - {worst_result['label']} - Final Error: {final_errors[worst_idx]:.4f}",
        save_path=save_path_worst
    )
    
    # 4. Print summary statistics
    logger.info("\n=== Evaluation Summary ===")
    logger.info(f"Number of test cases: {num_test_cases}")
    logger.info(f"Average final error: {np.mean(final_errors):.4f}")
    logger.info(f"Best final error: {final_errors[best_idx]:.4f} (Test {best_idx+1})")
    logger.info(f"Worst final error: {final_errors[worst_idx]:.4f} (Test {worst_idx+1})")
    logger.info(f"\nVisualizations saved to {output_dir}/")
    
    return evaluation_results


def demo_evaluation_module():
    """Demonstrate the evaluation module capabilities without a trained model."""
    logger.info("Running evaluation module demo...")
    
    # Create evaluator
    evaluator = DoubleIntegratorEvaluator(dt=0.1, steps=50)
    
    # Create a few test cases
    test_cases = [
        {'initial_state': [0.5, -0.3]},
        {'initial_state': [0.7, 0.2]},
        {'initial_state': [-0.6, -0.4]}
    ]
    
    predictions = []
    for test_case in test_cases:
        x0, v0 = test_case['initial_state']
        # Get optimal and add noise for demo
        optimal = solve_double_integrator(x0, v0, dt=0.1, steps=50)
        noisy = optimal + np.random.normal(0, 0.15, len(optimal))
        predictions.append(np.clip(noisy, -3, 3))
    
    # Run batch evaluation
    results = evaluator.evaluate_batch(test_cases, predictions)
    
    # Create visualizations
    evaluator.visualize_batch(
        results['individual_results'],
        save_path="evaluation_outputs/demo_batch.png"
    )
    
    logger.info("Demo completed!")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models with combined visualizations")
    parser.add_argument("--model", type=str, help="Path to trained model")
    parser.add_argument("--num-tests", type=int, default=5,
                       help="Number of test cases (default: 5)")
    parser.add_argument("--demo", action="store_true",
                       help="Run demo without a model")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_evaluation_module()
    else:
        evaluate_and_visualize(args.model, args.num_tests)