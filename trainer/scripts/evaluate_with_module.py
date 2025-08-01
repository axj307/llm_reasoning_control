#!/usr/bin/env python3
"""
Script to evaluate models using the new evaluation module.
"""

import argparse
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import EvaluationManager
from utils import parse_control_output
from data import create_dataset
from trainer_module_v2 import ControlTrainer
from logger import logger


def evaluate_model(model_path: str, system: str = "double_integrator", 
                  num_test_cases: int = 10, visualize: bool = True):
    """
    Evaluate a trained model using the evaluation module.
    
    Args:
        model_path: Path to the trained model
        system: Control system to evaluate on
        num_test_cases: Number of test cases to evaluate
        visualize: Whether to create visualizations
    """
    logger.info(f"Evaluating model: {model_path}")
    
    # Load model
    trainer = ControlTrainer(model_name=model_path)
    
    # Create test dataset
    test_dataset = create_dataset(num_samples=num_test_cases)
    
    # Initialize evaluation manager
    eval_manager = EvaluationManager()
    
    # Prepare test cases and predictions
    test_cases = []
    predictions = []
    
    for i in range(num_test_cases):
        sample = test_dataset[i]
        
        # Extract test case info
        metadata = sample.get('metadata', {})
        test_case = {
            'initial_state': metadata.get('initial_state', [0.0, 0.0]),
            'optimal_controls': metadata.get('control_sequence', []),
            'metadata': metadata
        }
        test_cases.append(test_case)
        
        # Generate prediction
        output = trainer.generate(sample['prompt'])
        
        # Parse controls
        controls = parse_control_output(output)
        if controls is None:
            logger.warning(f"Failed to parse controls for test case {i}")
            controls = [0.0] * metadata.get('steps', 50)
            
        predictions.append(controls)
        
    # Run evaluation
    model_name = os.path.basename(model_path)
    results = eval_manager.evaluate_model(
        model_name=model_name,
        system=system,
        test_cases=test_cases,
        predictions=predictions,
        visualize=visualize
    )
    
    # Generate and print report
    report = eval_manager.generate_report(model_name, system)
    print(report)
    
    # Save report
    report_path = os.path.join(eval_manager.output_dir, f"{model_name}_report.txt")
    eval_manager.generate_report(model_name, system, save_path=report_path)
    
    return results


def compare_models(model_paths: list, system: str = "double_integrator", 
                  num_test_cases: int = 10):
    """
    Compare multiple models.
    
    Args:
        model_paths: List of model paths to compare
        system: Control system to evaluate on
        num_test_cases: Number of test cases per model
    """
    eval_manager = EvaluationManager()
    
    # Evaluate each model
    model_names = []
    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        model_names.append(model_name)
        
        logger.info(f"Evaluating {model_name}...")
        evaluate_model(model_path, system, num_test_cases, visualize=False)
        
    # Compare models
    comparison = eval_manager.compare_models(model_names, system)
    
    # Print comparison
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(f"System: {comparison['system']}")
    print(f"Metric: {comparison['metric']}")
    print("\nRanking (best to worst):")
    
    for i, model_name in enumerate(comparison['ranking']):
        model_data = comparison['models'][model_name]
        print(f"\n{i+1}. {model_name}")
        print(f"   - {comparison['metric']}: {model_data['value']:.4f}")
        print(f"   - Success rate: {model_data['success_rate']:.2%}")
        print(f"   - Convergence rate: {model_data['convergence_rate']:.2%}")
        

def demo_visualization():
    """Demonstrate the comprehensive visualization capabilities."""
    from control import solve_double_integrator
    
    # Create example data
    x0, v0 = 0.5, -0.3
    dt = 0.1
    steps = 50
    
    # Generate optimal controls
    optimal_controls = solve_double_integrator(x0, v0, dt, steps)
    
    # Add some noise to create "predicted" controls
    np.random.seed(42)
    predicted_controls = optimal_controls + np.random.normal(0, 0.1, size=len(optimal_controls))
    predicted_controls = np.clip(predicted_controls, -3, 3)
    
    # Create evaluator and visualize
    eval_manager = EvaluationManager()
    evaluator = eval_manager.get_evaluator("double_integrator", dt, steps)
    
    # Single trajectory visualization
    evaluator.visualize_single(
        initial_state=np.array([x0, v0]),
        predicted_controls=predicted_controls,
        optimal_controls=optimal_controls,
        save_path="demo_single_trajectory.png"
    )
    
    logger.info("Demo visualization saved to: demo_single_trajectory.png")
    
    # Batch visualization demo
    test_cases = []
    predictions = []
    
    for i in range(5):
        # Random initial conditions
        x0 = np.random.uniform(-0.8, 0.8)
        v0 = np.random.uniform(-0.8, 0.8)
        
        # Solve for optimal
        optimal = solve_double_integrator(x0, v0, dt, steps)
        
        # Add noise
        predicted = optimal + np.random.normal(0, 0.05 * (i + 1), size=len(optimal))
        predicted = np.clip(predicted, -3, 3)
        
        test_cases.append({
            'initial_state': [x0, v0],
            'optimal_controls': optimal
        })
        predictions.append(predicted)
        
    # Run batch evaluation
    results = eval_manager.evaluate_model(
        model_name="demo_model",
        system="double_integrator",
        test_cases=test_cases,
        predictions=predictions,
        dt=dt,
        steps=steps,
        visualize=True
    )
    
    logger.info("Batch visualizations saved to: evaluation_results/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate control models")
    parser.add_argument("--model", type=str, help="Path to model to evaluate")
    parser.add_argument("--compare", nargs="+", help="List of models to compare")
    parser.add_argument("--system", type=str, default="double_integrator",
                       help="Control system (default: double_integrator)")
    parser.add_argument("--num-tests", type=int, default=10,
                       help="Number of test cases (default: 10)")
    parser.add_argument("--demo", action="store_true",
                       help="Run visualization demo")
    parser.add_argument("--no-viz", action="store_true",
                       help="Disable visualizations")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_visualization()
    elif args.compare:
        compare_models(args.compare, args.system, args.num_tests)
    elif args.model:
        evaluate_model(args.model, args.system, args.num_tests, 
                      visualize=not args.no_viz)
    else:
        parser.print_help()