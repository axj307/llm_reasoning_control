#!/usr/bin/env python3
"""
Demo script showing the new evaluation pipeline capabilities.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer_module_v3 import ControlTrainerV3
from data import create_dataset
from logger import logger


def main():
    """Run evaluation demo."""
    logger.info("Starting evaluation pipeline demo...")
    
    # 1. Create trainer (using base model for demo)
    trainer = ControlTrainerV3(model_name="unsloth/Qwen2.5-3B-Instruct-bnb-4bit")
    
    # 2. Create a small training dataset
    logger.info("Creating training dataset...")
    train_dataset = create_dataset(num_samples=50)
    
    # 3. Quick SFT training (optional - skip if you have a trained model)
    logger.info("Running quick SFT training for demo...")
    trainer.train_sft(train_dataset, output_dir="demo_sft_model")
    
    # 4. Evaluate on standard benchmark
    logger.info("\nEvaluating on standard benchmark...")
    results = trainer.evaluate_on_benchmark(
        benchmark_name="easy",
        visualize=True
    )
    
    # 5. Show aggregate metrics
    logger.info("\nEvaluation Results Summary:")
    aggregate = results['aggregate_metrics']
    logger.info(f"Success Rate: {aggregate['success_rate']:.2%}")
    logger.info(f"Average Final Error: {aggregate['final_state_error_mean']:.4f}")
    logger.info(f"Control Effort: {aggregate['control_effort_mean']:.2f}")
    
    # 6. Custom evaluation with specific test cases
    logger.info("\nRunning custom evaluation...")
    
    # Create specific test cases
    test_cases = [
        {"initial_state": [0.5, -0.3], "label": "Easy case"},
        {"initial_state": [0.8, 0.8], "label": "Corner case"},
        {"initial_state": [-0.7, 0.5], "label": "Mixed case"},
    ]
    
    # Generate optimal controls for comparison
    from control import solve_double_integrator
    for test_case in test_cases:
        x0, v0 = test_case['initial_state']
        optimal_controls = solve_double_integrator(x0, v0, dt=0.1, steps=50)
        test_case['optimal_controls'] = optimal_controls
    
    # Evaluate
    predictions = []
    for test_case in test_cases:
        x0, v0 = test_case['initial_state']
        prompt = f"Control a double integrator system with initial state [position={x0:.2f}, velocity={v0:.2f}] to reach the origin (0,0) in 5.0 seconds using 50 steps."
        
        output = trainer.generate(prompt)
        from utils import parse_control_output
        controls = parse_control_output(output)
        
        if controls is None:
            controls = [0.0] * 50
            
        predictions.append(controls)
    
    # Run evaluation
    custom_results = trainer.eval_manager.evaluate_model(
        model_name="demo_model",
        system="double_integrator",
        test_cases=test_cases,
        predictions=predictions,
        visualize=True
    )
    
    logger.info("\nDemo completed! Check 'evaluation_results/' directory for visualizations.")
    
    # Clean up
    trainer.cleanup()


if __name__ == "__main__":
    main()