#!/usr/bin/env python3
"""
Simple script to create benchmark-style visualizations from your evaluation results.
Use this after running your model evaluation.
"""

import numpy as np
import os
from utils_enhanced_v3 import create_benchmark_style_comparison


def visualize_your_results(trajectories, dt=0.1, output_dir="outputs"):
    """
    Create benchmark-style visualization from your evaluation results.
    
    Args:
        trajectories: List of (x0, v0, controls) tuples from your evaluation
        dt: Time step (default 0.1)
        output_dir: Where to save the plots
    """
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the benchmark-style visualization
    print("Creating benchmark-style visualization...")
    save_path = os.path.join(output_dir, "publication_comparison")
    
    create_benchmark_style_comparison(
        test_cases=trajectories,
        dt=dt,
        save_path=save_path,
        show_optimal=True
    )
    
    print(f"✓ Visualization saved to: {save_path}.png and {save_path}.pdf")
    
    # Also create without optimal for comparison
    save_path_model = os.path.join(output_dir, "model_only_comparison")
    create_benchmark_style_comparison(
        test_cases=trajectories,
        dt=dt,
        save_path=save_path_model,
        show_optimal=False
    )
    
    print(f"✓ Model-only version saved to: {save_path_model}.png and {save_path_model}.pdf")


# Example usage with your data structure
if __name__ == "__main__":
    # Example: After you've collected your trajectories
    # Replace this with your actual evaluation results
    
    # Sample trajectories (x0, v0, controls)
    example_trajectories = [
        (0.5, -0.3, np.random.randn(50).clip(-3, 3).tolist()),
        (0.7, 0.2, np.random.randn(50).clip(-3, 3).tolist()),
        (-0.6, -0.4, np.random.randn(50).clip(-3, 3).tolist()),
        (0.3, 0.6, np.random.randn(50).clip(-3, 3).tolist()),
        (-0.4, 0.5, np.random.randn(50).clip(-3, 3).tolist()),
    ]
    
    # Create visualizations
    visualize_your_results(example_trajectories)
    
    print("\nTo use with your actual results:")
    print("1. Collect your trajectories: [(x0, v0, controls), ...]")
    print("2. Call: visualize_your_results(trajectories)")
    print("\nThe function expects controls as a list of floats.")