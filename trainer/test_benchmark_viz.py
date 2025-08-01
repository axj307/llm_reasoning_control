#!/usr/bin/env python3
"""
Test benchmark-style visualization without requiring model loading.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control import solve_double_integrator
from utils_enhanced_v3 import create_benchmark_style_comparison, create_clean_comparison_plot


def test_benchmark_visualization():
    """Test benchmark-style visualization with sample data."""
    
    # Test cases
    test_cases_data = [
        (0.5, -0.3),
        (0.7, 0.2),
        (-0.6, -0.4),
        (0.3, 0.6),
        (-0.4, 0.5),
    ]
    
    # Generate trajectories with varying quality
    all_trajectories = []
    dt = 0.1
    steps = 50
    
    print("Generating test trajectories...")
    for i, (x0, v0) in enumerate(test_cases_data):
        # Get optimal controls
        optimal_controls = solve_double_integrator(x0, v0, dt, steps)
        
        # Add noise to simulate model performance
        if i < 2:
            noise_level = 0.1  # Good performance
        else:
            noise_level = 0.2  # Moderate performance
        
        predicted_controls = optimal_controls + np.random.normal(0, noise_level, len(optimal_controls))
        predicted_controls = np.clip(predicted_controls, -3, 3).tolist()
        
        all_trajectories.append((x0, v0, predicted_controls))
        print(f"  Test case {i+1}: x0={x0:.2f}, v0={v0:.2f}")
    
    # Create output directory
    output_dir = "benchmark_style_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test 1: Benchmark style (matching your example exactly)
    print("\n1. Creating benchmark-style comparison plot...")
    save_path = os.path.join(output_dir, "benchmark_comparison")
    fig1 = create_benchmark_style_comparison(
        test_cases=all_trajectories,
        dt=dt,
        save_path=save_path,
        show_optimal=True
    )
    print(f"   ✓ Saved to: {save_path}.png and {save_path}.pdf")
    
    # Test 2: Clean style (no IC labels)
    print("\n2. Creating clean comparison plot...")
    save_path = os.path.join(output_dir, "clean_comparison")
    fig2 = create_clean_comparison_plot(
        test_cases=all_trajectories,
        dt=dt,
        save_path=save_path,
        show_optimal=True
    )
    print(f"   ✓ Saved to: {save_path}.png and {save_path}.pdf")
    
    # Test 3: Model only (no optimal)
    print("\n3. Creating model-only plot...")
    save_path = os.path.join(output_dir, "model_only")
    fig3 = create_clean_comparison_plot(
        test_cases=all_trajectories,
        dt=dt,
        save_path=save_path,
        show_optimal=False
    )
    print(f"   ✓ Saved to: {save_path}.png and {save_path}.pdf")
    
    print("\n✅ All visualizations created successfully!")
    print(f"\nOutputs saved to: {output_dir}/")
    print("\nFeatures demonstrated:")
    print("- Benchmark-style with captions below subplots")
    print("- Clean legends (no individual IC labels)")
    print("- Professional color scheme (blue=optimal, red=model)")
    print("- Transparent legend boxes")
    print("- Early trajectory arrows")
    print("- Target markers and bounds")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run the test
    test_benchmark_visualization()