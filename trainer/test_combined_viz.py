#!/usr/bin/env python3
"""
Test script for combined visualization functionality.
"""

import numpy as np
import matplotlib.pyplot as plt
from control import solve_double_integrator
from utils_enhanced import visualize_all_trajectories_with_controls, visualize_all_trajectories

def test_combined_visualization():
    """Test the combined visualization with sample data."""
    
    # Test cases with different initial conditions
    test_cases = [
        (0.5, -0.3),   # Case 1
        (0.7, 0.2),    # Case 2
        (-0.6, -0.4),  # Case 3
        (0.3, 0.6),    # Case 4
        (-0.4, 0.5)    # Case 5
    ]
    
    # Generate controls for each test case
    all_trajectories = []
    dt = 0.1
    steps = 50
    
    print("Generating test trajectories...")
    for i, (x0, v0) in enumerate(test_cases):
        # Get optimal controls
        optimal_controls = solve_double_integrator(x0, v0, dt, steps)
        
        # Add some noise to simulate model predictions
        noise_level = 0.1 * (i + 1) / len(test_cases)  # Increasing noise
        predicted_controls = optimal_controls + np.random.normal(0, noise_level, len(optimal_controls))
        predicted_controls = np.clip(predicted_controls, -3, 3)
        
        all_trajectories.append((x0, v0, predicted_controls.tolist()))
        print(f"  Test case {i+1}: x0={x0:.2f}, v0={v0:.2f}, noise_level={noise_level:.3f}")
    
    # Test 1: Full visualization with controls
    print("\nCreating full combined visualization (with controls)...")
    visualize_all_trajectories_with_controls(
        test_cases=all_trajectories,
        dt=dt,
        save_path="test_combined_full.png",
        show_optimal=True
    )
    print("  Saved to: test_combined_full.png")
    
    # Test 2: Simple visualization without controls
    print("\nCreating simple combined visualization (without controls)...")
    visualize_all_trajectories(
        test_cases=all_trajectories,
        dt=dt,
        save_path="test_combined_simple.png",
        show_optimal=True
    )
    print("  Saved to: test_combined_simple.png")
    
    # Test 3: Create a minimal example with just 2 trajectories
    print("\nCreating minimal example with 2 trajectories...")
    minimal_cases = all_trajectories[:2]
    visualize_all_trajectories(
        test_cases=minimal_cases,
        dt=dt,
        save_path="test_minimal.png",
        show_optimal=True
    )
    print("  Saved to: test_minimal.png")
    
    print("\n✅ All visualization tests completed successfully!")
    print("\nThe visualizations show:")
    print("- Phase space trajectories for all test cases")
    print("- Position over time")
    print("- Velocity over time")
    print("- Control inputs (in full version)")
    print("- Both predicted (solid) and optimal (dashed) trajectories")
    print("- Different initial conditions marked with different colors")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run the test
    test_combined_visualization()