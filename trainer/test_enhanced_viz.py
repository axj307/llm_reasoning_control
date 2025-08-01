#!/usr/bin/env python3
"""
Test script for enhanced visualization functionality with publication-quality styling.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control import solve_double_integrator
from utils_enhanced_v2 import (
    create_publication_comparison_plot,
    create_model_trajectories_plot,
    create_control_dashboard
)


def test_enhanced_visualizations():
    """Test all enhanced visualization functions."""
    
    # Define test cases with diverse initial conditions
    test_cases_data = [
        (0.5, -0.3),   # Case 1: Positive position, negative velocity
        (0.7, 0.2),    # Case 2: Both positive
        (-0.6, -0.4),  # Case 3: Both negative
        (0.3, 0.6),    # Case 4: Small position, large velocity
        (-0.4, 0.5),   # Case 5: Negative position, positive velocity
        (0.8, -0.2),   # Case 6: Large position, small velocity
        (-0.5, -0.5),  # Case 7: Symmetric negative
    ]
    
    # Generate trajectories with varying quality
    all_trajectories = []
    dt = 0.1
    steps = 50
    
    print("Generating test trajectories with varying noise levels...")
    for i, (x0, v0) in enumerate(test_cases_data):
        # Get optimal controls
        optimal_controls = solve_double_integrator(x0, v0, dt, steps)
        
        # Add noise to simulate different model performance levels
        if i < 2:
            # Good performance - low noise
            noise_level = 0.05
            print(f"  Test {i+1}: Good performance (low noise)")
        elif i < 5:
            # Medium performance
            noise_level = 0.15
            print(f"  Test {i+1}: Medium performance (moderate noise)")
        else:
            # Poor performance - high noise
            noise_level = 0.25
            print(f"  Test {i+1}: Challenging case (higher noise)")
        
        predicted_controls = optimal_controls + np.random.normal(0, noise_level, len(optimal_controls))
        predicted_controls = np.clip(predicted_controls, -3, 3).tolist()
        
        all_trajectories.append((x0, v0, predicted_controls))
    
    # Create output directory
    output_dir = "enhanced_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test 1: Publication-quality comparison plot (first 5 cases)
    print("\n1. Creating publication-quality comparison plot...")
    save_path = os.path.join(output_dir, "test_publication_comparison")
    fig1 = create_publication_comparison_plot(
        test_cases=all_trajectories[:5],
        dt=dt,
        save_path=save_path,
        show_optimal=True,
        title="Enhanced Control System Comparison\nModel Predictions vs Optimal Solutions"
    )
    print(f"   ✓ Saved to: {save_path}.png and {save_path}.pdf")
    
    # Test 2: Model-only trajectories plot
    print("\n2. Creating model-only trajectories plot...")
    save_path = os.path.join(output_dir, "test_model_trajectories")
    fig2 = create_model_trajectories_plot(
        test_cases=all_trajectories[:4],
        dt=dt,
        save_path=save_path,
        title="Model Performance Across Different Initial Conditions",
        add_metrics=True
    )
    print(f"   ✓ Saved to: {save_path}.png and {save_path}.pdf")
    
    # Test 3: Comprehensive control dashboard (all cases)
    print("\n3. Creating comprehensive control dashboard...")
    save_path = os.path.join(output_dir, "test_control_dashboard")
    fig3 = create_control_dashboard(
        test_cases=all_trajectories,
        dt=dt,
        save_path=save_path,
        title="Control System Performance Analysis Dashboard"
    )
    print(f"   ✓ Saved to: {save_path}.png and {save_path}.pdf")
    
    # Test 4: Create a minimal example with just 2 trajectories
    print("\n4. Creating minimal example with 2 trajectories...")
    minimal_cases = all_trajectories[:2]
    save_path = os.path.join(output_dir, "test_minimal_comparison")
    fig4 = create_publication_comparison_plot(
        test_cases=minimal_cases,
        dt=dt,
        save_path=save_path,
        show_optimal=True,
        title="Minimal Example: Two Test Cases"
    )
    print(f"   ✓ Saved to: {save_path}.png and {save_path}.pdf")
    
    # Test 5: Model-only without metrics
    print("\n5. Creating clean model-only plot without metrics...")
    save_path = os.path.join(output_dir, "test_clean_model")
    fig5 = create_model_trajectories_plot(
        test_cases=all_trajectories[:3],
        dt=dt,
        save_path=save_path,
        title="Clean Model Trajectories",
        add_metrics=False
    )
    print(f"   ✓ Saved to: {save_path}.png and {save_path}.pdf")
    
    print("\n✅ All visualization tests completed successfully!")
    print(f"\nVisualizations saved to: {output_dir}/")
    print("\nKey features demonstrated:")
    print("- Publication-quality styling with professional colors")
    print("- Clear subplot labels (a), (b), (c), (d)")
    print("- Enhanced grid and box styling")
    print("- Directional arrows in phase space")
    print("- Performance metrics annotations")
    print("- Multiple export formats (PNG and PDF)")
    print("- Comparison with optimal solutions")
    print("- Comprehensive dashboard with error analysis")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run the test
    test_enhanced_visualizations()