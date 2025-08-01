#!/usr/bin/env python3
"""
Simple script to create clean publication-ready plots from your evaluation results.
"""

import argparse
import numpy as np
import os
from utils_enhanced_v2 import (
    create_publication_comparison_plot,
    create_model_trajectories_plot,
    create_control_dashboard
)
from control import solve_double_integrator


def create_clean_visualizations(trajectories, dt=0.1, output_dir="outputs"):
    """
    Create all three types of clean visualizations.
    
    Args:
        trajectories: List of (x0, v0, controls) tuples
        dt: Time step
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Publication comparison (with optimal)
    print("Creating publication comparison plot...")
    save_path = os.path.join(output_dir, "publication_comparison_clean")
    create_publication_comparison_plot(
        test_cases=trajectories,
        dt=dt,
        save_path=save_path,
        show_optimal=True
    )
    
    # 2. Model trajectories only
    print("Creating model trajectories plot...")
    save_path = os.path.join(output_dir, "model_trajectories_clean")
    create_model_trajectories_plot(
        test_cases=trajectories,
        dt=dt,
        save_path=save_path,
        add_metrics=False  # Clean version without metrics box
    )
    
    # 3. Control dashboard
    print("Creating control dashboard...")
    save_path = os.path.join(output_dir, "control_dashboard_clean")
    create_control_dashboard(
        test_cases=trajectories,
        dt=dt,
        save_path=save_path
    )
    
    print(f"\n✅ All clean visualizations saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create clean publication plots")
    parser.add_argument("--demo", action="store_true", 
                       help="Run with demo data")
    parser.add_argument("--output-dir", default="outputs", 
                       help="Output directory")
    args = parser.parse_args()
    
    if args.demo:
        # Demo with sample data
        print("Running with demo data...")
        np.random.seed(42)
        
        test_cases = [
            (0.5, -0.3),
            (0.7, 0.2),
            (-0.6, -0.4),
            (0.3, 0.6),
            (-0.4, 0.5),
        ]
        
        trajectories = []
        for x0, v0 in test_cases:
            optimal = solve_double_integrator(x0, v0, 0.1, 50)
            noisy = optimal + np.random.normal(0, 0.15, len(optimal))
            controls = np.clip(noisy, -3, 3).tolist()
            trajectories.append((x0, v0, controls))
        
        create_clean_visualizations(trajectories, output_dir=args.output_dir)
    else:
        print("To use with your data:")
        print("1. Import this function: from create_clean_plots import create_clean_visualizations")
        print("2. Call: create_clean_visualizations(your_trajectories)")
        print("\nFor demo: python create_clean_plots.py --demo")