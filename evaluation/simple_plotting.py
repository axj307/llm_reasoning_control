"""
Simple plotting utilities for control system evaluation and testing.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Optional, Tuple, Dict, Any

def visualize_lqr_solution(x0: float, v0: float, dt: float, steps: int, save_path: Optional[str] = None):
    """
    Create a simple LQR solution visualization.
    
    Args:
        x0: Initial position
        v0: Initial velocity  
        dt: Time step
        steps: Number of control steps
        save_path: Optional path to save the figure
    """
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    
    from core.solvers.lqr_solver import solve_double_integrator_lqr
    from environments.double_integrator import DoubleIntegrator
    
    # Solve optimal control
    initial_state = [x0, v0]
    controls = solve_double_integrator_lqr(initial_state, dt, steps)
    
    # Simulate trajectory
    env = DoubleIntegrator()
    states = [initial_state]
    current_state = np.array(initial_state)
    
    for control in controls:
        # Double integrator dynamics
        next_pos = current_state[0] + current_state[1] * dt + 0.5 * control * dt**2
        next_vel = current_state[1] + control * dt
        current_state = np.array([next_pos, next_vel])
        states.append(current_state.copy())
    
    states = np.array(states)
    times = np.arange(len(states)) * dt
    control_times = times[:-1]
    
    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'LQR Solution: x0={x0:.2f}, v0={v0:.2f}', fontsize=14, fontweight='bold')
    
    # Phase space plot
    ax1.plot(states[:, 0], states[:, 1], 'b-', linewidth=2, marker='o', markersize=3, label='Trajectory')
    ax1.scatter(states[0, 0], states[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
    ax1.scatter(states[-1, 0], states[-1, 1], c='red', s=100, marker='X', label='End', zorder=5)
    ax1.scatter(0, 0, c='orange', s=100, marker='*', label='Target', zorder=5)
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Velocity')
    ax1.set_title('Phase Space')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Position over time
    ax2.plot(times, states[:, 0], 'b-', linewidth=2, marker='o', markersize=3, label='Position')
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Target')
    ax2.fill_between(times, -1, 1, alpha=0.1, color='gray', label='Bounds')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position')
    ax2.set_title('Position vs Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Velocity over time
    ax3.plot(times, states[:, 1], 'b-', linewidth=2, marker='o', markersize=3, label='Velocity')
    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Target')
    ax3.fill_between(times, -1, 1, alpha=0.1, color='gray', label='Bounds')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity')
    ax3.set_title('Velocity vs Time')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Control inputs
    ax4.step(control_times, controls, 'b-', linewidth=2, where='post', marker='o', markersize=3, label='Control')
    ax4.fill_between(control_times, -3, 3, alpha=0.1, color='gray', label='Control Bounds')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Control Input')
    ax4.set_title('Control vs Time')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ LQR solution plot saved to: {save_path}")
    else:
        # Auto-save to figures directory
        os.makedirs("figures", exist_ok=True)
        auto_path = f"figures/lqr_solution_x{x0:.1f}_v{v0:.1f}.png"
        plt.savefig(auto_path, dpi=300, bbox_inches='tight')
        print(f"✅ LQR solution plot saved to: {auto_path}")
    
    plt.close()
    
    return states, controls

def create_evaluation_plots(model_results: Dict[str, Any], save_dir: str = "figures"):
    """
    Create evaluation plots for model performance.
    
    Args:
        model_results: Dictionary containing model evaluation results
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Example evaluation plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    if 'trajectories' in model_results:
        trajectories = model_results['trajectories']
        for i, traj in enumerate(trajectories[:3]):  # Plot first 3 trajectories
            states = np.array(traj.get('states', []))
            if len(states) > 0:
                ax.plot(states[:, 0], states[:, 1], label=f'Trajectory {i+1}', linewidth=2)
    
    ax.scatter(0, 0, c='red', s=100, marker='*', label='Target', zorder=5)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_title('Model Evaluation Results')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    save_path = os.path.join(save_dir, "model_evaluation.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Evaluation plot saved to: {save_path}")
    return save_path

def test_plotting_functions():
    """Test all plotting functions."""
    print("🎨 Testing plotting functions...")
    
    # Test LQR visualization
    visualize_lqr_solution(0.5, -0.3, 0.1, 50)
    visualize_lqr_solution(0.7, 0.2, 0.1, 50)
    
    # Test evaluation plots
    dummy_results = {
        'trajectories': [
            {'states': [[0.5, -0.3], [0.3, -0.1], [0.1, 0.1], [0.0, 0.0]]},
            {'states': [[0.7, 0.2], [0.5, 0.1], [0.2, 0.0], [0.0, 0.0]]},
        ]
    }
    create_evaluation_plots(dummy_results)
    
    print("✅ All plotting functions tested successfully!")

if __name__ == "__main__":
    test_plotting_functions()