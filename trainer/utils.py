"""
Utility functions for visualization and helpers.
"""

import matplotlib.pyplot as plt
from control import simulate_trajectory


def visualize_solution(x0, v0, controls, dt, save_path="control_solution.png"):
    """Visualize control solution trajectory."""
    # Simulate trajectory
    positions, velocities, times = simulate_trajectory(x0, v0, controls, dt)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Position plot
    plt.subplot(3, 1, 1)
    plt.plot(times, positions, 'b-o')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.grid(True)
    plt.ylabel('Position')
    plt.title('Control Solution Trajectory')
    
    # Velocity plot
    plt.subplot(3, 1, 2)
    plt.plot(times, velocities, 'g-o')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.grid(True)
    plt.ylabel('Velocity')
    
    # Control plot
    plt.subplot(3, 1, 3)
    plt.step(times[:-1], controls, 'r-o', where='post')
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Control Input')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return positions[-1], velocities[-1]


def plot_multiple_trajectories(initial_states, control_sequences, dt, save_path="multiple_trajectories.png"):
    """Plot multiple trajectories on the same figure."""
    plt.figure(figsize=(12, 12))
    
    for idx, (initial_state, controls) in enumerate(zip(initial_states, control_sequences)):
        x0, v0 = initial_state
        positions, velocities, times = simulate_trajectory(x0, v0, controls, dt)
        
        plt.subplot(3, 1, 1)
        plt.plot(times, positions, label=f'Traj {idx+1}')
        
        plt.subplot(3, 1, 2)
        plt.plot(times, velocities, label=f'Traj {idx+1}')
        
        plt.subplot(3, 1, 3)
        plt.step(times[:-1], controls, where='post', label=f'Traj {idx+1}')
    
    # Format position plot
    plt.subplot(3, 1, 1)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.grid(True)
    plt.ylabel('Position')
    plt.title('Multiple Control Trajectories')
    plt.legend()
    
    # Format velocity plot
    plt.subplot(3, 1, 2)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.grid(True)
    plt.ylabel('Velocity')
    plt.legend()
    
    # Format control plot
    plt.subplot(3, 1, 3)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.grid(True)
    plt.ylabel('Control Input')
    plt.xlabel('Time (s)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def parse_control_output(output):
    """Parse control values from model output."""
    import re
    from config import SOLUTION_START, SOLUTION_END
    
    # Extract control sequence
    control_match = re.search(rf"{SOLUTION_START}(.*?){SOLUTION_END}", output, re.DOTALL)
    if control_match:
        control_text = control_match.group(1).strip()
        try:
            return [float(x.strip()) for x in control_text.split(',')]
        except:
            return None
    return None