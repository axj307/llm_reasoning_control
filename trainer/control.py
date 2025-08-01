"""
Control system logic including LQR solver.
"""

import numpy as np
import scipy.linalg as la


def solve_double_integrator(x0, v0, dt, steps):
    """
    Compute LQR optimal control sequence for the double integrator.
    
    Args:
        x0: Initial position
        v0: Initial velocity
        dt: Time step
        steps: Number of control steps
        
    Returns:
        List of control values
    """
    # System dynamics for double integrator in discrete time
    A = np.array([[1, dt], [0, 1]])
    B = np.array([[0.5*dt**2], [dt]])
    
    # Cost matrices
    Q = np.diag([10.0, 10.0])  # State cost
    R = np.array([[0.1]])       # Control cost
    
    # Solve discrete time algebraic Riccati equation
    P = la.solve_discrete_are(A, B, Q, R)
    
    # Compute optimal feedback gain
    K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
    
    # Initial state
    x = np.array([[x0], [v0]])
    
    # Simulate the closed loop system
    controls = []
    
    for i in range(steps):
        # Compute optimal control
        u = -K @ x
        
        # Clamp control within bounds
        u_clamped = max(-3.0, min(3.0, float(u[0])))
        
        # Apply control to system
        x = A @ x + B * u_clamped
        
        # Clamp states if needed
        x[0,0] = max(-1.0, min(1.0, x[0,0]))  # Position
        x[1,0] = max(-1.0, min(1.0, x[1,0]))  # Velocity
        
        controls.append(u_clamped)
    
    return controls


def simulate_trajectory(x0, v0, controls, dt):
    """
    Simulate system trajectory given initial state and control sequence.
    
    Args:
        x0: Initial position
        v0: Initial velocity
        controls: List of control inputs
        dt: Time step
        
    Returns:
        positions: List of positions
        velocities: List of velocities
        times: List of time points
    """
    x, v = x0, v0
    positions = [x]
    velocities = [v]
    times = [0]
    
    for i, u in enumerate(controls):
        v = v + u * dt
        x = x + v * dt
        positions.append(x)
        velocities.append(v)
        times.append((i+1) * dt)
    
    return positions, velocities, times