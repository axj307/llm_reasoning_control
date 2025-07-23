"""LQR solver for double integrator system."""

import numpy as np
import scipy.linalg as la
from typing import List, Tuple, Optional


def solve_double_integrator_lqr(x0: float, v0: float, dt: float, steps: int,
                                Q: Optional[np.ndarray] = None, 
                                R: Optional[float] = None) -> List[float]:
    """
    Compute LQR optimal control sequence for the double integrator.
    
    Args:
        x0: Initial position
        v0: Initial velocity
        dt: Time step
        steps: Number of control steps
        Q: State cost matrix (2x2), default: diag([10.0, 10.0])
        R: Control cost scalar, default: 0.1
        
    Returns:
        List of control inputs
    """
    # Default LQR weights if not provided
    if Q is None:
        Q = np.diag([10.0, 10.0])  # Equal weight on position and velocity
    if R is None:
        R = np.array([[0.1]])  # Small control cost for aggressive control
    
    # System dynamics for double integrator in discrete time
    # x_{k+1} = A * x_k + B * u_k
    A = np.array([[1, dt], 
                  [0, 1]])
    
    B = np.array([[0.5 * dt**2], 
                  [dt]])
    
    # Solve discrete time algebraic Riccati equation
    P = la.solve_discrete_are(A, B, Q, R)
    
    # Compute optimal feedback gain
    # K = (R + B^T P B)^{-1} B^T P A
    K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
    
    # Initial state
    x = np.array([[x0], [v0]])
    
    # Simulate the closed loop system and get control sequence
    controls = []
    states = [x.copy()]
    
    # Control and state bounds
    u_min, u_max = -3.0, 3.0
    pos_min, pos_max = -1.0, 1.0
    vel_min, vel_max = -1.0, 1.0
    
    for i in range(steps):
        # Compute optimal control: u = -K*x
        u = -K @ x
        
        # Clamp control within bounds
        u_clamped = float(max(u_min, min(u_max, float(u[0]))))
        
        # Apply control to system
        x = A @ x + B * u_clamped
        
        # Clamp states if needed (though LQR should respect constraints if tuned well)
        x[0, 0] = max(pos_min, min(pos_max, x[0, 0]))  # Position
        x[1, 0] = max(vel_min, min(vel_max, x[1, 0]))  # Velocity
        
        # Save control
        controls.append(u_clamped)
        states.append(x.copy())
    
    return controls


def compute_lqr_cost(states: List[np.ndarray], controls: List[float],
                     Q: np.ndarray, R: float) -> float:
    """
    Compute the total LQR cost for a trajectory.
    
    Args:
        states: List of state vectors
        controls: List of control inputs
        Q: State cost matrix
        R: Control cost scalar
        
    Returns:
        Total cost
    """
    total_cost = 0.0
    
    # State costs
    for state in states[:-1]:  # Exclude final state (handled separately)
        total_cost += float(state.T @ Q @ state)
    
    # Final state cost (could use different weight)
    final_state = states[-1]
    total_cost += float(final_state.T @ Q @ final_state)
    
    # Control costs
    for u in controls:
        total_cost += R * u**2
    
    return total_cost


def solve_lqr_with_different_weights(x0: float, v0: float, dt: float, steps: int,
                                   weight_configs: List[Tuple[np.ndarray, float]]) -> List[List[float]]:
    """
    Solve LQR with multiple weight configurations.
    
    Args:
        x0, v0: Initial state
        dt, steps: Time parameters
        weight_configs: List of (Q, R) tuples
        
    Returns:
        List of control sequences for each weight configuration
    """
    solutions = []
    for Q, R in weight_configs:
        controls = solve_double_integrator_lqr(x0, v0, dt, steps, Q, R)
        solutions.append(controls)
    return solutions