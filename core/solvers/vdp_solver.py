"""
Van der Pol oscillator solver using numerical optimization.
"""

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize


def van_der_pol_dynamics(state, t, control_func, mu=1.0):
    """Van der Pol oscillator dynamics with control input."""
    x, x_dot = state
    u = control_func(t)
    
    # Van der Pol equation: ẍ - μ(1 - x²)ẋ + x = u
    x_ddot = mu * (1 - x**2) * x_dot - x + u
    
    return [x_dot, x_ddot]


def solve_van_der_pol(x0, v0, mu, dt, steps):
    """
    Solve Van der Pol oscillator optimal control using numerical optimization.
    
    Args:
        x0: Initial position
        v0: Initial velocity  
        dt: Time step
        steps: Number of control steps
        mu: Van der Pol parameter (default=1.0)
    
    Returns:
        controls: List of optimal control inputs
        states: List of system states
    """
    total_time = dt * steps
    time_points = np.linspace(0, total_time, steps + 1)
    
    def control_sequence_to_func(u_seq):
        """Convert discrete control sequence to continuous function."""
        def control_func(t):
            if t >= total_time:
                return u_seq[-1]
            idx = int(t / dt)
            if idx >= len(u_seq):
                return u_seq[-1]
            return u_seq[idx]
        return control_func
    
    def simulate_system(u_seq):
        """Simulate the Van der Pol system with given control sequence."""
        control_func = control_sequence_to_func(u_seq)
        
        # Integrate the system
        initial_state = [x0, v0]
        states = odeint(van_der_pol_dynamics, initial_state, time_points, 
                       args=(control_func, mu))
        
        return states
    
    def cost_function(u_seq):
        """Cost function for optimal control (LQR-style)."""
        states = simulate_system(u_seq)
        
        # Quadratic cost on states and controls
        state_cost = np.sum(10.0 * (states[:-1, 0]**2 + states[:-1, 1]**2))
        control_cost = np.sum(0.1 * np.array(u_seq)**2)
        
        # Terminal cost
        final_state = states[-1]
        terminal_cost = 100.0 * (final_state[0]**2 + final_state[1]**2)
        
        return state_cost + control_cost + terminal_cost
    
    # Initial guess for control sequence
    u_initial = np.zeros(steps)
    
    # Control bounds (reasonable for Van der Pol)
    bounds = [(-10.0, 10.0) for _ in range(steps)]
    
    # Optimize control sequence
    result = minimize(cost_function, u_initial, method='L-BFGS-B', bounds=bounds)
    
    if not result.success:
        print(f"Warning: Optimization did not converge. Using fallback control.")
        # Fallback: simple proportional control
        controls = []
        x, x_dot = x0, v0
        for _ in range(steps):
            u = -2.0 * x - 1.0 * x_dot  # Simple PD control
            controls.append(u)
            # Simple Euler integration for fallback
            x_ddot = mu * (1 - x**2) * x_dot - x + u
            x_dot += dt * x_ddot
            x += dt * x_dot
    else:
        controls = result.x.tolist()
    
    # Simulate final trajectory
    states = simulate_system(controls)
    
    return controls, states.tolist()