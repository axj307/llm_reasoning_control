"""Optimal solver for Van der Pol oscillator using numerical optimization."""

import numpy as np
from scipy.integrate import solve_ivp
import scipy.optimize as opt
from typing import List, Optional, Callable


def solve_van_der_pol_optimal_core(x0: float, v0: float, mu: float = 1.0, 
                                  dt: float = 0.1, steps: int = 50,
                                  Q: Optional[np.ndarray] = None,
                                  R: Optional[float] = None) -> List[float]:
    """
    Compute control sequence for the Van der Pol oscillator using numerical optimization.
    
    Args:
        x0: Initial position
        v0: Initial velocity
        mu: Van der Pol damping parameter
        dt: Time step
        steps: Number of control steps
        Q: State cost matrix (2x2), default: diag([10.0, 5.0])
        R: Control cost scalar, default: 0.1
        
    Returns:
        List of control inputs
    """
    # Default cost weights if not provided
    if Q is None:
        Q = np.diag([10.0, 5.0])  # Higher weight on position than velocity
    if R is None:
        R = 0.1
    
    # Define Van der Pol dynamics with control
    def van_der_pol_dynamics(t, state, u):
        """Van der Pol dynamics: ẍ - μ(1-x²)ẋ + x = u"""
        x, v = state
        dxdt = v
        dvdt = mu * (1 - x**2) * v - x + u
        return [dxdt, dvdt]
    
    # Define cost function for optimization
    def cost_function(u_sequence):
        """Compute total cost for a control sequence."""
        # Initialize state
        state = np.array([x0, v0])
        states = [state.copy()]
        total_cost = 0
        
        # Control bounds
        u_min, u_max = -5.0, 5.0
        
        # Simulate forward with these controls
        for i, u in enumerate(u_sequence):
            # Bound control
            u = max(u_min, min(u_max, u))
            
            # Simulate one step using RK45
            sol = solve_ivp(
                lambda t, y: van_der_pol_dynamics(t, y, u), 
                [0, dt], 
                state, 
                method='RK45', 
                t_eval=[dt]
            )
            state = sol.y[:, -1].tolist()
            states.append(state.copy())
            
            # Compute step cost (quadratic state and control costs)
            state_array = np.array([[state[0]], [state[1]]])
            state_cost = float(state_array.T @ Q @ state_array)
            control_cost = R * u**2
            
            # Higher weight for final states (last 10% of trajectory)
            time_weight = 1.0 if i < 0.9 * steps else 5.0
            step_cost = time_weight * (state_cost + control_cost)
            total_cost += step_cost
        
        # Extra penalty for final state not being at origin
        final_x, final_v = states[-1]
        final_cost = 50.0 * (final_x**2 + final_v**2)
        
        return total_cost + final_cost
    
    # Initial guess - start with damped controls
    initial_guess = np.zeros(steps)
    
    # Or use a simple feedback-like initial guess
    # initial_guess = []
    # x, v = x0, v0
    # for _ in range(steps):
    #     u_guess = -2.0 * x - 1.0 * v  # Simple PD-like control
    #     initial_guess.append(np.clip(u_guess, -5.0, 5.0))
    #     # Rough state prediction
    #     v = v + (mu * (1 - x**2) * v - x + u_guess) * dt
    #     x = x + v * dt
    # initial_guess = np.array(initial_guess)
    
    # Set up optimization bounds
    bounds = [(-5, 5)] * steps
    
    # Optimize control sequence
    result = opt.minimize(
        cost_function, 
        initial_guess, 
        method='SLSQP',  # Sequential Least Squares Programming
        bounds=bounds,
        options={
            'maxiter': 1000,
            'ftol': 1e-6,
            'disp': False
        }
    )
    
    # Return optimal control sequence
    controls = np.clip(result.x, -5.0, 5.0).tolist()
    return controls


def solve_van_der_pol_mpc(x0: float, v0: float, mu: float = 1.0,
                         dt: float = 0.1, steps: int = 50,
                         horizon: int = 10) -> List[float]:
    """
    Solve Van der Pol control using Model Predictive Control approach.
    
    Args:
        x0, v0: Initial state
        mu: Van der Pol parameter
        dt: Time step
        steps: Total number of steps
        horizon: MPC prediction horizon
        
    Returns:
        List of control inputs
    """
    controls = []
    state = [x0, v0]
    
    for step in range(steps):
        # Determine actual horizon (may be shorter near the end)
        actual_horizon = min(horizon, steps - step)
        
        # Solve for optimal control over horizon
        horizon_controls = solve_van_der_pol_optimal(
            state[0], state[1], mu, dt, actual_horizon
        )
        
        # Apply only the first control
        u = horizon_controls[0]
        controls.append(u)
        
        # Simulate one step forward
        sol = solve_ivp(
            lambda t, y: van_der_pol_dynamics(t, y, u, mu),
            [0, dt],
            state,
            method='RK45',
            t_eval=[dt]
        )
        state = sol.y[:, -1].tolist()
    
    return controls


def van_der_pol_dynamics(t, state, u, mu):
    """Van der Pol dynamics for use in simulation."""
    x, v = state
    dxdt = v
    dvdt = mu * (1 - x**2) * v - x + u
    return [dxdt, dvdt]


def analyze_van_der_pol_stability(mu: float = 1.0):
    """
    Analyze stability properties of Van der Pol oscillator.
    
    Returns information about limit cycles, equilibria, etc.
    """
    # The Van der Pol oscillator has:
    # - Unstable equilibrium at origin for mu > 0
    # - Stable limit cycle
    
    info = {
        'equilibrium': (0, 0),
        'equilibrium_stable': mu <= 0,
        'has_limit_cycle': mu > 0,
        'nonlinear_damping': f"μ(1-x²) with μ={mu}"
    }
    
    return info


def solve_van_der_pol_optimal(initial_state, dt: float, steps: int,
                              Q: Optional[np.ndarray] = None, 
                              R: Optional[float] = None) -> List[float]:
    """
    Wrapper function to match the expected data pipeline interface.
    
    Args:
        initial_state: Initial state [x0, v0] as list/array
        dt: Time step
        steps: Number of control steps
        Q: State cost matrix (2x2), default: diag([10.0, 5.0])
        R: Control cost scalar, default: 0.1
        
    Returns:
        List of control inputs
    """
    # Extract x0, v0 from initial_state
    if isinstance(initial_state, (list, tuple)):
        x0, v0 = initial_state[0], initial_state[1]
    elif isinstance(initial_state, np.ndarray):
        x0, v0 = float(initial_state[0]), float(initial_state[1])
    else:
        raise ValueError("initial_state must be a list/array with [x0, v0]")
    
    # Call the core function
    return solve_van_der_pol_optimal_core(x0, v0, mu=1.0, dt=dt, steps=steps, Q=Q, R=R)