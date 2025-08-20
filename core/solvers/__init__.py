"""Solver registry for optimal control solutions."""

from .lqr_solver import solve_double_integrator_lqr
from .vdp_solver import solve_van_der_pol

# Create alias for backward compatibility
solve_van_der_pol_optimal = solve_van_der_pol

# Solver registry - maps system names to their optimal solvers
SOLVER_REGISTRY = {
    "double_integrator": solve_double_integrator_lqr,
    "van_der_pol": solve_van_der_pol_optimal,
    # Future solvers:
    # "pendulum": solve_pendulum_swing_up,
    # "cartpole": solve_cartpole_lqr,
}

def get_solver(system_name):
    """Get the optimal solver for a system."""
    if system_name not in SOLVER_REGISTRY:
        raise ValueError(f"No solver registered for system: {system_name}")
    return SOLVER_REGISTRY[system_name]

def list_solvers():
    """List all available solvers."""
    return list(SOLVER_REGISTRY.keys())

__all__ = ["solve_double_integrator_lqr", "solve_van_der_pol_optimal",
           "get_solver", "list_solvers", "SOLVER_REGISTRY"]