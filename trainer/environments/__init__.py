"""
Control system environments.
"""

from .double_integrator import DoubleIntegratorEnvironment

# Environment registry for easy access
ENVIRONMENTS = {
    "double_integrator": DoubleIntegratorEnvironment,
    "di": DoubleIntegratorEnvironment,  # Alias
}

def get_environment(name: str, **kwargs):
    """
    Get environment by name.
    
    Args:
        name: Environment name
        **kwargs: Environment initialization arguments
        
    Returns:
        Environment instance
    """
    if name not in ENVIRONMENTS:
        raise ValueError(f"Unknown environment: {name}. Available: {list(ENVIRONMENTS.keys())}")
    return ENVIRONMENTS[name](**kwargs)

__all__ = [
    "DoubleIntegratorEnvironment",
    "ENVIRONMENTS",
    "get_environment"
]