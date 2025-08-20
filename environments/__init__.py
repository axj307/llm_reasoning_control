"""Environment registry for automatic discovery and registration of control environments."""

# Import all environment classes
from .base_environment import BaseEnvironment
from .double_integrator import DoubleIntegrator
from .van_der_pol import VanDerPol

# Environment registry - automatically extended when new environments are added
ENVIRONMENT_REGISTRY = {
    "double_integrator": DoubleIntegrator,
    "van_der_pol": VanDerPol,
    # Future environments will be added here:
    # "pendulum": Pendulum,
    # "cartpole": CartPole,
    # "quadrotor": Quadrotor,
}

# Backward compatibility - system registry
SYSTEM_REGISTRY = ENVIRONMENT_REGISTRY

def get_environment(name):
    """Get an environment class by name."""
    if name not in ENVIRONMENT_REGISTRY:
        raise ValueError(f"Unknown environment: {name}. Available environments: {list_environments()}")
    return ENVIRONMENT_REGISTRY[name]

def get_system(name):
    """Get a system class by name (backward compatibility)."""
    return get_environment(name)

def list_environments():
    """List all available environments."""
    return list(ENVIRONMENT_REGISTRY.keys())

def list_systems():
    """List all available systems (backward compatibility)."""
    return list_environments()

def create_environment(name, **kwargs):
    """Create an environment instance by name."""
    environment_class = get_environment(name)
    return environment_class(**kwargs)

def create_system(name, **kwargs):
    """Create a system instance by name (backward compatibility)."""
    return create_environment(name, **kwargs)

# For convenience, expose all environments at package level
__all__ = ["BaseEnvironment", "DoubleIntegrator", "VanDerPol", 
           "get_environment", "get_system", "list_environments", "list_systems",
           "create_environment", "create_system", "ENVIRONMENT_REGISTRY", "SYSTEM_REGISTRY"]