"""Core modules for the universal control LLM system."""

# Import classes only when needed to avoid dependency issues
def get_data_generator():
    """Get the data generator class (lazy import)."""
    from .data_pipeline import UniversalDataGenerator
    return UniversalDataGenerator

def get_model_manager():
    """Get the model manager class (lazy import)."""
    from .model_manager import UniversalModelManager
    return UniversalModelManager

# For backward compatibility
def UniversalDataGenerator(*args, **kwargs):
    """Create data generator instance."""
    cls = get_data_generator()
    return cls(*args, **kwargs)

def UniversalModelManager(*args, **kwargs):
    """Create model manager instance."""
    cls = get_model_manager()
    return cls(*args, **kwargs)

__all__ = ["get_data_generator", "get_model_manager", "UniversalDataGenerator", "UniversalModelManager"]