"""
Central Model Registry for DriftBench-TS.

All models must be registered here to ensure consistent model name mapping
and proper instantiation throughout the pipeline.

Model Names (used in configs and results):
- naive: Naive baseline (previous value)
- seasonal_naive: Seasonal naive baseline
- rf: Random Forest
- lgbm: LightGBM
- lstm: LSTM neural network
- ridge_features: Ridge Regression with temporal features
- tsmixer_legacy: Alias for ridge_features (backward compatibility)
"""

from typing import Dict, Type, Optional
import logging

logger = logging.getLogger(__name__)

MODEL_REGISTRY: Dict[str, Type['BaseModel']] = {}

AVAILABLE_MODELS = list(MODEL_REGISTRY.keys())


def register_model(name: str):
    """Decorator to register a model in the central registry."""
    def decorator(cls):
        if name in MODEL_REGISTRY:
            logger.warning(f"Model '{name}' already registered, overwriting")
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model(name: str, **kwargs) -> 'BaseModel':
    """
    Get a model instance by name from the registry.
    
    Parameters
    ----------
    name : str
        Model name (naive, seasonal_naive, rf, lgbm, lstm, ridge_features, tsmixer_legacy)
    **kwargs
        Model-specific parameters
        
    Returns
    -------
    BaseModel
        Instantiated model
    """
    # Backward compatibility mapping
    name_mapping = {
        'tsmixer': 'tsmixer_legacy',
    }
    name = name_mapping.get(name, name)
    
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    
    logger.info(f"Initializing model: {name}")
    model = MODEL_REGISTRY[name](**kwargs)
    logger.info(f"Model initialized: {name} ({model.name})")
    
    return model


def is_model_available(name: str) -> bool:
    """Check if a model is available (registered and dependencies satisfied)."""
    if name not in MODEL_REGISTRY:
        return False
    
    model_class = MODEL_REGISTRY[name]
    if hasattr(model_class, 'is_available'):
        return model_class.is_available()
    
    return True


def get_available_models() -> list:
    """Get list of all available model names."""
    return list(MODEL_REGISTRY.keys())


def get_model_info(name: str) -> dict:
    """Get information about a model."""
    if name not in MODEL_REGISTRY:
        return {}
    
    model_class = MODEL_REGISTRY[name]
    return {
        'name': name,
        'class': model_class.__name__,
        'available': is_model_available(name),
        'description': getattr(model_class, '__doc__', '') or ''
    }
