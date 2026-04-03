"""
Models module for DriftBench-TS.

This module provides forecasting models for time series prediction.
All models are registered in the central MODEL_REGISTRY.

Available models:
- naive: Naive baseline (previous value)
- seasonal_naive: Seasonal naive baseline
- rf: Random Forest
- lgbm: LightGBM
- lstm: LSTM neural network
- tsmixer: TSMixer (MLP-based)
"""

from driftbench.models.base import BaseModel

from driftbench.models.naive import NaiveModel
from driftbench.models.seasonal_naive import SeasonalNaiveModel
from driftbench.models.sklearn_rf import RandomForestModel
from driftbench.models.lgbm import LGBMModel
from driftbench.models.lstm import LSTMModel
from driftbench.models.tsmixer import RidgeFeatureModel

from driftbench.models.registry import (
    MODEL_REGISTRY,
    get_model,
    get_available_models,
    is_model_available,
    get_model_info
)

__all__ = [
    'BaseModel',
    'NaiveModel',
    'SeasonalNaiveModel',
    'RandomForestModel',
    'LGBMModel',
    'LSTMModel',
    'RidgeFeatureModel',
    'MODEL_REGISTRY',
    'get_model',
    'get_available_models',
    'is_model_available',
    'get_model_info'
]
