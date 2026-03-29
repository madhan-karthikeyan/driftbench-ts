"""
DriftBench-TS: Time series forecasting and drift detection framework.
"""

__version__ = "0.1.0"

from driftbench.datasets import load_electricity_dataset
from driftbench.models.seasonal_naive import SeasonalNaiveModel
from driftbench.models.lgbm import LGBMModel
from driftbench.models.lstm import LSTMModel

__all__ = [
    'load_electricity_dataset',
    'SeasonalNaiveModel',
    'LGBMModel',
    'LSTMModel'
]
