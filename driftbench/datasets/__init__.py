"""
Dataset loaders for time series forecasting benchmarking.
"""

from driftbench.datasets.base import BaseDataset
from driftbench.datasets.electricity import (
    load_electricity_dataset,
    get_entity_statistics
)
from driftbench.datasets.loaders import (
    load_traffic_dataset,
    load_weather_dataset,
    load_synthetic_drift_dataset,
    get_dataset,
    get_available_datasets
)

__all__ = [
    'BaseDataset',
    'load_electricity_dataset',
    'get_entity_statistics',
    'load_traffic_dataset',
    'load_weather_dataset',
    'load_synthetic_drift_dataset',
    'get_dataset',
    'get_available_datasets'
]
