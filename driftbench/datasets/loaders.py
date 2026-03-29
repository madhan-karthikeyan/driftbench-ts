"""
Additional dataset loaders for drift detection benchmarking.

This module provides loaders for multiple time series datasets:
- Electricity dataset (UCI)
- Traffic dataset (UCI)
- Weather dataset (synthetic or external)

All loaders return data in the same format:
(entity_id, timestamp, target)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import os


def load_electricity_dataset(
    data_path: str = "datasets/electricity.csv",
    sample_entities: Optional[int] = None,
    downsample_freq: str = "h"
) -> pd.DataFrame:
    """
    Load and preprocess the electricity dataset.

    Parameters
    ----------
    data_path : str
        Path to the electricity.csv file.
    sample_entities : int, optional
        If provided, randomly sample this many entities.
    downsample_freq : str
        Frequency to downsample to (default: 'h' for hourly).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: entity_id, timestamp, target
    """
    df = pd.read_csv(data_path)
    df = df.rename(columns={'date': 'timestamp'})
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    entity_cols = [col for col in df.columns if col != 'timestamp']
    df_long = df.melt(
        id_vars=['timestamp'],
        value_vars=entity_cols,
        var_name='entity_id',
        value_name='target'
    )
    df_long['entity_id'] = df_long['entity_id'].astype(str)
    df_long = df_long.sort_values(['entity_id', 'timestamp']).reset_index(drop=True)
    
    if sample_entities is not None:
        unique_entities = df_long['entity_id'].unique()
        sampled = pd.Series(unique_entities).sample(n=sample_entities, random_state=42).tolist()
        df_long = df_long[df_long['entity_id'].isin(sampled)]
    
    df_long['timestamp'] = pd.to_datetime(df_long['timestamp'])
    return df_long


def load_traffic_dataset(
    data_path: str = "datasets/traffic.csv",
    sample_entities: Optional[int] = None,
    downsample_freq: str = "h"
) -> pd.DataFrame:
    """
    Load and preprocess the Metro Interstate Traffic Volume dataset.
    
    This dataset contains hourly traffic volume from Minneapolis-St Paul, MN
    for westbound I-94. It's a single-location dataset that we treat as
    having a single entity.

    Parameters
    ----------
    data_path : str
        Path to the traffic.csv file.
    sample_entities : int, optional
        Not used for this dataset (single location).
    downsample_freq : str
        Frequency to downsample to (default: 'h' for hourly).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: entity_id, timestamp, target
    """
    df = pd.read_csv(data_path)
    
    df = df.rename(columns={'date_time': 'timestamp', 'traffic_volume': 'target'})
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df['entity_id'] = 'i94_westbound'
    
    df = df[['entity_id', 'timestamp', 'target']]
    df = df.sort_values(['entity_id', 'timestamp']).reset_index(drop=True)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def load_weather_dataset(
    start_date: str = "2010-01-01",
    end_date: str = "2014-12-31",
    n_entities: int = 10,
    seed: int = 42,
    include_seasonality: bool = True,
    include_trend: bool = True,
    noise_level: float = 0.1
) -> pd.DataFrame:
    """
    Generate a synthetic weather-like dataset with realistic patterns.
    
    This is useful for testing when real datasets are not available.

    Parameters
    ----------
    start_date : str
        Start date for the dataset.
    end_date : str
        End date for the dataset.
    n_entities : int
        Number of synthetic entities (e.g., weather stations).
    seed : int
        Random seed for reproducibility.
    include_seasonality : bool
        Whether to include seasonal patterns.
    include_trend : bool
        Whether to include trend.
    noise_level : float
        Level of random noise (0-1).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: entity_id, timestamp, target
    """
    np.random.seed(seed)
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='h')
    n_timesteps = len(date_range)
    
    entities = [f"station_{i}" for i in range(n_entities)]
    
    records = []
    
    for entity in entities:
        base_value = np.random.uniform(10, 30)
        seasonal_amp = np.random.uniform(5, 15)
        trend_slope = np.random.uniform(-0.001, 0.001) if include_trend else 0
        
        for i, ts in enumerate(date_range):
            value = base_value
            
            if include_seasonality:
                day_of_year = ts.dayofyear
                value += seasonal_amp * np.sin(2 * np.pi * day_of_year / 365)
                
                hour_of_day = ts.hour
                value += 3 * np.sin(2 * np.pi * hour_of_day / 24)
            
            if include_trend:
                value += trend_slope * i
            
            noise = np.random.normal(0, noise_level * base_value)
            value += noise
            
            value = max(0, value)
            
            records.append({
                'entity_id': entity,
                'timestamp': ts,
                'target': value
            })
    
    df = pd.DataFrame(records)
    return df


def load_synthetic_drift_dataset(
    n_timesteps: int = 10000,
    n_entities: int = 5,
    drift_type: str = 'sudden',
    drift_point: float = 0.7,
    drift_magnitude: float = 0.5,
    seed: int = 42
) -> Tuple[pd.DataFrame, List[dict]]:
    """
    Generate a synthetic dataset with known drift points.
    
    This is ideal for testing drift detection algorithms since
    the ground truth drift times are known.

    Parameters
    ----------
    n_timesteps : int
        Number of timesteps.
    n_entities : int
        Number of entities.
    drift_type : str
        Type of drift: 'sudden', 'gradual', 'incremental', 'recurring'.
    drift_point : float
        Fraction of data where drift starts (0.0-1.0).
    drift_magnitude : float
        Magnitude of drift (as fraction of mean).
    seed : int
        Random seed.

    Returns
    -------
    tuple
        (DataFrame with columns: entity_id, timestamp, target, has_drift),
        (list of dict with drift event info)
    """
    np.random.seed(seed)
    
    records = []
    drift_events = []
    drift_start_idx = int(n_timesteps * drift_point)
    
    for entity_id in range(n_entities):
        base_mean = np.random.uniform(100, 200)
        base_std = np.random.uniform(10, 20)
        
        for t in range(n_timesteps):
            ts = pd.Timestamp('2020-01-01') + pd.Timedelta(hours=t)
            
            mean = base_mean
            has_drift = False
            
            if t >= drift_start_idx:
                if drift_type == 'sudden':
                    drift_factor = drift_magnitude
                    has_drift = True
                elif drift_type == 'gradual':
                    progress = (t - drift_start_idx) / (n_timesteps - drift_start_idx)
                    drift_factor = drift_magnitude * progress
                    has_drift = progress > 0.1
                elif drift_type == 'incremental':
                    drift_factor = drift_magnitude * np.sin(np.pi * (t - drift_start_idx) / (n_timesteps - drift_start_idx))
                    has_drift = True
                elif drift_type == 'recurring':
                    cycle = np.sin(2 * np.pi * (t - drift_start_idx) / 500)
                    drift_factor = drift_magnitude * max(0, cycle)
                    has_drift = drift_factor > 0.1
                
                mean = base_mean * (1 + drift_factor)
            
            value = np.random.normal(mean, base_std)
            
            records.append({
                'entity_id': f"entity_{entity_id}",
                'timestamp': ts,
                'target': value,
                'has_drift': has_drift
            })
            
            if has_drift and (len(drift_events) == 0 or drift_events[-1]['end_idx'] < t):
                drift_events.append({
                    'start_idx': drift_start_idx,
                    'end_idx': t,
                    'start_time': pd.Timestamp('2020-01-01') + pd.Timedelta(hours=drift_start_idx),
                    'type': drift_type,
                    'magnitude': drift_magnitude,
                    'entity': f"entity_{entity_id}"
                })
    
    df = pd.DataFrame(records)
    return df, drift_events


def load_oil_prices_dataset(
    data_path: str = "datasets/brent_prices_tasks_resamp_week.csv",
    sample_entities: Optional[int] = None,
    downsample_freq: str = "w"
) -> pd.DataFrame:
    """
    Load oil prices dataset (Brent or WTI).
    
    Parameters
    ----------
    data_path : str
        Path to the oil prices CSV file.
    sample_entities : int, optional
        Not used for this dataset (single series).
    downsample_freq : str
        Frequency ('w' for weekly, 'd' for daily).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: entity_id, timestamp, target
    """
    df = pd.read_csv(data_path)
    
    price_col = [c for c in df.columns if c != 'DATE_DT'][0]
    entity_name = 'brent' if 'brent' in data_path.lower() else 'wti'
    
    df = df.rename(columns={'DATE_DT': 'timestamp', price_col: 'target'})
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['target'] = pd.to_numeric(df['target'], errors='coerce')
    df = df.dropna(subset=['target'])
    
    df['entity_id'] = entity_name
    df = df[['entity_id', 'timestamp', 'target']]
    df = df.sort_values(['entity_id', 'timestamp']).reset_index(drop=True)
    
    return df


def get_dataset(name: str, **kwargs) -> pd.DataFrame:
    """
    Get a dataset by name.
    
    Parameters
    ----------
    name : str
        Dataset name: 'electricity', 'traffic', 'weather', 'synthetic', 'brent', 'wti'.
    **kwargs
        Additional arguments for the dataset loader.
    
    Returns
    -------
    pd.DataFrame
        Dataset with columns: entity_id, timestamp, target
    """
    loaders = {
        'electricity': load_electricity_dataset,
        'traffic': load_traffic_dataset,
        'weather': load_weather_dataset,
        'synthetic': lambda **kw: load_synthetic_drift_dataset(**kw)[0],
        'brent': lambda **kw: load_oil_prices_dataset(
            data_path="datasets/brent_prices_tasks_resamp_week.csv", **kw
        ),
        'wti': lambda **kw: load_oil_prices_dataset(
            data_path="datasets/wti_prices_tasks_resamp_week.csv", **kw
        )
    }
    
    name = name.lower()
    
    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(loaders.keys())}")
    
    return loaders[name](**kwargs)


def get_available_datasets() -> List[str]:
    """Get list of available dataset names."""
    return ['electricity', 'traffic', 'weather', 'synthetic', 'brent', 'wti']
