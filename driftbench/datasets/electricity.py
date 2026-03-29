"""
Electricity dataset module for drift detection benchmarking.

This module handles loading and preprocessing of the electricity dataset
for time series forecasting and drift detection experiments.
"""

import pandas as pd
import os
from typing import Optional


def load_electricity_dataset(
    data_path: str = "datasets/electricity.csv",
    sample_entities: Optional[int] = None,
    downsample_freq: str = "h"
) -> pd.DataFrame:
    """
    Load and preprocess the electricity dataset.

    Converts wide format (date | client_0 | client_1 | ... | client_321)
    to long format (entity_id | timestamp | target).

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
    # Load the wide format data
    df = pd.read_csv(data_path)

    # Parse the timestamp column - use rename instead of insert to avoid fragmentation
    df = df.rename(columns={'date': 'timestamp'})
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Get entity columns (all columns except timestamp)
    entity_cols = [col for col in df.columns if col != 'timestamp']

    # Melt to long format
    df_long = df.melt(
        id_vars=['timestamp'],
        value_vars=entity_cols,
        var_name='entity_id',
        value_name='target'
    )

    # Convert entity_id to string
    df_long['entity_id'] = df_long['entity_id'].astype(str)

    # Sort by entity and timestamp
    df_long = df_long.sort_values(['entity_id', 'timestamp']).reset_index(drop=True)

    # Sample entities if requested
    if sample_entities is not None:
        unique_entities = df_long['entity_id'].unique()
        sampled = pd.Series(unique_entities).sample(n=sample_entities, random_state=42).tolist()
        df_long = df_long[df_long['entity_id'].isin(sampled)]

    # Data is already hourly, ensure proper datetime index
    df_long['timestamp'] = pd.to_datetime(df_long['timestamp'])

    return df_long


def get_entity_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get basic statistics per entity.

    Parameters
    ----------
    df : pd.DataFrame
        Long format DataFrame with entity_id, timestamp, target.

    Returns
    -------
    pd.DataFrame
        Statistics per entity.
    """
    return df.groupby('entity_id')['target'].agg(['mean', 'std', 'min', 'max', 'count'])
