"""
Missing value handling utilities.
"""

import pandas as pd
import numpy as np


def handle_missing_values(
    df: pd.DataFrame,
    method: str = 'forward_fill',
    groupby_cols: list = None
) -> pd.DataFrame:
    """
    Handle missing values in time series data.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with time series data.
    method : str
        Method to handle missing values: 'forward_fill', 'backward_fill', 'mean', 'zero'.
    groupby_cols : list, optional
        Columns to group by before applying the method.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values handled.
    """
    df = df.copy()

    if groupby_cols:
        if method == 'forward_fill':
            df = df.groupby(groupby_cols).ffill()
        elif method == 'backward_fill':
            df = df.groupby(groupby_cols).bfill()
        elif method == 'mean':
            df[groupby_cols] = df.groupby(groupby_cols).transform(
                lambda x: x.fillna(x.mean())
            )
        elif method == 'zero':
            df = df.fillna(0)
    else:
        if method == 'forward_fill':
            df = df.ffill()
        elif method == 'backward_fill':
            df = df.bfill()
        elif method == 'mean':
            df = df.fillna(df.mean())
        elif method == 'zero':
            df = df.fillna(0)

    return df


def detect_missing_patterns(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> dict:
    """
    Detect patterns in missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    timestamp_col : str
        Name of the timestamp column.

    Returns
    -------
    dict
        Dictionary containing missing value statistics.
    """
    total_rows = len(df)
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / total_rows) * 100

    return {
        'total_rows': total_rows,
        'missing_per_column': missing_counts.to_dict(),
        'missing_pct_per_column': missing_pct.to_dict(),
        'total_missing': missing_counts.sum()
    }
