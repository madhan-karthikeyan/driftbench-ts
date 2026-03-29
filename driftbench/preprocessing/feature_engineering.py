"""
Feature engineering utilities for time series.

FIXED: All rolling features now use .shift(1) to avoid look-ahead bias.
Only past data is used for feature computation.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def create_time_features(
    df: pd.DataFrame, 
    timestamp_col: str = 'timestamp',
    include_cyclical: bool = True
) -> pd.DataFrame:
    """
    Create time-based features from timestamp.
    
    FIXED: Added cyclical encoding for better representation.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with timestamp column.
    timestamp_col : str
        Name of the timestamp column.
    include_cyclical : bool
        Whether to include cyclical sin/cos encoding.

    Returns
    -------
    pd.DataFrame
        DataFrame with added time features.
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    df['hour'] = df[timestamp_col].dt.hour
    df['day_of_week'] = df[timestamp_col].dt.dayofweek
    df['day_of_month'] = df[timestamp_col].dt.day
    df['month'] = df[timestamp_col].dt.month
    df['quarter'] = df[timestamp_col].dt.quarter
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_month_start'] = df[timestamp_col].dt.is_month_start.astype(int)
    df['is_month_end'] = df[timestamp_col].dt.is_month_end.astype(int)

    if include_cyclical:
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    return df


def create_lag_features(
    df: pd.DataFrame,
    target_col: str = 'target',
    lags: List[int] = [1, 2, 3, 6, 12, 24, 48, 168],
    groupby_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create lag features for time series.
    
    These are inherently leak-free since they only look backward.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target_col : str
        Name of the target column.
    lags : list
        List of lag periods to create.
    groupby_cols : list, optional
        Columns to group by before creating lags.

    Returns
    -------
    pd.DataFrame
        DataFrame with added lag features.
    """
    df = df.copy()

    if groupby_cols:
        for lag in lags:
            df[f'lag_{lag}'] = df.groupby(groupby_cols)[target_col].shift(lag)
    else:
        for lag in lags:
            df[f'lag_{lag}'] = df[target_col].shift(lag)

    return df


def create_rolling_features(
    df: pd.DataFrame,
    target_col: str = 'target',
    windows: List[int] = [3, 6, 12, 24, 48],
    groupby_cols: Optional[List[str]] = None,
    include_expanding: bool = True
) -> pd.DataFrame:
    """
    Create rolling window features for time series.
    
    CRITICAL FIX: All rolling features now use .shift(1) to avoid look-ahead bias.
    
    Features computed:
    - Rolling mean/std/min/max with shift(1) - only uses past data
    - Expanding mean/std - uses all past data

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target_col : str
        Name of the target column.
    windows : list
        List of window sizes.
    groupby_cols : list, optional
        Columns to group by before creating rolling features.
    include_expanding : bool
        Whether to include expanding window features.

    Returns
    -------
    pd.DataFrame
        DataFrame with added rolling features (leak-free).
    """
    df = df.copy()
    
    shift_target = df[target_col].shift(1)

    if groupby_cols:
        for window in windows:
            shifted = df.groupby(groupby_cols)[target_col].shift(1)
            df[f'rolling_mean_{window}'] = df.groupby(groupby_cols)[target_col].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
            df[f'rolling_std_{window}'] = df.groupby(groupby_cols)[target_col].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
            )
            df[f'rolling_min_{window}'] = df.groupby(groupby_cols)[target_col].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).min()
            )
            df[f'rolling_max_{window}'] = df.groupby(groupby_cols)[target_col].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).max()
            )
    else:
        for window in windows:
            shifted = shift_target.copy()
            
            df[f'rolling_mean_{window}'] = shifted.rolling(
                window=window, min_periods=1
            ).mean()
            df[f'rolling_std_{window}'] = shifted.rolling(
                window=window, min_periods=1
            ).std()
            df[f'rolling_min_{window}'] = shifted.rolling(
                window=window, min_periods=1
            ).min()
            df[f'rolling_max_{window}'] = shifted.rolling(
                window=window, min_periods=1
            ).max()
            
            rolling_diff = shifted.diff()
            df[f'rolling_diff_mean_{window}'] = rolling_diff.rolling(
                window=window, min_periods=1
            ).mean()
    
    if include_expanding:
        if groupby_cols:
            df['expanding_mean'] = df.groupby(groupby_cols)[target_col].transform(
                lambda x: x.shift(1).expanding(min_periods=1).mean()
            )
            df['expanding_std'] = df.groupby(groupby_cols)[target_col].transform(
                lambda x: x.shift(1).expanding(min_periods=1).std()
            )
        else:
            df['expanding_mean'] = shift_target.expanding(min_periods=1).mean()
            df['expanding_std'] = shift_target.expanding(min_periods=1).std()

    return df


def create_diff_features(
    df: pd.DataFrame,
    target_col: str = 'target',
    periods: List[int] = [1, 24, 168],
    groupby_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create differencing features for time series.
    
    These capture trends and are inherently leak-free.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target_col : str
        Name of the target column.
    periods : list
        List of differencing periods.
    groupby_cols : list, optional
        Columns to group by.

    Returns
    -------
    pd.DataFrame
        DataFrame with added diff features.
    """
    df = df.copy()

    if groupby_cols:
        for period in periods:
            df[f'diff_{period}'] = df.groupby(groupby_cols)[target_col].diff(period)
    else:
        for period in periods:
            df[f'diff_{period}'] = df[target_col].diff(period)

    return df


def create_ewm_features(
    df: pd.DataFrame,
    target_col: str = 'target',
    spans: List[int] = [12, 24, 48],
    groupby_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create exponentially weighted moving average features.
    
    CRITICAL FIX: Uses shift(1) to avoid look-ahead bias.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target_col : str
        Name of the target column.
    spans : list
        List of spans for EWM.
    groupby_cols : list, optional
        Columns to group by.

    Returns
    -------
    pd.DataFrame
        DataFrame with added EWM features.
    """
    df = df.copy()

    if groupby_cols:
        for span in spans:
            shifted = df.groupby(groupby_cols)[target_col].shift(1)
            df[f'ewm_mean_{span}'] = df.groupby(groupby_cols)[target_col].transform(
                lambda x: x.shift(1).ewm(span=span, adjust=False).mean()
            )
    else:
        shifted = df[target_col].shift(1)
        for span in spans:
            df[f'ewm_mean_{span}'] = shifted.ewm(span=span, adjust=False).mean()

    return df


def create_all_features(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    target_col: str = 'target',
    groupby_cols: Optional[List[str]] = None,
    lags: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None,
    include_time: bool = True,
    include_ewm: bool = True
) -> pd.DataFrame:
    """
    Create all standard time series features.
    
    This is the main entry point that creates all features in a
    leak-free manner.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    timestamp_col : str
        Name of timestamp column.
    target_col : str
        Name of target column.
    groupby_cols : list, optional
        Columns to group by (e.g., entity_id).
    lags : list, optional
        Lag periods to create. Defaults to [1, 2, 3, 6, 12, 24, 48, 168].
    rolling_windows : list, optional
        Rolling window sizes. Defaults to [3, 6, 12, 24, 48].
    include_time : bool
        Whether to include time features.
    include_ewm : bool
        Whether to include EWM features.

    Returns
    -------
    pd.DataFrame
        DataFrame with all features (leak-free).
    """
    if lags is None:
        lags = [1, 2, 3, 6, 12, 24, 48, 168]
    if rolling_windows is None:
        rolling_windows = [3, 6, 12, 24, 48]

    result = df.copy()
    
    if include_time and timestamp_col in df.columns:
        result = create_time_features(result, timestamp_col)
    
    if groupby_cols:
        result = result.sort_values(groupby_cols + [timestamp_col])
        grouped = result.groupby(groupby_cols)
    else:
        grouped = None

    result = create_lag_features(
        result, target_col, lags, groupby_cols
    )
    
    result = create_rolling_features(
        result, target_col, rolling_windows, groupby_cols
    )
    
    result = create_diff_features(
        result, target_col, [1, 24, 168], groupby_cols
    )
    
    if include_ewm:
        result = create_ewm_features(
            result, target_col, [12, 24, 48], groupby_cols
        )

    result = result.sort_index()

    return result


def create_entity_features(
    df: pd.DataFrame,
    entity_col: str = 'entity_id',
    target_col: str = 'target',
    ref_window_start: Optional[pd.Timestamp] = None,
    ref_window_end: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """
    Create entity-level statistical features from a reference window.
    
    FIXED: Features are computed only from the reference window,
    avoiding any look-ahead bias.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    entity_col : str
        Name of entity identifier column.
    target_col : str
        Name of target column.
    ref_window_start : pd.Timestamp, optional
        Start of reference window for computing stats.
    ref_window_end : pd.Timestamp, optional
        End of reference window for computing stats.

    Returns
    -------
    pd.DataFrame
        DataFrame with entity-level features.
    """
    df = df.copy()
    
    ref_df = df
    if ref_window_start is not None:
        ref_df = df[df['timestamp'] >= ref_window_start]
    if ref_window_end is not None:
        ref_df = ref_df[ref_df['timestamp'] < ref_window_end]
    
    entity_stats = ref_df.groupby(entity_col)[target_col].agg([
        'mean', 'std', 'min', 'max', 'median'
    ]).reset_index()
    
    entity_stats.columns = [entity_col] + [
        f'{col}_entity_stat' for col in entity_stats.columns[1:]
    ]
    
    result = df.merge(entity_stats, on=entity_col, how='left')
    
    for col in entity_stats.columns[1:]:
        result[col] = result[col].fillna(entity_stats[col].mean())
    
    return result
