"""
Forecasting metrics for evaluating model performance.
"""

import numpy as np
import pandas as pd
from typing import Union


def mae(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series]
) -> float:
    """
    Mean Absolute Error.

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        MAE value.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Remove pairs with NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return np.nan

    return np.mean(np.abs(y_true - y_pred))


def rmse(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series]
) -> float:
    """
    Root Mean Squared Error.

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        RMSE value.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Remove pairs with NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return np.nan

    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def smape(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series]
) -> float:
    """
    Symmetric Mean Absolute Percentage Error.

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        SMAPE value (as percentage).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Remove pairs with NaN or zeros
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | (y_true == 0) | (y_pred == 0))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return np.nan

    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2

    # Avoid division by zero
    mask = denominator != 0
    numerator = numerator[mask]
    denominator = denominator[mask]

    if len(numerator) == 0:
        return np.nan

    return 100 * np.mean(numerator / denominator)


def mape(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series]
) -> float:
    """
    Mean Absolute Percentage Error.

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        MAPE value (as percentage).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Remove pairs with NaN or zeros
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | (y_true == 0))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return np.nan

    return 100 * np.mean(np.abs((y_true - y_pred) / y_true))


def compute_all_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series]
) -> dict:
    """
    Compute all forecasting metrics.

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    dict
        Dictionary containing all metrics.
    """
    return {
        'mae': mae(y_true, y_pred),
        'rmse': rmse(y_true, y_pred),
        'smape': smape(y_true, y_pred),
        'mape': mape(y_true, y_pred)
    }
