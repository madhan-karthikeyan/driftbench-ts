"""
Extended evaluation metrics for drift detection benchmarking.

This module provides metrics for evaluating drift detection performance
and model adaptation effectiveness.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass


@dataclass
class DriftEvent:
    """Represents a detected or true drift event."""
    timestamp: pd.Timestamp
    step: int
    magnitude: Optional[float] = None
    detected: bool = False
    detected_at: Optional[pd.Timestamp] = None


def detection_delay(
    true_drift_times: List[pd.Timestamp],
    detected_drift_times: List[pd.Timestamp],
    step_duration_hours: float = 24.0
) -> Dict[str, float]:
    """
    Compute detection delay metrics.

    Measures the time between true drift occurrence and detection.

    Parameters
    ----------
    true_drift_times : list
        List of true drift timestamps.
    detected_drift_times : list
        List of detected drift timestamps.
    step_duration_hours : float
        Duration of each step in hours (default: 24).

    Returns
    -------
    dict
        Dictionary with detection delay metrics.
    """
    if not true_drift_times:
        return {
            'mean_detection_delay_steps': np.nan,
            'mean_detection_delay_hours': np.nan,
            'max_detection_delay_steps': np.nan,
            'max_detection_delay_hours': np.nan
        }

    delays = []

    for true_time in true_drift_times:
        # Find closest detection
        if detected_drift_times:
            # Find first detection after true drift
            after_drift = [d for d in detected_drift_times if d >= true_time]
            if after_drift:
                delay = (after_drift[0] - true_time).total_seconds() / 3600
                delays.append(delay / step_duration_hours)

    if not delays:
        return {
            'mean_detection_delay_steps': np.nan,
            'mean_detection_delay_hours': np.nan,
            'max_detection_delay_steps': np.nan,
            'max_detection_delay_hours': np.nan
        }

    return {
        'mean_detection_delay_steps': np.mean(delays),
        'mean_detection_delay_hours': np.mean(delays) * step_duration_hours,
        'max_detection_delay_steps': np.max(delays),
        'max_detection_delay_hours': np.max(delays) * step_duration_hours
    }


def false_positive_rate(
    true_drift_times: List[pd.Timestamp],
    detected_drift_times: List[pd.Timestamp],
    total_timesteps: int,
    window_before_drift: int = 10
) -> Dict[str, float]:
    """
    Compute false positive rate.

    Parameters
    ----------
    true_drift_times : list
        List of true drift timestamps.
    detected_drift_times : list
        List of detected drift timestamps.
    total_timesteps : int
        Total number of timesteps.
    window_before_drift : int
        Window before true drift to exclude from FP counting.

    Returns
    -------
    dict
        Dictionary with false positive metrics.
    """
    # Count false positives
    false_positives = 0

    for detected_time in detected_drift_times:
        is_false_positive = True

        # Check if detection is near a true drift
        for true_time in true_drift_times:
            # Exclude window before true drift
            window_start = true_time - pd.Timedelta(steps=window_before_drift)
            if window_start <= detected_time <= true_time:
                is_false_positive = False
                break
            # Detection shortly after drift is considered true positive
            if true_time < detected_time <= true_time + pd.Timedelta(steps=window_before_drift):
                is_false_positive = False
                break

        if is_false_positive:
            false_positives += 1

    # FPR = FP / (FP + TN)
    # TN = timesteps without drift detection that are not near true drift
    # Simplified: FPR = FP / total non-drift timesteps
    non_drift_timesteps = total_timesteps - len(true_drift_times)

    if non_drift_timesteps > 0:
        fpr = false_positives / total_timesteps
    else:
        fpr = 0.0 if false_positives == 0 else np.nan

    # Also compute detection rate (true positive rate)
    true_positives = len(detected_drift_times) - false_positives
    detection_rate = true_positives / len(true_drift_times) if true_drift_times else 0.0

    return {
        'false_positives': false_positives,
        'false_positive_rate': fpr,
        'detection_rate': detection_rate
    }


def recovery_time(
    drift_time: pd.Timestamp,
    error_before_drift: float,
    error_series: pd.Series,
    timestamps: pd.Series,
    recovery_threshold: float = 0.1
) -> Optional[float]:
    """
    Compute recovery time after drift.

    Recovery is defined as the time when error returns to within
    threshold of pre-drift baseline.

    Parameters
    ----------
    drift_time : pd.Timestamp
        Time of drift occurrence.
    error_before_drift : float
        Error before drift (baseline).
    error_series : pd.Series
        Series of errors over time.
    timestamps : pd.Series
        Series of timestamps.
    recovery_threshold : float
        Threshold for recovery (as fraction of baseline error increase).

    Returns
    -------
    float
        Recovery time in steps, or None if not recovered.
    """
    # Find data after drift
    after_drift = timestamps >= drift_time

    if not after_drift.any():
        return None

    errors_after = error_series[after_drift]

    # Calculate acceptable error range
    max_acceptable_error = error_before_drift * (1 + recovery_threshold)

    # Find first point where error returns to acceptable range
    recovered = errors_after <= max_acceptable_error

    if not recovered.any():
        return None

    first_recovered_idx = recovered.idxmax()
    recovery_idx = error_series.index.get_loc(first_recovered_idx)
    drift_idx = timestamps[timestamps >= drift_time].index[0]
    actual_drift_idx = error_series.index.get_loc(drift_idx)

    return recovery_idx - actual_drift_idx


def performance_degradation_area(
    timestamps: pd.Series,
    errors: pd.Series,
    drift_time: pd.Timestamp,
    baseline_error: float
) -> Dict[str, float]:
    """
    Compute the area of performance degradation after drift.

    Parameters
    ----------
    timestamps : pd.Series
        Series of timestamps.
    errors : pd.Series
        Series of errors.
    drift_time : pd.Timestamp
        Time of drift occurrence.
    baseline_error : float
        Pre-drift baseline error.

    Returns
    -------
    dict
        Dictionary with degradation metrics.
    """
    # Get post-drift errors
    after_drift = timestamps >= drift_time
    post_errors = errors[after_drift]

    if len(post_errors) == 0:
        return {
            'degradation_area': 0.0,
            'max_error_increase': 0.0,
            'mean_error_increase': 0.0
        }

    # Calculate error increases
    error_increases = post_errors - baseline_error
    error_increases = error_increases.clip(lower=0)  # Only count increases

    # Area under the error increase curve (using trapezoidal rule)
    degradation_area = np.trapz(error_increases.values)

    return {
        'degradation_area': float(degradation_area),
        'max_error_increase': float(error_increases.max()),
        'mean_error_increase': float(error_increases.mean())
    }


def correlation_drift_magnitude_error(
    drift_magnitudes: List[float],
    error_increases: List[float]
) -> Dict[str, float]:
    """
    Compute correlation between drift magnitude and error increase.

    Parameters
    ----------
    drift_magnitudes : list
        List of drift magnitudes.
    error_increases : list
        Corresponding error increases.

    Returns
    -------
    dict
        Dictionary with correlation metrics.
    """
    if len(drift_magnitudes) != len(error_increases):
        return {
            'pearson_correlation': np.nan,
            'spearman_correlation': np.nan
        }

    if len(drift_magnitudes) < 2:
        return {
            'pearson_correlation': np.nan,
            'spearman_correlation': np.nan
        }

    from scipy.stats import pearsonr, spearmanr

    pearson_corr, _ = pearsonr(drift_magnitudes, error_increases)
    spearman_corr, _ = spearmanr(drift_magnitudes, error_increases)

    return {
        'pearson_correlation': float(pearson_corr),
        'spearman_correlation': float(spearman_corr)
    }


def compute_drift_benchmark_metrics(
    metrics_df: pd.DataFrame,
    drift_events: List[DriftEvent],
    timestamps_col: str = 'timestamp',
    error_col: str = 'error',
    step_duration_hours: float = 24.0
) -> Dict[str, Any]:
    """
    Compute comprehensive drift benchmark metrics.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame with metrics over time.
    drift_events : list
        List of DriftEvent objects (true drift events).
    timestamps_col : str
        Name of timestamp column.
    error_col : str
        Name of error column.
    step_duration_hours : float
        Duration of each step in hours.

    Returns
    -------
    dict
        Dictionary with all benchmark metrics.
    """
    timestamps = pd.to_datetime(metrics_df[timestamps_col])
    errors = metrics_df[error_col].values

    # Extract true and detected drift times
    true_drift_times = [e.timestamp for e in drift_events if not e.detected]
    detected_drift_times = [e.detected_at for e in drift_events if e.detected and e.detected_at is not None]

    # Compute metrics
    result = {}

    # Detection delay
    result.update(detection_delay(true_drift_times, detected_drift_times, step_duration_hours))

    # False positive rate
    result.update(false_positive_rate(
        true_drift_times,
        detected_drift_times,
        len(metrics_df)
    ))

    # Recovery times
    recovery_times = []
    for drift_event in drift_events:
        if drift_event.detected_at is not None:
            # Get baseline error (average before drift)
            before_drift = timestamps < drift_event.timestamp
            if before_drift.any():
                baseline = errors[before_drift].mean()
                rec_time = recovery_time(
                    drift_event.timestamp,
                    baseline,
                    pd.Series(errors, index=timestamps.index),
                    timestamps,
                    recovery_threshold=0.1
                )
                if rec_time is not None:
                    recovery_times.append(rec_time)

    if recovery_times:
        result['mean_recovery_time_steps'] = np.mean(recovery_times)
        result['mean_recovery_time_hours'] = np.mean(recovery_times) * step_duration_hours
        result['max_recovery_time_steps'] = np.max(recovery_times)
    else:
        result['mean_recovery_time_steps'] = np.nan
        result['mean_recovery_time_hours'] = np.nan
        result['max_recovery_time_steps'] = np.nan

    # Performance degradation
    total_degradation = 0.0
    total_max_increase = 0.0
    n_drift_events = 0

    for drift_event in drift_events:
        before_drift = timestamps < drift_event.timestamp
        if before_drift.any():
            baseline = errors[before_drift].mean()
            deg_metrics = performance_degradation_area(
                timestamps, pd.Series(errors, index=timestamps.index),
                drift_event.timestamp, baseline
            )
            total_degradation += deg_metrics['degradation_area']
            total_max_increase += deg_metrics['max_error_increase']
            n_drift_events += 1

    if n_drift_events > 0:
        result['avg_degradation_area'] = total_degradation / n_drift_events
        result['avg_max_error_increase'] = total_max_increase / n_drift_events
    else:
        result['avg_degradation_area'] = np.nan
        result['avg_max_error_increase'] = np.nan

    return result


def compute_extended_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    timestamps: Optional[Union[np.ndarray, pd.Series]] = None
) -> dict:
    """
    Compute extended forecasting metrics including drift-aware metrics.

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.
    timestamps : array-like, optional
        Timestamps for each prediction.

    Returns
    -------
    dict
        Dictionary with all metrics.
    """
    from driftbench.metrics.forecasting_metrics import mae, rmse, smape, mape

    # Base metrics
    result = {
        'mae': mae(y_true, y_pred),
        'rmse': rmse(y_true, y_pred),
        'smape': smape(y_true, y_pred),
        'mape': mape(y_true, y_pred)
    }

    # Additional error-based metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    errors = np.abs(y_true - y_pred)

    result['error_mean'] = float(np.mean(errors))
    result['error_std'] = float(np.std(errors))
    result['error_max'] = float(np.max(errors))

    # Error autocorrelation (indicates systematic patterns)
    if len(errors) > 1:
        result['error_autocorr_lag1'] = float(np.corrcoef(errors[:-1], errors[1:])[0, 1])
    else:
        result['error_autocorr_lag1'] = np.nan

    # Relative errors
    mask = y_true != 0
    if mask.any():
        relative_errors = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
        result['relative_error_mean'] = float(np.mean(relative_errors))
    else:
        result['relative_error_mean'] = np.nan

    return result
