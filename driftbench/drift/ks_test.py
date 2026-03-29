"""
Kolmogorov-Smirnov test for drift detection.

This module implements drift detection using the two-sample KS test.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Optional


def ks_test(
    reference: np.ndarray,
    current: np.ndarray,
    alpha: float = 0.05
) -> Tuple[float, float, bool]:
    """
    Perform two-sample Kolmogorov-Smirnov test for drift detection.

    Parameters
    ----------
    reference : np.ndarray
        Reference (baseline) data distribution.
    current : np.ndarray
        Current data distribution to compare.
    alpha : float
        Significance level for the test (default: 0.05).

    Returns
    -------
    tuple
        (statistic, p_value, drift_detected)
        - statistic: KS statistic (maximum difference between CDFs)
        - p_value: p-value from the test
        - drift_detected: True if drift is detected (p_value < alpha)
    """
    # Remove NaN values
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]

    if len(reference) == 0 or len(current) == 0:
        return 0.0, 1.0, False

    # Perform two-sample KS test
    statistic, p_value = stats.ks_2samp(reference, current)

    # Drift is detected if p-value is below significance level
    drift_detected = p_value < alpha

    return statistic, p_value, drift_detected


def compute_ks_drift_score(
    reference: np.ndarray,
    current: np.ndarray
) -> float:
    """
    Compute KS drift score (just the statistic, no hypothesis test).

    Parameters
    ----------
    reference : np.ndarray
        Reference data distribution.
    current : np.ndarray
        Current data distribution.

    Returns
    -------
    float
        KS drift score (0 = no drift, 1 = maximum drift).
    """
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]

    if len(reference) == 0 or len(current) == 0:
        return 0.0

    statistic, _ = stats.ks_2samp(reference, current)
    return statistic


def detect_drift_in_window(
    df: pd.DataFrame,
    reference_window: pd.DataFrame,
    target_col: str = 'target',
    alpha: float = 0.05
) -> dict:
    """
    Detect drift between a reference window and current window.

    Parameters
    ----------
    df : pd.DataFrame
        Current window data.
    reference_window : pd.DataFrame
        Reference window data.
    target_col : str
        Column name for target values.
    alpha : float
        Significance level for KS test.

    Returns
    -------
    dict
        Dictionary containing drift detection results.
    """
    reference_values = reference_window[target_col].values
    current_values = df[target_col].values

    statistic, p_value, drift_detected = ks_test(
        reference_values, current_values, alpha
    )

    return {
        'drift_detected': drift_detected,
        'ks_statistic': statistic,
        'p_value': p_value,
        'alpha': alpha,
        'reference_size': len(reference_values),
        'current_size': len(current_values)
    }


class DriftDetector:
    """
    Drift detector using KS test.
    """

    def __init__(self, alpha: float = 0.05):
        """
        Initialize the drift detector.

        Parameters
        ----------
        alpha : float
            Significance level for drift detection.
        """
        self.alpha = alpha
        self.reference_window = None

    def set_reference(self, data: np.ndarray):
        """Set the reference window for comparison."""
        self.reference_window = data

    def detect_drift(self, current: np.ndarray) -> dict:
        """
        Detect drift in current data compared to reference.

        Parameters
        ----------
        current : np.ndarray
            Current data to check for drift.

        Returns
        -------
        dict
            Drift detection results.
        """
        if self.reference_window is None:
            raise ValueError("Reference window not set. Call set_reference first.")

        statistic, p_value, drift_detected = ks_test(
            self.reference_window, current, self.alpha
        )

        return {
            'drift_detected': drift_detected,
            'ks_statistic': statistic,
            'p_value': p_value
        }
