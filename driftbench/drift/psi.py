"""
Population Stability Index (PSI) for drift detection.
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass, field
import pandas as pd
from .base_detector import BaseDriftDetector, DriftDetectionResult


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    buckets: int = 10
) -> float:
    """
    Compute PSI between expected (reference) and actual (current) distributions.
    
    PSI < 0.1: No significant drift (green)
    PSI 0.1-0.2: Moderate drift, monitor (yellow)  
    PSI > 0.2: Significant drift (red)
    
    Parameters
    ----------
    expected : np.ndarray
        Reference/expected distribution (training data).
    actual : np.ndarray
        Current/actual distribution to compare.
    buckets : int
        Number of buckets for distribution comparison (default: 10).
    
    Returns
    -------
    float
        PSI value.
    """
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    
    if len(expected) < 2 or len(actual) < 2:
        return 0.0
    
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints[-1] = breakpoints[-1] + 1
    
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]
    
    expected_percents = expected_counts / len(expected)
    actual_percents = actual_counts / len(actual)
    
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
    
    contribution = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
    
    return np.sum(contribution)


class PSIDriftDetector(BaseDriftDetector):
    """
    Drift detector using Population Stability Index (PSI).
    
    PSI measures how much the distribution of a variable has changed
    over time. Common thresholds:
    - PSI < 0.1: No significant drift
    - PSI 0.1-0.2: Moderate drift, monitor
    - PSI > 0.2: Significant drift, action required
    
    This class conforms to the BaseDriftDetector interface.
    """
    
    def __init__(
        self,
        threshold: float = 0.2,
        min_samples: int = 100,
        window_size: int = 1000,
        name: str = "psi"
    ):
        """
        Initialize PSI detector.
        
        Parameters
        ----------
        threshold : float
            PSI threshold for drift detection (default: 0.2).
        min_samples : int
            Minimum samples required for reliable comparison (default: 100).
        window_size : int
            Size of sliding window for current data (default: 1000).
        name : str
            Detector name for logging.
        """
        super().__init__(name=name)
        self.threshold = threshold
        self.min_samples = min_samples
        self.window_size = window_size
        self.current_window: list = []
    
    def fit(self, reference_data: np.ndarray) -> 'PSIDriftDetector':
        """Set reference distribution from training data."""
        reference_data = np.asarray(reference_data)
        reference_data = reference_data[~np.isnan(reference_data)]
        
        if len(reference_data) < self.min_samples:
            raise ValueError(
                f"Reference data too small: {len(reference_data)} < {self.min_samples}"
            )
        
        self.reference_data = reference_data
        self.is_fitted = True
        return self
    
    def detect(self, new_data: np.ndarray) -> DriftDetectionResult:
        """
        Detect drift by comparing current data to reference distribution.
        
        Parameters
        ----------
        new_data : np.ndarray
            New observations to check for drift.
        
        Returns
        -------
        DriftDetectionResult
            Drift detection result with psi_value, drift_detected, etc.
        """
        new_data = np.asarray(new_data)
        new_data = new_data[~np.isnan(new_data)]
        
        if not self.is_fitted or self.reference_data is None:
            return DriftDetectionResult(
                drift_detected=False,
                drift_score=0.0,
                detector_name=self.name,
                metadata={'status': 'not_fitted'}
            )
        
        self.current_window.extend(new_data.tolist())
        
        if len(self.current_window) > self.window_size:
            self.current_window = self.current_window[-self.window_size:]
        
        if len(self.current_window) < self.min_samples:
            return DriftDetectionResult(
                drift_detected=False,
                drift_score=0.0,
                detector_name=self.name,
                metadata={
                    'status': 'warming_up',
                    'current_size': len(self.current_window),
                    'required': self.min_samples
                }
            )
        
        current_array = np.array(self.current_window)
        psi_value = compute_psi(self.reference_data, current_array)
        
        drift_detected = psi_value > self.threshold
        
        if drift_detected:
            self.reference_data = current_array.copy()
        
        result = DriftDetectionResult(
            drift_detected=drift_detected,
            drift_score=float(psi_value),
            detector_name=self.name,
            reference_size=len(self.reference_data),
            current_size=len(current_array),
            metadata={
                'category': self._categorize_psi(psi_value),
                'threshold': self.threshold
            }
        )
        
        self.drift_history.append(result)
        return result
    
    def _categorize_psi(self, psi: float) -> str:
        """Categorize PSI value for interpretation."""
        if psi < 0.1:
            return 'stable'
        elif psi < 0.2:
            return 'moderate'
        else:
            return 'significant'
    
    def get_drift_score(self) -> float:
        """Get current drift score (PSI value)."""
        if not self.drift_history:
            return 0.0
        return self.drift_history[-1].drift_score
    
    def reset(self):
        """Reset detector state."""
        super().reset()
        self.current_window = []
