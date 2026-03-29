"""
Base detector interface for drift detection.

This module defines the abstract base class for all drift detectors
and provides common functionality.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum


class DetectorType(Enum):
    """Enumeration of available detector types."""
    ADWIN = "adwin"
    PAGE_HINKLEY = "page_hinkley"
    KS_TEST = "ks_test"
    WASSERSTEIN = "wasserstein"
    RESIDUAL_KS = "residual_ks"


class DetectionMode(Enum):
    """Mode for drift detection."""
    FEATURE = "feature"
    RESIDUAL = "residual"
    EMBEDDING = "embedding"


@dataclass
class DriftDetectionResult:
    """Result of drift detection."""
    drift_detected: bool
    drift_score: float
    timestamp: Optional[pd.Timestamp] = None
    p_value: Optional[float] = None
    reference_size: int = 0
    current_size: int = 0
    detector_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseDriftDetector(ABC):
    """
    Abstract base class for drift detectors.

    All detectors must implement:
    - fit(reference_data): Set reference window
    - detect(new_data): Check for drift
    - get_drift_score(): Get current drift score
    - reset(): Reset detector state
    """

    def __init__(self, name: str = "base", **kwargs):
        """
        Initialize the detector.

        Parameters
        ----------
        name : str
            Detector name.
        **kwargs
            Additional parameters.
        """
        self.name = name
        self.reference_data: Optional[np.ndarray] = None
        self.is_fitted = False
        self.drift_history: List[DriftDetectionResult] = []

    @abstractmethod
    def fit(self, reference_data: np.ndarray) -> 'BaseDriftDetector':
        """
        Fit the detector on reference data.

        Parameters
        ----------
        reference_data : np.ndarray
            Reference (baseline) data distribution.

        Returns
        -------
        self
            The fitted detector.
        """
        pass

    @abstractmethod
    def detect(self, new_data: np.ndarray) -> DriftDetectionResult:
        """
        Detect drift in new data.

        Parameters
        ----------
        new_data : np.ndarray
            New data to check for drift.

        Returns
        -------
        DriftDetectionResult
            Drift detection result.
        """
        pass

    @abstractmethod
    def get_drift_score(self) -> float:
        """
        Get the current drift score.

        Returns
        -------
        float
            Drift score (higher = more drift).
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset the detector to initial state."""
        pass

    def get_detection_history(self) -> pd.DataFrame:
        """
        Get detection history as a DataFrame.

        Returns
        -------
        pd.DataFrame
            History of drift detections.
        """
        if not self.drift_history:
            return pd.DataFrame()

        records = []
        for result in self.drift_history:
            records.append({
                'timestamp': result.timestamp,
                'drift_detected': result.drift_detected,
                'drift_score': result.drift_score,
                'p_value': result.p_value,
                'detector': result.detector_name
            })

        return pd.DataFrame(records)


class StatisticalDriftDetector(BaseDriftDetector):
    """
    Base class for statistical drift detectors.

    Provides common functionality for statistical tests.
    """

    def __init__(self, name: str, alpha: float = 0.05, **kwargs):
        """
        Initialize statistical detector.

        Parameters
        ----------
        name : str
            Detector name.
        alpha : float
            Significance level for drift detection.
        **kwargs
            Additional parameters.
        """
        super().__init__(name, **kwargs)
        self.alpha = alpha

    def fit(self, reference_data: np.ndarray) -> 'StatisticalDriftDetector':
        """Fit detector on reference data."""
        self.reference_data = np.asarray(reference_data)
        # Remove NaN values
        self.reference_data = self.reference_data[~np.isnan(self.reference_data)]
        self.is_fitted = True
        return self

    def reset(self):
        """Reset detector state."""
        self.reference_data = None
        self.is_fitted = False
        self.drift_history = []
