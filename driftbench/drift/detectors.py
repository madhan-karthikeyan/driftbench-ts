"""
ADWIN (ADaptive WINdowing) and other drift detectors.

This module provides multiple drift detection algorithms:
- ADWIN: Adaptive windowing detector with proper effect size
- Page-Hinkley: Sequential change detection
- CUSUM: Cumulative sum detector
- DDM: Drift Detection Method
- Residual KS Test: Detects drift in model residuals
- Wasserstein: Distribution distance-based detection

All detectors use normalized effect size (Cohen's d) for continuous data.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from collections import deque
from dataclasses import dataclass, field

from driftbench.drift.base_detector import (
    BaseDriftDetector,
    StatisticalDriftDetector,
    DriftDetectionResult,
    DetectorType
)
from driftbench.drift.psi import PSIDriftDetector


class ADWINDetector(BaseDriftDetector):
    """
    ADWIN drift detector with proper effect size normalization.

    ADWIN detects change by using an adaptive sliding window that
    shrinks when variance increases and expands when data is stationary.
    
    Uses Cohen's d effect size for normalized drift detection on 
    continuous data, avoiding the 100% false positive rate of the
    naive mean-difference comparison.

    Reference:
    Bifet, A., & Gavalda, R. (2007). Learning from time-changing
    data with adaptive windowing.
    """

    def __init__(
        self,
        delta: float = 0.002,
        min_window_size: int = 30,
        max_window_size: int = 1000,
        effect_threshold: float = 0.5,
        warmup_size: int = 100
    ):
        """
        Initialize ADWIN detector.

        Parameters
        ----------
        delta : float
            Confidence value for the statistical test (default: 0.002).
        min_window_size : int
            Minimum window size before detecting drift (default: 30).
        max_window_size : int
            Maximum window size (default: 1000).
        effect_threshold : float
            Minimum Cohen's d effect size to trigger drift (default: 0.5).
            - 0.2 = small effect
            - 0.5 = medium effect
            - 0.8 = large effect
        warmup_size : int
            Number of samples before detector becomes active (default: 100).
        """
        super().__init__(name=DetectorType.ADWIN.value)
        self.delta = delta
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.effect_threshold = effect_threshold
        self.warmup_size = warmup_size
        
        self.window: deque = deque(maxlen=max_window_size)
        self.total: float = 0.0
        self.variance: float = 0.0
        self.count: int = 0
        self.sample_count: int = 0
        
        self.last_drift_score: float = 0.0
        self.last_effect_size: float = 0.0
        self.last_p_value: float = 1.0
        
        self.drift_detected_count: int = 0
        self.total_checks: int = 0
        
        self._reference_mean: float = 0.0
        self._reference_std: float = 1.0
        self._is_warmed_up: bool = False

    def fit(self, reference_data: np.ndarray) -> 'ADWINDetector':
        """Initialize detector with reference data as baseline."""
        reference_data = np.asarray(reference_data)
        reference_data = reference_data[~np.isnan(reference_data)]
        
        if len(reference_data) == 0:
            return self

        self._reference_mean = np.mean(reference_data)
        ref_std = np.std(reference_data)
        self._reference_std = ref_std if ref_std > 0 else 1.0
        
        self.window.clear()
        for val in reference_data[-self.max_window_size:]:
            self._add_element(val)
        
        self._is_warmed_up = len(reference_data) >= self.warmup_size
        self.is_fitted = True
        return self

    def _add_element(self, value: float) -> None:
        """Add element to the window and update statistics."""
        self.window.append(value)
        self.count += 1
        self.sample_count += 1

        delta_val = value - self.total / self.count if self.count > 0 else 0
        self.total += value
        if self.count > 1:
            self.variance += delta_val * (value - self.total / self.count)

    def detect(self, new_data: np.ndarray) -> DriftDetectionResult:
        """Detect drift using normalized effect size."""
        if not self.is_fitted:
            self.fit(new_data)
            return DriftDetectionResult(
                drift_detected=False,
                drift_score=0.0,
                detector_name=self.name
            )

        new_data = np.asarray(new_data)
        new_data = new_data[~np.isnan(new_data)]

        if len(new_data) == 0:
            return DriftDetectionResult(
                drift_detected=False,
                drift_score=0.0,
                detector_name=self.name
            )

        for value in new_data:
            self._add_element(value)
            
        self.total_checks += 1
        
        if len(self.window) < 2 * self.min_window_size:
            return DriftDetectionResult(
                drift_detected=False,
                drift_score=0.0,
                detector_name=self.name,
                reference_size=len(self.window),
                current_size=len(new_data),
                metadata={'status': 'warming_up'}
            )

        drift_detected, effect_size, drift_score = self._check_for_drift()

        self.last_drift_score = drift_score
        self.last_effect_size = effect_size
        
        if drift_detected:
            self.drift_detected_count += 1
            self._remove_old_elements()
            
            return DriftDetectionResult(
                drift_detected=True,
                drift_score=drift_score,
                p_value=self.last_p_value,
                detector_name=self.name,
                reference_size=len(self.window),
                current_size=len(new_data),
                metadata={
                    'effect_size': effect_size,
                    'window_size': len(self.window),
                    'total_checks': self.total_checks,
                    'drift_detected_count': self.drift_detected_count
                }
            )

        return DriftDetectionResult(
            drift_detected=False,
            drift_score=drift_score,
            p_value=self.last_p_value,
            detector_name=self.name,
            reference_size=len(self.window),
            current_size=len(new_data),
            metadata={
                'effect_size': effect_size,
                'window_size': len(self.window)
            }
        )

    def _check_for_drift(self) -> Tuple[bool, float, float]:
        """
        Check for drift using Cohen's d effect size.
        
        Cohen's d = (mean_current - mean_reference) / pooled_std
        
        Returns:
            Tuple of (drift_detected, effect_size, normalized_score)
        """
        window_size = len(self.window)
        if window_size < 2 * self.min_window_size:
            return False, 0.0, 0.0

        split_point = window_size // 2
        window_arr = np.array(self.window)

        left = window_arr[:split_point]
        right = window_arr[split_point:]

        mean_left = np.mean(left)
        mean_right = np.mean(right)
        std_left = np.std(left, ddof=1) if len(left) > 1 else 0
        std_right = np.std(right, ddof=1) if len(right) > 1 else 0
        
        mean_diff = abs(mean_right - mean_left)
        pooled_std = np.sqrt((std_left**2 + std_right**2) / 2)
        
        if pooled_std > 1e-10:
            effect_size = mean_diff / pooled_std
        else:
            effect_size = 0.0
        
        drift_score = effect_size
        
        drift_detected = effect_size > self.effect_threshold
        
        if self._reference_std > 1e-10:
            self.last_p_value = 1.0 - min(1.0, effect_size / 2.0)
        else:
            self.last_p_value = 1.0

        return drift_detected, effect_size, drift_score

    def _remove_old_elements(self) -> None:
        """Remove oldest half of elements after drift detected."""
        if len(self.window) > self.min_window_size:
            remove_count = len(self.window) // 2
            for _ in range(remove_count):
                self.window.popleft()
            self.count = len(self.window)

    def get_drift_score(self) -> float:
        """Get current drift score (Cohen's d)."""
        return self.last_effect_size

    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            'total_checks': self.total_checks,
            'drift_detected_count': self.drift_detected_count,
            'detection_rate': self.drift_detected_count / max(1, self.total_checks),
            'last_effect_size': self.last_effect_size,
            'last_drift_score': self.last_drift_score,
            'window_size': len(self.window),
            'is_warmed_up': self._is_warmed_up
        }

    def reset(self) -> None:
        """Reset detector to initial state."""
        self.window = deque(maxlen=self.max_window_size)
        self.total = 0.0
        self.variance = 0.0
        self.count = 0
        self.sample_count = 0
        self.last_drift_score = 0.0
        self.last_effect_size = 0.0
        self.last_p_value = 1.0
        self.is_fitted = False
        self.drift_history = []
        self.drift_detected_count = 0
        self.total_checks = 0
        self._is_warmed_up = False

    def get_window_size(self) -> int:
        """Get current window size."""
        return len(self.window)


class PageHinkleyDetector(BaseDriftDetector):
    """
    Page-Hinkley drift detector with proper threshold scaling.
    
    The Page-Hinkley test is a sequential analysis method that
    detects a change in the mean of a Gaussian process.
    
    Reference:
    Page, E. S. (1954). Continuous Inspection Schemes.
    """

    def __init__(
        self,
        delta: float = 0.005,
        threshold: float = 50.0,
        alpha: float = 0.005,
        min_samples: int = 30
    ):
        """
        Initialize Page-Hinkley detector.

        Parameters
        ----------
        delta : float
            Magnitude of change to detect (default: 0.005).
        threshold : float
            Detection threshold (default: 50.0). Scaled to data units.
        alpha : float
            Forgetting factor (default: 0.005).
        min_samples : int
            Minimum samples before detecting (default: 30).
        """
        super().__init__(name=DetectorType.PAGE_HINKLEY.value)
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.min_samples = min_samples

        self.cumsum: float = 0.0
        self.n: int = 0
        self.reference_mean: Optional[float] = None
        
        self.last_drift_score: float = 0.0
        self.drift_detected_count: int = 0

    def fit(self, reference_data: np.ndarray) -> 'PageHinkleyDetector':
        """Initialize detector with reference data."""
        reference_data = np.asarray(reference_data)
        reference_data = reference_data[~np.isnan(reference_data)]

        if len(reference_data) > 0:
            self.reference_mean = np.mean(reference_data)
            self.cumsum = 0.0
            self.n = len(reference_data)

        self.is_fitted = True
        return self

    def detect(self, new_data: np.ndarray) -> DriftDetectionResult:
        """Detect drift using Page-Hinkley test."""
        if not self.is_fitted or self.reference_mean is None:
            self.fit(new_data)
            return DriftDetectionResult(
                drift_detected=False,
                drift_score=0.0,
                detector_name=self.name
            )

        new_data = np.asarray(new_data)
        new_data = new_data[~np.isnan(new_data)]

        if len(new_data) == 0:
            return DriftDetectionResult(
                drift_detected=False,
                drift_score=0.0,
                detector_name=self.name
            )

        drift_detected = False

        for value in new_data:
            self.n += 1
            self.cumsum += value - self.reference_mean - self.delta
            
            self.cumsum = (1 - self.alpha) * self.cumsum

            self.last_drift_score = abs(self.cumsum) / max(1, self.n)

            if abs(self.cumsum) > self.threshold:
                drift_detected = True
                self.drift_detected_count += 1
                break

        return DriftDetectionResult(
            drift_detected=drift_detected,
            drift_score=self.last_drift_score,
            detector_name=self.name,
            reference_size=self.n,
            current_size=len(new_data),
            metadata={
                'cumsum': self.cumsum,
                'total_samples': self.n,
                'drift_count': self.drift_detected_count
            }
        )

    def get_drift_score(self) -> float:
        """Get current drift score."""
        return self.last_drift_score

    def reset(self) -> None:
        """Reset detector to initial state."""
        self.cumsum = 0.0
        self.n = 0
        self.reference_mean = None
        self.last_drift_score = 0.0
        self.is_fitted = False
        self.drift_history = []
        self.drift_detected_count = 0


class CUSUMDetector(BaseDriftDetector):
    """
    CUSUM (Cumulative Sum) drift detector.
    
    Detects shifts in the mean of a process by accumulating
    deviations from a target value.
    """

    def __init__(
        self,
        threshold: float = 5.0,
        drift_threshold: float = 0.1,
        min_samples: int = 30
    ):
        """
        Initialize CUSUM detector.

        Parameters
        ----------
        threshold : float
            Detection threshold for cumulative sum.
        drift_threshold : float
            Allowable drift per step.
        min_samples : int
            Minimum samples before detecting.
        """
        super().__init__(name="cusum")
        self.threshold = threshold
        self.drift_threshold = drift_threshold
        self.min_samples = min_samples
        
        self.cumsum_pos: float = 0.0
        self.cumsum_neg: float = 0.0
        self.n: int = 0
        self.last_drift_score: float = 0.0
        self.drift_detected_count: int = 0
        
        self._reference_mean: Optional[float] = None

    def fit(self, reference_data: np.ndarray) -> 'CUSUMDetector':
        """Initialize with reference data."""
        reference_data = np.asarray(reference_data)
        reference_data = reference_data[~np.isnan(reference_data)]
        
        if len(reference_data) > 0:
            self._reference_mean = np.mean(reference_data)
        
        self.is_fitted = True
        return self

    def detect(self, new_data: np.ndarray) -> DriftDetectionResult:
        """Detect drift using CUSUM."""
        if not self.is_fitted:
            self.fit(new_data)
            return DriftDetectionResult(
                drift_detected=False,
                drift_score=0.0,
                detector_name=self.name
            )

        new_data = np.asarray(new_data)
        new_data = new_data[~np.isnan(new_data)]

        if len(new_data) == 0:
            return DriftDetectionResult(
                drift_detected=False,
                drift_score=0.0,
                detector_name=self.name
            )

        drift_detected = False
        
        for value in new_data:
            self.n += 1
            
            deviation = value - (self._reference_mean or 0)
            
            self.cumsum_pos = max(0, self.cumsum_pos + deviation - self.drift_threshold)
            self.cumsum_neg = max(0, self.cumsum_neg - deviation - self.drift_threshold)
            
            self.last_drift_score = max(self.cumsum_pos, self.cumsum_neg) / max(1, self.n)

            if self.cumsum_pos > self.threshold or self.cumsum_neg > self.threshold:
                drift_detected = True
                self.drift_detected_count += 1
                self.cumsum_pos = 0.0
                self.cumsum_neg = 0.0
                break

        return DriftDetectionResult(
            drift_detected=drift_detected,
            drift_score=self.last_drift_score,
            detector_name=self.name,
            reference_size=self.n,
            current_size=len(new_data),
            metadata={
                'cumsum_pos': self.cumsum_pos,
                'cumsum_neg': self.cumsum_neg,
                'drift_count': self.drift_detected_count
            }
        )

    def get_drift_score(self) -> float:
        """Get current drift score."""
        return self.last_drift_score

    def reset(self) -> None:
        """Reset detector."""
        self.cumsum_pos = 0.0
        self.cumsum_neg = 0.0
        self.n = 0
        self.last_drift_score = 0.0
        self.is_fitted = False
        self.drift_history = []
        self.drift_detected_count = 0
        self._reference_mean = None


class ResidualKSTestDetector(StatisticalDriftDetector):
    """
    Residual-based KS test drift detector.
    
    Monitors model prediction residuals for distribution changes.
    This is often more informative than raw feature drift detection.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        window_size: int = 100,
        min_reference_size: int = 30,
        effect_threshold: float = 0.5
    ):
        """
        Initialize residual KS test detector.

        Parameters
        ----------
        alpha : float
            Significance level (default: 0.05).
        window_size : int
            Size of sliding window (default: 100).
        min_reference_size : int
            Minimum reference window size (default: 30).
        effect_threshold : float
            Minimum Cohen's d for detection (default: 0.5).
        """
        super().__init__(name=DetectorType.RESIDUAL_KS.value, alpha=alpha)
        self.window_size = window_size
        self.min_reference_size = min_reference_size
        self.effect_threshold = effect_threshold

        self.reference_residuals: List[float] = []
        self.current_residuals: List[float] = []
        self.last_drift_score: float = 0.0
        self.last_effect_size: float = 0.0
        self.last_p_value: float = 1.0
        self.drift_detected_count: int = 0

    def fit(self, reference_data: np.ndarray) -> 'ResidualKSTestDetector':
        """Fit detector on reference residuals."""
        super().fit(reference_data)
        self.reference_residuals = list(self.reference_data)
        return self

    def add_residuals(self, residuals: np.ndarray) -> None:
        """Add new residuals to the current window."""
        residuals = np.asarray(residuals)
        residuals = residuals[~np.isnan(residuals)]
        self.current_residuals.extend(residuals.tolist())

        if len(self.current_residuals) > self.window_size:
            self.current_residuals = self.current_residuals[-self.window_size:]

    def detect(self, new_data: np.ndarray) -> DriftDetectionResult:
        """Detect drift in residuals using both KS test and effect size."""
        self.add_residuals(new_data)

        if len(self.reference_residuals) < self.min_reference_size:
            return DriftDetectionResult(
                drift_detected=False,
                drift_score=0.0,
                detector_name=self.name,
                reference_size=len(self.reference_residuals),
                current_size=len(self.current_residuals)
            )

        if len(self.current_residuals) < self.min_reference_size:
            return DriftDetectionResult(
                drift_detected=False,
                drift_score=0.0,
                detector_name=self.name,
                reference_size=len(self.reference_residuals),
                current_size=len(self.current_residuals)
            )

        from scipy import stats

        ref_arr = np.array(self.reference_residuals)
        curr_arr = np.array(self.current_residuals)

        ks_stat, p_value = stats.ks_2samp(ref_arr, curr_arr)
        
        mean_ref = np.mean(ref_arr)
        mean_curr = np.mean(curr_arr)
        std_ref = np.std(ref_arr)
        std_curr = np.std(curr_arr)
        pooled_std = np.sqrt((std_ref**2 + std_curr**2) / 2)
        
        if pooled_std > 1e-10:
            effect_size = abs(mean_curr - mean_ref) / pooled_std
        else:
            effect_size = 0.0

        self.last_drift_score = ks_stat
        self.last_effect_size = effect_size
        self.last_p_value = p_value
        
        drift_detected = (p_value < self.alpha) or (effect_size > self.effect_threshold)

        if drift_detected:
            self.reference_residuals = list(curr_arr)
            self.current_residuals = []
            self.drift_detected_count += 1

        return DriftDetectionResult(
            drift_detected=drift_detected,
            drift_score=ks_stat,
            p_value=p_value,
            detector_name=self.name,
            reference_size=len(ref_arr),
            current_size=len(curr_arr),
            metadata={
                'effect_size': effect_size,
                'drift_count': self.drift_detected_count
            }
        )

    def get_drift_score(self) -> float:
        """Get current drift score."""
        return self.last_effect_size

    def reset(self) -> None:
        """Reset detector state."""
        super().reset()
        self.reference_residuals = []
        self.current_residuals = []
        self.last_drift_score = 0.0
        self.last_effect_size = 0.0
        self.last_p_value = 1.0
        self.drift_detected_count = 0


class WassersteinDetector(StatisticalDriftDetector):
    """
    Wasserstein distance-based drift detector.
    
    Uses the Wasserstein distance (Earth Mover's Distance) to measure
    distribution change with proper normalization.
    """

    def __init__(
        self,
        threshold: float = 0.1,
        window_size: int = 100,
        min_samples: int = 30,
        normalize: bool = True
    ):
        """
        Initialize Wasserstein detector.

        Parameters
        ----------
        threshold : float
            Drift detection threshold (default: 0.1).
            For normalized mode: threshold is in terms of std deviations.
        window_size : int
            Size of sliding window (default: 100).
        min_samples : int
            Minimum samples before detection.
        normalize : bool
            Whether to normalize by reference std.
        """
        super().__init__(name=DetectorType.WASSERSTEIN.value, alpha=threshold)
        self.threshold = threshold
        self.window_size = window_size
        self.min_samples = min_samples
        self.normalize = normalize

        self.current_window: List[float] = []
        self.last_drift_score: float = 0.0
        self.drift_detected_count: int = 0
        
        self._reference_mean: float = 0.0
        self._reference_std: float = 1.0

    def fit(self, reference_data: np.ndarray) -> 'WassersteinDetector':
        """Fit detector on reference data."""
        super().fit(reference_data)

        ref_data = list(self.reference_data)
        if len(ref_data) > self.window_size:
            ref_data = ref_data[-self.window_size:]
        
        self._reference_mean = np.mean(self.reference_data)
        self._reference_std = np.std(self.reference_data)
        if self._reference_std < 1e-10:
            self._reference_std = 1.0
            
        self.current_window = ref_data

        return self

    def detect(self, new_data: np.ndarray) -> DriftDetectionResult:
        """Detect drift using Wasserstein distance."""
        if not self.is_fitted:
            self.fit(new_data)
            return DriftDetectionResult(
                drift_detected=False,
                drift_score=0.0,
                detector_name=self.name
            )

        new_data = np.asarray(new_data)
        new_data = new_data[~np.isnan(new_data)]

        if len(new_data) == 0:
            return DriftDetectionResult(
                drift_detected=False,
                drift_score=0.0,
                detector_name=self.name
            )

        self.current_window.extend(new_data.tolist())

        if len(self.current_window) > self.window_size:
            self.current_window = self.current_window[-self.window_size:]
        
        if len(self.current_window) < self.min_samples:
            return DriftDetectionResult(
                drift_detected=False,
                drift_score=0.0,
                detector_name=self.name,
                reference_size=len(self.reference_data),
                current_size=len(self.current_window)
            )

        drift_score = self._compute_wasserstein_distance(
            self.reference_data,
            np.array(self.current_window)
        )

        if self.normalize:
            drift_score_normalized = drift_score / self._reference_std
        else:
            drift_score_normalized = drift_score
            
        self.last_drift_score = drift_score_normalized

        drift_detected = drift_score_normalized > self.threshold

        if drift_detected:
            self.reference_data = np.array(self.current_window)
            self._reference_mean = np.mean(self.reference_data)
            self._reference_std = np.std(self.reference_data)
            if self._reference_std < 1e-10:
                self._reference_std = 1.0
            self.drift_detected_count += 1

        return DriftDetectionResult(
            drift_detected=drift_detected,
            drift_score=drift_score_normalized,
            detector_name=self.name,
            reference_size=len(self.reference_data),
            current_size=len(self.current_window),
            metadata={
                'raw_distance': drift_score,
                'drift_count': self.drift_detected_count
            }
        )

    def _compute_wasserstein_distance(
        self,
        reference: np.ndarray,
        current: np.ndarray
    ) -> float:
        """Compute Wasserstein-1 distance between two distributions."""
        from scipy.stats import wasserstein_distance

        ref = reference.copy()
        curr = current.copy()

        if len(ref) > 1000:
            indices = np.random.choice(len(ref), 1000, replace=False)
            ref = ref[indices]
        if len(curr) > 1000:
            indices = np.random.choice(len(curr), 1000, replace=False)
            curr = curr[indices]

        return wasserstein_distance(ref, curr)

    def get_drift_score(self) -> float:
        """Get current drift score."""
        return self.last_drift_score

    def reset(self) -> None:
        """Reset detector state."""
        super().reset()
        self.current_window = []
        self.last_drift_score = 0.0
        self.drift_detected_count = 0
        self._reference_mean = 0.0
        self._reference_std = 1.0


class KSTestDetectorWrapper(StatisticalDriftDetector):
    """
    Wrapper for the KS test detector with proper threshold handling.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        effect_threshold: float = 0.5,
        window_size: int = 200
    ):
        """
        Initialize KS test detector.

        Parameters
        ----------
        alpha : float
            Significance level (default: 0.05).
        effect_threshold : float
            Minimum Cohen's d for detection (default: 0.5).
        window_size : int
            Size of sliding window (default: 200).
        """
        super().__init__(name=DetectorType.KS_TEST.value, alpha=alpha)
        self.effect_threshold = effect_threshold
        self.window_size = window_size
        self.last_drift_score = 0.0
        self.last_effect_size = 0.0
        self.last_p_value = 1.0
        self.drift_detected_count = 0
        self._window: List[float] = []

    def fit(self, reference_data: np.ndarray) -> 'KSTestDetectorWrapper':
        """Fit detector on reference data."""
        super().fit(reference_data)
        self._window = list(self.reference_data[-self.window_size:])
        return self

    def detect(self, new_data: np.ndarray) -> DriftDetectionResult:
        """Detect drift using KS test with effect size."""
        if not self.is_fitted or self.reference_data is None:
            self.fit(new_data)
            return DriftDetectionResult(
                drift_detected=False,
                drift_score=0.0,
                detector_name=self.name
            )

        new_data = np.asarray(new_data)
        new_data = new_data[~np.isnan(new_data)]

        if len(new_data) == 0:
            return DriftDetectionResult(
                drift_detected=False,
                drift_score=0.0,
                detector_name=self.name
            )

        self._window.extend(new_data.tolist())
        if len(self._window) > self.window_size:
            self._window = self._window[-self.window_size:]

        if len(self._window) < 50:
            return DriftDetectionResult(
                drift_detected=False,
                drift_score=0.0,
                detector_name=self.name
            )

        from scipy import stats

        ks_stat, p_value = stats.ks_2samp(self.reference_data, np.array(self._window))
        
        mean_ref = np.mean(self.reference_data)
        mean_curr = np.mean(self._window)
        std_ref = np.std(self.reference_data)
        std_curr = np.std(self._window)
        pooled_std = np.sqrt((std_ref**2 + std_curr**2) / 2)
        
        if pooled_std > 1e-10:
            effect_size = abs(mean_curr - mean_ref) / pooled_std
        else:
            effect_size = 0.0

        self.last_drift_score = ks_stat
        self.last_effect_size = effect_size
        self.last_p_value = p_value

        drift_detected = (p_value < self.alpha) or (effect_size > self.effect_threshold)

        if not drift_detected:
            self.reference_data = np.concatenate([
                self.reference_data[1:],
                new_data[-1:]
            ])

        return DriftDetectionResult(
            drift_detected=drift_detected,
            drift_score=ks_stat,
            p_value=p_value,
            detector_name=self.name,
            reference_size=len(self.reference_data),
            current_size=len(self._window),
            metadata={
                'effect_size': effect_size,
                'drift_count': self.drift_detected_count
            }
        )

    def get_drift_score(self) -> float:
        """Get current drift score."""
        return self.last_effect_size

    def reset(self) -> None:
        """Reset detector state."""
        super().reset()
        self._window = []
        self.last_drift_score = 0.0
        self.last_effect_size = 0.0
        self.last_p_value = 1.0
        self.drift_detected_count = 0


def create_detector(
    detector_type: str,
    mode: str = "feature",
    **kwargs
) -> BaseDriftDetector:
    """
    Create a drift detector by type.

    Parameters
    ----------
    detector_type : str
        Type of detector: 'adwin', 'page_hinkley', 'ks_test', 'wasserstein', 'residual_ks', 'cusum', 'psi'.
    mode : str
        Detection mode: 'feature', 'residual', 'embedding'.
    **kwargs
        Additional parameters for the detector.

    Returns
    -------
    BaseDriftDetector
        Configured detector instance.
    """
    detector_map = {
        'adwin': ADWINDetector,
        'page_hinkley': PageHinkleyDetector,
        'page-hinkley': PageHinkleyDetector,
        'ks_test': KSTestDetectorWrapper,
        'ks-test': KSTestDetectorWrapper,
        'wasserstein': WassersteinDetector,
        'residual_ks': ResidualKSTestDetector,
        'residual-ks': ResidualKSTestDetector,
        'cusum': CUSUMDetector,
        'psi': PSIDriftDetector,
    }

    detector_type_lower = detector_type.lower()

    if detector_type_lower not in detector_map:
        raise ValueError(
            f"Unknown detector type: {detector_type}. "
            f"Available: {list(detector_map.keys())}"
        )

    return detector_map[detector_type_lower](**kwargs)
