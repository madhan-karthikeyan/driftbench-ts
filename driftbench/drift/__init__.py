"""
Drift detection module.

This module provides drift detection algorithms and utilities.
"""

from driftbench.drift.base_detector import (
    BaseDriftDetector,
    StatisticalDriftDetector,
    DriftDetectionResult,
    DetectorType,
    DetectionMode
)

from driftbench.drift.detectors import (
    ADWINDetector,
    PageHinkleyDetector,
    ResidualKSTestDetector,
    WassersteinDetector,
    KSTestDetectorWrapper,
    create_detector
)

from driftbench.drift.ks_test import (
    ks_test,
    compute_ks_drift_score,
    detect_drift_in_window,
    DriftDetector
)

__all__ = [
    'BaseDriftDetector',
    'StatisticalDriftDetector',
    'DriftDetectionResult',
    'DetectorType',
    'DetectionMode',
    'ADWINDetector',
    'PageHinkleyDetector',
    'ResidualKSTestDetector',
    'WassersteinDetector',
    'KSTestDetectorWrapper',
    'create_detector',
    'ks_test',
    'compute_ks_drift_score',
    'detect_drift_in_window',
    'DriftDetector',
]
