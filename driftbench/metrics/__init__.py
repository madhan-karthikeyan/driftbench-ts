"""
Metrics module.

This module provides metrics for evaluating forecasting models
and drift detection performance.
"""

from driftbench.metrics.forecasting_metrics import (
    mae,
    rmse,
    smape,
    mape,
    compute_all_metrics
)

from driftbench.metrics.extended_metrics import (
    detection_delay,
    false_positive_rate,
    recovery_time,
    performance_degradation_area,
    correlation_drift_magnitude_error,
    compute_drift_benchmark_metrics,
    compute_extended_metrics,
    DriftEvent
)

from driftbench.metrics.stat_tests import (
    diebold_mariano_test,
    paired_bootstrap_test,
    wilcoxon_signed_rank_test,
    compare_models,
    compare_retraining_strategies,
    compare_drift_detectors,
    TestResult
)

__all__ = [
    # Forecasting metrics
    'mae',
    'rmse',
    'smape',
    'mape',
    'compute_all_metrics',
    # Extended metrics
    'detection_delay',
    'false_positive_rate',
    'recovery_time',
    'performance_degradation_area',
    'correlation_drift_magnitude_error',
    'compute_drift_benchmark_metrics',
    'compute_extended_metrics',
    'DriftEvent',
    # Statistical tests
    'diebold_mariano_test',
    'paired_bootstrap_test',
    'wilcoxon_signed_rank_test',
    'compare_models',
    'compare_retraining_strategies',
    'compare_drift_detectors',
    'TestResult',
]
