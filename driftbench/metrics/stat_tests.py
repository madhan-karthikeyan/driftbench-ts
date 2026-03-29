"""
Statistical testing framework for model comparison.

This module provides statistical tests for comparing forecasting models,
retraining strategies, and drift detection methods.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple, Union
from scipy import stats
from scipy.stats import norm
from dataclasses import dataclass


@dataclass
class TestResult:
    """Result of a statistical test."""
    statistic: float
    p_value: float
    significant: bool
    confidence_level: float = 0.05
    test_name: str = ""
    alternative: str = ""
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


def diebold_mariano_test(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    h: int = 1,
    crit: str = "MAE"
) -> TestResult:
    """
    Diebold-Mariano test for comparing predictive accuracy.

    Tests whether one forecast is significantly more accurate than another.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred1 : np.ndarray
        Predictions from model 1.
    y_pred2 : np.ndarray
        Predictions from model 2.
    h : int
        Forecast horizon (default: 1).
    crit : str
        Loss criterion: "MAE", "MSE", or "MAPE" (default: "MAE").

    Returns
    -------
    TestResult
        Test result with statistic and p-value.
    """
    n = len(y_true)

    # Compute forecast errors
    e1 = y_true - y_pred1
    e2 = y_true - y_pred2

    # Compute loss differential
    if crit == "MAE":
        d = np.abs(e1) - np.abs(e2)
    elif crit == "MSE":
        d = e1**2 - e2**2
    elif crit == "MAPE":
        # Avoid division by zero
        mask = y_true != 0
        d = np.abs(e1[mask] / y_true[mask]) - np.abs(e2[mask] / y_true[mask])
    else:
        raise ValueError(f"Unknown criterion: {crit}")

    # Mean and variance of loss differential
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)

    # Account for serial correlation (simplified version)
    # For h > 1, we should use autocovariance
    gamma = 0
    for k in range(1, h):
        gamma += np.cov(d[k:], d[:-k])[0, 1]

    # Variance of mean (with correction for serial correlation)
    var_mean_d = (var_d + 2 * gamma) / n

    if var_mean_d <= 0:
        return TestResult(
            statistic=np.nan,
            p_value=np.nan,
            significant=False,
            test_name="Diebold-Mariano",
            details={"error": "Non-positive variance"}
        )

    # DM statistic
    dm_stat = mean_d / np.sqrt(var_mean_d)

    # Two-sided p-value (using normal approximation)
    p_value = 2 * (1 - norm.cdf(np.abs(dm_stat)))

    return TestResult(
        statistic=dm_stat,
        p_value=p_value,
        significant=p_value < 0.05,
        test_name="Diebold-Mariano",
        alternative=f"Model 1 {'<' if dm_stat < 0 else '>'} Model 2",
        details={
            "mean_loss_diff": mean_d,
            "var_loss_diff": var_d,
            "criterion": crit,
            "horizon": h,
            "n_observations": n
        }
    )


def paired_bootstrap_test(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    n_bootstrap: int = 1000,
    crit: str = "MAE",
    confidence_level: float = 0.05
) -> TestResult:
    """
    Paired bootstrap test for comparing predictive accuracy.

    Uses bootstrap resampling to compute confidence interval
    for the difference in performance.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred1 : np.ndarray
        Predictions from model 1.
    y_pred2 : np.ndarray
        Predictions from model 2.
    n_bootstrap : int
        Number of bootstrap iterations (default: 1000).
    crit : str
        Loss criterion: "MAE", "MSE", or "RMSE" (default: "MAE").
    confidence_level : float
        Confidence level for interval (default: 0.05).

    Returns
    -------
    TestResult
        Test result with bootstrap confidence interval.
    """
    n = len(y_true)

    # Compute errors for each model
    e1 = y_true - y_pred1
    e2 = y_true - y_pred2

    # Compute loss for each model
    if crit == "MAE":
        loss1 = np.abs(e1)
        loss2 = np.abs(e2)
    elif crit in ("MSE", "RMSE"):
        loss1 = e1**2
        loss2 = e2**2
    else:
        raise ValueError(f"Unknown criterion: {crit}")

    # Observed difference in mean loss
    observed_diff = np.mean(loss1 - loss2)

    # Bootstrap
    bootstrap_diffs = []
    rng = np.random.default_rng()

    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = rng.integers(0, n, size=n)
        boot_diff = np.mean(loss1[indices] - loss2[indices])
        bootstrap_diffs.append(boot_diff)

    bootstrap_diffs = np.array(bootstrap_diffs)

    # Compute p-value (two-sided)
    p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))

    # Confidence interval
    alpha = confidence_level
    ci_lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))

    return TestResult(
        statistic=observed_diff,
        p_value=p_value,
        significant=p_value < confidence_level,
        confidence_level=confidence_level,
        test_name="Paired Bootstrap",
        alternative="Model 1 != Model 2" if p_value < confidence_level else "Model 1 = Model 2",
        details={
            "mean_loss_diff": observed_diff,
            "bootstrap_mean": np.mean(bootstrap_diffs),
            "bootstrap_std": np.std(bootstrap_diffs),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_bootstrap": n_bootstrap,
            "criterion": crit
        }
    )


def wilcoxon_signed_rank_test(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    alternative: str = "two-sided"
) -> TestResult:
    """
    Wilcoxon signed-rank test for paired samples.

    Non-parametric test for comparing paired samples.
    More robust than t-test for non-normal distributions.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred1 : np.ndarray
        Predictions from model 1.
    y_pred2 : np.ndarray
        Predictions from model 2.
    alternative : str
        Alternative hypothesis: "two-sided", "less", or "greater".

    Returns
    -------
    TestResult
        Test result with statistic and p-value.
    """
    # Compute errors
    e1 = y_true - y_pred1
    e2 = y_true - y_pred2

    # Compute absolute errors
    loss1 = np.abs(e1)
    loss2 = np.abs(e2)

    # Compute differences
    diff = loss1 - loss2

    # Remove zeros (ties)
    diff = diff[diff != 0]

    if len(diff) < 10:
        return TestResult(
            statistic=np.nan,
            p_value=np.nan,
            significant=False,
            test_name="Wilcoxon Signed-Rank",
            details={"warning": "Too few non-zero differences"}
        )

    # Perform test
    statistic, p_value = stats.wilcoxon(
        diff,
        alternative=alternative
    )

    return TestResult(
        statistic=statistic,
        p_value=p_value,
        significant=p_value < 0.05,
        test_name="Wilcoxon Signed-Rank",
        alternative=alternative,
        details={
            "n_pairs": len(diff),
            "median_diff": np.median(diff)
        }
    )


def compare_models(
    predictions_dict: Dict[str, np.ndarray],
    y_true: np.ndarray,
    test_type: str = "diebold_mariano",
    crit: str = "MAE",
    confidence_level: float = 0.05
) -> Dict[str, Any]:
    """
    Compare multiple models using statistical tests.

    Parameters
    ----------
    predictions_dict : dict
        Dictionary mapping model names to predictions.
    y_true : np.ndarray
        True values.
    test_type : str
        Test to use: "diebold_mariano", "bootstrap", or "wilcoxon".
    crit : str
        Loss criterion for DM and bootstrap tests.
    confidence_level : float
        Confidence level for tests.

    Returns
    -------
    dict
        Dictionary with pairwise comparison results and summary.
    """
    model_names = list(predictions_dict.keys())

    if len(model_names) < 2:
        return {"error": "At least two models required"}

    results = {}

    # Pairwise comparisons
    for i, model1 in enumerate(model_names):
        for model2 in model_names[i+1:]:
            key = f"{model1}_vs_{model2}"

            if test_type == "diebold_mariano":
                test_result = diebold_mariano_test(
                    y_true,
                    predictions_dict[model1],
                    predictions_dict[model2],
                    crit=crit
                )
            elif test_type == "bootstrap":
                test_result = paired_bootstrap_test(
                    y_true,
                    predictions_dict[model1],
                    predictions_dict[model2],
                    crit=crit,
                    confidence_level=confidence_level
                )
            elif test_type == "wilcoxon":
                test_result = wilcoxon_signed_rank_test(
                    y_true,
                    predictions_dict[model1],
                    predictions_dict[model2]
                )
            else:
                raise ValueError(f"Unknown test type: {test_type}")

            results[key] = {
                "statistic": test_result.statistic,
                "p_value": test_result.p_value,
                "significant": test_result.significant,
                "winner": model1 if test_result.statistic < 0 else model2 if test_result.statistic > 0 else None,
                "details": test_result.details
            }

    # Summary
    win_counts = {m: 0 for m in model_names}

    for result in results.values():
        if result["significant"] and result["winner"]:
            win_counts[result["winner"]] += 1

    summary = {
        "test_type": test_type,
        "criterion": crit,
        "win_counts": win_counts,
        "best_model": max(win_counts, key=win_counts.get) if any(win_counts.values()) else None
    }

    return {
        "pairwise_comparisons": results,
        "summary": summary
    }


def compare_retraining_strategies(
    strategy_results: Dict[str, Dict[str, float]],
    metric: str = "mae"
) -> Dict[str, Any]:
    """
    Compare retraining strategies using statistical tests.

    Parameters
    ----------
    strategy_results : dict
        Dictionary mapping strategy names to metric dictionaries.
        Each value should have key 'metric' with the metric value.
    metric : str
        Metric to compare.

    Returns
    -------
    dict
        Comparison results.
    """
    strategies = list(strategy_results.keys())

    if len(strategies) < 2:
        return {"error": "At least two strategies required"}

    # Get metric values for each strategy
    metrics = {s: strategy_results[s].get(metric, np.nan) for s in strategies}

    # Simple comparison: ranks
    sorted_strategies = sorted(metrics.items(), key=lambda x: x[1])

    ranks = {}
    current_rank = 1
    for i, (strategy, value) in enumerate(sorted_strategies):
        if i > 0 and value == sorted_strategies[i-1][1]:
            ranks[strategy] = ranks[sorted_strategies[i-1][0]]
        else:
            ranks[strategy] = current_rank
        current_rank += 1

    return {
        "strategies": strategies,
        "metrics": metrics,
        "ranks": ranks,
        "best_strategy": sorted_strategies[0][0] if sorted_strategies else None,
        "best_value": sorted_strategies[0][1] if sorted_strategies else None
    }


def compare_drift_detectors(
    detector_results: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compare drift detection methods.

    Parameters
    ----------
    detector_results : dict
        Dictionary mapping detector names to their results.
        Should include: detection_rate, false_positive_rate, detection_delay.

    Returns
    -------
    dict
        Comparison results with rankings.
    """
    detectors = list(detector_results.keys())

    if len(detectors) < 2:
        return {"error": "At least two detectors required"}

    # Extract metrics
    detection_rates = {d: detector_results[d].get('detection_rate', 0) for d in detectors}
    fprs = {d: detector_results[d].get('false_positive_rate', 1) for d in detectors}
    delays = {d: detector_results[d].get('mean_detection_delay_steps', np.inf) for d in detectors}

    # Rank detectors
    # Higher detection rate is better
    dr_sorted = sorted(detection_rates.items(), key=lambda x: x[1], reverse=True)
    dr_ranks = {d: i+1 for i, (d, _) in enumerate(dr_sorted)}

    # Lower FPR is better
    fpr_sorted = sorted(fprs.items(), key=lambda x: x[1])
    fpr_ranks = {d: i+1 for i, (d, _) in enumerate(fpr_sorted)}

    # Lower delay is better
    delay_sorted = sorted(delays.items(), key=lambda x: x[1])
    delay_ranks = {d: i+1 for i, (d, _) in enumerate(delay_sorted)}

    # Combined ranking (average rank)
    combined_ranks = {}
    for d in detectors:
        avg_rank = (dr_ranks[d] + fpr_ranks[d] + delay_ranks[d]) / 3
        combined_ranks[d] = avg_rank

    best_detector = min(combined_ranks, key=combined_ranks.get)

    return {
        "detectors": detectors,
        "detection_rates": detection_rates,
        "false_positive_rates": fprs,
        "detection_delays": delays,
        "detection_rate_ranks": dr_ranks,
        "fpr_ranks": fpr_ranks,
        "delay_ranks": delay_ranks,
        "combined_ranks": combined_ranks,
        "best_detector": best_detector
    }
