"""
Advanced retraining policies for model adaptation.

This module implements various retraining strategies that can be used
with the rolling window simulator for adaptive model updates.

FIXED: Added proper cooldown mechanism and reset logic to prevent
every-step retraining.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class RetrainingPolicyType(Enum):
    """Enumeration of available retraining policies."""
    FIXED_SCHEDULE = "fixed_schedule"
    DRIFT_TRIGGERED = "drift_triggered"
    BUDGET_AWARE = "budget_aware"
    ERROR_THRESHOLD_BASED = "error_threshold_based"
    NO_RETRAINING = "no_retraining"


@dataclass
class RetrainingState:
    """State tracking for retraining decisions."""
    current_step: int = 0
    total_retrains: int = 0
    retraining_timestamps: List[datetime] = field(default_factory=list)
    drift_detected: bool = False
    drift_scores: List[float] = field(default_factory=list)
    recent_errors: List[float] = field(default_factory=list)
    last_retrain_step: int = -1
    steps_since_retrain: int = 0
    compute_budget_used: float = 0.0
    current_date: Optional[pd.Timestamp] = None
    
    consecutive_drift_count: int = 0
    last_drift_step: int = -1


@dataclass
class RetrainingDecision:
    """Result of retraining decision."""
    should_retrain: bool
    reason: str
    confidence: float = 1.0


class BaseRetrainingPolicy(ABC):
    """
    Abstract base class for retraining policies.
    
    FIXED: All policies now properly implement cooldown mechanisms
    and include the last_retrain_step in state for cooldown checking.
    """

    def __init__(self, policy_type: str = "base", min_steps_between_retrain: int = 5):
        """
        Initialize the policy.

        Parameters
        ----------
        policy_type : str
            Type identifier for the policy.
        min_steps_between_retrain : int
            Minimum steps between retraining (cooldown period).
            Default: 5 steps.
        """
        self.policy_type = policy_type
        self.state = RetrainingState()
        self.min_steps_between_retrain = min_steps_between_retrain

    @abstractmethod
    def should_retrain(self, current_state: RetrainingState) -> RetrainingDecision:
        """
        Determine if model should be retrained.

        Parameters
        ----------
        current_state : RetrainingState
            Current state information.

        Returns
        -------
        RetrainingDecision
            Decision with should_retrain flag, reason, and confidence.
        """
        pass

    def _check_cooldown(self, current_step: int) -> bool:
        """
        Check if we're in cooldown period.
        
        FIXED: This is the key fix - ensure minimum gap between retrains.
        """
        steps_since = current_step - self.state.last_retrain_step
        return steps_since < self.min_steps_between_retrain

    def reset(self) -> None:
        """Reset policy to initial state."""
        self.state = RetrainingState()

    def get_config(self) -> Dict[str, Any]:
        """Get policy configuration."""
        return {
            "policy_type": self.policy_type,
            "min_steps_between_retrain": self.min_steps_between_retrain
        }


class NoRetrainingPolicy(BaseRetrainingPolicy):
    """
    No retraining policy - train once and never retrain.
    
    This is a critical baseline for comparing retraining strategies.
    """

    def __init__(self):
        super().__init__(policy_type=RetrainingPolicyType.NO_RETRAINING.value, min_steps_between_retrain=999999)
        self._trained_once = False

    def should_retrain(self, current_state: RetrainingState) -> RetrainingDecision:
        """Never retrain."""
        if not self._trained_once:
            self._trained_once = True
            return RetrainingDecision(
                should_retrain=True,
                reason="initial_training",
                confidence=1.0
            )
        return RetrainingDecision(
            should_retrain=False,
            reason="no_retraining_policy",
            confidence=1.0
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "policy_type": self.policy_type,
            "description": "Train once, never retrain"
        }


class FixedSchedulePolicy(BaseRetrainingPolicy):
    """
    Fixed schedule retraining policy.
    
    FIXED: Now respects min_steps_between_retrain for proper cooldown.
    """

    def __init__(
        self,
        retrain_interval: int = 1,
        retrain_every_n_steps: Optional[int] = None,
        min_steps_between_retrain: int = 1
    ):
        """
        Initialize fixed schedule policy.

        Parameters
        ----------
        retrain_interval : int
            Number of steps between retraining (default: 1 = every step).
        retrain_every_n_steps : int, optional
            Alternative: retrain every N steps.
        min_steps_between_retrain : int
            Minimum steps between retraining (cooldown).
        """
        super().__init__(
            policy_type=RetrainingPolicyType.FIXED_SCHEDULE.value,
            min_steps_between_retrain=min_steps_between_retrain
        )
        self.retrain_interval = retrain_interval
        self.retrain_every_n_steps = retrain_every_n_steps

    def should_retrain(self, current_state: RetrainingState) -> RetrainingDecision:
        """Determine if retraining is needed based on schedule."""
        if self._check_cooldown(current_state.current_step):
            return RetrainingDecision(
                should_retrain=False,
                reason="cooldown_period",
                confidence=1.0
            )

        step = current_state.current_step

        if self.retrain_every_n_steps:
            if step % self.retrain_every_n_steps == 0:
                return RetrainingDecision(
                    should_retrain=True,
                    reason=f"fixed_schedule_every_{self.retrain_every_n_steps}_steps",
                    confidence=1.0
                )
        elif step % self.retrain_interval == 0:
            return RetrainingDecision(
                should_retrain=True,
                reason=f"fixed_schedule_interval_{self.retrain_interval}",
                confidence=1.0
            )

        return RetrainingDecision(
            should_retrain=False,
            reason="not_scheduled",
            confidence=1.0
        )

    def get_config(self) -> Dict[str, Any]:
        """Get policy configuration."""
        return {
            "policy_type": self.policy_type,
            "retrain_interval": self.retrain_interval,
            "retrain_every_n_steps": self.retrain_every_n_steps,
            "min_steps_between_retrain": self.min_steps_between_retrain
        }


class DriftTriggeredPolicy(BaseRetrainingPolicy):
    """
    Drift-triggered retraining policy.
    
    FIXED: 
    - Added proper cooldown mechanism
    - Fixed consecutive drift counter logic
    - Added decay for consecutive count
    """

    def __init__(
        self,
        drift_threshold: float = 0.5,
        min_steps_between_retrain: int = 5,
        require_consecutive: int = 2,
        decay_consecutive: bool = True,
        consecutive_decay_rate: float = 0.5
    ):
        """
        Initialize drift-triggered policy.

        Parameters
        ----------
        drift_threshold : float
            Drift score threshold for triggering retraining (default: 0.5).
            This is the Cohen's d effect size threshold.
        min_steps_between_retrain : int
            Minimum steps between retraining to avoid thrashing (default: 5).
        require_consecutive : int
            Number of consecutive drift detections to trigger retrain (default: 2).
        decay_consecutive : bool
            Whether to decay consecutive count over time (default: True).
        consecutive_decay_rate : float
            How much to decay consecutive count per step without drift (default: 0.5).
        """
        super().__init__(
            policy_type=RetrainingPolicyType.DRIFT_TRIGGERED.value,
            min_steps_between_retrain=min_steps_between_retrain
        )
        self.drift_threshold = drift_threshold
        self.require_consecutive = require_consecutive
        self.decay_consecutive = decay_consecutive
        self.consecutive_decay_rate = consecutive_decay_rate
        self._consecutive_drift_count: float = 0.0

    def should_retrain(self, current_state: RetrainingState) -> RetrainingDecision:
        """Determine if retraining is needed based on drift detection."""
        step = current_state.current_step
        
        if self._check_cooldown(step):
            return RetrainingDecision(
                should_retrain=False,
                reason="cooldown_period",
                confidence=1.0
            )

        drift_detected = current_state.drift_detected
        
        if drift_detected:
            self._consecutive_drift_count += 1.0
            current_state.consecutive_drift_count = int(self._consecutive_drift_count)
        else:
            if self.decay_consecutive:
                self._consecutive_drift_count = max(0, self._consecutive_drift_count - self.consecutive_decay_rate)
            current_state.consecutive_drift_count = int(self._consecutive_drift_count)

        drift_score = current_state.drift_scores[-1] if current_state.drift_scores else 0.0
        score_triggers = drift_score > self.drift_threshold

        if score_triggers and self._consecutive_drift_count >= self.require_consecutive:
            self._consecutive_drift_count = 0.0
            return RetrainingDecision(
                should_retrain=True,
                reason=f"drift_detected_consecutive_{int(self._consecutive_drift_count + 1)}_threshold_{drift_score:.3f}",
                confidence=min(1.0, self._consecutive_drift_count / self.require_consecutive)
            )

        return RetrainingDecision(
            should_retrain=False,
            reason=f"no_drift_consecutive_{int(self._consecutive_drift_count)}/{self.require_consecutive}",
            confidence=0.5
        )

    def reset(self) -> None:
        """Reset policy to initial state."""
        super().reset()
        self._consecutive_drift_count = 0.0

    def get_config(self) -> Dict[str, Any]:
        """Get policy configuration."""
        return {
            "policy_type": self.policy_type,
            "drift_threshold": self.drift_threshold,
            "min_steps_between_retrain": self.min_steps_between_retrain,
            "require_consecutive": self.require_consecutive,
            "decay_consecutive": self.decay_consecutive,
            "consecutive_decay_rate": self.consecutive_decay_rate
        }


class BudgetAwarePolicy(BaseRetrainingPolicy):
    """
    Budget-aware retraining policy.
    
    FIXED: Properly handles cooldown and budget constraints.
    """

    def __init__(
        self,
        max_retrains_per_year: int = 4,
        initial_budget: Optional[int] = None,
        min_steps_between_retrain: int = 5,
        require_drift: bool = True
    ):
        """
        Initialize budget-aware policy.

        Parameters
        ----------
        max_retrains_per_year : int
            Maximum retrainings allowed per year (default: 4 = quarterly).
        initial_budget : int, optional
            Initial budget (defaults to max_retrains_per_year).
        min_steps_between_retrain : int
            Minimum steps between retraining (cooldown).
        require_drift : bool
            Whether to require drift detection (default: True).
        """
        super().__init__(
            policy_type=RetrainingPolicyType.BUDGET_AWARE.value,
            min_steps_between_retrain=min_steps_between_retrain
        )
        self.max_retrains_per_year = max_retrains_per_year
        self.initial_budget = initial_budget or max_retrains_per_year
        self.require_drift = require_drift

        self.budget = float(self.initial_budget)
        self.last_refill_step = 0

        self.budget_history: List[Dict[str, Any]] = []

    def should_retrain(self, current_state: RetrainingState) -> RetrainingDecision:
        """Determine if retraining is needed within budget constraints."""
        step = current_state.current_step
        
        if self._check_cooldown(step):
            return RetrainingDecision(
                should_retrain=False,
                reason="cooldown_period",
                confidence=1.0
            )

        has_budget = self.budget >= 1.0
        drift_triggered = current_state.drift_detected

        if self.require_drift and not drift_triggered:
            return RetrainingDecision(
                should_retrain=False,
                reason="no_drift_no_budget_spend",
                confidence=1.0
            )

        if has_budget and (not self.require_drift or drift_triggered):
            self.budget -= 1.0
            self.budget_history.append({
                'step': step,
                'budget_before': self.budget + 1.0,
                'budget_after': self.budget,
                'drift_detected': drift_triggered
            })
            return RetrainingDecision(
                should_retrain=True,
                reason=f"budget_aware_drift_{drift_triggered}_budget_{self.budget + 1:.0f}",
                confidence=1.0
            )

        return RetrainingDecision(
            should_retrain=False,
            reason=f"no_budget_{self.budget:.1f}/{self.initial_budget}",
            confidence=1.0
        )

    def get_config(self) -> Dict[str, Any]:
        """Get policy configuration."""
        return {
            "policy_type": self.policy_type,
            "max_retrains_per_year": self.max_retrains_per_year,
            "initial_budget": self.initial_budget,
            "current_budget": self.budget,
            "min_steps_between_retrain": self.min_steps_between_retrain,
            "require_drift": self.require_drift
        }

    def reset(self) -> None:
        """Reset policy to initial state."""
        super().reset()
        self.budget = float(self.initial_budget)
        self.last_refill_step = 0
        self.budget_history = []


class ErrorThresholdBasedPolicy(BaseRetrainingPolicy):
    """
    Error-threshold-based retraining policy.
    
    FIXED: Added cooldown and improved baseline computation.
    """

    def __init__(
        self,
        error_threshold: float = 0.1,
        error_metric: str = "mae",
        window_size: int = 10,
        min_error_exceedences: int = 2,
        relative_to_baseline: bool = True,
        baseline_error: Optional[float] = None,
        min_steps_between_retrain: int = 5
    ):
        """
        Initialize error-threshold-based policy.

        Parameters
        ----------
        error_threshold : float
            Error threshold for triggering retraining (default: 0.1 = 10% increase).
        error_metric : str
            Which error metric to use (default: 'mae').
        window_size : int
            Number of recent errors to consider (default: 10).
        min_error_exceedences : int
            Minimum number of times error must exceed threshold (default: 2).
        relative_to_baseline : bool
            Whether threshold is relative to baseline error (default: True).
        baseline_error : float, optional
            Baseline error for relative threshold. If None, computed from initial window.
        min_steps_between_retrain : int
            Minimum steps between retraining (cooldown).
        """
        super().__init__(
            policy_type=RetrainingPolicyType.ERROR_THRESHOLD_BASED.value,
            min_steps_between_retrain=min_steps_between_retrain
        )
        self.error_threshold = error_threshold
        self.error_metric = error_metric
        self.window_size = window_size
        self.min_error_exceedences = min_error_exceedences
        self.relative_to_baseline = relative_to_baseline
        self.baseline_error = baseline_error

        self._error_exceedence_count: float = 0.0
        self._baseline_computed = False
        self.steps_since_retrain = 0

    def should_retrain(self, current_state: RetrainingState) -> RetrainingDecision:
        """Determine if retraining is needed based on error threshold."""
        errors = current_state.recent_errors

        if len(errors) == 0:
            return RetrainingDecision(
                should_retrain=False,
                reason="no_errors_yet",
                confidence=1.0
            )

        if self._check_cooldown(current_state.current_step):
            return RetrainingDecision(
                should_retrain=False,
                reason="cooldown_period",
                confidence=1.0
            )

        if self.relative_to_baseline and not self._baseline_computed:
            if len(errors) >= self.window_size:
                self.baseline_error = np.mean(errors[:self.window_size])
                self._baseline_computed = True

        effective_threshold = self.error_threshold
        if self.relative_to_baseline and self.baseline_error is not None:
            effective_threshold = self.baseline_error * (1 + self.error_threshold)

        recent_errors = errors[-self.window_size:] if len(errors) >= self.window_size else errors
        exceedences = sum(1 for e in recent_errors if e > effective_threshold)

        if exceedences >= self.min_error_exceedences:
            self._error_exceedence_count += float(exceedences)
            current_state.recent_errors = []
            return RetrainingDecision(
                should_retrain=True,
                reason=f"error_threshold_exceeded_{exceedences}/{self.min_error_exceedences}",
                confidence=min(1.0, exceedences / (self.min_error_exceedences * 2))
            )

        self._error_exceedence_count = max(0, self._error_exceedence_count - 0.5)

        return RetrainingDecision(
            should_retrain=False,
            reason=f"error_normal_exceedences_{exceedences}/{self.min_error_exceedences}",
            confidence=0.5
        )

    def get_config(self) -> Dict[str, Any]:
        """Get policy configuration."""
        return {
            "policy_type": self.policy_type,
            "error_threshold": self.error_threshold,
            "error_metric": self.error_metric,
            "window_size": self.window_size,
            "min_error_exceedences": self.min_error_exceedences,
            "relative_to_baseline": self.relative_to_baseline,
            "baseline_error": self.baseline_error,
            "min_steps_between_retrain": self.min_steps_between_retrain
        }

    def reset(self) -> None:
        """Reset policy to initial state."""
        super().reset()
        self._error_exceedence_count = 0.0
        self._baseline_computed = False
        self.steps_since_retrain = 0


class RetrainingSimulator:
    """
    Simulator for managing model retraining with various policies.
    
    FIXED: Properly tracks last_retrain_step and updates state correctly.
    """

    def __init__(self, policy: BaseRetrainingPolicy):
        """
        Initialize the retraining simulator.

        Parameters
        ----------
        policy : BaseRetrainingPolicy
            The retraining policy to use.
        """
        self.policy = policy
        self.retraining_log: List[Dict[str, Any]] = []
        self.compute_proxy: float = 0.0

    def should_retrain(
        self,
        step: int,
        drift_detected: bool = False,
        drift_score: float = 0.0,
        error: float = 0.0,
        current_date: Optional[pd.Timestamp] = None
    ) -> RetrainingDecision:
        """
        Determine if model should be retrained.

        Parameters
        ----------
        step : int
            Current step number.
        drift_detected : bool
            Whether drift was detected.
        drift_score : float
            Current drift score.
        error : float
            Current prediction error.
        current_date : pd.Timestamp
            Current timestamp.

        Returns
        -------
        RetrainingDecision
            Decision with should_retrain flag, reason, and confidence.
        """
        self.policy.state.current_step = step
        self.policy.state.drift_detected = drift_detected
        self.policy.state.current_date = current_date
        
        if drift_score > 0:
            self.policy.state.drift_scores.append(drift_score)
            if len(self.policy.state.drift_scores) > 100:
                self.policy.state.drift_scores = self.policy.state.drift_scores[-100:]
        
        if error > 0:
            self.policy.state.recent_errors.append(error)
            if len(self.policy.state.recent_errors) > 100:
                self.policy.state.recent_errors = self.policy.state.recent_errors[-100:]

        decision = self.policy.should_retrain(self.policy.state)

        if decision.should_retrain:
            self._log_retraining(
                step=step,
                drift_detected=drift_detected,
                drift_score=drift_score,
                error=error,
                current_date=current_date,
                reason=decision.reason
            )

            self.compute_proxy += 1.0
            self.policy.state.total_retrains += 1
            self.policy.state.last_retrain_step = step
            self.policy.state.steps_since_retrain = 0

            if current_date:
                self.policy.state.retraining_timestamps.append(current_date)

        self.policy.state.steps_since_retrain += 1

        return decision

    def _log_retraining(
        self,
        step: int,
        drift_detected: bool,
        drift_score: float,
        error: float,
        current_date: Optional[pd.Timestamp],
        reason: str
    ) -> None:
        """Log a retraining event."""
        event = {
            'step': step,
            'timestamp': current_date,
            'drift_detected': drift_detected,
            'drift_score': drift_score,
            'error': error,
            'reason': reason,
            'total_retrains': self.policy.state.total_retrains
        }
        self.retraining_log.append(event)

    def get_retraining_log(self) -> pd.DataFrame:
        """
        Get retraining log as DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with retraining events.
        """
        if not self.retraining_log:
            return pd.DataFrame()
        return pd.DataFrame(self.retraining_log)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get retraining statistics.

        Returns
        -------
        dict
            Statistics about retraining events.
        """
        log = self.retraining_log

        stats = {
            'total_retrains': len(log),
            'compute_proxy': self.compute_proxy,
            'drift_triggered_retrains': sum(1 for e in log if e.get('drift_detected', False)),
            'total_steps': self.policy.state.current_step + 1 if log else 0
        }
        
        if log:
            stats['retrain_rate'] = len(log) / max(1, stats['total_steps'])
            
            timestamps = [e['timestamp'] for e in log if e.get('timestamp') is not None]
            if len(timestamps) >= 2:
                time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() / 3600
                             for i in range(len(timestamps)-1)]
                stats['avg_hours_between_retrains'] = np.mean(time_diffs)
                stats['median_hours_between_retrains'] = np.median(time_diffs)

        return stats

    def reset(self) -> None:
        """Reset the simulator."""
        self.policy.reset()
        self.retraining_log = []
        self.compute_proxy = 0.0


def create_retraining_policy(config: Dict[str, Any]) -> BaseRetrainingPolicy:
    """
    Create a retraining policy from configuration.

    Parameters
    ----------
    config : dict
        Configuration dictionary with policy parameters.

    Returns
    -------
    BaseRetrainingPolicy
        Configured retraining policy.
    """
    policy_type = config.get('policy', 'fixed_schedule').lower()
    min_gap = config.get('min_steps_between_retrain', 5)

    if policy_type == 'no_retraining' or policy_type == 'none':
        return NoRetrainingPolicy()

    elif policy_type == 'fixed_schedule' or policy_type == 'fixed':
        return FixedSchedulePolicy(
            retrain_interval=config.get('retrain_interval', 1),
            retrain_every_n_steps=config.get('retrain_every_n_steps', None),
            min_steps_between_retrain=config.get('min_steps_between_retrain', 1)
        )

    elif policy_type == 'drift_triggered' or policy_type == 'drift':
        return DriftTriggeredPolicy(
            drift_threshold=config.get('drift_threshold', 0.5),
            min_steps_between_retrain=min_gap,
            require_consecutive=config.get('require_consecutive', 2),
            decay_consecutive=config.get('decay_consecutive', True),
            consecutive_decay_rate=config.get('consecutive_decay_rate', 0.5)
        )

    elif policy_type == 'budget_aware':
        return BudgetAwarePolicy(
            max_retrains_per_year=config.get('max_retrains_per_year', 4),
            initial_budget=config.get('initial_budget', None),
            min_steps_between_retrain=min_gap,
            require_drift=config.get('require_drift', True)
        )

    elif policy_type == 'error_threshold_based' or policy_type == 'error_threshold':
        return ErrorThresholdBasedPolicy(
            error_threshold=config.get('error_threshold', 0.1),
            error_metric=config.get('error_metric', 'mae'),
            window_size=config.get('window_size', 10),
            min_error_exceedences=config.get('min_error_exceedences', 2),
            relative_to_baseline=config.get('relative_to_baseline', True),
            baseline_error=config.get('baseline_error', None),
            min_steps_between_retrain=min_gap
        )

    else:
        raise ValueError(f"Unknown policy type: {policy_type}")
