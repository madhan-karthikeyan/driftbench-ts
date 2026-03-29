"""
Simulator module.

This module provides rolling window simulation with drift detection
and adaptive retraining capabilities.
"""

from driftbench.simulator.rolling import RollingSimulator

from driftbench.simulator.advanced_rolling import (
    AdvancedRollingSimulator,
    create_advanced_simulator
)

from driftbench.simulator.drift_injection import (
    DriftInjector,
    DriftConfig,
    DriftType,
    create_drift_injector
)

from driftbench.simulator.retraining import (
    BaseRetrainingPolicy,
    FixedSchedulePolicy,
    DriftTriggeredPolicy,
    BudgetAwarePolicy,
    ErrorThresholdBasedPolicy,
    RetrainingSimulator,
    RetrainingState,
    RetrainingPolicyType,
    create_retraining_policy
)

__all__ = [
    'RollingSimulator',
    'AdvancedRollingSimulator',
    'create_advanced_simulator',
    'DriftInjector',
    'DriftConfig',
    'DriftType',
    'create_drift_injector',
    'BaseRetrainingPolicy',
    'FixedSchedulePolicy',
    'DriftTriggeredPolicy',
    'BudgetAwarePolicy',
    'ErrorThresholdBasedPolicy',
    'RetrainingSimulator',
    'RetrainingState',
    'RetrainingPolicyType',
    'create_retraining_policy',
]
