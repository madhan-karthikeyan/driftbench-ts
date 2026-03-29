"""
Synthetic drift injection for time series benchmarking.

This module provides configurable drift injection for controlled
drift detection and model evaluation experiments.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Literal
from enum import Enum
from dataclasses import dataclass


class DriftType(Enum):
    """Enumeration of supported drift types."""
    SUDDEN_MEAN_SHIFT = "sudden_mean_shift"
    GRADUAL_DRIFT = "gradual_drift"
    VARIANCE_INCREASE = "variance_increase"
    SEASONAL_PHASE_SHIFT = "seasonal_phase_shift"
    CONCEPT_DRIFT = "concept_drift"


@dataclass
class DriftConfig:
    """Configuration for drift injection."""
    enabled: bool = False
    type: str = "sudden_mean_shift"
    drift_start: Optional[str] = None
    drift_end: Optional[str] = None
    magnitude: float = 0.2
    slope: float = 0.01
    features: Optional[List[str]] = None  # None means apply to all
    entity_specific: bool = False  # Apply different drift per entity


class DriftInjector:
    """
    Inject synthetic drift into time series data.

    Supports multiple drift types:
    - sudden_mean_shift: Immediate change in mean
    - gradual_drift: Slowly increasing drift over time
    - variance_increase: Increase in data variance
    - seasonal_phase_shift: Shift in seasonal pattern
    - concept_drift: Change in relationship between features and target

    No data leakage: drift is injected only at inference time,
    training data remains clean.
    """

    def __init__(self, config: Optional[DriftConfig] = None):
        """
        Initialize the drift injector.

        Parameters
        ----------
        config : DriftConfig, optional
            Configuration for drift injection.
        """
        self.config = config or DriftConfig()
        self.drift_type = DriftType(self.config.type)
        self._rng = np.random.default_rng()

    def inject(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        target_col: str = 'target',
        entity_col: str = 'entity_id',
        feature_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Inject drift into the dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Input time series data.
        timestamp_col : str
            Name of timestamp column.
        target_col : str
            Name of target column.
        entity_col : str
            Name of entity identifier column.
        feature_cols : list, optional
            List of feature columns for concept drift.

        Returns
        -------
        pd.DataFrame
            Data with drift injected (in-place modification warning).
        """
        if not self.config.enabled:
            return df.copy()

        df = df.copy()

        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        # Get drift start/end times
        drift_start = pd.to_datetime(self.config.drift_start) if self.config.drift_start else None
        drift_end = pd.to_datetime(self.config.drift_end) if self.config.drift_end else None

        # Determine which rows are in drift period
        in_drift_period = self._get_drift_mask(df, timestamp_col, drift_start, drift_end)

        # Get unique entities
        entities = df[entity_col].unique()

        for entity in entities:
            entity_mask = df[entity_col] == entity
            entity_in_drift = in_drift_period & entity_mask

            if not entity_in_drift.any():
                continue

            # Apply drift based on type
            self._inject_entity_drift(
                df, entity, entity_in_drift, target_col, feature_cols, timestamp_col,
                drift_start, drift_end
            )

        return df

    def _get_drift_mask(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
        drift_start: Optional[pd.Timestamp],
        drift_end: Optional[pd.Timestamp]
    ) -> pd.Series:
        """Create boolean mask for rows in drift period."""
        if drift_start is None:
            # Apply to second half of data
            n = len(df)
            return df.index >= n // 2

        mask = df[timestamp_col] >= drift_start
        if drift_end is not None:
            mask = mask & (df[timestamp_col] < drift_end)
        return mask

    def _inject_entity_drift(
        self,
        df: pd.DataFrame,
        entity: Any,
        drift_mask: pd.Series,
        target_col: str,
        feature_cols: Optional[List[str]],
        timestamp_col: str,
        drift_start: Optional[pd.Timestamp],
        drift_end: Optional[pd.Timestamp]
    ):
        """Apply drift injection to a single entity."""
        entity_idx = df[df['entity_id'] == entity].index
        drift_idx = entity_idx[drift_mask.loc[entity_idx]]

        if len(drift_idx) == 0:
            return

        values = df.loc[drift_idx, target_col].values.copy()

        if self.drift_type == DriftType.SUDDEN_MEAN_SHIFT:
            values = self._apply_sudden_mean_shift(values)

        elif self.drift_type == DriftType.GRADUAL_DRIFT:
            values = self._apply_gradual_drift(
                values, drift_start, drift_end, df.loc[drift_idx, timestamp_col]
            )

        elif self.drift_type == DriftType.VARIANCE_INCREASE:
            values = self._apply_variance_increase(values)

        elif self.drift_type == DriftType.SEASONAL_PHASE_SHIFT:
            values = self._apply_seasonal_phase_shift(
                values, df.loc[drift_idx, timestamp_col]
            )

        elif self.drift_type == DriftType.CONCEPT_DRIFT:
            if feature_cols:
                values = self._apply_concept_drift(
                    df.loc[drift_idx], feature_cols, target_col
                )

        df.loc[drift_idx, target_col] = values

    def _apply_sudden_mean_shift(self, values: np.ndarray) -> np.ndarray:
        """Apply sudden mean shift drift."""
        mean_val = np.mean(values)
        shift = mean_val * self.config.magnitude
        return values + shift

    def _apply_gradual_drift(
        self,
        values: np.ndarray,
        drift_start: Optional[pd.Timestamp],
        drift_end: Optional[pd.Timestamp],
        timestamps: pd.Series
    ) -> np.ndarray:
        """Apply gradual drift over time."""
        result = values.copy()
        n = len(values)

        for i in range(n):
            # Linear increase in drift magnitude
            progress = i / n
            shift = values[i] * self.config.magnitude * self.config.slope * progress
            result[i] = values[i] + shift

        return result

    def _apply_variance_increase(self, values: np.ndarray) -> np.ndarray:
        """Apply variance increase drift."""
        std_val = np.std(values)
        if std_val == 0:
            return values

        # Scale factor for variance increase
        scale = 1 + self.config.magnitude
        return (values - np.mean(values)) * scale + np.mean(values)

    def _apply_seasonal_phase_shift(
        self,
        values: np.ndarray,
        timestamps: pd.Series
    ) -> np.ndarray:
        """Apply seasonal phase shift drift."""
        # Assume hourly data with daily seasonality
        hours = timestamps.dt.hour.values

        # Shift phase by magnitude (in hours)
        shift_hours = int(self.config.magnitude * 12)  # Max 12 hour shift
        shifted_hours = (hours + shift_hours) % 24

        # Interpolate based on shifted hours (simplified approach)
        result = values.copy()

        # Simple phase shift: swap values based on shifted time
        # This is a simplified version; real implementation would use
        # proper time series decomposition
        for i in range(len(values)):
            target_hour = shifted_hours[i]
            # Find similar hour in nearby window
            hour_diffs = np.abs(hours - target_hour)
            nearest_idx = np.argmin(hour_diffs)
            result[i] = values[nearest_idx]

        return result

    def _apply_concept_drift(
        self,
        entity_df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str
    ) -> np.ndarray:
        """Apply concept drift (change in feature-target relationship)."""
        # This simulates a change in the relationship between features and target
        # Simple approach: add interaction terms with drift magnitude
        values = entity_df[target_col].values.copy()

        if not feature_cols or len(feature_cols) == 0:
            return values

        # Create a simple linear combination with modified weights
        # In real concept drift, the underlying function changes
        base_value = entity_df[feature_cols[0]].values if feature_cols else values
        mean_val = np.mean(base_value)

        # Modify the target generation process
        shift = mean_val * self.config.magnitude
        noise = np.std(values) * 0.1 * self._rng.standard_normal(len(values))

        return values + shift + noise

    def get_drift_period_info(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp'
    ) -> Dict[str, Any]:
        """
        Get information about drift period in the data.

        Parameters
        ----------
        df : pd.DataFrame
            Data to check.
        timestamp_col : str
            Name of timestamp column.

        Returns
        -------
        dict
            Information about drift period.
        """
        if not self.config.enabled:
            return {"drift_enabled": False}

        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df = df.copy()
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        drift_start = pd.to_datetime(self.config.drift_start) if self.config.drift_start else None
        drift_end = pd.to_datetime(self.config.drift_end) if self.config.drift_end else None

        mask = self._get_drift_mask(df, timestamp_col, drift_start, drift_end)

        return {
            "drift_enabled": True,
            "drift_type": self.config.type,
            "drift_start": drift_start,
            "drift_end": drift_end,
            "drift_period_rows": mask.sum(),
            "total_rows": len(df),
            "drift_fraction": mask.sum() / len(df)
        }


def create_drift_injector(config_dict: Dict[str, Any]) -> DriftInjector:
    """
    Create a DriftInjector from a configuration dictionary.

    Parameters
    ----------
    config_dict : dict
        Configuration dictionary with drift parameters.

    Returns
    -------
    DriftInjector
        Configured drift injector.
    """
    if not config_dict.get('enabled', False):
        return DriftInjector(DriftConfig(enabled=False))

    drift_config = DriftConfig(
        enabled=config_dict.get('enabled', False),
        type=config_dict.get('type', 'sudden_mean_shift'),
        drift_start=config_dict.get('drift_start', None),
        drift_end=config_dict.get('drift_end', None),
        magnitude=config_dict.get('magnitude', 0.2),
        slope=config_dict.get('slope', 0.01),
        features=config_dict.get('features', None),
        entity_specific=config_dict.get('entity_specific', False)
    )

    return DriftInjector(drift_config)
